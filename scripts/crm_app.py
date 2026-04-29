"""
crm_app.py — Loan Propensity CRM Application
=============================================
Flask web application. Run: python crm_app.py
Opens automatically in your browser at http://localhost:5000

Three pages:
  /           → Page 1: Client CRM (360-degree client view)
  /analytics  → Page 2: Analytics Dashboard (portfolio-level insights)
  /prediction → Page 3: Prediction & Scoring (model results + threshold simulator)

Package as executable:
  pip install pyinstaller
  pyinstaller --onefile --add-data "data:data" crm_app.py
"""

import os, json, pickle, webbrowser, threading
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template_string, request

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "../data"
PORT     = 5000

app = Flask(__name__)

def load_data():
    d = {}
    d["clients"]    = pd.read_csv(f"{DATA_DIR}/clients.csv")
    d["tx"]         = pd.read_csv(f"{DATA_DIR}/transactions.csv")
    d["app"]        = pd.read_csv(f"{DATA_DIR}/application.csv")
    d["credit"]     = pd.read_csv(f"{DATA_DIR}/credit.csv")
    d["payments"]   = pd.read_csv(f"{DATA_DIR}/payments.csv")
    d["scores_cs"]  = pd.read_csv(f"{DATA_DIR}/credit_scores.csv")
    d["scores_ml"]  = pd.read_csv(f"{DATA_DIR}/model_scores.csv")
    d["features"]   = pd.read_csv(f"{DATA_DIR}/features.csv")
    d["importance"] = pd.read_csv(f"{DATA_DIR}/feature_importance.csv")
    d["tx"]["dt"]   = pd.to_datetime(d["tx"]["transaction_date"])
    d["app"]["dt"]  = pd.to_datetime(d["app"]["timestamp"])
    return d

DATA = load_data()

# ── Load SHAP values from model.pkl ──────────────────────────────────────────
SHAP_DATA = None
try:
    with open(f"{DATA_DIR}/model.pkl", "rb") as _pf:
        _pkg = pickle.load(_pf)
        if "shap_values" in _pkg and "expected_value" in _pkg:
            SHAP_DATA = {
                "values":         np.array(_pkg["shap_values"]),
                "expected_value": float(_pkg["expected_value"]),
                "features":       _pkg.get("features", []),
            }
            print(f"[SHAP] Loaded: {SHAP_DATA['values'].shape} values, "
                  f"{len(SHAP_DATA['features'])} features")
        else:
            print("[SHAP] model.pkl found but no SHAP data — retrain model.py")
except Exception as _e:
    print(f"[SHAP] Could not load: {_e}")

MCC_NAMES = {5211:"Repair",1021:"Electronics",5680:"Clothing",3001:"Travel",
             5411:"Supermarket",5812:"Restaurant",5912:"Pharmacy",6011:"ATM/Cash"}

TIER_COLORS = {"High":"#e74c3c","Medium":"#e67e22","Low":"#3498db","Very Low":"#2ecc71"}
PHASE_COLORS = {"Phase 1 Normal":"#3498db","Phase 2 Spike":"#e74c3c",
                "Phase 3 Repaying":"#2ecc71","Phase 4 Target":"#9b59b6"}

def phase_of(dt):
    if dt < pd.Timestamp("2024-07-01"): return "Phase 1 Normal"
    if dt < pd.Timestamp("2024-10-01"): return "Phase 2 Spike"
    if dt < pd.Timestamp("2025-10-01"): return "Phase 3 Repaying"
    return "Phase 4 Target"

# ─────────────────────────────────────────────────────────────────────────────
# SHARED HTML TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

BASE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Loan Propensity CRM</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
<style>
  :root{--primary:#2c3e50;--accent:#3498db;--danger:#e74c3c}
  body{background:#f0f2f5;font-family:'Segoe UI',sans-serif}
  .navbar{background:var(--primary)!important}
  .navbar-brand{font-weight:700;font-size:1.2rem;letter-spacing:.5px}
  .nav-link{color:rgba(255,255,255,.75)!important;font-weight:500}
  .nav-link.active,.nav-link:hover{color:#fff!important}
  .card{border:none;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,.08)}
  .card-header{background:var(--primary);color:#fff;border-radius:12px 12px 0 0!important;font-weight:600}
  .badge-high{background:#e74c3c;color:#fff;padding:4px 10px;border-radius:20px;font-size:.8rem}
  .badge-medium{background:#e67e22;color:#fff;padding:4px 10px;border-radius:20px;font-size:.8rem}
  .badge-low{background:#3498db;color:#fff;padding:4px 10px;border-radius:20px;font-size:.8rem}
  .badge-verylow{background:#2ecc71;color:#fff;padding:4px 10px;border-radius:20px;font-size:.8rem}
  .stat-card{border-radius:12px;padding:20px;color:#fff;margin-bottom:16px}
  .stat-card h2{font-size:2rem;font-weight:700;margin:0}
  .stat-card p{margin:0;opacity:.85;font-size:.9rem}
  .chart-container{background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,.06);margin-bottom:20px}
  .score-gauge{text-align:center;padding:10px}
  .client-row{cursor:pointer;transition:background .15s}
  .client-row:hover{background:#f8f9fa}
  .client-row.selected{background:#e8f4fd}
  #threshold-val{font-size:1.4rem;font-weight:700;color:var(--accent)}
  .table th{background:#f8f9fa;font-weight:600;font-size:.85rem}
  .outcome-tp{color:#27ae60;font-weight:600}
  .outcome-tn{color:#3498db;font-weight:600}
  .outcome-fp{color:#e67e22;font-weight:600}
  .outcome-fn{color:#e74c3c;font-weight:600}
  .scrollable{max-height:420px;overflow-y:auto}
  .phase-badge{font-size:.7rem;padding:2px 6px;border-radius:10px;color:#fff}
  .spinner-container{display:flex;justify-content:center;padding:40px}
</style>
</head>
<body>
<nav class="navbar navbar-expand-lg navbar-dark">
  <div class="container-fluid px-4">
    <a class="navbar-brand" href="/">🏦 Loan Propensity CRM</a>
    <div class="collapse navbar-collapse">
      <ul class="navbar-nav ms-auto">
        <li class="nav-item"><a class="nav-link {a1}" href="/">👤 Client CRM</a></li>
        <li class="nav-item"><a class="nav-link {a2}" href="/analytics">📊 Analytics</a></li>
        <li class="nav-item"><a class="nav-link {a3}" href="/prediction">🤖 Prediction</a></li>
      </ul>
    </div>
  </div>
</nav>
{content}
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body></html>"""

def page(content, active=1):
    a = ["","",""]
    a[active-1] = "active"
    return (BASE_HTML
        .replace("{content}", content)
        .replace("{a1}", a[0])
        .replace("{a2}", a[1])
        .replace("{a3}", a[2]))

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — CLIENT CRM
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def client_crm():
    content = """
<div class="container-fluid px-4 py-4">
<div class="row">
  <!-- LEFT: Client List -->
  <div class="col-md-3">
    <div class="card mb-3">
      <div class="card-header">👥 Clients (120)</div>
      <div class="card-body p-2">
        <input class="form-control form-control-sm mb-2" id="searchBox"
               placeholder="🔍 Search by name..." onkeyup="filterClients()">
        <!-- Quick filter buttons -->
        <div class="d-flex gap-1 mb-2">
          <button class="btn btn-sm btn-outline-secondary flex-fill active" id="btnAll"
                  onclick="quickFilter('all')">All</button>
          <button class="btn btn-sm btn-outline-danger flex-fill" id="btnHigh"
                  onclick="quickFilter('high')">🔴 High</button>
          <button class="btn btn-sm btn-outline-primary flex-fill" id="btnTop"
                  onclick="quickFilter('top20')">Top 20</button>
        </div>
        <div class="scrollable" id="clientList" style="max-height:75vh"></div>
      </div>
    </div>
  </div>
  <!-- RIGHT: Client Detail -->
  <div class="col-md-9" id="clientDetail">
    <div class="d-flex align-items-center justify-content-center" style="height:60vh">
      <div class="text-center text-muted">
        <div style="font-size:4rem">👈</div>
        <p class="mt-2">Select a client from the list to view their profile</p>
      </div>
    </div>
  </div>
</div>
</div>

<script>
let allClients = [];
let selectedId = null;

fetch('/api/clients').then(r=>r.json()).then(data=>{
  allClients = data;
  renderList(data);
});

function renderList(clients){
  const list = document.getElementById('clientList');
  list.innerHTML = clients.map(c=>`
    <div class="client-row p-2 border-bottom ${c.client_id==selectedId?'selected':''}"
         onclick="loadClient(${c.client_id})" id="row-${c.client_id}">
      <div class="d-flex justify-content-between align-items-center">
        <span class="fw-semibold" style="font-size:.9rem">${c.first_name} ${c.last_name}</span>
        <span class="badge-${c.risk_tier.toLowerCase().replace(' ','')}">${c.risk_tier}</span>
      </div>
      <div class="text-muted" style="font-size:.75rem">
        ${(c.oof_probability*100).toFixed(0)}% propensity &bull; ID ${c.client_id}
      </div>
    </div>`).join('');
}

function filterClients(){
  const q = document.getElementById('searchBox').value.toLowerCase();
  renderList(allClients.filter(c=>(c.first_name+' '+c.last_name).toLowerCase().includes(q)));
}

function quickFilter(mode){
  ['all','high','top20'].forEach(m=>{
    document.getElementById('btn'+m.charAt(0).toUpperCase()+m.slice(1))?.classList.remove('active');
  });
  document.getElementById(mode==='all'?'btnAll':mode==='high'?'btnHigh':'btnTop')?.classList.add('active');
  if(mode==='all') renderList(allClients);
  else if(mode==='high') renderList(allClients.filter(c=>c.risk_tier==='High'));
  else renderList(allClients.slice(0,20));
}

function loadClient(id){
  selectedId = id;
  document.querySelectorAll('.client-row').forEach(r=>r.classList.remove('selected'));
  const row = document.getElementById('row-'+id);
  if(row) row.classList.add('selected');
  document.getElementById('clientDetail').innerHTML =
    '<div class="spinner-container"><div class="spinner-border text-primary"></div></div>';
  fetch('/api/client/'+id).then(r=>r.json()).then(renderClient);
}

function tierBadge(tier){
  return `<span class="badge-${tier.toLowerCase().replace(' ','')}">${tier}</span>`;
}

function renderClient(d){
  const c = d.client, sc = d.credit_score, ml = d.model_score, cr = d.credit, pay = d.payments;
  const outcomeColor = {'TP':'#27ae60','TN':'#3498db','FP':'#e67e22','FN':'#e74c3c'}[ml.outcome]||'#666';

  document.getElementById('clientDetail').innerHTML = `
  <!-- Header bar -->
  <div class="d-flex align-items-center mb-3 p-3 bg-white rounded-3 shadow-sm">
    <div style="width:56px;height:56px;border-radius:50%;background:var(--primary);
                display:flex;align-items:center;justify-content:center;color:#fff;font-size:1.4rem;margin-right:16px">
      ${c.first_name[0]+c.last_name[0]}
    </div>
    <div class="flex-grow-1">
      <h4 class="mb-0">${c.first_name} ${c.last_name}</h4>
      <small class="text-muted">${c.employment_type} &bull; ${c.city} &bull; Age ${c.age}</small>
    </div>
    <div class="text-center me-3">
      <div class="fw-bold" style="font-size:1.6rem;color:${ml.oof_probability>=.5?'#e74c3c':'#27ae60'}">
        ${(ml.oof_probability*100).toFixed(1)}%
      </div>
      <small>Loan Propensity</small>
    </div>
    <div>${tierBadge(ml.risk_tier)}</div>
    <!-- Urgency level -->
    <div class="text-center ms-3">
      <div class="fw-bold" style="font-size:.95rem;color:${
        ml.oof_probability>=.65?'#e74c3c':ml.oof_probability>=.40?'#e67e22':'#27ae60'}">
        ${ml.oof_probability>=.65?'🔴 High (0–7 days)':ml.oof_probability>=.40?'🟡 Medium (7–30 days)':'🟢 Low'}
      </div>
      <small class="text-muted">Est. ${ml.oof_probability>=.65?'~5 days':ml.oof_probability>=.40?'~18 days':'> 30 days'}</small>
    </div>
  </div>

  <!-- Recommended Action + Top Reasons -->
  <div class="row g-3 mb-3">
    <div class="col-md-4">
      <div class="card h-100 text-center">
        <div class="card-header">⚡ Recommended Action</div>
        <div class="card-body d-flex flex-column align-items-center justify-content-center">
          <div style="font-size:1.3rem;font-weight:700;color:${
            ml.oof_probability>=.65?'#27ae60':ml.oof_probability>=.40?'#e67e22':'#e74c3c'}">
            ${ml.oof_probability>=.65?'✅ Offer Credit Now':ml.oof_probability>=.40?'🟡 Monitor (7–14 days)':'❌ Do Not Offer'}
          </div>
          <small class="text-muted mt-1">OOF: ${(ml.oof_probability*100).toFixed(1)}%  ·  Final: ${(ml.final_probability*100).toFixed(1)}%</small>
        </div>
      </div>
    </div>
    <div class="col-md-8">
      <div class="card h-100">
        <div class="card-header">🔎 Top Reasons</div>
        <div class="card-body" id="topReasons">
          <div class="text-muted text-center py-2">Loading…</div>
        </div>
      </div>
    </div>
  </div>

  <!-- SHAP Waterfall -->
  <div class="chart-container mb-3">
    <h6 class="text-muted mb-1">🧠 SHAP Explanation — why this prediction?</h6>
    <div id="shapWaterfall" style="height:280px"></div>
  </div>

  <!-- Local Feature Impact chips -->
  <div class="card mb-3">
    <div class="card-header">📊 Local Feature Impact</div>
    <div class="card-body d-flex flex-wrap gap-2" id="localImpact">
      <span class="text-muted">Loading…</span>
    </div>
  </div>

  <div class="row g-3 mb-3">
    <!-- Client info -->
    <div class="col-md-4">
      <div class="card h-100">
        <div class="card-header">📋 Client Profile</div>
        <div class="card-body">
          ${[['Salary', (c.monthly_salary||0).toLocaleString()+' AMD/mo'],
             ['Bank Account', c.bank_account],
             ['Account Since', c.account_open_date],
             ['Marital Status', c.marital_status],
             ['Dependants', c.dependants],
             ['Phone', c.phone]].map(([k,v])=>
            `<div class="d-flex justify-content-between border-bottom py-1">
               <span class="text-muted" style="font-size:.85rem">${k}</span>
               <span style="font-size:.85rem;font-weight:500">${v}</span>
             </div>`).join('')}
        </div>
      </div>
    </div>

    <!-- Credit Score Gauge -->
    <div class="col-md-4">
      <div class="card h-100">
        <div class="card-header">📈 Credit Score</div>
        <div class="card-body p-2">
          <div id="gaugeChart" style="height:160px"></div>
          <div class="row text-center mt-1">
            ${[['Payment Hist.', sc.payment_history_pts],
               ['Utilization', sc.credit_utilization_pts],
               ['Inquiries', sc.credit_inquiries_pts],
               ['DTI', sc.dti_pts],
               ['Relationship', sc.relationship_pts]].map(([k,v])=>
              `<div class="col" style="font-size:.7rem">
                 <div class="fw-bold">${(v||0).toFixed(0)}</div>
                 <div class="text-muted">${k}</div>
               </div>`).join('')}
          </div>
        </div>
      </div>
    </div>

    <!-- Model Score -->
    <div class="col-md-4">
      <div class="card h-100">
        <div class="card-header">🤖 Model Assessment</div>
        <div class="card-body">
          ${[['OOF Probability', (ml.oof_probability*100).toFixed(2)+'%'],
             ['Final Probability', (ml.final_probability*100).toFixed(2)+'%'],
             ['Predicted Label', ml.predicted_label==1?'Will Seek Loan':'No Loan'],
             ['Actual Label', ml.actual_label==1?'Sought Loan ✓':'No Loan'],
             ['Outcome', `<span style="color:${outcomeColor};font-weight:600">${ml.outcome}</span>`],
             ['Risk Tier', tierBadge(ml.risk_tier)]].map(([k,v])=>
            `<div class="d-flex justify-content-between border-bottom py-1">
               <span class="text-muted" style="font-size:.85rem">${k}</span>
               <span style="font-size:.85rem">${v}</span>
             </div>`).join('')}
        </div>
      </div>
    </div>
  </div>

  <div class="row g-3 mb-3">
    <!-- Balance timeline -->
    <div class="col-md-8">
      <div class="chart-container">
        <h6 class="text-muted mb-2">💰 Balance Over Time (all phases)</h6>
        <div id="balanceChart" style="height:240px"></div>
      </div>
    </div>
    <!-- Spending donut -->
    <div class="col-md-4">
      <div class="chart-container">
        <h6 class="text-muted mb-2">🍩 Spend by Category</h6>
        <div id="donutChart" style="height:240px"></div>
      </div>
    </div>
  </div>

  <div class="row g-3 mb-3">
    <!-- App activity -->
    <div class="col-md-4">
      <div class="card">
        <div class="card-header">📱 App Activity</div>
        <div class="card-body">
          ${[['Total Sessions', d.app.total_sessions],
             ['Balance Checks', d.app.balance_checks],
             ['Check Ratio', (d.app.check_balance_ratio*100).toFixed(1)+'%'],
             ['Active Days', d.app.active_days],
             ['Sessions Last 3m', d.app.sessions_last_3m]].map(([k,v])=>
            `<div class="d-flex justify-content-between border-bottom py-1">
               <span class="text-muted" style="font-size:.85rem">${k}</span>
               <span class="fw-semibold" style="font-size:.85rem">${v}</span>
             </div>`).join('')}
        </div>
      </div>
    </div>
    <!-- Credit history -->
    <div class="col-md-8">
      <div class="card">
        <div class="card-header">🏦 Credit History</div>
        <div class="card-body">
          ${cr ? `
            <div class="row mb-2">
              ${[['Amount', (cr.credit_amount||0).toLocaleString()+' AMD'],
                 ['Rate', cr.annual_rate_pct+'%'],
                 ['Term', cr.term_months+' months'],
                 ['Status', cr.status],
                 ['Purpose', cr.purpose]].map(([k,v])=>
                `<div class="col-4 border-bottom py-1">
                   <small class="text-muted d-block">${k}</small>
                   <span class="fw-semibold" style="font-size:.85rem">${v}</span>
                 </div>`).join('')}
            </div>
            <h6 class="text-muted mt-2 mb-1" style="font-size:.8rem">PAYMENT SCHEDULE</h6>
            <div class="scrollable" style="max-height:200px">
              <table class="table table-sm table-hover mb-0" style="font-size:.8rem">
                <thead><tr><th>#</th><th>Due</th><th>Paid</th><th>Days Late</th><th>Status</th></tr></thead>
                <tbody>
                  ${pay.map(p=>{
                    const sc = {'PAID_ON_TIME':'#27ae60','PAID_LATE':'#e67e22',
                                'MISSED':'#e74c3c','OVERDUE':'#c0392b','SCHEDULED':'#95a5a6'}[p.status]||'#666';
                    return `<tr>
                      <td>${p.payment_number}</td>
                      <td>${p.due_date||''}</td>
                      <td>${p.payment_date||'—'}</td>
                      <td>${p.days_late!==null&&p.days_late!==undefined?p.days_late+'d':'—'}</td>
                      <td style="color:${sc};font-weight:600">${p.status}</td>
                    </tr>`;}).join('')}
                </tbody>
              </table>
            </div>` :
            '<p class="text-muted text-center py-3">No credit record for this client</p>'}
        </div>
      </div>
    </div>
  </div>

  <!-- Transaction history -->
  <div class="chart-container">
    <div class="d-flex justify-content-between align-items-center mb-2">
      <h6 class="text-muted mb-0">📄 Transaction History (${d.tx.length} transactions)</h6>
      <select class="form-select form-select-sm w-auto" id="phaseFilter" onchange="filterTx()">
        <option value="">All Phases</option>
        <option>Phase 1 Normal</option>
        <option>Phase 2 Spike</option>
        <option>Phase 3 Repaying</option>
        <option>Phase 4 Target</option>
      </select>
    </div>
    <div class="scrollable" style="max-height:280px">
      <table class="table table-sm table-hover mb-0" id="txTable" style="font-size:.8rem">
        <thead><tr><th>Date</th><th>Merchant</th><th>Category</th>
                   <th class="text-end">Amount</th><th>Status</th><th>Balance</th></tr></thead>
        <tbody id="txBody"></tbody>
      </table>
    </div>
  </div>`;

  // Charts
  const balData = d.balance_chart;
  const phases = [...new Set(balData.map(r=>r.phase))];
  const phaseOrder = ['Phase 1 Normal','Phase 2 Spike','Phase 3 Repaying','Phase 4 Target'];
  const colors = {'Phase 1 Normal':'#3498db','Phase 2 Spike':'#e74c3c',
                  'Phase 3 Repaying':'#2ecc71','Phase 4 Target':'#9b59b6'};
  const traces = phaseOrder.filter(p=>phases.includes(p)).map(p=>{
    const pts = balData.filter(r=>r.phase===p);
    return {x:pts.map(r=>r.date), y:pts.map(r=>r.balance),
            mode:'lines+markers', name:p, line:{color:colors[p],width:2},
            marker:{size:4}};
  });
  Plotly.newPlot('balanceChart', traces,
    {margin:{t:10,r:10,b:30,l:60}, showlegend:true,
     legend:{orientation:'h',y:-0.25}, yaxis:{tickformat:',d'}},
    {responsive:true,displayModeBar:false});

  const spendData = d.spend_donut;
  Plotly.newPlot('donutChart',
    [{type:'pie', labels:spendData.map(r=>r.category),
      values:spendData.map(r=>r.amount), hole:.45,
      marker:{colors:['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22','#95a5a6']}}],
    {margin:{t:10,r:10,b:10,l:10}, showlegend:true,
     legend:{orientation:'h',y:-0.1,font:{size:10}}},
    {responsive:true,displayModeBar:false});

  Plotly.newPlot('gaugeChart',
    [{type:'indicator', mode:'gauge+number',
      value:sc.credit_score,
      gauge:{axis:{range:[300,850]},
             bar:{color:sc.credit_score>=750?'#27ae60':sc.credit_score>=670?'#3498db':
                       sc.credit_score>=580?'#e67e22':'#e74c3c'},
             steps:[{range:[300,500],color:'#fadbd8'},{range:[500,670],color:'#fdebd0'},
                    {range:[670,750],color:'#d5f5e3'},{range:[750,850],color:'#a9dfbf'}]},
      number:{suffix:'', font:{size:28}}}],
    {margin:{t:5,r:20,b:5,l:20}, height:160},
    {responsive:true,displayModeBar:false});

  // Transaction table
  window._txAll = d.tx;
  filterTx();

  // ── Top Reasons ──────────────────────────────────────────────────────────
  const fr = d.features || {};
  const reasons = [];
  const burnAcc = fr.burn_acceleration||1;
  const balDrop = fr.balance_drop_pct||0;
  const chkSpk  = fr.balance_check_spike||1;
  const spkMax  = Math.max(fr.spike_repair||0,fr.spike_electronics||0,fr.spike_travel||0,fr.spike_clothing||0);
  const zeroR   = fr.zero_day_rate||0;
  const crScore = fr.credit_score||600;
  const chkR    = fr.check_ratio_last_3m||0;
  if(burnAcc>=1.3) reasons.push({sign:'+',txt:`Spend increased recently (×${burnAcc.toFixed(1)})`,col:'#27ae60'});
  else if(burnAcc<0.8) reasons.push({sign:'–',txt:'Spending slowed down',col:'#95a5a6'});
  if(balDrop>=0.2) reasons.push({sign:'+',txt:`Balance declining (${(balDrop*100).toFixed(0)}% drop)`,col:'#e67e22'});
  if(chkSpk>=1.4) reasons.push({sign:'+',txt:`App activity spiked (×${chkSpk.toFixed(1)} checks)`,col:'#27ae60'});
  if(spkMax>=2.0) reasons.push({sign:'+',txt:`Category spike detected (×${spkMax.toFixed(1)})`,col:'#27ae60'});
  if(zeroR>=0.08) reasons.push({sign:'–',txt:`Near-zero balance events (${(zeroR*100).toFixed(0)}%)`,col:'#e74c3c'});
  if(crScore<580) reasons.push({sign:'–',txt:`Credit score low (${Math.round(crScore)})`,col:'#e74c3c'});
  else if(crScore>=700) reasons.push({sign:'+',txt:`Good credit history (${Math.round(crScore)})`,col:'#27ae60'});
  if(chkR>=0.5) reasons.push({sign:'+',txt:`High check ratio (3m) ${(chkR*100).toFixed(0)}%`,col:'#e67e22'});
  const rBox = reasons.length?reasons.slice(0,5):
    [{sign:'·',txt:'No strong signal available',col:'#95a5a6'}];
  document.getElementById('topReasons').innerHTML = rBox.map(r=>
    `<div class="d-flex align-items-center gap-2 py-1 border-bottom">
       <span style="font-weight:700;color:${r.col};min-width:16px">${r.sign}</span>
       <span style="font-size:.87rem">${r.txt}</span> 
     </div>`).join('');

  // ── SHAP Waterfall ───────────────────────────────────────────────────────
  if(d.shap && d.shap.values && d.shap.values.length>0){
    const sv = d.shap.values;
    const fn = d.shap.feature_names;
    const ev = d.shap.expected_value;
    const top=12, pairs=fn.map((n,i)=>({n,v:sv[i]}))
      .sort((a,b)=>Math.abs(b.v)-Math.abs(a.v)).slice(0,top);
    const basePct = (1/(1+Math.exp(-ev))*100).toFixed(1);
    const finalPct= (ml.final_probability*100).toFixed(1);
    Plotly.newPlot('shapWaterfall',[{
      type:'bar', orientation:'h',
      x: pairs.map(p=>p.v),
      y: pairs.map(p=>p.n),
      marker:{color:pairs.map(p=>p.v>0?'#e74c3c':'#3498db'),opacity:.85},
      text: pairs.map(p=>(p.v>0?'+':'')+p.v.toFixed(4)),
      textposition:'outside', textfont:{size:9},
      hovertemplate:'<b>%{y}</b><br>SHAP: %{x:.4f}<br>%{customdata}<extra></extra>',
      customdata: pairs.map(p=>p.v>0?'↑ increases loan probability':'↓ decreases loan probability'),
    }],{
      margin:{t:30,r:80,b:40,l:160},
      xaxis:{title:'SHAP value (contribution to log-odds of loan)',zeroline:true,zerolinecolor:'#aaa'},
      title:{text:`Base rate: ${basePct}%  →  Final: ${finalPct}%  ${parseFloat(finalPct)>=50?'🔴 Loan likely':'🟢 No loan likely'}`,
             font:{size:11},x:0.5},
    },{responsive:true,displayModeBar:false});
  } else {
    document.getElementById('shapWaterfall').innerHTML =
      '<div class="text-muted text-center py-4">SHAP data not available — retrain model with updated model.py</div>';
  }

  // ── Local Feature Impact chips ───────────────────────────────────────────
  const impItems=[
    {n:'check_ratio_last_3m', v:fr.check_ratio_last_3m||0,   w:.30, hi:true},
    {n:'balance_check_spike', v:fr.balance_check_spike||1,    w:.20, hi:true},
    {n:'spike (max cat.)',    v:spkMax,                        w:.15, hi:true},
    {n:'urgency_score',       v:fr.urgency_score||0,           w:.12, hi:true},
    {n:'failed_tx_rate',      v:fr.failed_tx_rate||0,          w:.10, hi:false},
    {n:'balance_drop_pct',    v:fr.balance_drop_pct||0,        w:.08, hi:true},
  ].map(item=>{
    const centred = item.v<=1?(item.v-0.5):(item.v-3)/6;
    let imp = Math.round(centred*item.w*1000)/1000;
    if(!item.hi) imp=-imp;
    return {n:item.n, imp};
  }).sort((a,b)=>Math.abs(b.imp)-Math.abs(a.imp));
  document.getElementById('localImpact').innerHTML = impItems.map(it=>{
    const col=it.imp>=0?'#27ae60':'#e74c3c';
    const bg=it.imp>=0?'#eafaf1':'#fdecea';
    const sign=it.imp>=0?'+':'–';
    return `<span class="px-2 py-1 rounded" style="background:${bg};border:1px solid ${col};font-size:.8rem">
      <span style="color:${col};font-weight:700">${sign}${Math.abs(it.imp).toFixed(3)}</span>
      <span class="text-muted ms-1">${it.n}</span>
    </span>`;
  }).join('');
}
function filterTx(){
  const phase = document.getElementById('phaseFilter')?.value||'';
  const rows = (window._txAll||[]).filter(r=>!phase||r.phase===phase);
  const sc = {'SUCCESS':'#27ae60','FAILED':'#e74c3c'};
  document.getElementById('txBody').innerHTML = rows.slice(0,200).map(r=>`
    <tr>
      <td>${r.transaction_date||''}</td>
      <td>${r.merchant_name||''}</td>
      <td><small>${r.category||''}</small></td>
      <td class="text-end">${(r.amount||0).toLocaleString()}</td>
      <td style="color:${sc[r.status]||'#666'};font-weight:600">${r.status}</td>
      <td class="text-end">${(r.balance||0).toLocaleString()}</td>
    </tr>`).join('');
}
</script>"""
    return page(content, active=1)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — ANALYTICS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/analytics")
def analytics():
    content = """
<div class="container-fluid px-4 py-4">
  <!-- KPI row -->
  <div class="row g-3 mb-4" id="kpiRow">
    <div class="col-md-3"><div class="stat-card" style="background:#2c3e50">
      <p>Total Clients</p><h2 id="kpi1">—</h2></div></div>
    <div class="col-md-3"><div class="stat-card" style="background:#e74c3c">
      <p>High Risk</p><h2 id="kpi2">—</h2></div></div>
    <div class="col-md-3"><div class="stat-card" style="background:#3498db">
      <p>Loans Issued</p><h2 id="kpi3">—</h2></div></div>
    <div class="col-md-3"><div class="stat-card" style="background:#27ae60">
      <p>Avg Credit Score</p><h2 id="kpi4">—</h2></div></div>
  </div>
  <div class="row g-3 mb-3">
    <div class="col-md-6">
      <div class="chart-container">
        <h6 class="text-muted">Balance Drop % Distribution (by label)</h6>
        <div id="balDropChart" style="height:260px"></div>
      </div>
    </div>
    <div class="col-md-6">
      <div class="chart-container">
        <h6 class="text-muted">Burn Acceleration vs Burn Rate Ratio (by label)</h6>
        <div id="burnChart" style="height:260px"></div>
      </div>
    </div>
  </div>
  <div class="row g-3 mb-3">
    <div class="col-md-6">
      <div class="chart-container">
        <h6 class="text-muted">Category Spike Ratios by Label (avg)</h6>
        <div id="spikeChart" style="height:260px"></div>
      </div>
    </div>
    <div class="col-md-6">
      <div class="chart-container">
        <h6 class="text-muted">Urgency Score Distribution (label 0 vs 1)</h6>
        <div id="urgencyChart" style="height:260px"></div>
      </div>
    </div>
  </div>
  <div class="row g-3 mb-3">
    <div class="col-md-6">
      <div class="chart-container">
        <h6 class="text-muted">Balance-Check Ratio: Phase 3 Late vs Historical</h6>
        <div id="checkSpikeChart" style="height:260px"></div>
      </div>
    </div>
    <div class="col-md-6">
      <div class="chart-container">
        <h6 class="text-muted">Credit Health — Payment Status Breakdown</h6>
        <div id="payHealthChart" style="height:260px"></div>
      </div>
    </div>
  </div>
  <div class="row g-3">
    <div class="col-12">
      <div class="chart-container">
        <h6 class="text-muted">Phase Transition: Balance-Check Frequency (avg by phase, label 0 vs 1)</h6>
        <div id="phaseChart" style="height:260px"></div>
      </div>
    </div>
  </div>
</div>

<script>
fetch('/api/analytics').then(r=>r.json()).then(d=>{
  document.getElementById('kpi1').textContent = d.kpi.total;
  document.getElementById('kpi2').textContent = d.kpi.high_risk;
  document.getElementById('kpi3').textContent = d.kpi.loans;
  document.getElementById('kpi4').textContent = d.kpi.avg_score;

  // Balance drop histogram
  Plotly.newPlot('balDropChart',[
    {x:d.bal_drop[0],type:'histogram',name:'No Loan',opacity:.65,
     marker:{color:'#3498db'},nbinsx:15},
    {x:d.bal_drop[1],type:'histogram',name:'Loan',opacity:.65,
     marker:{color:'#e74c3c'},nbinsx:15}],
    {barmode:'overlay',margin:{t:10,r:10,b:40,l:50},
     xaxis:{title:'Balance Drop %'},yaxis:{title:'Count'},
     legend:{x:.75,y:1}},{responsive:true,displayModeBar:false});

  // Burn scatter
  Plotly.newPlot('burnChart',[
    {x:d.burn[0].acc,y:d.burn[0].ratio,mode:'markers',name:'No Loan',
     marker:{color:'#3498db',size:8,opacity:.7}},
    {x:d.burn[1].acc,y:d.burn[1].ratio,mode:'markers',name:'Loan',
     marker:{color:'#e74c3c',size:8,opacity:.7}}],
    {margin:{t:10,r:10,b:50,l:60},
     xaxis:{title:'Burn Acceleration'},yaxis:{title:'Burn Rate Ratio'},
     legend:{x:.75,y:1}},{responsive:true,displayModeBar:false});

  // Category spikes grouped bar
  const cats = ['Repair','Electronics','Clothing','Travel'];
  Plotly.newPlot('spikeChart',[
    {x:cats,y:d.spikes[0],type:'bar',name:'No Loan',marker:{color:'#3498db'}},
    {x:cats,y:d.spikes[1],type:'bar',name:'Loan',marker:{color:'#e74c3c'}}],
    {barmode:'group',margin:{t:10,r:10,b:40,l:50},
     yaxis:{title:'Avg Spike Ratio'},legend:{x:.75,y:1}},
    {responsive:true,displayModeBar:false});

  // Urgency score density (as histogram)
  Plotly.newPlot('urgencyChart',[
    {x:d.urgency[0],type:'histogram',name:'No Loan',histnorm:'probability density',
     opacity:.65,marker:{color:'#3498db'},nbinsx:20},
    {x:d.urgency[1],type:'histogram',name:'Loan',histnorm:'probability density',
     opacity:.65,marker:{color:'#e74c3c'},nbinsx:20}],
    {barmode:'overlay',margin:{t:10,r:10,b:40,l:60},
     xaxis:{title:'Urgency Score'},yaxis:{title:'Density'},
     legend:{x:.75,y:1}},{responsive:true,displayModeBar:false});

  // Balance check spike scatter
  Plotly.newPlot('checkSpikeChart',[
    {x:d.check_spike[0].hist,y:d.check_spike[0].recent,mode:'markers',name:'No Loan',
     marker:{color:'#3498db',size:8,opacity:.7}},
    {x:d.check_spike[1].hist,y:d.check_spike[1].recent,mode:'markers',name:'Loan',
     marker:{color:'#e74c3c',size:8,opacity:.7}}],
    {margin:{t:10,r:10,b:50,l:60},
     xaxis:{title:'Historical Check Ratio'},yaxis:{title:'Recent Check Ratio (last 3m)'},
     legend:{x:.75,y:1}},{responsive:true,displayModeBar:false});

  // Payment health stacked bar
  const statuses = ['PAID_ON_TIME','PAID_LATE','MISSED','OVERDUE','SCHEDULED'];
  const cols = ['#27ae60','#e67e22','#e74c3c','#c0392b','#95a5a6'];
  Plotly.newPlot('payHealthChart',
    statuses.map((s,i)=>({
      x:d.pay_health.clients, y:d.pay_health[s]||d.pay_health.clients.map(()=>0),
      type:'bar', name:s, marker:{color:cols[i]}
    })),
    {barmode:'stack',margin:{t:10,r:10,b:60,l:50},
     xaxis:{title:'Client ID',tickangle:45,nticks:15},
     yaxis:{title:'Payments'},legend:{orientation:'h',y:1.1}},
    {responsive:true,displayModeBar:false});

  // Phase transition line chart
  const phases = ['Phase 1 Normal','Phase 2 Spike','Phase 3 Early','Phase 3 Late','Phase 4 Target'];
  Plotly.newPlot('phaseChart',[
    {x:phases,y:d.phase_check[0],type:'scatter',mode:'lines+markers',name:'No Loan',
     line:{color:'#3498db',width:2},marker:{size:8}},
    {x:phases,y:d.phase_check[1],type:'scatter',mode:'lines+markers',name:'Loan',
     line:{color:'#e74c3c',width:2},marker:{size:8}}],
    {margin:{t:10,r:10,b:80,l:60},
     xaxis:{title:'Timeline Phase'},yaxis:{title:'Avg Balance-Check Ratio'},
     legend:{x:.8,y:1}},{responsive:true,displayModeBar:false});
});
</script>"""
    return page(content, active=2)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — PREDICTION & SCORING
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/prediction")
def prediction():
    content = """
<div class="container-fluid px-4 py-4">
  <div class="row g-3 mb-3">
    <!-- Score table -->
    <div class="col-md-6">
      <div class="card">
        <div class="card-header d-flex justify-content-between align-items-center">
          🏆 Client Scores
          <div>
            <select class="form-select form-select-sm d-inline w-auto me-1" id="tierFilter" onchange="filterTable()">
              <option value="">All Tiers</option>
              <option>High</option><option>Medium</option>
              <option>Low</option><option>Very Low</option>
            </select>
            <button class="btn btn-sm btn-outline-light" onclick="exportCSV()">⬇️ Export</button>
          </div>
        </div>
        <div class="card-body p-0">
          <div class="scrollable" style="max-height:500px">
            <table class="table table-sm table-hover mb-0" style="font-size:.8rem">
              <thead style="position:sticky;top:0;z-index:1">
                <tr>
                  <th onclick="sortTable('name')" style="cursor:pointer">Name ↕</th>
                  <th onclick="sortTable('prob')" style="cursor:pointer">OOF Prob ↕</th>
                  <th>Tier</th>
                  <th>Actual</th>
                  <th>Outcome</th>
                </tr>
              </thead>
              <tbody id="scoreBody"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- Right side charts -->
    <div class="col-md-6">
      <div class="chart-container mb-3">
        <h6 class="text-muted mb-1">📈 ROC Curve</h6>
        <div id="rocChart" style="height:200px"></div>
      </div>
      <div class="chart-container">
        <h6 class="text-muted mb-1">🎯 Precision-Recall Curve</h6>
        <div id="prChart" style="height:200px"></div>
      </div>
    </div>
  </div>

  <!-- Threshold simulator -->
  <div class="chart-container mb-3">
    <div class="row align-items-center">
      <div class="col-md-6">
        <h6 class="text-muted">🎚️ Threshold Simulator</h6>
        <div class="d-flex align-items-center gap-3">
          <input type="range" class="form-range flex-grow-1" id="threshSlider"
                 min="0" max="1" step="0.01" value="0.5" oninput="updateThreshold(this.value)">
          <div id="threshold-val">0.50</div>
        </div>
        <div class="row text-center mt-2" id="threshMetrics">
          <div class="col"><div class="fw-bold" id="th-prec">—</div><small class="text-muted">Precision</small></div>
          <div class="col"><div class="fw-bold" id="th-rec">—</div><small class="text-muted">Recall</small></div>
          <div class="col"><div class="fw-bold" id="th-f1">—</div><small class="text-muted">F1</small></div>
          <div class="col"><div class="fw-bold" id="th-flagged">—</div><small class="text-muted">Flagged</small></div>
          <div class="col"><div class="fw-bold" id="th-tp">—</div><small class="text-muted">True Pos</small></div>
        </div>
      </div>
      <div class="col-md-6">
        <div id="threshChart" style="height:180px"></div>
      </div>
    </div>
  </div>

  <div class="row g-3">
    <!-- Feature importance -->
    <div class="col-md-6">
      <div class="chart-container">
        <h6 class="text-muted">🔍 Feature Importance (top 15)</h6>
        <div id="impChart" style="height:320px"></div>
      </div>
    </div>
    <!-- Risk tier donut + confusion matrix -->
    <div class="col-md-3">
      <div class="chart-container mb-3">
        <h6 class="text-muted">🎯 Risk Tier Distribution</h6>
        <div id="tierChart" style="height:200px"></div>
      </div>
      <div class="chart-container">
        <h6 class="text-muted mb-2">📊 Confusion Matrix</h6>
        <div id="cmChart" style="height:180px"></div>
      </div>
    </div>
    <div class="col-md-3">
      <div class="card">
        <div class="card-header">📋 Model Summary</div>
        <div class="card-body" id="modelSummary"></div>
      </div>
    </div>
  </div>
</div>

<script>
let scoreData = [];
let predData  = {};
let sortCol = 'prob', sortAsc = false;

fetch('/api/prediction_data').then(r=>r.json()).then(d=>{
  predData  = d;
  scoreData = d.scores;
  renderTable();

  // ROC curve
  Plotly.newPlot('rocChart',[
    {x:d.roc.fpr,y:d.roc.tpr,type:'scatter',mode:'lines',
     name:`AUC=${d.roc.auc.toFixed(4)}`,line:{color:'#e74c3c',width:2}},
    {x:[0,1],y:[0,1],mode:'lines',name:'Random',
     line:{color:'#aaa',dash:'dash',width:1}}],
    {margin:{t:10,r:10,b:40,l:50},
     xaxis:{title:'FPR'},yaxis:{title:'TPR'},legend:{x:.6,y:.05}},
    {responsive:true,displayModeBar:false});

  // PR curve
  Plotly.newPlot('prChart',[
    {x:d.pr.recall,y:d.pr.precision,type:'scatter',mode:'lines',
     name:`AP=${d.pr.ap.toFixed(4)}`,line:{color:'#3498db',width:2}}],
    {margin:{t:10,r:10,b:40,l:50},
     xaxis:{title:'Recall'},yaxis:{title:'Precision'},legend:{x:.05,y:.05}},
    {responsive:true,displayModeBar:false});

  // Feature importance
  const imp = d.importance.slice(0,15);
  const grpColor = f => f.includes('check')||f.includes('session')||f.includes('ratio')? '#9b59b6':
                        f.includes('spike')||f.includes('share')? '#e74c3c':
                        f.includes('balance')||f.includes('burn')? '#3498db':
                        f.includes('credit')||f.includes('repay')? '#27ae60':'#e67e22';
  Plotly.newPlot('impChart',[{
    y:imp.map(r=>r.feature), x:imp.map(r=>r.importance),
    type:'bar',orientation:'h',
    marker:{color:imp.map(r=>grpColor(r.feature))}}],
    {margin:{t:10,r:20,b:30,l:180},yaxis:{autorange:'reversed'}},
    {responsive:true,displayModeBar:false});

  // Risk tier donut
  const tiers = d.tiers;
  Plotly.newPlot('tierChart',[{
    type:'pie', labels:Object.keys(tiers), values:Object.values(tiers),
    hole:.45, marker:{colors:['#e74c3c','#e67e22','#3498db','#2ecc71']}}],
    {margin:{t:5,r:5,b:5,l:5},showlegend:true,
     legend:{orientation:'h',y:-0.1,font:{size:9}}},
    {responsive:true,displayModeBar:false});

  // Confusion matrix heatmap
  const cm = d.cm;
  Plotly.newPlot('cmChart',[{
    z:[[cm.tn,cm.fp],[cm.fn,cm.tp]],
    x:['Pred: No Loan','Pred: Loan'], y:['Act: No Loan','Act: Loan'],
    type:'heatmap', colorscale:[[0,'#f8f9fa'],[1,'#2c3e50']],
    text:[[`TN: ${cm.tn}`,`FP: ${cm.fp}`],[`FN: ${cm.fn}`,`TP: ${cm.tp}`]],
    texttemplate:'%{text}', showscale:false}],
    {margin:{t:5,r:5,b:40,l:80}},{responsive:true,displayModeBar:false});

  // Threshold chart (initial)
  updateThreshold(0.5);

  // Model summary
  document.getElementById('modelSummary').innerHTML = `
    ${[['Algorithm','Gradient Boosting'],
       ['Folds','5-fold Stratified CV'],
       ['OOF ROC-AUC', d.metrics.auc.toFixed(4)],
       ['OOF Avg Precision', d.metrics.ap.toFixed(4)],
       ['Accuracy (0.50)', d.metrics.acc.toFixed(1)+'%'],
       ['Precision', d.metrics.prec.toFixed(1)+'%'],
       ['Recall', d.metrics.rec.toFixed(1)+'%'],
       ['F1-Score', d.metrics.f1.toFixed(3)],
       ['Features Used', d.metrics.n_features]].map(([k,v])=>
      `<div class="d-flex justify-content-between border-bottom py-1">
         <span class="text-muted" style="font-size:.8rem">${k}</span>
         <span class="fw-semibold" style="font-size:.8rem">${v}</span>
       </div>`).join('')}`;
});

function renderTable(){
  const tier = document.getElementById('tierFilter').value;
  let rows = scoreData.filter(r=>!tier||r.risk_tier===tier);
  if(sortCol==='prob') rows.sort((a,b)=>sortAsc?a.oof_probability-b.oof_probability:b.oof_probability-a.oof_probability);
  if(sortCol==='name') rows.sort((a,b)=>sortAsc?a.name.localeCompare(b.name):b.name.localeCompare(a.name));
  const oc = {TP:'outcome-tp',FP:'outcome-fp',FN:'outcome-fn',TN:'outcome-tn'};
  const tc = {High:'badge-high',Medium:'badge-medium',Low:'badge-low','Very Low':'badge-verylow'};
  document.getElementById('scoreBody').innerHTML = rows.map(r=>`
    <tr>
      <td>${r.name}</td>
      <td class="fw-semibold">${(r.oof_probability*100).toFixed(1)}%</td>
      <td><span class="${tc[r.risk_tier]||''}">${r.risk_tier}</span></td>
      <td>${r.actual_label===1?'✅ Loan':'❌ No'}</td>
      <td class="${oc[r.outcome]||''}">${r.outcome}</td>
    </tr>`).join('');
}

function filterTable(){ renderTable(); }
function sortTable(col){ if(sortCol===col) sortAsc=!sortAsc; else {sortCol=col;sortAsc=false}; renderTable(); }

function updateThreshold(t){
  t = parseFloat(t);
  document.getElementById('threshold-val').textContent = t.toFixed(2);
  if(!predData.scores) return;
  const probs = predData.scores.map(r=>r.oof_probability);
  const actual= predData.scores.map(r=>r.actual_label);
  const pred  = probs.map(p=>p>=t?1:0);
  const tp = pred.filter((_,i)=>pred[i]===1&&actual[i]===1).length;
  const fp = pred.filter((_,i)=>pred[i]===1&&actual[i]===0).length;
  const fn = pred.filter((_,i)=>pred[i]===0&&actual[i]===1).length;
  const flagged = tp+fp;
  const prec = flagged?tp/flagged:0, rec = tp+fn?tp/(tp+fn):0;
  const f1   = prec+rec?2*prec*rec/(prec+rec):0;
  document.getElementById('th-prec').textContent = (prec*100).toFixed(1)+'%';
  document.getElementById('th-rec').textContent  = (rec*100).toFixed(1)+'%';
  document.getElementById('th-f1').textContent   = f1.toFixed(3);
  document.getElementById('th-flagged').textContent = flagged;
  document.getElementById('th-tp').textContent   = tp;

  const thresholds = Array.from({length:101},(_,i)=>i/100);
  const precVals=[],recVals=[],f1Vals=[];
  for(const th of thresholds){
    const p2=probs.map(v=>v>=th?1:0);
    const tp2=p2.filter((_,i)=>p2[i]===1&&actual[i]===1).length;
    const fp2=p2.filter((_,i)=>p2[i]===1&&actual[i]===0).length;
    const fn2=p2.filter((_,i)=>p2[i]===0&&actual[i]===1).length;
    const fl2=tp2+fp2;
    const pr=fl2?tp2/fl2:0, re=tp2+fn2?tp2/(tp2+fn2):0;
    precVals.push((pr*100).toFixed(1));
    recVals.push((re*100).toFixed(1));
    f1Vals.push((pr+re?2*pr*re/(pr+re):0)*100);
  }
  Plotly.newPlot('threshChart',[
    {x:thresholds,y:precVals,name:'Precision',line:{color:'#e74c3c'}},
    {x:thresholds,y:recVals,name:'Recall',line:{color:'#3498db'}},
    {x:thresholds,y:f1Vals,name:'F1×100',line:{color:'#2ecc71',dash:'dot'}},
    {x:[t,t],y:[0,100],mode:'lines',name:'Threshold',
     line:{color:'#333',dash:'dash',width:1}}],
    {margin:{t:5,r:10,b:30,l:40},legend:{orientation:'h',y:1.1,font:{size:10}},
     yaxis:{range:[0,105]}},
    {responsive:true,displayModeBar:false});
}

function exportCSV(){
  const rows = [['Client ID','Name','OOF Probability','Risk Tier','Predicted','Actual','Outcome']];
  scoreData.forEach(r=>rows.push([r.client_id,r.name,r.oof_probability,
    r.risk_tier,r.predicted_label,r.actual_label,r.outcome]));
  const csv = rows.map(r=>r.join(',')).join('\\n');
  const a = document.createElement('a');
  a.href = 'data:text/csv;charset=utf-8,'+encodeURIComponent(csv);
  a.download = 'model_scores_export.csv'; a.click();
}
</script>"""
    return page(content, active=3)


# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/clients")
def api_clients():
    cl  = DATA["clients"]
    ml  = DATA["scores_ml"]
    merged = cl.merge(ml[["client_id","oof_probability","risk_tier"]], on="client_id", how="left")
    merged["oof_probability"] = merged["oof_probability"].fillna(0)
    merged["risk_tier"] = merged["risk_tier"].fillna("Very Low")
    merged = merged.sort_values("oof_probability", ascending=False)
    return jsonify(merged[["client_id","first_name","last_name","oof_probability","risk_tier"]].to_dict("records"))


@app.route("/api/client/<int:cid>")
def api_client(cid):
    cl  = DATA["clients"]
    tx  = DATA["tx"]
    app = DATA["app"]
    cr  = DATA["credit"]
    pay = DATA["payments"]
    sc  = DATA["scores_cs"]
    ml  = DATA["scores_ml"]

    client = cl[cl.client_id==cid].iloc[0].to_dict()
    model_score = ml[ml.client_id==cid].iloc[0].to_dict() if cid in ml.client_id.values else {}
    cs = sc[sc.client_id==cid].iloc[0].to_dict() if cid in sc.client_id.values else {}

    # Credit
    cr_row = cr[cr.client_id==cid]
    credit_dict = cr_row.iloc[0].to_dict() if not cr_row.empty else None

    # Payments
    pay_rows = pay[pay.client_id==cid].sort_values("due_date")
    pay_list = pay_rows.fillna("").to_dict("records")

    # App summary
    client_app = app[app.client_id==cid]
    cutoff3 = pd.Timestamp("2025-07-01")
    app_summary = {
        "total_sessions": len(client_app),
        "balance_checks": int((client_app["action"]=="check_balance").sum()),
        "check_balance_ratio": round(float((client_app["action"]=="check_balance").mean()), 3),
        "active_days": int(client_app["dt"].dt.date.nunique()),
        "sessions_last_3m": int((client_app["dt"] >= cutoff3).sum()),
    }

    # Transactions
    client_tx = tx[tx.client_id==cid].sort_values("dt", ascending=False)
    client_tx["phase"] = client_tx["dt"].apply(phase_of)
    tx_list = client_tx[["transaction_date","merchant_name","category",
                          "amount","status","balance","phase"]].fillna("").to_dict("records")

    # Balance chart data
    bal_chart = client_tx.sort_values("dt")[["dt","balance","phase"]].copy()
    bal_chart["date"] = bal_chart["dt"].dt.strftime("%Y-%m-%d")
    bal_list = bal_chart[["date","balance","phase"]].to_dict("records")

    # Spend donut
    spend = (client_tx[client_tx.status=="SUCCESS"]
             .groupby("category")["amount"].sum().reset_index())
    spend_list = spend.to_dict("records")

    # Features row — needed for Top Reasons and Local Impact in the UI
    feat_row = DATA["features"][DATA["features"].client_id == cid]
    features_dict = {}
    if not feat_row.empty:
        for k, v in feat_row.iloc[0].items():
            try:
                if pd.isna(v):
                    features_dict[k] = None
                elif isinstance(v, (np.integer,)):
                    features_dict[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    features_dict[k] = float(v)
                else:
                    features_dict[k] = v
            except Exception:
                features_dict[k] = None

    # SHAP — per-client values for the waterfall chart
    shap_payload = None
    if SHAP_DATA is not None and not feat_row.empty:
        try:
            pos = DATA["features"].index.get_loc(feat_row.index[0])
            sv  = SHAP_DATA["values"][pos]
            shap_payload = {
                "values":         [round(float(v), 6) for v in sv],
                "feature_names":  SHAP_DATA["features"],
                "expected_value": round(SHAP_DATA["expected_value"], 6),
            }
        except Exception:
            pass

    # Clean NaN + numpy types for scalar dicts
    def clean(d):
        if not d: return d
        out = {}
        for k, v in d.items():
            try:
                if pd.isna(v):
                    out[k] = None
                elif isinstance(v, (np.integer,)):
                    out[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    out[k] = float(v)
                else:
                    out[k] = v
            except Exception:
                out[k] = None
        return out

    return jsonify({
        "client":        clean(client),
        "credit_score":  clean(cs),
        "model_score":   clean(model_score),
        "credit":        clean(credit_dict),
        "payments":      pay_list,
        "app":           app_summary,
        "tx":            tx_list[:500],
        "balance_chart": bal_list,
        "spend_donut":   spend_list,
        "features":      features_dict,
        "shap":          shap_payload,
    })


@app.route("/api/analytics")
def api_analytics():
    feat = DATA["features"]
    ml   = DATA["scores_ml"]
    pay  = DATA["payments"]
    sc   = DATA["scores_cs"]
    cr   = DATA["credit"]
    cl   = DATA["clients"]
    app  = DATA["app"]
    tx   = DATA["tx"]

    f0 = feat[feat.will_seek_loan==0]
    f1 = feat[feat.will_seek_loan==1]

    # KPIs
    kpi = {
        "total": len(cl),
        "high_risk": int((ml.risk_tier=="High").sum()),
        "loans": int((cr.status.isin(["ACTIVE","CLOSED"])).sum()),
        "avg_score": int(sc.credit_score.mean()),
    }

    # Balance drop
    bal_drop = [f0.balance_drop_pct.tolist(), f1.balance_drop_pct.tolist()]

    # Burn
    burn = [
        {"acc": f0.burn_acceleration.clip(0,5).tolist(), "ratio": f0.burn_rate_ratio.clip(0,3).tolist()},
        {"acc": f1.burn_acceleration.clip(0,5).tolist(), "ratio": f1.burn_rate_ratio.clip(0,3).tolist()},
    ]

    # Spikes by category
    cats = ["spike_repair","spike_electronics","spike_clothing","spike_travel"]
    spikes = [
        [round(f0[c].clip(0,5).mean(),2) for c in cats],
        [round(f1[c].clip(0,5).mean(),2) for c in cats],
    ]

    # Urgency
    urgency = [f0.urgency_score.tolist(), f1.urgency_score.tolist()]

    # Check spike scatter
    hist_col = "check_balance_ratio" if "check_balance_ratio" in feat.columns else "check_ratio_last_3m"
    check_spike = [
        {"hist": f0[hist_col].tolist(), "recent": f0.check_ratio_last_3m.tolist()},
        {"hist": f1[hist_col].tolist(), "recent": f1.check_ratio_last_3m.tolist()},
    ]

    # Payment health stacked bar
    if len(pay) > 0:
        paid_clients = sorted(pay.client_id.unique())[:30]
        ph = {}
        ph["clients"] = [str(c) for c in paid_clients]
        for s in ["PAID_ON_TIME","PAID_LATE","MISSED","OVERDUE","SCHEDULED"]:
            ph[s] = [int((pay[pay.client_id==c].status==s).sum()) for c in paid_clients]
    else:
        ph = {"clients":[], "PAID_ON_TIME":[], "PAID_LATE":[], "MISSED":[], "OVERDUE":[], "SCHEDULED":[]}

    # Phase transition: avg check ratio per phase per label
    def phase_check_ratio(client_ids, app_df):
        windows = {
            "Phase 1 Normal":  (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-07-01")),
            "Phase 2 Spike":   (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-10-01")),
            "Phase 3 Early":   (pd.Timestamp("2024-10-01"), pd.Timestamp("2025-04-01")),
            "Phase 3 Late":    (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-10-01")),
            "Phase 4 Target":  (pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-01")),
        }
        result = []
        sub = app_df[app_df.client_id.isin(client_ids)]
        for name, (s, e) in windows.items():
            w = sub[(sub.dt >= s) & (sub.dt < e)]
            if len(w) == 0: result.append(0.0)
            else: result.append(round(float((w.action=="check_balance").mean()), 3))
        return result

    phase_check = [
        phase_check_ratio(f0.client_id.tolist(), app),
        phase_check_ratio(f1.client_id.tolist(), app),
    ]

    return jsonify({
        "kpi": kpi, "bal_drop": bal_drop, "burn": burn,
        "spikes": spikes, "urgency": urgency, "check_spike": check_spike,
        "pay_health": ph, "phase_check": phase_check,
    })


@app.route("/api/prediction_data")
def api_prediction_data():
    from sklearn.metrics import (roc_curve, precision_recall_curve,
                                 roc_auc_score, average_precision_score,
                                 accuracy_score, precision_score, recall_score, f1_score)
    ml   = DATA["scores_ml"]
    imp  = DATA["importance"]
    cl   = DATA["clients"]

    merged = ml.merge(cl[["client_id","first_name","last_name"]], on="client_id", how="left")
    merged["name"] = merged["first_name"] + " " + merged["last_name"]
    scores = merged[["client_id","name","oof_probability","final_probability",
                      "predicted_label","actual_label","risk_tier","outcome"]].to_dict("records")

    y      = ml.actual_label.values
    y_prob = ml.oof_probability.values
    y_pred = ml.predicted_label.values

    fpr, tpr, _ = roc_curve(y, y_prob)
    prec, rec, _= precision_recall_curve(y, y_prob)
    auc_val = roc_auc_score(y, y_prob)
    ap_val  = average_precision_score(y, y_prob)

    cm_vals = {
        "tp": int(((y_pred==1)&(y==1)).sum()),
        "fp": int(((y_pred==1)&(y==0)).sum()),
        "fn": int(((y_pred==0)&(y==1)).sum()),
        "tn": int(((y_pred==0)&(y==0)).sum()),
    }
    tiers = ml.risk_tier.value_counts().to_dict()

    metrics = {
        "auc":        auc_val,
        "ap":         ap_val,
        "acc":        accuracy_score(y, y_pred)*100,
        "prec":       precision_score(y, y_pred, zero_division=0)*100,
        "rec":        recall_score(y, y_pred, zero_division=0)*100,
        "f1":         f1_score(y, y_pred, zero_division=0),
        "n_features": len(DATA["features"].columns) - 4,
    }

    return jsonify({
        "scores":     scores,
        "roc":        {"fpr":fpr.tolist(),"tpr":tpr.tolist(),"auc":auc_val},
        "pr":         {"precision":prec.tolist(),"recall":rec.tolist(),"ap":ap_val},
        "importance": imp.head(15).to_dict("records"),
        "tiers":      tiers,
        "cm":         cm_vals,
        "metrics":    metrics,
    })


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
#
# def open_browser():
#     import time; time.sleep(1.2)
#     webbrowser.open(f"http://localhost:{PORT}")

if __name__ == "__main__":
    print("="*55)
    print("  Loan Propensity CRM — starting...")
    print(f"  URL : http://localhost:{PORT}")
    print("  Stop: Ctrl+C")
    print("="*55)
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(port=PORT, debug=False)