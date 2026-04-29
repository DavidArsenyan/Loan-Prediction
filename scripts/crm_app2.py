"""
crm_app2.py  —  Loan Propensity CRM  (Desktop Edition)
======================================================
Requirements:
    pip install PyQt6 matplotlib pandas numpy scikit-learn lightgbm shap

Run:
    python crm_app2.py

Build .exe automatically (first run with --build flag):
    python crm_app2.py --build

Or manually:
    pip install pyinstaller
    pyinstaller --onefile --windowed --name "LoanCRM" --add-data "data:data" crm_app2.py

New in this version
-------------------
  • SHAP explanations: per-client waterfall chart on client page.
  • SHAP analytics: beeswarm (all clients) + summary bar in analytics page.
  • KeyError 'oof_probability' fixed — column names now match model.py output.
  • Hover tooltips on ALL charts.
  • Self-building: run with --build.
"""

import sys, os, pickle, warnings, subprocess
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SELF-BUILD MODE
# ─────────────────────────────────────────────────────────────────────────────

if "--build" in sys.argv:
    print("=" * 55)
    print("  LoanCRM — self-build mode")
    print("=" * 55)
    script = os.path.abspath(__file__)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(script)), "data")

    print("[1/3] Installing PyInstaller...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller", "-q"])

    sep = ";" if sys.platform == "win32" else ":"
    add_data = f"{data_dir}{sep}data"

    print("[2/3] Running PyInstaller...")
    subprocess.check_call([
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        "--name", "LoanCRM",
        "--add-data", add_data,
        "--hidden-import", "sklearn.utils._cython_blas",
        "--hidden-import", "sklearn.neighbors._typedefs",
        "--hidden-import", "sklearn.tree._classes",
        "--hidden-import", "lightgbm",
        "--hidden-import", "shap",
        script,
    ])
    print("[3/3] Done!")
    exe = "dist\\LoanCRM.exe" if sys.platform == "win32" else "dist/LoanCRM"
    print(f"\n  Executable: {os.path.abspath(exe)}")
    print("=" * 55)
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# NORMAL RUN
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QPushButton, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QScrollArea, QFrame, QTableWidget, QTableWidgetItem,
    QSlider, QSplitter, QGridLayout, QHeaderView, QComboBox, QSizePolicy,
    QAbstractItemView, QToolTip,
)
from PyQt6.QtCore import Qt, QSize, QPoint, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette

# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────

BG      = "#0f1117"
CARD    = "#1c1f2e"
CARD2   = "#252844"
BORDER  = "#2e3250"
ACCENT  = "#4f6ef7"
ACCENT2 = "#7c5ef5"
SUCCESS = "#22c55e"
WARNING = "#f59e0b"
DANGER  = "#ef4444"
TEXT    = "#e8eaf0"
MUTED   = "#8b91a8"

TIER_COLORS  = {"High": DANGER, "Medium": WARNING, "Low": ACCENT, "Very Low": SUCCESS}
PHASE_COLORS = {
    "Phase 1 Normal":   "#3b82f6",
    "Phase 2 Spike":    "#ef4444",
    "Phase 3 Repaying": "#22c55e",
    "Phase 4 Target":   "#a855f7",
}

def mpl_style():
    plt.rcParams.update({
        "figure.facecolor": CARD,  "axes.facecolor": CARD,
        "axes.edgecolor":   BORDER,"axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color":  TEXT,  "grid.color":  BORDER,
        "grid.linestyle": "--", "grid.alpha": 0.4,
        "legend.facecolor": CARD2, "legend.edgecolor": BORDER,
        "legend.labelcolor": TEXT, "font.size": 9,
    })

mpl_style()

# Resolve data directory (works both in dev and PyInstaller bundle)
if getattr(sys, "frozen", False):
    _base = sys._MEIPASS
    DATA_DIR = os.path.join(_base, "data")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

# ─────────────────────────────────────────────────────────────────────────────
# HOVER TOOLTIP ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class HoverAnnotation:
    def __init__(self, canvas: "MplCanvas", ax, fmt=None):
        self.canvas = canvas
        self.ax     = ax
        self.fmt    = fmt
        self.annot  = ax.annotate(
            "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc=CARD2, ec=ACCENT, lw=1, alpha=0.95),
            fontsize=8, color=TEXT,
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8),
        )
        self.annot.set_visible(False)
        canvas.mpl_connect("motion_notify_event", self._on_move)
        canvas.mpl_connect("axes_leave_event",    self._on_leave)

    def _fmt_default(self, label, x, y):
        xl = self.ax.get_xlabel() or "x"
        yl = self.ax.get_ylabel() or "y"
        parts = []
        if label and label not in ("_nolegend_", ""):
            parts.append(label)
        parts.append(f"{xl}: {x:,.3g}")
        parts.append(f"{yl}: {y:,.3g}")
        return "\n".join(parts)

    def _on_leave(self, event):
        self.annot.set_visible(False)
        self.canvas.draw_idle()

    def _on_move(self, event):
        if event.inaxes != self.ax:
            self.annot.set_visible(False)
            self.canvas.draw_idle()
            return

        hit = False

        for line in self.ax.get_lines():
            if not line.get_visible(): continue
            cont, ind = line.contains(event)
            if cont and len(ind["ind"]) > 0:
                i  = ind["ind"][0]
                xd = line.get_xdata(); yd = line.get_ydata()
                x, y = xd[i], yd[i]
                lbl = line.get_label()
                txt = self.fmt(lbl, x, y) if self.fmt else self._fmt_default(lbl, x, y)
                self.annot.xy = (x, y)
                self.annot.set_text(txt)
                self.annot.set_visible(True)
                self.canvas.draw_idle()
                hit = True
                break

        if hit: return

        for bar_cont in self.ax.containers:
            for patch in bar_cont:
                if not hasattr(patch, "get_x"): continue
                if patch.contains_point((event.x, event.y)):
                    xc = patch.get_x() + patch.get_width() / 2
                    h  = patch.get_height()
                    lbl = bar_cont.get_label() if hasattr(bar_cont, "get_label") else ""
                    yl = self.ax.get_ylabel() or "value"
                    txt = (f"{lbl}\n" if lbl and lbl != "_nolegend_" else "")
                    txt += f"{yl}: {h:,.3g}"
                    self.annot.xy = (xc, h)
                    self.annot.set_text(txt)
                    self.annot.set_visible(True)
                    self.canvas.draw_idle()
                    hit = True
                    break
            if hit: break

        if hit: return

        for patch in self.ax.patches:
            if hasattr(patch, "get_width") and hasattr(patch, "get_height"):
                if patch.contains_point((event.x, event.y)):
                    xc = patch.get_x() + patch.get_width() / 2
                    h  = patch.get_height()
                    yl = self.ax.get_ylabel() or "value"
                    txt = f"{yl}: {h:,.3g}"
                    self.annot.xy = (xc, h)
                    self.annot.set_text(txt)
                    self.annot.set_visible(True)
                    self.canvas.draw_idle()
                    hit = True
                    break

        if hit: return

        for col in self.ax.collections:
            cont, ind = col.contains(event)
            if cont and len(ind["ind"]) > 0:
                i        = ind["ind"][0]
                offsets  = col.get_offsets()
                x, y     = float(offsets[i][0]), float(offsets[i][1])
                lbl      = col.get_label()
                txt = self.fmt(lbl, x, y) if self.fmt else self._fmt_default(lbl, x, y)
                self.annot.xy = (x, y)
                self.annot.set_text(txt)
                self.annot.set_visible(True)
                self.canvas.draw_idle()
                hit = True
                break

        if not hit:
            self.annot.set_visible(False)
            self.canvas.draw_idle()


def add_hover(canvas: "MplCanvas", ax, fmt=None) -> HoverAnnotation:
    return HoverAnnotation(canvas, ax, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# DATA STORE
# ─────────────────────────────────────────────────────────────────────────────

class DataStore:
    def __init__(self):
        self.clients    = pd.read_csv(f"{DATA_DIR}/clients.csv")
        self.tx         = pd.read_csv(f"{DATA_DIR}/transactions.csv")
        self.app        = pd.read_csv(f"{DATA_DIR}/application.csv")
        self.credit     = pd.read_csv(f"{DATA_DIR}/credit.csv")
        self.payments   = pd.read_csv(f"{DATA_DIR}/payments.csv")
        self.scores_cs  = pd.read_csv(f"{DATA_DIR}/credit_scores.csv")
        self.scores_ml  = pd.read_csv(f"{DATA_DIR}/model_scores.csv")
        self.features   = pd.read_csv(f"{DATA_DIR}/features.csv")
        self.importance = pd.read_csv(f"{DATA_DIR}/feature_importance.csv")
        self.tx["dt"]   = pd.to_datetime(self.tx["transaction_date"])
        self.app["dt"]  = pd.to_datetime(self.app["timestamp"])
        self.scores_ml  = self.scores_ml.merge(
            self.clients[["client_id","first_name","last_name"]], on="client_id", how="left"
        )
        self.scores_ml["name"] = (self.scores_ml["first_name"] + " "
                                  + self.scores_ml["last_name"])

        # ── Load SHAP data from model pickle ─────────────────────────────
        self.shap_data = None
        try:
            with open(f"{DATA_DIR}/model.pkl", "rb") as pf:
                pkg = pickle.load(pf)
                if "shap_values" in pkg and "expected_value" in pkg:
                    self.shap_data = {
                        "values":         np.array(pkg["shap_values"]),
                        "expected_value": float(pkg["expected_value"]),
                        "features":       pkg.get("features", []),
                    }
        except Exception as e:
            print(f"[WARN] Could not load SHAP data: {e}")

    def phase_of(self, dt):
        if dt < pd.Timestamp("2024-07-01"): return "Phase 1 Normal"
        if dt < pd.Timestamp("2024-10-01"): return "Phase 2 Spike"
        if dt < pd.Timestamp("2025-10-01"): return "Phase 3 Repaying"
        return "Phase 4 Target"

DS = DataStore()

# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def lbl(text, size=10, bold=False, color=TEXT, wrap=False):
    l = QLabel(str(text))
    f = QFont("Segoe UI", size)
    f.setBold(bold)
    l.setFont(f)
    l.setStyleSheet(f"color: {color}; background: transparent;")
    if wrap:
        l.setWordWrap(True)
    return l

def make_card(radius=10):
    f = QFrame()
    f.setStyleSheet(
        f"QFrame {{ background: {CARD}; border-radius: {radius}px;"
        f" border: 1px solid {BORDER}; }}"
    )
    return f

def stat_chip(value, subtitle, color=ACCENT):
    card = QFrame()
    card.setStyleSheet(
        f"QFrame {{ background: {CARD2}; border-radius: 8px;"
        f" border: 1px solid {BORDER}; }}"
    )
    lay = QVBoxLayout(card)
    lay.setContentsMargins(12, 10, 12, 10)
    lay.setSpacing(2)
    v = lbl(str(value), 18, True, color)
    v.setAlignment(Qt.AlignmentFlag.AlignCenter)
    s = lbl(subtitle, 8, False, MUTED)
    s.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(v)
    lay.addWidget(s)
    return card

def make_row_widget(hlay):
    w = QWidget()
    w.setStyleSheet("background: transparent;")
    w.setLayout(hlay)
    return w

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, figsize=(5, 3), dpi=90):
        self.fig = Figure(figsize=figsize, dpi=dpi, facecolor=CARD)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: transparent; border: none;")
        self._hovers = []

    def ax(self, **kw):
        return self.fig.add_subplot(111, **kw)

    def redraw(self):
        try:
            self.fig.tight_layout(pad=0.8)
        except Exception:
            pass
        self.draw()

    def attach_hover(self, ax, fmt=None):
        h = add_hover(self, ax, fmt)
        self._hovers.append(h)
        return h


TABLE_STYLE = (
    f"QTableWidget {{ background: {CARD}; gridline-color: {BORDER};"
    f" color: {TEXT}; font-size: 9px; border: none; }}"
    f"QTableWidget::item {{ padding: 4px 6px; }}"
    f"QTableWidget::item:selected {{ background: {ACCENT}33; }}"
    f"QHeaderView::section {{ background: {CARD2}; color: {MUTED};"
    f" border: 1px solid {BORDER}; padding: 4px; font-size: 8px; }}"
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1  —  CLIENT CRM
# ─────────────────────────────────────────────────────────────────────────────

class ClientListPanel(QWidget):
    client_selected = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background: {CARD}; border-radius: 10px;")
        self.setFixedWidth(255)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 12, 10, 12)
        lay.setSpacing(8)
        lay.addWidget(lbl("CLIENTS", 8, True, MUTED))

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search by name…")
        self.search.setStyleSheet(
            f"QLineEdit {{ background: {BG}; color: {TEXT};"
            f" border: 1px solid {BORDER}; border-radius: 6px;"
            f" padding: 6px 8px; font-size: 9px; }}"
            f"QLineEdit:focus {{ border-color: {ACCENT}; }}"
        )
        self.search.textChanged.connect(self._filter)
        lay.addWidget(self.search)

        self.lw = QListWidget()
        self.lw.setStyleSheet(
            f"QListWidget {{ background: transparent; border: none; outline: none; }}"
            f"QListWidget::item {{ background: transparent; border-radius: 6px;"
            f" padding: 0; margin: 2px 0; }}"
            f"QListWidget::item:selected {{ background: {ACCENT}22;"
            f" border: 1px solid {ACCENT}44; }}"
            f"QListWidget::item:hover {{ background: {CARD2}; }}"
        )
        self.lw.setSpacing(2)
        self.lw.currentItemChanged.connect(self._on_sel)
        lay.addWidget(self.lw)

        qf = QHBoxLayout(); qf.setSpacing(4)
        btn_style = (
            f"QPushButton {{ background: {CARD2}; color: {MUTED}; border: 1px solid {BORDER};"
            f" border-radius: 5px; padding: 3px 6px; font-size: 8px; }}"
            f"QPushButton:checked {{ background: {ACCENT}22; color: {ACCENT};"
            f" border-color: {ACCENT}; }}"
            f"QPushButton:hover {{ background: {CARD2}; color: {TEXT}; }}"
        )
        self.btn_all  = QPushButton("All")
        self.btn_high = QPushButton("🔴 High")
        self.btn_top  = QPushButton("Top 20")
        for b in [self.btn_all, self.btn_high, self.btn_top]:
            b.setCheckable(True); b.setStyleSheet(btn_style)
            qf.addWidget(b)
        self.btn_all.setChecked(True)
        self.btn_all.clicked.connect(lambda: self._quick_filter("all"))
        self.btn_high.clicked.connect(lambda: self._quick_filter("high"))
        self.btn_top.clicked.connect(lambda: self._quick_filter("top20"))
        lay.addLayout(qf)

        self._all = []
        self._populate()

    def _quick_filter(self, mode):
        for b, m in [(self.btn_all,"all"),(self.btn_high,"high"),(self.btn_top,"top20")]:
            b.setChecked(m == mode)
        if mode == "all":
            self._render(self._all)
        elif mode == "high":
            self._render([r for r in self._all if r["risk_tier"] == "High"])
        else:
            self._render(self._all[:20])

    def _populate(self):
        ml = DS.scores_ml.sort_values("oof_probability", ascending=False)
        self._all = ml[["client_id","name","oof_probability","risk_tier"]].to_dict("records")
        self._render(self._all)

    def _render(self, rows):
        self.lw.clear()
        for r in rows:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, r["client_id"])
            item.setSizeHint(QSize(0, 56))
            w = QWidget()
            w.setStyleSheet("background: transparent;")
            wl = QVBoxLayout(w)
            wl.setContentsMargins(8, 5, 8, 5)
            wl.setSpacing(2)
            row1 = QHBoxLayout()
            nl = lbl(r["name"], 10, True, TEXT)
            tc = TIER_COLORS.get(r["risk_tier"], MUTED)
            tl = lbl(r["risk_tier"], 8, True, tc)
            row1.addWidget(nl); row1.addStretch(); row1.addWidget(tl)
            sub = lbl(f"{r['oof_probability']*100:.0f}% propensity  ·  ID {r['client_id']}",
                      8, False, MUTED)
            wl.addLayout(row1); wl.addWidget(sub)
            self.lw.addItem(item)
            self.lw.setItemWidget(item, w)

    def _filter(self, text):
        q = text.lower()
        self._render([r for r in self._all if q in r["name"].lower()])

    def _on_sel(self, cur, _):
        if cur:
            cid = cur.data(Qt.ItemDataRole.UserRole)
            if cid:
                self.client_selected.emit(int(cid))


class ClientDetailPanel(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setStyleSheet(f"background: {BG}; border: none;")
        self._placeholder()

    def _placeholder(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ico = lbl("👈", 40)
        ico.setAlignment(Qt.AlignmentFlag.AlignCenter)
        txt = lbl("Select a client from the list", 12, False, MUTED)
        txt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l.addWidget(ico); l.addWidget(txt)
        self.setWidget(w)

    def load(self, client_id):
        cid  = int(client_id)
        cl   = DS.clients[DS.clients.client_id == cid].iloc[0]
        ml   = DS.scores_ml[DS.scores_ml.client_id == cid].iloc[0]
        cs   = DS.scores_cs[DS.scores_cs.client_id == cid].iloc[0]
        cr   = DS.credit[DS.credit.client_id == cid]
        pay  = DS.payments[DS.payments.client_id == cid].sort_values("due_date")
        ctx  = DS.tx[DS.tx.client_id == cid].copy()
        capp = DS.app[DS.app.client_id == cid]
        ctx["phase"] = ctx["dt"].apply(DS.phase_of)

        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        # ── Header ────────────────────────────────────────────────────────
        header = QFrame()
        header.setStyleSheet(
            f"background: {CARD}; border-radius: 10px; border: 1px solid {BORDER};"
        )
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 12, 16, 12)

        av = QLabel(cl.first_name[0] + cl.last_name[0])
        av.setFixedSize(50, 50)
        av.setAlignment(Qt.AlignmentFlag.AlignCenter)
        av.setStyleSheet(
            f"background: {ACCENT}; border-radius: 25px;"
            f" color: white; font-size: 16px; font-weight: bold;"
        )
        hl.addWidget(av)

        info = QVBoxLayout()
        info.setSpacing(2)
        info.addWidget(lbl(f"{cl.first_name} {cl.last_name}", 14, True))
        info.addWidget(lbl(f"{cl.employment_type}  ·  {cl.city}  ·  Age {cl.age}",
                           9, False, MUTED))
        hl.addLayout(info)
        hl.addStretch()

        prob_pct   = ml.oof_probability * 100
        prob_color = DANGER if prob_pct >= 50 else SUCCESS
        pv = QVBoxLayout()
        pv.setSpacing(0)
        pl2 = lbl(f"{prob_pct:.1f}%", 20, True, prob_color)
        pl2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ps  = lbl("Loan Propensity", 8, False, MUTED)
        ps.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pv.addWidget(pl2); pv.addWidget(ps)
        hl.addLayout(pv)

        tc_color = TIER_COLORS.get(ml.risk_tier, MUTED)
        tl2 = lbl(ml.risk_tier, 10, True, tc_color)
        tl2.setStyleSheet(
            f"color: {tc_color}; background: {tc_color}22;"
            f" border-radius: 10px; padding: 3px 10px;"
        )
        hl.addWidget(tl2)

        # ── Urgency ───────────────────────────────────────────────────────
        prob_val = float(ml.oof_probability)
        feat_row = DS.features[DS.features.client_id == cid]
        urgency_val = float(feat_row["urgency_score"].iloc[0]) if not feat_row.empty else 0.0
        check_spike = float(feat_row["balance_check_spike"].iloc[0]) if not feat_row.empty else 1.0
        if prob_val >= 0.65 or (urgency_val >= 0.6 and check_spike >= 1.5):
            urgency_txt   = "🔴 High (0–7 days)"
            urgency_color = DANGER
            est_days      = "~5 days"
        elif prob_val >= 0.40:
            urgency_txt   = "🟡 Medium (7–30 days)"
            urgency_color = WARNING
            est_days      = "~18 days"
        else:
            urgency_txt   = "🟢 Low"
            urgency_color = SUCCESS
            est_days      = "> 30 days"

        uv = QVBoxLayout(); uv.setSpacing(1)
        ul1 = lbl(urgency_txt, 9, True, urgency_color)
        ul1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ul2 = lbl(f"Est. {est_days}", 7, False, MUTED)
        ul2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        uv.addWidget(ul1); uv.addWidget(ul2)
        hl.addLayout(uv)
        lay.addWidget(header)

        # ── Row 0: Recommended Action + Reasoning ─────────────────────────
        r0 = QHBoxLayout(); r0.setSpacing(12)

        ac_card = make_card()
        ac_lay  = QVBoxLayout(ac_card)
        ac_lay.setContentsMargins(14, 10, 14, 10); ac_lay.setSpacing(4)
        ac_lay.addWidget(lbl("RECOMMENDED ACTION", 8, True, MUTED))
        if prob_val >= 0.65:
            action_txt   = "✅  Offer Credit Now"
            action_color = SUCCESS
        elif prob_val >= 0.40:
            action_txt   = "🟡  Monitor (7–14 days)"
            action_color = WARNING
        else:
            action_txt   = "❌  Do Not Offer"
            action_color = DANGER
        al = lbl(action_txt, 11, True, action_color)
        al.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ac_lay.addWidget(al)
        ac_lay.addWidget(lbl(f"OOF prob: {prob_val*100:.1f}%  ·  "
                              f"Final: {float(ml.final_probability)*100:.1f}%",
                              7, False, MUTED))
        r0.addWidget(ac_card, 1)

        rs_card = make_card()
        rs_lay  = QVBoxLayout(rs_card)
        rs_lay.setContentsMargins(14, 10, 14, 10); rs_lay.setSpacing(3)
        rs_lay.addWidget(lbl("TOP REASONS", 8, True, MUTED))
        rs_lay.addSpacing(2)

        reasons = []
        if not feat_row.empty:
            fr = feat_row.iloc[0]
            burn_acc    = float(fr.get("burn_acceleration", 1.0))
            bal_chk_spk = float(fr.get("balance_check_spike", 1.0))
            bal_drop    = float(fr.get("balance_drop_pct", 0.0))
            spk_max     = max(float(fr.get("spike_repair", 0)),
                              float(fr.get("spike_electronics", 0)),
                              float(fr.get("spike_travel", 0)),
                              float(fr.get("spike_clothing", 0)))
            zero_rate   = float(fr.get("zero_day_rate", 0.0))
            cr_score    = float(fr.get("credit_score", 600))
            chk_ratio   = float(fr.get("check_ratio_last_3m", 0.0))

            if burn_acc >= 1.3:
                reasons.append(("+", f"Spend increased recently  (×{burn_acc:.1f})", SUCCESS))
            elif burn_acc < 0.8:
                reasons.append(("–", f"Spending slowed down", MUTED))
            if bal_drop >= 0.2:
                reasons.append(("+", f"Balance declining  ({bal_drop*100:.0f}% drop)", WARNING))
            if bal_chk_spk >= 1.4:
                reasons.append(("+", f"App activity spiked  (×{bal_chk_spk:.1f} checks)", SUCCESS))
            if spk_max >= 2.0:
                reasons.append(("+", f"Category spike detected  (×{spk_max:.1f})", SUCCESS))
            if zero_rate >= 0.08:
                reasons.append(("–", f"Near-zero balance events  ({zero_rate*100:.0f}%)", DANGER))
            if cr_score < 580:
                reasons.append(("–", f"Credit score low  ({int(cr_score)})", DANGER))
            elif cr_score >= 700:
                reasons.append(("+", f"Good credit history  ({int(cr_score)})", SUCCESS))
            if chk_ratio >= 0.5:
                reasons.append(("+", f"High check ratio (3m)  {chk_ratio*100:.0f}%", WARNING))

        for sign, txt, col in (reasons[:5] if reasons else
                                [("·", "No strong signal available", MUTED)]):
            rs_lay.addWidget(lbl(f"{sign}  {txt}", 8, sign == "+" and True or False, col))
        r0.addWidget(rs_card, 2)
        lay.addWidget(make_row_widget(r0))

        # ── Row 1: Profile + Credit score gauge + Model assessment ────────
        r1 = QHBoxLayout(); r1.setSpacing(12)

        pc = make_card()
        pl = QVBoxLayout(pc)
        pl.setContentsMargins(14, 12, 14, 12); pl.setSpacing(2)
        pl.addWidget(lbl("CLIENT PROFILE", 8, True, MUTED))
        pl.addSpacing(6)
        for k, v in [
            ("Salary",        f"{int(cl.monthly_salary):,} AMD/mo"),
            ("Account Since", str(cl.account_open_date)),
            ("Bank Account",  str(cl.bank_account)),
            ("Marital",       str(cl.marital_status)),
            ("Dependants",    str(cl.dependants)),
            ("Phone",         str(cl.phone)),
        ]:
            row = QHBoxLayout()
            row.addWidget(lbl(k, 8, False, MUTED))
            row.addStretch()
            vl = lbl(v, 8, True, TEXT)
            vl.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(vl)
            pl.addLayout(row)
            sep = QFrame()
            sep.setStyleSheet(f"background: {BORDER}; max-height: 1px; margin: 1px 0;")
            pl.addWidget(sep)
        r1.addWidget(pc, 1)

        gc = make_card()
        gl = QVBoxLayout(gc)
        gl.setContentsMargins(14, 12, 14, 12); gl.setSpacing(6)
        gl.addWidget(lbl("CREDIT SCORE", 8, True, MUTED))
        gauge = MplCanvas(figsize=(2.8, 2.0), dpi=90)
        ax = gauge.fig.add_subplot(111, facecolor=CARD)
        score = int(cs.credit_score)
        snorm = (score - 300) / 550
        gcol  = SUCCESS if score >= 700 else (WARNING if score >= 580 else DANGER)
        ax.add_patch(mpatches.Wedge(
            (0.5, 0.1), 0.38, 0, 180, width=0.10,
            facecolor=BORDER, transform=ax.transAxes
        ))
        ax.add_patch(mpatches.Wedge(
            (0.5, 0.1), 0.38, 0, 180 * snorm, width=0.10,
            facecolor=gcol, transform=ax.transAxes
        ))
        ax.text(0.5, 0.28, str(score), transform=ax.transAxes,
                ha="center", va="center", fontsize=20, fontweight="bold", color=TEXT)
        ax.text(0.5, 0.13, str(cs.rating), transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=gcol)
        ax.set_xlim(0,1); ax.set_ylim(0, 0.52); ax.axis("off")
        gauge.draw()
        gl.addWidget(gauge)
        comps = [
            ("Payment",     cs.payment_history_pts),
            ("Utilization", cs.credit_utilization_pts),
            ("Inquiries",   cs.credit_inquiries_pts),
            ("DTI",         cs.dti_pts),
            ("Relation.",   cs.relationship_pts),
        ]
        for name, pts in comps:
            rr = QHBoxLayout()
            rr.addWidget(lbl(name, 7, False, MUTED))
            bg = QFrame(); bg.setFixedHeight(5)
            bg.setStyleSheet(f"background: {BORDER}; border-radius: 2px;")
            fill = QFrame(bg); fill.setFixedHeight(5)
            fill.setFixedWidth(max(3, int(min(float(pts), 110) / 110 * 90)))
            fill.setStyleSheet(f"background: {ACCENT}; border-radius: 2px;")
            pl2 = lbl(f"{float(pts):.0f}", 7, True, TEXT)
            pl2.setFixedWidth(26)
            pl2.setAlignment(Qt.AlignmentFlag.AlignRight)
            rr.addWidget(bg); rr.addWidget(pl2)
            gl.addLayout(rr)
        r1.addWidget(gc, 1)

        mc = make_card()
        ml_lay = QVBoxLayout(mc)
        ml_lay.setContentsMargins(14, 12, 14, 12); ml_lay.setSpacing(2)
        ml_lay.addWidget(lbl("MODEL ASSESSMENT", 8, True, MUTED))
        ml_lay.addSpacing(6)
        oc = {"TP": SUCCESS, "TN": ACCENT, "FP": WARNING, "FN": DANGER}
        for k, v, vc in [
            ("OOF Probability",   f"{ml.oof_probability*100:.2f}%",   prob_color),
            ("Final Probability",  f"{ml.final_probability*100:.2f}%", prob_color),
            ("Predicted",          "Will Seek Loan" if ml.predicted_label else "No Loan", TEXT),
            ("Actual",             "Sought Loan ✓" if ml.actual_label else "No Loan", TEXT),
            ("Outcome",            str(ml.outcome), oc.get(ml.outcome, TEXT)),
            ("Correct",            "Yes ✓" if ml.correct else "No ✗",
             SUCCESS if ml.correct else DANGER),
        ]:
            row = QHBoxLayout()
            row.addWidget(lbl(k, 8, False, MUTED))
            row.addStretch()
            vl = lbl(v, 8, True, vc)
            vl.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(vl)
            ml_lay.addLayout(row)
            sep = QFrame()
            sep.setStyleSheet(f"background: {BORDER}; max-height: 1px; margin: 1px 0;")
            ml_lay.addWidget(sep)
        r1.addWidget(mc, 1)
        lay.addLayout(r1)

        # ── SHAP Waterfall — why this exact prediction? ───────────────────
        shap_card = make_card()
        shap_vlay = QVBoxLayout(shap_card)
        shap_vlay.setContentsMargins(14, 10, 14, 8); shap_vlay.setSpacing(4)
        shap_vlay.addWidget(lbl("SHAP EXPLANATION  —  why this prediction?", 8, True, MUTED))

        shap_cv = MplCanvas(figsize=(9, 2.8), dpi=90)
        ax_shap = shap_cv.ax()

        # Find the positional index of this client in DS.features (same order as model training)
        feat_idx_rows = DS.features[DS.features.client_id == cid]
        shap_ok = (
            DS.shap_data is not None
            and not feat_idx_rows.empty
            and len(DS.shap_data["features"]) > 0
        )

        if shap_ok:
            pos = feat_idx_rows.index[0]           # integer position in features.csv
            sv  = DS.shap_data["values"][pos]      # (n_features,)
            ev  = DS.shap_data["expected_value"]
            feat_names = DS.shap_data["features"]

            # Top 12 features by |SHAP|
            top_n = min(12, len(sv))
            order = np.argsort(np.abs(sv))[::-1][:top_n]
            sv_top = sv[order]
            fn_top = [feat_names[i] for i in order]

            # Plot horizontal waterfall bars (positive=red pushes toward loan, negative=blue)
            colors = [DANGER if v > 0 else ACCENT for v in sv_top]
            y_pos  = np.arange(len(sv_top))

            bars = ax_shap.barh(y_pos, sv_top, color=colors, alpha=0.85, height=0.65)
            ax_shap.set_yticks(y_pos)
            ax_shap.set_yticklabels(fn_top, fontsize=7)
            ax_shap.axvline(0, color=MUTED, linewidth=0.9, linestyle="--", alpha=0.7)
            ax_shap.set_xlabel("SHAP value  (contribution to log-odds of loan)", fontsize=7)
            ax_shap.tick_params(labelsize=7)
            ax_shap.grid(True, alpha=0.15, axis="x")

            # Value labels on bars
            for bar, val in zip(bars, sv_top):
                offset = 0.003 if val >= 0 else -0.003
                ha = "left" if val >= 0 else "right"
                ax_shap.text(
                    val + offset,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha=ha,
                    fontsize=6, color=TEXT, alpha=0.9
                )

            base_prob = 1.0 / (1.0 + np.exp(-ev))
            final_prob = float(ml.final_probability)
            ax_shap.set_title(
                f"Base rate: {base_prob:.1%}   →   Final prediction: {final_prob:.1%}   "
                f"({'🔴 Loan likely' if final_prob >= 0.5 else '🟢 No loan likely'})",
                fontsize=7.5, color=TEXT, pad=5
            )

            # Hover on waterfall
            _shap_annot = ax_shap.annotate(
                "", xy=(0, 0), xytext=(10, 5), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.4", fc=CARD2, ec=ACCENT, lw=1, alpha=0.95),
                fontsize=8, color=TEXT
            )
            _shap_annot.set_visible(False)
            def _shap_hover(event, _ax=ax_shap, _ann=_shap_annot, _cv=shap_cv,
                            _bars=bars, _svt=sv_top, _fnt=fn_top):
                if event.inaxes != _ax:
                    _ann.set_visible(False); _cv.draw_idle(); return
                hit = False
                for bar, val, fname in zip(_bars, _svt, _fnt):
                    if bar.contains_point((event.x, event.y)):
                        direction = "increases" if val > 0 else "decreases"
                        _ann.xy = (val, bar.get_y() + bar.get_height() / 2)
                        _ann.set_text(
                            f"{fname}\nSHAP: {val:+.4f}\n"
                            f"→ {direction} loan probability"
                        )
                        _ann.set_visible(True)
                        hit = True
                        break
                if not hit:
                    _ann.set_visible(False)
                _cv.draw_idle()

            shap_cv.mpl_connect("motion_notify_event", _shap_hover)
            shap_cv.mpl_connect("axes_leave_event",
                                 lambda e: (_shap_annot.set_visible(False), shap_cv.draw_idle()))
        else:
            ax_shap.text(0.5, 0.5,
                         "SHAP data not available.\nRetrain the model with the updated model.py.",
                         ha="center", va="center", color=MUTED, fontsize=9,
                         transform=ax_shap.transAxes)

        shap_cv.redraw()
        shap_vlay.addWidget(shap_cv)
        lay.addWidget(shap_card)

        # ── Balance timeline ──────────────────────────────────────────────
        bc = make_card()
        bl = QVBoxLayout(bc)
        bl.setContentsMargins(14, 12, 14, 8)
        bl.addWidget(lbl("BALANCE OVER TIME  (hover for details)", 8, True, MUTED))
        bal_cv = MplCanvas(figsize=(9, 2.5), dpi=90)
        ax = bal_cv.ax()
        suc = ctx[ctx.status == "SUCCESS"].sort_values("dt")
        _dates  = suc["dt"].values
        _bals   = suc["balance"].values
        _phases = suc["phase"].values
        for phase, color in PHASE_COLORS.items():
            seg = suc[suc.phase == phase]
            if not seg.empty:
                ax.plot(seg.dt, seg.balance, color=color, linewidth=1.8,
                        label=phase, alpha=0.9)
        for dts, col in [("2024-07-01", PHASE_COLORS["Phase 2 Spike"]),
                          ("2024-10-01", PHASE_COLORS["Phase 3 Repaying"]),
                          ("2025-10-01", PHASE_COLORS["Phase 4 Target"])]:
            ax.axvline(pd.Timestamp(dts), color=col, linewidth=0.8,
                       linestyle="--", alpha=0.5)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k")
        )
        ax.legend(fontsize=7, loc="upper right", framealpha=0.4)
        ax.grid(True, alpha=0.2); ax.tick_params(labelsize=7)
        _bal_annot = ax.annotate(
            "", xy=(0,0), xytext=(12, 12), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc=CARD2, ec=ACCENT, lw=1, alpha=0.95),
            fontsize=8, color=TEXT,
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8),
        )
        _bal_annot.set_visible(False)
        def _bal_hover(event):
            if event.inaxes != ax or len(_dates) == 0:
                _bal_annot.set_visible(False); bal_cv.draw_idle(); return
            try:
                xf = matplotlib.dates.date2num(pd.Timestamp(event.xdata))
                xs = matplotlib.dates.date2num(pd.to_datetime(_dates))
                idx = int(np.argmin(np.abs(xs - xf)))
                dt_str = str(pd.Timestamp(_dates[idx]).date())
                bal_v  = int(_bals[idx])
                phase  = _phases[idx]
                _bal_annot.xy = (_dates[idx], _bals[idx])
                _bal_annot.set_text(f"{dt_str}\nBalance: {bal_v:,} AMD\n{phase}")
                _bal_annot.set_visible(True)
            except Exception:
                _bal_annot.set_visible(False)
            bal_cv.draw_idle()
        bal_cv.mpl_connect("motion_notify_event", _bal_hover)
        bal_cv.mpl_connect("axes_leave_event", lambda e: (_bal_annot.set_visible(False), bal_cv.draw_idle()))
        bal_cv.redraw()
        bl.addWidget(bal_cv)
        lay.addWidget(bc)

        # ── Row 2: spend donut + app activity + credit history ────────────
        r2 = QHBoxLayout(); r2.setSpacing(12)

        dc = make_card()
        dl = QVBoxLayout(dc)
        dl.setContentsMargins(14, 12, 14, 8)
        dl.addWidget(lbl("SPEND BY CATEGORY  (hover for %, AMD)", 8, True, MUTED))
        spend = ctx[ctx.status == "SUCCESS"].groupby("category")["amount"].sum()
        dn    = MplCanvas(figsize=(3.0, 2.4), dpi=90)
        ax2   = dn.ax()
        cols  = [ACCENT, ACCENT2, SUCCESS, WARNING, DANGER,
                 "#06b6d4", "#ec4899", "#84cc16"]
        total_spend = spend.values.sum()
        wedges, texts, autotexts = ax2.pie(
            spend.values, labels=spend.index, autopct="%1.0f%%",
            colors=cols[:len(spend)], startangle=90,
            wedgeprops={"edgecolor": CARD, "linewidth": 2},
            pctdistance=0.72,
            textprops={"fontsize": 7, "color": TEXT},
        )
        for at in autotexts:
            at.set_fontsize(7); at.set_color(CARD)
        ax2.add_artist(plt.Circle((0,0), 0.4, fc=CARD))
        _pie_annot = ax2.annotate(
            "", xy=(0,0), xytext=(20, 20), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc=CARD2, ec=ACCENT, lw=1, alpha=0.95),
            fontsize=8, color=TEXT,
        )
        _pie_annot.set_visible(False)
        _pie_cats = spend.index.tolist()
        _pie_vals = spend.values.tolist()
        def _pie_hover(event):
            if event.inaxes != ax2:
                _pie_annot.set_visible(False); dn.draw_idle(); return
            hit = False
            for i, wedge in enumerate(wedges):
                if wedge.contains_point([event.x, event.y]):
                    pct = _pie_vals[i] / total_spend * 100 if total_spend else 0
                    _pie_annot.xy = (0, 0)
                    _pie_annot.set_text(
                        f"{_pie_cats[i]}\n{int(_pie_vals[i]):,} AMD\n{pct:.1f}%"
                    )
                    _pie_annot.set_visible(True)
                    hit = True
                    break
            if not hit:
                _pie_annot.set_visible(False)
            dn.draw_idle()
        dn.mpl_connect("motion_notify_event", _pie_hover)
        dn.mpl_connect("axes_leave_event", lambda e: (_pie_annot.set_visible(False), dn.draw_idle()))
        dn.redraw()
        dl.addWidget(dn)
        r2.addWidget(dc, 2)

        ac = make_card()
        al = QVBoxLayout(ac)
        al.setContentsMargins(14, 12, 14, 12); al.setSpacing(4)
        al.addWidget(lbl("APP ACTIVITY", 8, True, MUTED))
        al.addSpacing(4)
        cutoff3   = pd.Timestamp("2025-07-01")
        tot_sess  = len(capp)
        checks    = int((capp.action == "check_balance").sum())
        ch_ratio  = checks / tot_sess if tot_sess else 0
        act_days  = int(capp.dt.dt.date.nunique())
        sess_3m   = int((capp.dt >= cutoff3).sum())
        for k, v, vc in [
            ("Total Sessions",  tot_sess, TEXT),
            ("Balance Checks",  checks, TEXT),
            ("Check Ratio",     f"{ch_ratio*100:.1f}%",
             DANGER if ch_ratio > 0.6 else TEXT),
            ("Active Days",     act_days, TEXT),
            ("Sessions (3m)",   sess_3m,
             WARNING if sess_3m > 50 else TEXT),
        ]:
            row = QHBoxLayout()
            row.addWidget(lbl(k, 8, False, MUTED))
            row.addStretch()
            row.addWidget(lbl(str(v), 9, True, vc))
            al.addLayout(row)
            sep = QFrame()
            sep.setStyleSheet(f"background: {BORDER}; max-height: 1px; margin: 1px 0;")
            al.addWidget(sep)
        al.addStretch()
        r2.addWidget(ac, 1)

        cc = make_card()
        cl2 = QVBoxLayout(cc)
        cl2.setContentsMargins(14, 12, 14, 12)
        cl2.addWidget(lbl("CREDIT HISTORY", 8, True, MUTED))
        cl2.addSpacing(4)
        if not cr.empty:
            c0 = cr.iloc[0]
            st_col = (SUCCESS if c0.status == "CLOSED"
                      else ACCENT if c0.status == "ACTIVE" else DANGER)
            for k, v, vc in [
                ("Amount",  f"{int(c0.credit_amount):,} AMD", TEXT),
                ("Rate",    f"{c0.annual_rate_pct}%", WARNING),
                ("Term",    f"{int(c0.term_months)} months", TEXT),
                ("Status",  str(c0.status), st_col),
                ("Purpose", str(c0.purpose), TEXT),
            ]:
                row = QHBoxLayout()
                row.addWidget(lbl(k, 8, False, MUTED))
                row.addStretch()
                row.addWidget(lbl(str(v), 8, True, vc))
                cl2.addLayout(row)
            if not pay.empty:
                cl2.addSpacing(4)
                cl2.addWidget(lbl("PAYMENTS", 7, True, MUTED))
                on_time = int((pay.status == "PAID_ON_TIME").sum())
                late    = int((pay.status == "PAID_LATE").sum())
                missed  = int((pay.status == "MISSED").sum())
                for k, v, vc in [("On Time", on_time, SUCCESS),
                                  ("Late",    late,    WARNING),
                                  ("Missed",  missed,  DANGER)]:
                    row = QHBoxLayout()
                    row.addWidget(lbl(k, 8, False, MUTED))
                    row.addStretch()
                    row.addWidget(lbl(str(v), 9, True, vc))
                    cl2.addLayout(row)
        else:
            cl2.addWidget(lbl("No credit record", 9, False, MUTED))
        cl2.addStretch()
        r2.addWidget(cc, 1)
        lay.addLayout(r2)

        # ── Transaction table ─────────────────────────────────────────────
        tc2 = make_card()
        tl2 = QVBoxLayout(tc2)
        tl2.setContentsMargins(14, 12, 14, 12)
        tl2.addWidget(lbl(f"TRANSACTIONS  ({len(ctx)} total)", 8, True, MUTED))
        tl2.addSpacing(4)
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(
            ["Date","Merchant","Category","Amount","Status","Balance"]
        )
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setStyleSheet(TABLE_STYLE)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setFixedHeight(200)
        ctx_s = ctx.sort_values("dt", ascending=False).head(300)
        table.setRowCount(len(ctx_s))
        for i, (_, rw) in enumerate(ctx_s.iterrows()):
            sc = SUCCESS if rw.status == "SUCCESS" else DANGER
            for j, (txt, col) in enumerate([
                (str(rw.transaction_date)[:16], TEXT),
                (str(rw.merchant_name), TEXT),
                (str(rw.category), MUTED),
                (f"{int(rw.amount):,}", TEXT),
                (str(rw.status), sc),
                (f"{int(rw.balance):,}", TEXT),
            ]):
                cell = QTableWidgetItem(txt)
                cell.setForeground(QColor(col))
                table.setItem(i, j, cell)
        tl2.addWidget(table)
        lay.addWidget(tc2)
        lay.addStretch()
        self.setWidget(w)


class ClientCRMPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background: {BG};")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(14)
        self.list_panel   = ClientListPanel()
        self.detail_panel = ClientDetailPanel()
        self.list_panel.client_selected.connect(self.detail_panel.load)
        lay.addWidget(self.list_panel)
        lay.addWidget(self.detail_panel, 1)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2  —  ANALYTICS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

class AnalyticsPage(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setStyleSheet(f"background: {BG}; border: none;")
        w = QWidget(); w.setStyleSheet(f"background: {BG};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 16, 16, 16); lay.setSpacing(12)

        feat = DS.features
        ml   = DS.scores_ml
        cr   = DS.credit
        sc   = DS.scores_cs

        kpi = QHBoxLayout(); kpi.setSpacing(10)
        kpi.addWidget(stat_chip(len(DS.clients), "Total Clients", ACCENT))
        kpi.addWidget(stat_chip(int((ml.risk_tier=="High").sum()), "High Risk", DANGER))
        kpi.addWidget(stat_chip(
            int(cr.status.isin(["ACTIVE","CLOSED"]).sum()), "Loans Issued", SUCCESS
        ))
        kpi.addWidget(stat_chip(int(sc.credit_score.mean()), "Avg Credit Score", WARNING))
        kpi.addWidget(stat_chip(f"{ml['correct'].mean()*100:.0f}%", "Model Accuracy", ACCENT2))
        lay.addLayout(kpi)

        # ── Business Impact ───────────────────────────────────────────────
        bi = make_card()
        bi_lay = QHBoxLayout(bi)
        bi_lay.setContentsMargins(16, 10, 16, 10); bi_lay.setSpacing(20)
        bi_lay.addWidget(lbl("📈  BUSINESS IMPACT", 9, True, ACCENT))
        ml_scores = DS.scores_ml
        n_high     = int((ml_scores.risk_tier == "High").sum())
        n_total    = len(ml_scores)
        base_conv  = 0.25
        model_conv = float(ml_scores[ml_scores.risk_tier=="High"]["actual_label"].mean()) if n_high else 0
        conv_lift  = (model_conv - base_conv) / base_conv * 100 if base_conv else 0
        potential_loans = int(n_high * model_conv)
        for txt, val, col in [
            ("High-probability clients",  str(n_high),                DANGER),
            ("Expected true positives",   str(potential_loans),       SUCCESS),
            ("Conversion lift vs random", f"+{conv_lift:.0f}%",       WARNING),
            ("Coverage (% of portfolio)", f"{n_high/n_total*100:.0f}%", ACCENT),
        ]:
            cv = QVBoxLayout()
            vl = lbl(val, 14, True, col)
            vl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tl = lbl(txt, 7, False, MUTED)
            tl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cv.addWidget(vl); cv.addWidget(tl)
            bi_lay.addLayout(cv)
        bi_lay.addStretch()
        lay.addWidget(bi)

        f0 = feat[feat.will_seek_loan == 0]
        f1 = feat[feat.will_seek_loan == 1]

        def row_of_charts(*cards):
            r = QHBoxLayout(); r.setSpacing(10)
            for c in cards: r.addWidget(c)
            lay.addLayout(r)

        def chart_card(title, figsize=(4.5, 2.7)):
            c = make_card()
            l = QVBoxLayout(c)
            l.setContentsMargins(12, 10, 12, 6)
            l.addWidget(lbl(title.upper(), 7, True, MUTED))
            l.addSpacing(4)
            cv = MplCanvas(figsize=figsize, dpi=90)
            l.addWidget(cv)
            return c, cv

        # ── Row 1: Balance drop + Category spikes ─────────────────────────
        c1, cv1 = chart_card("Balance Drop % by Label  (hover bars)")
        ax1 = cv1.ax()
        ax1.hist(f0.balance_drop_pct, bins=14, color=ACCENT, alpha=0.65, label="No Loan")
        ax1.hist(f1.balance_drop_pct, bins=14, color=DANGER, alpha=0.65, label="Loan")
        ax1.set_xlabel("Balance Drop %", fontsize=8)
        ax1.legend(fontsize=7); ax1.grid(True, alpha=0.2)
        cv1.redraw()
        cv1.attach_hover(ax1, fmt=lambda lbl_, x, y:
                         f"{lbl_}\nBalance Drop: {x:.3f}\nCount: {int(y)}")

        c3, cv3 = chart_card("Avg Category Spike by Label  (hover bars)")
        ax3 = cv3.ax()
        cats  = ["spike_repair","spike_electronics","spike_clothing","spike_travel"]
        names = ["Repair","Elec.","Cloth.","Travel"]
        xp = np.arange(4)
        v0 = [f0[c].clip(0,5).mean() for c in cats]
        v1 = [f1[c].clip(0,5).mean() for c in cats]
        bars0 = ax3.bar(xp-0.2, v0, 0.38, color=ACCENT, alpha=0.8, label="No Loan")
        bars1 = ax3.bar(xp+0.2, v1, 0.38, color=DANGER, alpha=0.8, label="Loan")
        ax3.set_xticks(xp); ax3.set_xticklabels(names, fontsize=8)
        ax3.set_ylabel("Avg Spike Ratio", fontsize=8)
        ax3.legend(fontsize=7); ax3.grid(True, alpha=0.2, axis="y")
        cv3.redraw()
        _ba3 = ax3.annotate("", xy=(0,0), xytext=(8,8), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc=CARD2, ec=ACCENT, lw=1, alpha=0.95),
            fontsize=8, color=TEXT)
        _ba3.set_visible(False)
        def _b3h(event, _ax=ax3, _a=_ba3, _cv=cv3):
            if event.inaxes != _ax: _a.set_visible(False); _cv.draw_idle(); return
            hit = False
            for bar, grp in [(bars0,"No Loan"),(bars1,"Loan")]:
                for rect, cat in zip(bar, names):
                    if rect.contains_point((event.x, event.y)):
                        h = rect.get_height()
                        _a.xy = (rect.get_x()+rect.get_width()/2, h)
                        _a.set_text(f"{grp} — {cat}\nAvg Spike: {h:.3f}")
                        _a.set_visible(True); hit = True; break
                if hit: break
            if not hit: _a.set_visible(False)
            _cv.draw_idle()
        cv3.mpl_connect("motion_notify_event", _b3h)
        cv3.mpl_connect("axes_leave_event", lambda e: (_ba3.set_visible(False), cv3.draw_idle()))
        row_of_charts(c1, c3)

        # ── Row 2: Urgency distribution + Phase transition ────────────────
        c4, cv4 = chart_card("Urgency Score Distribution  (hover bars)")
        ax4 = cv4.ax()
        ax4.hist(f0.urgency_score, bins=16, color=ACCENT, alpha=0.65, label="No Loan", density=True)
        ax4.hist(f1.urgency_score, bins=16, color=DANGER, alpha=0.65, label="Loan", density=True)
        ax4.set_xlabel("Urgency Score", fontsize=8)
        ax4.set_ylabel("Density", fontsize=8)
        ax4.legend(fontsize=7); ax4.grid(True, alpha=0.2)
        cv4.redraw()
        cv4.attach_hover(ax4, fmt=lambda lbl_, x, y:
                         f"{lbl_}\nScore: {x:.3f}\nDensity: {y:.3f}")

        c6, cv6 = chart_card("Phase Transition: Balance-Check Ratio  (hover points)")
        ax6 = cv6.ax()
        phase_labels = ["Ph.1\nNormal","Ph.2\nSpike","Ph.3\nEarly","Ph.3\nLate","Ph.4\nTarget"]
        windows = [
            (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-07-01")),
            (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-10-01")),
            (pd.Timestamp("2024-10-01"), pd.Timestamp("2025-04-01")),
            (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-10-01")),
            (pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-01")),
        ]
        for label_val, color, name in [(0, ACCENT, "No Loan"), (1, DANGER, "Loan")]:
            cids = feat[feat.will_seek_loan == label_val].client_id.tolist()
            sub  = DS.app[DS.app.client_id.isin(cids)]
            vals = []
            for s, e in windows:
                ww = sub[(sub.dt >= s) & (sub.dt < e)]
                vals.append(float((ww.action == "check_balance").mean()) if len(ww) else 0.0)
            ax6.plot(range(5), vals, marker="o", linewidth=2, color=color, markersize=6, label=name)
        ax6.set_xticks(range(5)); ax6.set_xticklabels(phase_labels, fontsize=7)
        ax6.set_ylabel("Avg Check Ratio", fontsize=8)
        ax6.legend(fontsize=7); ax6.grid(True, alpha=0.2)
        cv6.redraw()
        cv6.attach_hover(ax6, fmt=lambda lbl_, x, y:
                         f"{lbl_}\n{phase_labels[int(round(x))] if 0<=round(x)<=4 else ''}\nCheck Ratio: {y:.3f}")
        row_of_charts(c4, c6)

        # ── Row 3: Burn acceleration scatter + Balance-check scatter ──────
        c2, cv2 = chart_card("Burn Acceleration vs Rate Ratio  (hover points)")
        ax2 = cv2.ax()
        ax2.scatter(f0.burn_acceleration.clip(0,5), f0.burn_rate_ratio.clip(0,3),
                    c=ACCENT, alpha=0.65, s=30, label="No Loan")
        ax2.scatter(f1.burn_acceleration.clip(0,5), f1.burn_rate_ratio.clip(0,3),
                    c=DANGER, alpha=0.65, s=30, label="Loan")
        ax2.set_xlabel("Burn Acceleration", fontsize=8)
        ax2.set_ylabel("Burn Rate Ratio", fontsize=8)
        ax2.legend(fontsize=7); ax2.grid(True, alpha=0.2)
        cv2.redraw()
        cv2.attach_hover(ax2, fmt=lambda lbl_, x, y:
                         f"{lbl_}\nAcceleration: {x:.2f}\nBurn Ratio: {y:.2f}")

        c5, cv5 = chart_card("Balance-Check: Historical vs Recent 3m  (hover points)")
        ax5 = cv5.ax()
        ax5.scatter(f0.balance_check_ratio, f0.check_ratio_last_3m,
                    c=ACCENT, alpha=0.65, s=30, label="No Loan")
        ax5.scatter(f1.balance_check_ratio, f1.check_ratio_last_3m,
                    c=DANGER, alpha=0.65, s=30, label="Loan")
        ax5.plot([0,1],[0,1], "--", color=MUTED, linewidth=0.8, alpha=0.5)
        ax5.set_xlabel("Historical Check Ratio", fontsize=8)
        ax5.set_ylabel("Recent Check Ratio (3m)", fontsize=8)
        ax5.legend(fontsize=7); ax5.grid(True, alpha=0.2)
        cv5.redraw()
        cv5.attach_hover(ax5, fmt=lambda lbl_, x, y:
                         f"{lbl_}\nHistorical: {x:.3f}\nRecent 3m: {y:.3f}")
        row_of_charts(c2, c5)

        # ── Row 4: SHAP Analytics (beeswarm + summary bar) ────────────────
        if DS.shap_data is not None and len(DS.shap_data["features"]) > 0:
            shap_hdr = make_card()
            shap_hdr_lay = QHBoxLayout(shap_hdr)
            shap_hdr_lay.setContentsMargins(14, 8, 14, 8)
            shap_hdr_lay.addWidget(lbl("🔍  SHAP — MODEL EXPLANATION ANALYTICS", 9, True, ACCENT))
            shap_hdr_lay.addStretch()
            shap_hdr_lay.addWidget(lbl(
                "Red = high feature value  ·  Blue = low feature value  ·  "
                "Positive SHAP = increases loan probability", 7, False, MUTED
            ))
            lay.addWidget(shap_hdr)

            sv_all     = DS.shap_data["values"]       # (n_clients, n_features)
            feat_names = DS.shap_data["features"]
            # Get feature values matrix (align columns to SHAP feature order)
            try:
                X_vals = DS.features[feat_names].values.astype(float)
            except Exception:
                X_vals = np.zeros_like(sv_all)

            mean_abs = np.abs(sv_all).mean(axis=0)
            top_n_bee = min(15, len(feat_names))
            order_bee  = np.argsort(mean_abs)[-top_n_bee:]   # bottom=least important

            # ── Beeswarm ──────────────────────────────────────────────────
            c_bs, cv_bs = chart_card(
                "SHAP Beeswarm — per-client impact for each feature  (hover for details)",
                figsize=(4.5, 4.0)
            )
            ax_bs = cv_bs.ax()
            np.random.seed(0)
            _bee_pts = []   # for hover: (scatter_obj, feat_idx, feat_name)
            for plot_i, feat_i in enumerate(order_bee):
                sv_feat  = sv_all[:, feat_i]
                x_feat   = X_vals[:, feat_i]
                x_min, x_max = x_feat.min(), x_feat.max()
                x_norm   = (x_feat - x_min) / (x_max - x_min + 1e-9)
                jitter   = np.random.uniform(-0.3, 0.3, len(sv_feat))
                sc = ax_bs.scatter(
                    sv_feat, plot_i + jitter,
                    c=x_norm, cmap="RdBu_r",
                    alpha=0.65, s=12, vmin=0, vmax=1,
                    label=feat_names[feat_i]
                )
                _bee_pts.append((sc, feat_i, feat_names[feat_i], sv_feat, x_feat))
            ax_bs.set_yticks(range(top_n_bee))
            ax_bs.set_yticklabels([feat_names[i] for i in order_bee], fontsize=6)
            ax_bs.axvline(0, color=MUTED, linewidth=0.9, linestyle="--", alpha=0.6)
            ax_bs.set_xlabel("SHAP value  (→ increases loan probability)", fontsize=7)
            ax_bs.grid(True, alpha=0.15, axis="x")
            ax_bs.tick_params(labelsize=6)
            cv_bs.redraw()

            # Hover on beeswarm
            _bee_annot = ax_bs.annotate(
                "", xy=(0, 0), xytext=(10, 5), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.4", fc=CARD2, ec=ACCENT, lw=1, alpha=0.95),
                fontsize=8, color=TEXT
            )
            _bee_annot.set_visible(False)
            def _bee_hover(event, _ax=ax_bs, _ann=_bee_annot, _cv=cv_bs, _pts=_bee_pts):
                if event.inaxes != _ax:
                    _ann.set_visible(False); _cv.draw_idle(); return
                hit = False
                for sc, feat_i, fname, sv_feat, x_feat in _pts:
                    cont, ind = sc.contains(event)
                    if cont and len(ind["ind"]) > 0:
                        idx  = ind["ind"][0]
                        sval = sv_feat[idx]
                        fval = x_feat[idx]
                        _ann.xy = (sval, sc.get_offsets()[idx][1])
                        _ann.set_text(
                            f"{fname}\nSHAP: {sval:+.4f}\nFeature value: {fval:.3g}"
                        )
                        _ann.set_visible(True)
                        hit = True
                        break
                if not hit:
                    _ann.set_visible(False)
                _cv.draw_idle()
            cv_bs.mpl_connect("motion_notify_event", _bee_hover)
            cv_bs.mpl_connect("axes_leave_event",
                               lambda e: (_bee_annot.set_visible(False), cv_bs.draw_idle()))

            # ── SHAP Summary Bar ──────────────────────────────────────────
            c_sb, cv_sb = chart_card(
                "SHAP Summary Bar — mean |SHAP| per feature  (hover bars)",
                figsize=(4.5, 4.0)
            )
            ax_sb = cv_sb.ax()
            order_sb  = np.argsort(mean_abs)[-top_n_bee:]
            top_names = [feat_names[i] for i in order_sb]
            top_vals  = mean_abs[order_sb]
            mean_mean = top_vals.mean()
            bar_colors = [DANGER if v >= mean_mean else ACCENT for v in top_vals]
            bars_sb = ax_sb.barh(range(top_n_bee), top_vals,
                                 color=bar_colors, alpha=0.85, height=0.65)
            ax_sb.set_yticks(range(top_n_bee))
            ax_sb.set_yticklabels(top_names, fontsize=6)
            ax_sb.set_xlabel("Mean |SHAP value|  (average absolute impact)", fontsize=7)
            ax_sb.tick_params(labelsize=6)
            ax_sb.grid(True, alpha=0.2, axis="x")
            ax_sb.axvline(mean_mean, color=WARNING, linewidth=0.8,
                          linestyle=":", alpha=0.7, label="mean")
            ax_sb.legend(fontsize=6)
            cv_sb.redraw()

            _sb_annot = ax_sb.annotate(
                "", xy=(0, 0), xytext=(8, 4), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.4", fc=CARD2, ec=ACCENT, lw=1, alpha=0.95),
                fontsize=8, color=TEXT
            )
            _sb_annot.set_visible(False)
            def _sb_hover(event, _ax=ax_sb, _ann=_sb_annot, _cv=cv_sb,
                          _bars=bars_sb, _names=top_names, _vals=top_vals):
                if event.inaxes != _ax:
                    _ann.set_visible(False); _cv.draw_idle(); return
                hit = False
                for bar, name, val in zip(_bars, _names, _vals):
                    if bar.contains_point((event.x, event.y)):
                        _ann.xy = (val, bar.get_y() + bar.get_height() / 2)
                        _ann.set_text(
                            f"{name}\nMean |SHAP|: {val:.4f}\n"
                            f"Rank: #{list(reversed(_names)).index(name)+1}"
                        )
                        _ann.set_visible(True)
                        hit = True
                        break
                if not hit:
                    _ann.set_visible(False)
                _cv.draw_idle()
            cv_sb.mpl_connect("motion_notify_event", _sb_hover)
            cv_sb.mpl_connect("axes_leave_event",
                               lambda e: (_sb_annot.set_visible(False), cv_sb.draw_idle()))

            row_of_charts(c_bs, c_sb)

        lay.addStretch()
        self.setWidget(w)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3  —  PREDICTION & SCORING
# ─────────────────────────────────────────────────────────────────────────────

class PredictionPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background: {BG};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16); lay.setSpacing(12)

        top = QHBoxLayout(); top.setSpacing(12)

        tc = make_card()
        tl = QVBoxLayout(tc)
        tl.setContentsMargins(14, 12, 14, 12); tl.setSpacing(6)
        th = QHBoxLayout()
        th.addWidget(lbl("CLIENT SCORES", 8, True, MUTED))
        th.addStretch()
        self.tier_cb = QComboBox()
        self.tier_cb.addItems(["All Tiers","High","Medium","Low","Very Low"])
        self.tier_cb.setStyleSheet(
            f"QComboBox {{ background: {CARD2}; color: {TEXT};"
            f" border: 1px solid {BORDER}; border-radius: 5px;"
            f" padding: 3px 8px; font-size: 9px; }}"
            f"QComboBox::drop-down {{ border: none; }}"
            f"QComboBox QAbstractItemView {{ background: {CARD2}; color: {TEXT}; }}"
        )
        self.tier_cb.currentTextChanged.connect(self._filter_table)
        th.addWidget(self.tier_cb)
        tl.addLayout(th)

        self.score_table = QTableWidget()
        self.score_table.setColumnCount(5)
        self.score_table.setHorizontalHeaderLabels(
            ["Name","OOF Prob","Tier","Actual","Outcome"]
        )
        self.score_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.score_table.setStyleSheet(TABLE_STYLE)
        self.score_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.score_table.verticalHeader().setVisible(False)
        self.score_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        tl.addWidget(self.score_table)

        # ── Target List Output ────────────────────────────────────────────
        tgt_bar = QFrame()
        tgt_bar.setStyleSheet(
            f"QFrame {{ background: {CARD2}; border-radius: 6px; border: 1px solid {BORDER}; }}"
        )
        tgt_lay = QHBoxLayout(tgt_bar)
        tgt_lay.setContentsMargins(10, 6, 10, 6); tgt_lay.setSpacing(16)
        self._tgt_lbl_count = lbl("—", 10, True, DANGER)
        self._tgt_lbl_tp    = lbl("—", 10, True, SUCCESS)
        self._tgt_lbl_cov   = lbl("—", 10, True, WARNING)
        for val_lbl, caption in [
            (self._tgt_lbl_count, "Clients above threshold"),
            (self._tgt_lbl_tp,    "Expected true positives"),
            (self._tgt_lbl_cov,   "Coverage of total pool"),
        ]:
            col = QVBoxLayout(); col.setSpacing(0)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            sub = lbl(caption, 7, False, MUTED)
            sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(val_lbl); col.addWidget(sub)
            tgt_lay.addLayout(col)
        tgt_lay.addStretch()
        tgt_lay.addWidget(lbl("← updates with slider", 7, False, MUTED))
        tl.addWidget(tgt_bar)
        top.addWidget(tc, 3)

        rp_col = QVBoxLayout(); rp_col.setSpacing(10)
        for attr, title, fs in [
            ("roc_cv", "ROC CURVE  (hover for FPR/TPR)", (3.0, 2.0)),
            ("pr_cv",  "PRECISION-RECALL  (hover for P/R)", (3.0, 2.0)),
        ]:
            card = make_card()
            cl2  = QVBoxLayout(card)
            cl2.setContentsMargins(12, 10, 12, 6)
            cl2.addWidget(lbl(title, 7, True, MUTED))
            cv = MplCanvas(figsize=fs, dpi=90)
            setattr(self, attr, cv)
            cl2.addWidget(cv)
            rp_col.addWidget(card)
        top.addLayout(rp_col, 2)
        lay.addLayout(top)

        # ── Threshold simulator ───────────────────────────────────────────
        thresh = make_card()
        thr_lay = QHBoxLayout(thresh)
        thr_lay.setContentsMargins(16, 14, 16, 14); thr_lay.setSpacing(16)

        left = QVBoxLayout(); left.setSpacing(8)
        left.addWidget(lbl("THRESHOLD SIMULATOR  (drag slider)", 8, True, MUTED))

        sr = QHBoxLayout()
        self.thr_lbl = lbl("0.50", 14, True, ACCENT)
        self.thr_lbl.setFixedWidth(48)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100); self.slider.setValue(50)
        self.slider.setStyleSheet(
            f"QSlider::groove:horizontal {{ background: {BORDER};"
            f" height: 6px; border-radius: 3px; }}"
            f"QSlider::handle:horizontal {{ background: {ACCENT};"
            f" width: 15px; height: 15px; border-radius: 7px; margin: -4px 0; }}"
            f"QSlider::sub-page:horizontal {{ background: {ACCENT};"
            f" border-radius: 3px; }}"
        )
        self.slider.valueChanged.connect(self._update_threshold)
        sr.addWidget(self.thr_lbl); sr.addWidget(self.slider)
        left.addLayout(sr)

        chips = QHBoxLayout(); chips.setSpacing(8)
        self.mchips = {}
        for m in ["Precision","Recall","F1","Flagged","True Pos"]:
            chip = stat_chip("—", m, ACCENT)
            chip.setFixedHeight(62)
            self.mchips[m] = chip.findChild(QLabel)
            chips.addWidget(chip)
        left.addLayout(chips)
        thr_lay.addLayout(left, 2)

        thr_c = make_card()
        tcl   = QVBoxLayout(thr_c)
        tcl.setContentsMargins(8,8,8,8)
        self.thr_cv = MplCanvas(figsize=(4.5, 2.1), dpi=90)
        tcl.addWidget(self.thr_cv)
        thr_lay.addWidget(thr_c, 3)
        lay.addWidget(thresh)

        # ── Bottom: importance + tier donut + cm + summary ────────────────
        bot = QHBoxLayout(); bot.setSpacing(12)

        imp_c = make_card()
        il    = QVBoxLayout(imp_c)
        il.setContentsMargins(14, 12, 14, 8)
        il.addWidget(lbl("FEATURE IMPORTANCE TOP 15  (hover bars)", 7, True, MUTED))
        self.imp_cv = MplCanvas(figsize=(4.5, 3.4), dpi=90)
        il.addWidget(self.imp_cv)
        bot.addWidget(imp_c, 5)

        tier_c = make_card()
        trl    = QVBoxLayout(tier_c)
        trl.setContentsMargins(14, 12, 14, 8)
        trl.addWidget(lbl("RISK TIER DISTRIBUTION  (hover wedges)", 7, True, MUTED))
        self.tier_cv = MplCanvas(figsize=(2.4, 2.4), dpi=90)
        trl.addWidget(self.tier_cv)
        bot.addWidget(tier_c, 3)

        rc = QVBoxLayout(); rc.setSpacing(10)
        cm_c = make_card()
        cml  = QVBoxLayout(cm_c)
        cml.setContentsMargins(14, 12, 14, 8)
        cml.addWidget(lbl("CONFUSION MATRIX  (hover cells)", 7, True, MUTED))
        self.cm_cv = MplCanvas(figsize=(2.4, 2.1), dpi=90)
        cml.addWidget(self.cm_cv)
        rc.addWidget(cm_c)

        from sklearn.metrics import (accuracy_score, precision_score,
                                     recall_score, f1_score)
        ml = DS.scores_ml
        y2 = ml.actual_label.values; yp = ml.predicted_label.values
        oof_auc, oof_ap = 0.0, 0.0
        try:
            with open(f"{DATA_DIR}/model.pkl","rb") as pf:
                pkg = pickle.load(pf)
                oof_auc = pkg.get("oof_auc", 0)
                oof_ap  = pkg.get("oof_ap",  0)
        except Exception:
            pass

        sum_c = make_card()
        sl    = QVBoxLayout(sum_c)
        sl.setContentsMargins(14, 12, 14, 12); sl.setSpacing(3)
        sl.addWidget(lbl("MODEL SUMMARY", 8, True, MUTED))
        sl.addSpacing(4)
        for k, v in [
            ("Algorithm",    "LightGBM"),
            ("CV Strategy",  "5-Fold Stratified"),
            ("OOF AUC",      f"{oof_auc:.4f}"),
            ("OOF Avg Prec", f"{oof_ap:.4f}"),
            ("Accuracy",     f"{accuracy_score(y2,yp)*100:.1f}%"),
            ("Precision",    f"{precision_score(y2,yp,zero_division=0)*100:.1f}%"),
            ("Recall",       f"{recall_score(y2,yp,zero_division=0)*100:.1f}%"),
            ("F1",           f"{f1_score(y2,yp,zero_division=0):.3f}"),
        ]:
            row = QHBoxLayout()
            row.addWidget(lbl(k, 7, False, MUTED))
            row.addStretch()
            row.addWidget(lbl(v, 8, True, TEXT))
            sl.addLayout(row)
            sep = QFrame()
            sep.setStyleSheet(f"background: {BORDER}; max-height: 1px; margin: 1px 0;")
            sl.addWidget(sep)
        sl.addStretch()
        rc.addWidget(sum_c)
        bot.addLayout(rc, 3)
        lay.addLayout(bot)

        self._fill_table(DS.scores_ml.sort_values("oof_probability", ascending=False))
        self._render_static_charts()
        self._update_threshold(50)

    def _fill_table(self, df):
        oc = {"TP": SUCCESS, "TN": ACCENT, "FP": WARNING, "FN": DANGER}
        self.score_table.setRowCount(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            for j, (txt, col) in enumerate([
                (str(row["name"]),  TEXT),
                (f"{row['oof_probability']*100:.1f}%",
                 DANGER if row["oof_probability"] >= 0.5 else SUCCESS),
                (str(row["risk_tier"]),   TIER_COLORS.get(row["risk_tier"], MUTED)),
                ("✓ Loan" if row["actual_label"] else "No Loan", TEXT),
                (str(row["outcome"]),     oc.get(row["outcome"], TEXT)),
            ]):
                cell = QTableWidgetItem(txt)
                cell.setForeground(QColor(col))
                self.score_table.setItem(i, j, cell)

    def _filter_table(self, tier):
        ml = DS.scores_ml.sort_values("oof_probability", ascending=False)
        self._fill_table(ml if tier == "All Tiers" else ml[ml.risk_tier == tier])

    def _render_static_charts(self):
        from sklearn.metrics import (roc_curve, precision_recall_curve,
                                     roc_auc_score, average_precision_score)
        ml   = DS.scores_ml
        y    = ml.actual_label.values
        prob = ml.oof_probability.values

        # ROC
        fpr, tpr, _ = roc_curve(y, prob)
        auc_val      = roc_auc_score(y, prob)
        ax = self.roc_cv.ax()
        ax.plot(fpr, tpr, color=ACCENT, linewidth=2, label=f"AUC={auc_val:.3f}")
        ax.plot([0,1],[0,1],"--", color=MUTED, linewidth=0.8)
        ax.fill_between(fpr, tpr, alpha=0.12, color=ACCENT)
        ax.set_xlabel("FPR", fontsize=8); ax.set_ylabel("TPR", fontsize=8)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
        self.roc_cv.redraw()
        self.roc_cv.attach_hover(ax, fmt=lambda lbl_, x, y:
                                  f"ROC\nFPR: {x:.3f}\nTPR: {y:.3f}")

        # PR
        prec, rec, _ = precision_recall_curve(y, prob)
        ap_val        = average_precision_score(y, prob)
        ax2 = self.pr_cv.ax()
        ax2.plot(rec, prec, color=SUCCESS, linewidth=2, label=f"AP={ap_val:.3f}")
        ax2.fill_between(rec, prec, alpha=0.12, color=SUCCESS)
        ax2.set_xlabel("Recall", fontsize=8); ax2.set_ylabel("Precision", fontsize=8)
        ax2.legend(fontsize=7); ax2.grid(True, alpha=0.2)
        self.pr_cv.redraw()
        self.pr_cv.attach_hover(ax2, fmt=lambda lbl_, x, y:
                                  f"P-R Curve\nRecall: {x:.3f}\nPrecision: {y:.3f}")

        # Feature importance (gain-based from feature_importance.csv)
        imp = DS.importance.head(15)
        def feat_color(f):
            if "check" in f or "session" in f: return ACCENT2
            if "spike" in f or "share" in f:   return DANGER
            if "balance" in f or "burn" in f:  return ACCENT
            if "credit" in f or "dti" in f:    return SUCCESS
            return WARNING
        ax3 = self.imp_cv.ax()
        bars = ax3.barh(imp.feature, imp.importance,
                        color=[feat_color(f) for f in imp.feature], alpha=0.85)
        ax3.invert_yaxis()
        ax3.set_xlabel("Importance (Gain)", fontsize=8)
        ax3.tick_params(labelsize=7)
        ax3.grid(True, alpha=0.2, axis="x")
        self.imp_cv.redraw()
        _imp_annot = ax3.annotate("", xy=(0,0), xytext=(8,4), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc=CARD2, ec=ACCENT, lw=1, alpha=0.95),
            fontsize=8, color=TEXT)
        _imp_annot.set_visible(False)
        _imp_feats = imp.feature.tolist()
        _imp_gains = imp.importance.tolist()
        def _imp_hover(event):
            if event.inaxes != ax3:
                _imp_annot.set_visible(False); self.imp_cv.draw_idle(); return
            hit = False
            for bar, feat_name, gain in zip(bars, _imp_feats, _imp_gains):
                if bar.contains_point((event.x, event.y)):
                    _imp_annot.xy = (gain, bar.get_y() + bar.get_height()/2)
                    _imp_annot.set_text(f"{feat_name}\nGain: {gain:.4f}  ({gain/sum(_imp_gains)*100:.1f}%)")
                    _imp_annot.set_visible(True)
                    hit = True; break
            if not hit: _imp_annot.set_visible(False)
            self.imp_cv.draw_idle()
        self.imp_cv.mpl_connect("motion_notify_event", _imp_hover)
        self.imp_cv.mpl_connect("axes_leave_event",
                                 lambda e: (_imp_annot.set_visible(False), self.imp_cv.draw_idle()))

        # Tier donut
        tiers = ml.risk_tier.value_counts()
        ax4   = self.tier_cv.ax()
        _pct_total = tiers.values.sum()
        _pie_labels = [n if tiers.values[i]/_pct_total >= 0.05 else ""
                       for i, n in enumerate(tiers.index)]
        _tier_wedges, _tier_texts, _tier_autos = ax4.pie(
            tiers.values, labels=_pie_labels,
            colors=[TIER_COLORS.get(t, MUTED) for t in tiers.index],
            autopct=lambda p: f"{p:.0f}%" if p >= 5 else "",
            startangle=90,
            wedgeprops={"edgecolor": CARD, "linewidth": 2},
            textprops={"fontsize": 7, "color": TEXT},
            pctdistance=0.75)
        ax4.add_artist(plt.Circle((0,0), 0.4, fc=CARD))
        self.tier_cv.redraw()
        _tier_names  = tiers.index.tolist()
        _tier_vals   = tiers.values.tolist()
        _tier_total  = sum(_tier_vals)
        _tier_annot  = ax4.annotate("", xy=(0,0), xytext=(15,15),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc=CARD2, ec=ACCENT, lw=1, alpha=0.95),
            fontsize=8, color=TEXT)
        _tier_annot.set_visible(False)
        def _tier_hover(event):
            if event.inaxes != ax4:
                _tier_annot.set_visible(False); self.tier_cv.draw_idle(); return
            hit = False
            for i, wedge in enumerate(_tier_wedges):
                if wedge.contains_point([event.x, event.y]):
                    pct = _tier_vals[i] / _tier_total * 100 if _tier_total else 0
                    _tier_annot.xy = (0, 0)
                    _tier_annot.set_text(
                        f"{_tier_names[i]}\n{_tier_vals[i]} clients\n{pct:.0f}%"
                    )
                    _tier_annot.set_visible(True)
                    hit = True; break
            if not hit: _tier_annot.set_visible(False)
            self.tier_cv.draw_idle()
        self.tier_cv.mpl_connect("motion_notify_event", _tier_hover)
        self.tier_cv.mpl_connect("axes_leave_event",
                                  lambda e: (_tier_annot.set_visible(False), self.tier_cv.draw_idle()))

        # Confusion matrix
        pred = ml.predicted_label.values
        tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum())
        fn = int(((pred==0)&(y==1)).sum()); tn = int(((pred==0)&(y==0)).sum())
        cm = np.array([[tn,fp],[fn,tp]])
        _cm_labels = [["TN\n(Correct: No Loan)", "FP\n(Wrong: False Alarm)"],
                      ["FN\n(Missed Loan)",       "TP\n(Correct: Caught Loan)"]]
        ax5 = self.cm_cv.ax()
        ax5.imshow(cm, cmap="Blues", aspect="auto", vmin=0, vmax=cm.max()*1.5)
        cm_max = cm.max() or 1
        for i in range(2):
            for j in range(2):
                brightness = cm[i,j] / cm_max
                txt_color  = "#0f1117" if brightness > 0.55 else TEXT
                ax5.text(j, i, str(cm[i,j]), ha="center", va="center",
                         fontsize=13, fontweight="bold", color=txt_color)
        ax5.set_xticks([0,1])
        ax5.set_xticklabels(["Pred:No","Pred:Yes"], fontsize=7)
        ax5.set_yticks([0,1])
        ax5.set_yticklabels(["Act:No","Act:Yes"], fontsize=7)
        self.cm_cv.redraw()
        _cm_annot = ax5.annotate("", xy=(0,0), xytext=(8,8), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc=CARD2, ec=ACCENT, lw=1, alpha=0.95),
            fontsize=8, color=TEXT)
        _cm_annot.set_visible(False)
        def _cm_hover(event):
            if event.inaxes != ax5:
                _cm_annot.set_visible(False); self.cm_cv.draw_idle(); return
            col = int(round(event.xdata)) if event.xdata is not None else -1
            row = int(round(event.ydata)) if event.ydata is not None else -1
            if 0 <= row <= 1 and 0 <= col <= 1:
                _cm_annot.xy = (col, row)
                _cm_annot.set_text(f"{_cm_labels[row][col]}\nCount: {cm[row,col]}")
                _cm_annot.set_visible(True)
            else:
                _cm_annot.set_visible(False)
            self.cm_cv.draw_idle()
        self.cm_cv.mpl_connect("motion_notify_event", _cm_hover)
        self.cm_cv.mpl_connect("axes_leave_event",
                                lambda e: (_cm_annot.set_visible(False), self.cm_cv.draw_idle()))

    def _update_threshold(self, val):
        t    = val / 100
        self.thr_lbl.setText(f"{t:.2f}")
        ml   = DS.scores_ml
        prob = ml.oof_probability.values
        y    = ml.actual_label.values
        pred = (prob >= t).astype(int)
        tp   = int(((pred==1)&(y==1)).sum())
        fp   = int(((pred==1)&(y==0)).sum())
        fn   = int(((pred==0)&(y==1)).sum())
        flagged = tp + fp
        prec = tp/flagged if flagged else 0.0
        rec  = tp/(tp+fn) if (tp+fn) else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        for k, v in [
            ("Precision", f"{prec*100:.1f}%"),
            ("Recall",    f"{rec*100:.1f}%"),
            ("F1",        f"{f1:.3f}"),
            ("Flagged",   str(flagged)),
            ("True Pos",  str(tp)),
        ]:
            lbl_w = self.mchips.get(k)
            if lbl_w: lbl_w.setText(v)

        n_total = len(ml)
        self._tgt_lbl_count.setText(str(flagged))
        self._tgt_lbl_tp.setText(str(tp))
        self._tgt_lbl_cov.setText(f"{flagged/n_total*100:.0f}%")

        thresholds = np.linspace(0, 1, 101)
        precs, recs, f1s = [], [], []
        for th in thresholds:
            p2 = (prob >= th).astype(int)
            tp2 = int(((p2==1)&(y==1)).sum())
            fp2 = int(((p2==1)&(y==0)).sum())
            fn2 = int(((p2==0)&(y==1)).sum())
            fl  = tp2+fp2
            pr2 = tp2/fl if fl else 0.0
            re2 = tp2/(tp2+fn2) if (tp2+fn2) else 0.0
            precs.append(pr2*100); recs.append(re2*100)
            f1s.append(2*pr2*re2/(pr2+re2)*100 if (pr2+re2) else 0.0)

        self.thr_cv.fig.clear()
        ax = self.thr_cv.fig.add_subplot(111)
        ax.set_facecolor(CARD)
        ax.plot(thresholds, precs, color=DANGER,  linewidth=2, label="Precision %")
        ax.plot(thresholds, recs,  color=ACCENT,  linewidth=2, label="Recall %")
        ax.plot(thresholds, f1s,   color=SUCCESS, linewidth=1.5, linestyle="--", label="F1×100")
        ax.axvline(t, color=TEXT, linewidth=1.2, linestyle=":", alpha=0.7,
                   label=f"t={t:.2f}")
        ax.set_xlim(0,1); ax.set_ylim(0,105)
        ax.set_xlabel("Threshold", fontsize=8); ax.set_ylabel("%", fontsize=8)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        _thr_annot = ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc=CARD2, ec=ACCENT, lw=1, alpha=0.95),
            fontsize=8, color=TEXT)
        _thr_annot.set_visible(False)
        def _thr_hover(event):
            if event.inaxes != ax or event.xdata is None:
                _thr_annot.set_visible(False); self.thr_cv.draw_idle(); return
            xi = int(np.clip(round(event.xdata * 100), 0, 100))
            _thr_annot.xy = (thresholds[xi], event.ydata or 50)
            _thr_annot.set_text(
                f"Threshold: {thresholds[xi]:.2f}\n"
                f"Precision: {precs[xi]:.1f}%\n"
                f"Recall: {recs[xi]:.1f}%\n"
                f"F1×100: {f1s[xi]:.1f}"
            )
            _thr_annot.set_visible(True)
            self.thr_cv.draw_idle()
        self.thr_cv.mpl_connect("motion_notify_event", _thr_hover)
        self.thr_cv.mpl_connect("axes_leave_event",
                                 lambda e: (_thr_annot.set_visible(False), self.thr_cv.draw_idle()))
        self.thr_cv.redraw()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Loan Propensity CRM")
        self.setMinimumSize(1280, 820)

        central = QWidget()
        central.setStyleSheet(f"background: {BG};")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        sidebar = QWidget()
        sidebar.setFixedWidth(68)
        sidebar.setStyleSheet(
            f"background: {CARD}; border-right: 1px solid {BORDER};"
        )
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(8, 14, 8, 14); sb.setSpacing(6)

        logo = QLabel("🏦")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setStyleSheet("font-size: 26px; background: transparent;")
        sb.addWidget(logo); sb.addSpacing(14)

        self.nav_btns = []
        for i, (ico, tip) in enumerate([
            ("👤","Client CRM"), ("📊","Analytics"), ("🤖","Prediction")
        ]):
            btn = QPushButton(ico)
            btn.setFixedSize(50, 50)
            btn.setToolTip(tip)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, idx=i: self._nav(idx))
            btn.setStyleSheet(
                f"QPushButton {{ background: transparent; border: none;"
                f" border-radius: 10px; font-size: 20px; color: {MUTED}; }}"
                f"QPushButton:checked {{ background: {ACCENT}22; color: {ACCENT};"
                f" border: 1px solid {ACCENT}44; }}"
                f"QPushButton:hover:!checked {{ background: {CARD2}; }}"
            )
            sb.addWidget(btn, alignment=Qt.AlignmentFlag.AlignHCenter)
            self.nav_btns.append(btn)

        sb.addStretch()
        ver = lbl("v3.0", 7, False, MUTED)
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sb.addWidget(ver)
        root.addWidget(sidebar)

        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"background: {BG};")
        self.stack.addWidget(ClientCRMPage())
        self.stack.addWidget(AnalyticsPage())
        self.stack.addWidget(PredictionPage())
        root.addWidget(self.stack, 1)

        self._nav(0)

    def _nav(self, idx):
        self.stack.setCurrentIndex(idx)
        for i, btn in enumerate(self.nav_btns):
            btn.setChecked(i == idx)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Loan Propensity CRM")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,     QColor(BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base,       QColor(CARD))
    palette.setColor(QPalette.ColorRole.Text,       QColor(TEXT))
    app.setPalette(palette)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# HOW TO RUN & PACKAGE
# ─────────────────────────────────────────────────────────────────────────────
# 1. Install requirements (one-time):
#       pip install PyQt6 matplotlib pandas numpy scikit-learn lightgbm shap
#
# 2. Run directly:
#       python crm_app2.py
#       (data/ folder must be one level above scripts/)
#
# 3. Build .exe automatically (one command):
#       python crm_app2.py --build
#
#    This installs PyInstaller and builds dist/LoanCRM.exe
#    The .exe runs on any Windows machine without Python installed.