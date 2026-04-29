"""
feature_engineering.py
=======================
Builds the feature matrix for the loan propensity model.

The model answers: "Will this client take a loan in the next month?"
The label (target) = 1 if client had a phase 4 spike (Oct–Nov 2025),
which is the signal that they are about to need credit.

All features are computed from the observation window BEFORE the target
period, so there is no data leakage. The observation window is:
  Jan 2024 – Sep 2025 (phases 1, 2, 3)
The target period is:
  Oct–Nov 2025 (phase 4)

Feature groups
--------------
  1. Balance dynamics     — how the balance moves over time
  2. Burn rate            — how fast money is being spent
  3. Volatility           — how erratic the spending/balance is
  4. Category spikes      — unusual concentration in one MCC category
  5. Zero-day count       — days when balance hits the minimum floor
  6. Urgency score        — composite signal of financial stress
  7. Credit analysis      — repayment quality from previous credit history
  8. App behaviour        — balance-check frequency (anxiety signal)
  9. Client profile       — static demographic/relationship features

Inputs
------
  ../data/transactions.csv
  ../data/application.csv        (app sessions)
  ../data/clients.csv
  ../data/credit.csv
  ../data/payments.csv
  ../data/credit_scores.csv
  ../data/client_mcc_assignments.csv

Output
------
  ../data/features.csv           — one row per client, ready for model
"""

import pandas as pd
import numpy as np
import os
from datetime import date

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR    = "../data"
OUT_PATH    = f"{DATA_DIR}/features.csv"

# Observation window: everything before the prediction target
OBS_START   = date(2024,  1,  1)
OBS_END     = date(2025,  9, 30)

# Recent window: last 3 months of observation (for trend features)
RECENT_START = date(2025,  7,  1)
RECENT_END   = date(2025,  9, 30)

# Phase 2 window: the historical stress spike
P2_START    = date(2024,  7,  1)
P2_END      = date(2024,  9, 30)

# Target period: phase 4 spike = what we are predicting
TARGET_START = date(2025, 10,  1)
TARGET_END   = date(2025, 11, 30)

# Minimum balance floor used in transaction generation
MIN_BALANCE  = 10_000

MCC_NAMES = {
    5211: "repair",
    1021: "electronics",
    5680: "clothing",
    3001: "travel",
    5411: "supermarket",
    5812: "restaurant",
    5912: "pharmacy",
    6011: "atm",
}

TRIGGER_MCCS = [5211, 1021, 5680, 3001]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> dict:
    print("[LOAD] Reading source files ...")
    tx       = pd.read_csv(f"{DATA_DIR}/transactions.csv")
    app      = pd.read_csv(f"{DATA_DIR}/application.csv")
    clients  = pd.read_csv(f"{DATA_DIR}/clients.csv")
    credit   = pd.read_csv(f"{DATA_DIR}/credit.csv")
    payments = pd.read_csv(f"{DATA_DIR}/payments.csv")
    scores   = pd.read_csv(f"{DATA_DIR}/credit_scores.csv")
    mcc_asgn = pd.read_csv(f"{DATA_DIR}/client_mcc_assignments.csv")

    tx["transaction_date"]  = pd.to_datetime(tx["transaction_date"])
    app["timestamp"]        = pd.to_datetime(app["timestamp"])
    credit["application_date"] = pd.to_datetime(credit["application_date"])

    print(f"  transactions : {len(tx):,} rows")
    print(f"  app sessions : {len(app):,} rows")
    print(f"  clients      : {len(clients):,} rows")
    print(f"  credits      : {len(credit):,} rows")
    print(f"  payments     : {len(payments):,} rows")
    print(f"  credit scores: {len(scores):,} rows")
    return dict(tx=tx, app=app, clients=clients, credit=credit,
                payments=payments, scores=scores, mcc_asgn=mcc_asgn)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 1 — BALANCE DYNAMICS
# ─────────────────────────────────────────────────────────────────────────────

def feat_balance_dynamics(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Captures how the balance evolves over the observation window.

    Features:
      balance_mean          — average balance across all transactions
      balance_min           — lowest balance seen (floor proximity)
      balance_max           — peak balance (capacity)
      balance_last          — most recent balance (current state)
      balance_trend         — linear regression slope of balance over time
                              positive = growing, negative = shrinking
      balance_drop_pct      — (max - last) / max: how far balance has fallen
                              from peak → high value = financial pressure
      balance_mean_recent   — average balance in last 3 months
      balance_change_3m     — balance_mean_recent - balance_mean_full
                              negative = recent deterioration
    """
    obs = tx[
        (tx["transaction_date"].dt.date >= OBS_START) &
        (tx["transaction_date"].dt.date <= OBS_END) &
        (tx["status"] == "SUCCESS")
    ].copy()

    obs["ts"] = obs["transaction_date"].astype(np.int64) // 10**9  # unix seconds

    def _slope(g):
        if len(g) < 2:
            return 0.0
        x = g["ts"].values.astype(float)
        y = g["balance"].values.astype(float)
        x -= x.mean()
        denom = (x ** 2).sum()
        return float((x * y).sum() / denom) if denom > 0 else 0.0

    trend = obs.groupby("client_id").apply(_slope).reset_index(name="balance_trend")

    agg = obs.groupby("client_id").agg(
        balance_mean  = ("balance", "mean"),
        balance_min   = ("balance", "min"),
        balance_max   = ("balance", "max"),
        balance_last  = ("balance", "last"),
    ).reset_index()

    recent = obs[obs["transaction_date"].dt.date >= RECENT_START]
    recent_agg = recent.groupby("client_id").agg(
        balance_mean_recent = ("balance", "mean")
    ).reset_index()

    out = (agg
           .merge(trend,       on="client_id", how="left")
           .merge(recent_agg,  on="client_id", how="left"))

    out["balance_drop_pct"]    = (
        (out["balance_max"] - out["balance_last"]) /
        out["balance_max"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)

    out["balance_change_3m"]   = out["balance_mean_recent"] - out["balance_mean"]

    return out[[
        "client_id", "balance_mean", "balance_min", "balance_max",
        "balance_last", "balance_trend", "balance_drop_pct",
        "balance_mean_recent", "balance_change_3m"
    ]].fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 2 — BURN RATE
# ─────────────────────────────────────────────────────────────────────────────

def feat_burn_rate(tx: pd.DataFrame, clients: pd.DataFrame) -> pd.DataFrame:
    """
    How fast is the client spending relative to their income?

    Features:
      monthly_spend_mean    — average monthly outflow (successful tx only)
      monthly_spend_recent  — average monthly outflow last 3 months
      burn_rate_ratio       — monthly_spend_mean / monthly_salary
                              > 1 means spending more than earning
      burn_acceleration     — monthly_spend_recent / monthly_spend_mean
                              > 1 means spending is accelerating
      tx_count_mean_monthly — average number of transactions per month
      tx_count_recent       — tx count in last 3 months
    """
    obs = tx[
        (tx["transaction_date"].dt.date >= OBS_START) &
        (tx["transaction_date"].dt.date <= OBS_END) &
        (tx["status"] == "SUCCESS")
    ].copy()
    obs["month"] = obs["transaction_date"].dt.to_period("M")

    monthly_spend = (
        obs.groupby(["client_id", "month"])["amount"].sum()
        .reset_index()
        .groupby("client_id")["amount"]
        .mean()
        .reset_index(name="monthly_spend_mean")
    )

    monthly_count = (
        obs.groupby(["client_id", "month"])["transaction_id"].count()
        .reset_index()
        .groupby("client_id")["transaction_id"]
        .mean()
        .reset_index(name="tx_count_mean_monthly")
    )

    recent = obs[obs["transaction_date"].dt.date >= RECENT_START].copy()
    recent["month"] = recent["transaction_date"].dt.to_period("M")
    n_recent_months = max(1, len(recent["month"].unique()))

    recent_spend = (
        recent.groupby("client_id")["amount"].sum()
        / n_recent_months
    ).reset_index(name="monthly_spend_recent")

    recent_count = (
        recent.groupby("client_id")["transaction_id"].count()
    ).reset_index(name="tx_count_recent")

    out = (monthly_spend
           .merge(monthly_count,  on="client_id", how="left")
           .merge(recent_spend,   on="client_id", how="left")
           .merge(recent_count,   on="client_id", how="left")
           .merge(clients[["client_id", "monthly_salary"]], on="client_id", how="left"))

    out["burn_rate_ratio"]   = (
        out["monthly_spend_mean"] / out["monthly_salary"].replace(0, np.nan)
    ).fillna(0)

    out["burn_acceleration"] = (
        out["monthly_spend_recent"] / out["monthly_spend_mean"].replace(0, np.nan)
    ).fillna(1).clip(0, 5)

    return out[[
        "client_id", "monthly_spend_mean", "monthly_spend_recent",
        "burn_rate_ratio", "burn_acceleration",
        "tx_count_mean_monthly", "tx_count_recent"
    ]].fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 3 — VOLATILITY
# ─────────────────────────────────────────────────────────────────────────────

def feat_volatility(tx: pd.DataFrame) -> pd.DataFrame:
    """
    How erratic is the client's financial behaviour?

    Features:
      balance_std           — standard deviation of balance readings
      spend_std             — std of individual transaction amounts
      monthly_spend_std     — std of monthly totals (month-to-month volatility)
      cv_balance            — coefficient of variation of balance (std/mean)
                              high = very volatile relative to level
      failed_tx_rate        — fraction of transactions that failed
      failed_tx_rate_recent — failed rate in last 3 months (worsening?)
    """
    obs = tx[
        (tx["transaction_date"].dt.date >= OBS_START) &
        (tx["transaction_date"].dt.date <= OBS_END)
    ].copy()
    obs["month"] = obs["transaction_date"].dt.to_period("M")

    bal_vol = obs.groupby("client_id").agg(
        balance_std  = ("balance", "std"),
        balance_mean = ("balance", "mean"),
        spend_std    = ("amount",  "std"),
    ).reset_index()
    bal_vol["cv_balance"] = (
        bal_vol["balance_std"] / bal_vol["balance_mean"].replace(0, np.nan)
    ).fillna(0)

    monthly_vol = (
        obs[obs["status"] == "SUCCESS"]
        .groupby(["client_id", "month"])["amount"].sum()
        .reset_index()
        .groupby("client_id")["amount"]
        .std()
        .reset_index(name="monthly_spend_std")
    )

    fail_all = obs.groupby("client_id").apply(
        lambda g: (g["status"] == "FAILED").mean()
    ).reset_index(name="failed_tx_rate")

    recent_obs = obs[obs["transaction_date"].dt.date >= RECENT_START]
    fail_recent = recent_obs.groupby("client_id").apply(
        lambda g: (g["status"] == "FAILED").mean()
    ).reset_index(name="failed_tx_rate_recent")

    out = (bal_vol[["client_id", "balance_std", "spend_std", "cv_balance"]]
           .merge(monthly_vol,  on="client_id", how="left")
           .merge(fail_all,     on="client_id", how="left")
           .merge(fail_recent,  on="client_id", how="left"))

    return out.fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 4 — CATEGORY SPIKES
# ─────────────────────────────────────────────────────────────────────────────

def feat_category_spikes(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Detects unusual concentration of spend in a trigger MCC category.
    The key insight: a spike in repair/electronics/travel/clothing spend
    relative to the client's normal pattern is the strongest loan signal.

    Features (per trigger MCC: repair, electronics, clothing, travel):
      spend_{cat}_total     — total spend in that category (obs window)
      spend_{cat}_share     — share of total spend in that category
      spend_{cat}_recent    — spend in last 3 months
      spike_{cat}           — recent share / baseline share (ratio > 1 = spike)
                              > 2 means recent spend in this category doubled
                              vs their historical norm

    Also:
      max_single_trigger_tx — largest single transaction in any trigger MCC
      dominant_trigger_mcc  — which trigger MCC has the highest total spend
    """
    obs = tx[
        (tx["transaction_date"].dt.date >= OBS_START) &
        (tx["transaction_date"].dt.date <= OBS_END) &
        (tx["status"] == "SUCCESS")
    ].copy()

    recent = obs[obs["transaction_date"].dt.date >= RECENT_START].copy()

    total_spend = obs.groupby("client_id")["amount"].sum().reset_index(name="total_spend_obs")

    rows = {}
    for mcc, name in [(5211, "repair"), (1021, "electronics"),
                      (5680, "clothing"), (3001, "travel")]:
        cat_obs    = obs[obs["mcc_code"] == mcc].groupby("client_id")["amount"].sum().reset_index(name=f"spend_{name}_total")
        cat_recent = recent[recent["mcc_code"] == mcc].groupby("client_id")["amount"].sum().reset_index(name=f"spend_{name}_recent")

        if name not in rows:
            rows[name] = cat_obs.merge(cat_recent, on="client_id", how="outer").fillna(0)
        rows[name] = rows[name].merge(total_spend, on="client_id", how="left")

        # Baseline share = total category / total obs window
        rows[name][f"spend_{name}_share"] = (
            rows[name][f"spend_{name}_total"] /
            rows[name]["total_spend_obs"].replace(0, np.nan)
        ).fillna(0)

        # Recent share = recent category spend / recent total spend
        recent_total = recent.groupby("client_id")["amount"].sum().reset_index(name="recent_total")
        rows[name] = rows[name].merge(recent_total, on="client_id", how="left").fillna(0)
        recent_share = (
            rows[name][f"spend_{name}_recent"] /
            rows[name]["recent_total"].replace(0, np.nan)
        ).fillna(0)

        # Spike ratio: how much more concentrated recently vs historically
        rows[name][f"spike_{name}"] = (
            recent_share / rows[name][f"spend_{name}_share"].replace(0, np.nan)
        ).fillna(1).clip(0, 10)

        rows[name] = rows[name].drop(columns=["total_spend_obs", "recent_total"])

    # Merge all category frames
    out = rows["repair"]
    for name in ["electronics", "clothing", "travel"]:
        out = out.merge(rows[name], on="client_id", how="outer")
    out = out.fillna(0)

    # Largest single trigger transaction
    trigger_obs = obs[obs["mcc_code"].isin(TRIGGER_MCCS)]
    max_single  = trigger_obs.groupby("client_id")["amount"].max().reset_index(name="max_single_trigger_tx")
    out = out.merge(max_single, on="client_id", how="left").fillna(0)

    # Which trigger MCC dominates (encoded as integer for tree models)
    spend_cols  = {5211: "spend_repair_total", 1021: "spend_electronics_total",
                   5680: "spend_clothing_total", 3001: "spend_travel_total"}
    def _dominant(row):
        best_mcc, best_val = 0, -1
        for mcc, col in spend_cols.items():
            if col in row and row[col] > best_val:
                best_val = row[col]
                best_mcc = mcc
        return best_mcc

    out["dominant_trigger_mcc"] = out.apply(_dominant, axis=1)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 5 — ZERO-DAY COUNT
# ─────────────────────────────────────────────────────────────────────────────

def feat_zero_day_count(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Counts how often the client's balance was at or near the minimum floor.
    A client repeatedly hitting the floor is under severe financial pressure.

    Features:
      zero_day_count        — number of transactions where balance <= MIN_BALANCE * 1.5
      zero_day_rate         — zero_day_count / total tx count
      zero_day_count_recent — same count but only in last 3 months
      consecutive_zero_max  — longest consecutive run of near-zero balance tx
                              signals sustained (not just occasional) distress
    """
    obs = tx[
        (tx["transaction_date"].dt.date >= OBS_START) &
        (tx["transaction_date"].dt.date <= OBS_END)
    ].copy().sort_values(["client_id", "transaction_date"])

    threshold = MIN_BALANCE * 1.5

    obs["near_zero"] = (obs["balance"] <= threshold).astype(int)

    total_counts = obs.groupby("client_id")["transaction_id"].count().reset_index(name="tx_total")
    zero_counts  = obs.groupby("client_id")["near_zero"].sum().reset_index(name="zero_day_count")

    recent = obs[obs["transaction_date"].dt.date >= RECENT_START]
    zero_recent = recent.groupby("client_id")["near_zero"].sum().reset_index(name="zero_day_count_recent")

    # Longest consecutive run of near-zero transactions per client
    def _max_run(group):
        vals = group["near_zero"].values
        max_run, current = 0, 0
        for v in vals:
            if v == 1:
                current += 1
                max_run  = max(max_run, current)
            else:
                current = 0
        return max_run

    max_runs = obs.groupby("client_id").apply(_max_run).reset_index(name="consecutive_zero_max")

    out = (total_counts
           .merge(zero_counts,   on="client_id", how="left")
           .merge(zero_recent,   on="client_id", how="left")
           .merge(max_runs,      on="client_id", how="left"))

    out["zero_day_rate"] = (
        out["zero_day_count"] / out["tx_total"].replace(0, np.nan)
    ).fillna(0)

    return out[[
        "client_id", "zero_day_count", "zero_day_rate",
        "zero_day_count_recent", "consecutive_zero_max"
    ]].fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 6 — URGENCY SCORE
# ─────────────────────────────────────────────────────────────────────────────

def feat_urgency_score(tx: pd.DataFrame, app: pd.DataFrame) -> pd.DataFrame:
    """
    A composite signal of financial stress and intent to borrow.
    Combines transaction behaviour + app behaviour into a single index.

    Components:
      balance_check_freq    — avg daily balance checks (app sessions)
      balance_check_spike   — ratio of recent checks to historical average
                              sharply higher = client is monitoring anxiously
      high_amount_tx_share  — share of tx that are unusually large
                              (> 2x the client's own median amount)
      failed_spike          — failed_rate_recent / failed_rate_overall
                              > 1 means things are getting worse
      urgency_score         — weighted composite [0, 1]
                              0.35 * balance_check_spike_norm
                            + 0.30 * high_amount_tx_share
                            + 0.20 * failed_spike_norm
                            + 0.15 * zero proximity (from balance_drop_pct)
    """
    obs = tx[
        (tx["transaction_date"].dt.date >= OBS_START) &
        (tx["transaction_date"].dt.date <= OBS_END)
    ].copy()

    # App-side: balance check frequency
    obs_app = app[
        (app["timestamp"].dt.date >= OBS_START) &
        (app["timestamp"].dt.date <= OBS_END)
    ].copy()

    n_obs_days = (OBS_END - OBS_START).days + 1

    check_all = (
        obs_app[obs_app["action"] == "check_balance"]
        .groupby("client_id")["session_id"].count()
        / n_obs_days
    ).reset_index(name="balance_check_freq")

    n_recent_days = (RECENT_END - RECENT_START).days + 1
    recent_app = obs_app[obs_app["timestamp"].dt.date >= RECENT_START]
    check_recent = (
        recent_app[recent_app["action"] == "check_balance"]
        .groupby("client_id")["session_id"].count()
        / n_recent_days
    ).reset_index(name="balance_check_freq_recent")

    app_feats = check_all.merge(check_recent, on="client_id", how="outer").fillna(0)
    app_feats["balance_check_spike"] = (
        app_feats["balance_check_freq_recent"] /
        app_feats["balance_check_freq"].replace(0, np.nan)
    ).fillna(1).clip(0, 10)

    # Transaction-side: unusually large transactions
    median_amount = obs[obs["status"] == "SUCCESS"].groupby("client_id")["amount"].median().reset_index(name="median_amount")
    obs_succ = obs[obs["status"] == "SUCCESS"].merge(median_amount, on="client_id", how="left")
    obs_succ["is_large"] = (obs_succ["amount"] > obs_succ["median_amount"] * 2).astype(int)

    large_share = obs_succ.groupby("client_id").apply(
        lambda g: g["is_large"].mean()
    ).reset_index(name="high_amount_tx_share")

    # Failed rate overall vs recent
    fail_all = obs.groupby("client_id").apply(
        lambda g: (g["status"] == "FAILED").mean()
    ).reset_index(name="failed_rate_all")

    recent_tx = obs[obs["transaction_date"].dt.date >= RECENT_START]
    fail_recent = recent_tx.groupby("client_id").apply(
        lambda g: (g["status"] == "FAILED").mean()
    ).reset_index(name="failed_rate_recent")

    fail_feats = fail_all.merge(fail_recent, on="client_id", how="left").fillna(0)
    fail_feats["failed_spike"] = (
        fail_feats["failed_rate_recent"] /
        fail_feats["failed_rate_all"].replace(0, np.nan)
    ).fillna(1).clip(0, 5)

    # Balance drop (recomputed here for urgency weighting)
    bal_agg = obs[obs["status"] == "SUCCESS"].groupby("client_id").agg(
        balance_max  = ("balance", "max"),
        balance_last = ("balance", "last"),
    ).reset_index()
    bal_agg["balance_drop_pct"] = (
        (bal_agg["balance_max"] - bal_agg["balance_last"]) /
        bal_agg["balance_max"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)

    # Combine
    out = (app_feats
           .merge(large_share, on="client_id", how="outer")
           .merge(fail_feats,  on="client_id", how="outer")
           .merge(bal_agg[["client_id", "balance_drop_pct"]], on="client_id", how="outer")
           .fillna(0))

    # Normalise each component to [0, 1] before combining
    def _norm(series):
        mn, mx = series.min(), series.max()
        return (series - mn) / (mx - mn + 1e-9)

    out["urgency_score"] = (
        0.35 * _norm(out["balance_check_spike"])
      + 0.30 * _norm(out["high_amount_tx_share"])
      + 0.20 * _norm(out["failed_spike"])
      + 0.15 * _norm(out["balance_drop_pct"])
    ).clip(0, 1)

    return out[[
        "client_id", "balance_check_freq", "balance_check_spike",
        "high_amount_tx_share", "failed_spike", "urgency_score"
    ]].fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 7 — CREDIT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def feat_credit_analysis(credit: pd.DataFrame,
                          payments: pd.DataFrame,
                          scores: pd.DataFrame) -> pd.DataFrame:
    """
    Repayment quality and credit profile from the existing credit history
    (the phase 2 credit taken in Oct 2024 and paid off by Sep 2025).

    Features:
      has_prior_credit         — 1 if client ever had a credit
      prior_credit_approved    — 1 if most recent credit was approved
      prior_credit_closed      — 1 if credit is fully paid off
      payments_on_time_rate    — on-time payments / total past payments
      payments_late_rate       — late payments / total past payments
      payments_missed_rate     — missed payments / total past payments
      avg_days_late            — average days late across all late payments
      max_days_late            — worst single late payment
      credit_utilization       — remaining balance / credit limit (from scores)
      credit_score             — final credit score (300–850)
      dti_ratio                — debt-to-income from credit score calc
      payment_history_pts      — component score (0–110)
    """
    # Only look at credits from the observation window
    obs_credit = credit[
        credit["application_date"].dt.date <= pd.Timestamp(OBS_END).date()
    ].copy()

    has_credit = obs_credit.groupby("client_id").size().reset_index(name="n_credits")
    has_credit["has_prior_credit"] = 1

    approved = obs_credit[obs_credit["status"] != "DECLINED"]
    approved_flag = (
        approved.groupby("client_id").size().reset_index(name="_")
        .assign(prior_credit_approved=1)[["client_id", "prior_credit_approved"]]
    )

    closed_flag = (
        obs_credit[obs_credit["status"] == "CLOSED"]
        .groupby("client_id").size().reset_index(name="_")
        .assign(prior_credit_closed=1)[["client_id", "prior_credit_closed"]]
    )

    # Payment quality (only past/resolved payments, not SCHEDULED)
    past_pay = payments[payments["status"] != "SCHEDULED"].copy()
    if "days_late" in past_pay.columns:
        past_pay["days_late"] = pd.to_numeric(past_pay["days_late"], errors="coerce")

    pay_agg = past_pay.groupby("client_id").apply(lambda g: pd.Series({
        "payments_on_time_rate":  (g["status"] == "PAID_ON_TIME").mean(),
        "payments_late_rate":     (g["status"].isin(["PAID_LATE", "OVERDUE"])).mean(),
        "payments_missed_rate":   (g["status"] == "MISSED").mean(),
        "avg_days_late":          g.loc[g["status"].isin(["PAID_LATE","OVERDUE"]), "days_late"].mean(),
        "max_days_late":          g["days_late"].max(),
    })).reset_index()

    pay_agg = pay_agg.fillna(0)

    # Credit score components
    score_cols = ["client_id", "credit_score", "credit_utilization",
                  "dti_ratio", "payment_history_pts", "dti_pts",
                  "payments_on_time", "payments_late", "payments_missed"]
    score_cols = [c for c in score_cols if c in scores.columns]
    score_feats = scores[score_cols].copy()

    # Merge everything
    all_clients = credit["client_id"].unique()
    base = pd.DataFrame({"client_id": all_clients})

    out = (base
           .merge(has_credit[["client_id", "has_prior_credit"]], on="client_id", how="left")
           .merge(approved_flag,  on="client_id", how="left")
           .merge(closed_flag,    on="client_id", how="left")
           .merge(pay_agg,        on="client_id", how="left")
           .merge(score_feats,    on="client_id", how="left"))

    out["has_prior_credit"]       = out["has_prior_credit"].fillna(0)
    out["prior_credit_approved"]  = out["prior_credit_approved"].fillna(0)
    out["prior_credit_closed"]    = out["prior_credit_closed"].fillna(0)

    return out.fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 8 — APP BEHAVIOUR
# ─────────────────────────────────────────────────────────────────────────────

def feat_app_behaviour(app: pd.DataFrame) -> pd.DataFrame:
    """
    App usage patterns are a strong leading indicator of financial anxiety.
    Phase 2 and phase 4 both show elevated balance-check frequency.

    Features:
      total_sessions            — total app opens in observation window
      total_balance_checks      — total 'check_balance' actions
      balance_check_ratio       — balance checks / total sessions
      avg_session_duration_sec  — mean session length
      sessions_per_month        — engagement level
      sessions_last_3m          — sessions in last 3 months
      balance_checks_last_3m    — balance checks in last 3 months
      check_ratio_last_3m       — balance check ratio in last 3 months
                                  (higher than overall = rising anxiety)
    """
    obs = app[
        (app["timestamp"].dt.date >= OBS_START) &
        (app["timestamp"].dt.date <= OBS_END)
    ].copy()

    n_obs_months = ((OBS_END.year - OBS_START.year) * 12 +
                    OBS_END.month - OBS_START.month + 1)

    agg = obs.groupby("client_id").agg(
        total_sessions           = ("session_id", "count"),
        total_balance_checks     = ("action", lambda x: (x == "check_balance").sum()),
        avg_session_duration_sec = ("duration_sec", "mean"),
    ).reset_index()

    agg["balance_check_ratio"]  = (
        agg["total_balance_checks"] / agg["total_sessions"].replace(0, np.nan)
    ).fillna(0)
    agg["sessions_per_month"]   = agg["total_sessions"] / n_obs_months

    recent = obs[obs["timestamp"].dt.date >= RECENT_START]
    recent_agg = recent.groupby("client_id").agg(
        sessions_last_3m       = ("session_id", "count"),
        balance_checks_last_3m = ("action", lambda x: (x == "check_balance").sum()),
    ).reset_index()
    recent_agg["check_ratio_last_3m"] = (
        recent_agg["balance_checks_last_3m"] /
        recent_agg["sessions_last_3m"].replace(0, np.nan)
    ).fillna(0)

    return agg.merge(recent_agg, on="client_id", how="left").fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 9 — CLIENT PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def feat_client_profile(clients: pd.DataFrame) -> pd.DataFrame:
    """
    Static demographic and relationship features.

    Features:
      age                   — client age
      monthly_salary        — declared monthly salary
      employment_type_enc   — employment encoded (EMPLOYED=0, SELF_EMPLOYED=1,
                              STUDENT=2, RETIRED=3)
      is_employed           — binary flag for employed
      dependants            — number of dependants (affects financial pressure)
      account_age_years     — how long they've been a client
      marital_encoded       — SINGLE=0, MARRIED=1, DIVORCED=2, WIDOWED=3
    """
    emp_map     = {"EMPLOYED": 0, "SELF_EMPLOYED": 1, "STUDENT": 2, "RETIRED": 3}
    marital_map = {"SINGLE": 0, "MARRIED": 1, "DIVORCED": 2, "WIDOWED": 3}

    out = clients[["client_id", "age", "monthly_salary", "employment_type",
                   "dependants", "account_open_date"]].copy()

    out["employment_type_enc"] = out["employment_type"].map(emp_map).fillna(0)
    out["is_employed"]         = (out["employment_type"] == "EMPLOYED").astype(int)

    if "marital_status" in clients.columns:
        out["marital_encoded"] = clients["marital_status"].map(marital_map).fillna(0)
    else:
        out["marital_encoded"] = 0

    ref_date = pd.Timestamp(OBS_END)
    out["account_age_years"] = (
        (ref_date - pd.to_datetime(out["account_open_date"])).dt.days / 365.25
    ).clip(lower=0)

    return out[[
        "client_id", "age", "monthly_salary", "employment_type_enc",
        "is_employed", "dependants", "account_age_years", "marital_encoded"
    ]].fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# TARGET LABEL
# ─────────────────────────────────────────────────────────────────────────────
"""
feature_engineering.py
=======================
Builds the feature matrix for the loan propensity model.

The model answers: "Will this client take a loan in the next month?"
The label (target) = 1 if client had a phase 4 spike (Oct–Nov 2025),
which is the signal that they are about to need credit.

All features are computed from the observation window BEFORE the target
period, so there is no data leakage. The observation window is:
  Jan 2024 – Sep 2025 (phases 1, 2, 3)
The target period is:
  Oct–Nov 2025 (phase 4)

Feature groups
--------------
  1. Balance dynamics     — how the balance moves over time
  2. Burn rate            — how fast money is being spent
  3. Volatility           — how erratic the spending/balance is
  4. Category spikes      — unusual concentration in one MCC category
  5. Zero-day count       — days when balance hits the minimum floor
  6. Urgency score        — composite signal of financial stress
  7. Credit analysis      — repayment quality from previous credit history
  8. App behaviour        — balance-check frequency (anxiety signal)
  9. Client profile       — static demographic/relationship features

Inputs
------
  ../data/transactions.csv
  ../data/application.csv        (app sessions)
  ../data/clients.csv
  ../data/credit.csv
  ../data/payments.csv
  ../data/credit_scores.csv
  ../data/client_mcc_assignments.csv

Output
------
  ../data/features.csv           — one row per client, ready for model
"""

import pandas as pd
import numpy as np
import os
from datetime import date

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR    = "../data"
OUT_PATH    = f"{DATA_DIR}/features.csv"

# Observation window: everything before the prediction target
OBS_START   = date(2024,  1,  1)
OBS_END     = date(2025,  9, 30)

# Recent window: last 3 months of observation (for trend features)
RECENT_START = date(2025,  7,  1)
RECENT_END   = date(2025,  9, 30)

# Phase 2 window: the historical stress spike
P2_START    = date(2024,  7,  1)
P2_END      = date(2024,  9, 30)

# Target period: phase 4 spike = what we are predicting
TARGET_START = date(2025, 10,  1)
TARGET_END   = date(2025, 11, 30)

# Minimum balance floor used in transaction generation
MIN_BALANCE  = 10_000

MCC_NAMES = {
    5211: "repair",
    1021: "electronics",
    5680: "clothing",
    3001: "travel",
    5411: "supermarket",
    5812: "restaurant",
    5912: "pharmacy",
    6011: "atm",
}

TRIGGER_MCCS = [5211, 1021, 5680, 3001]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> dict:
    print("[LOAD] Reading source files ...")
    tx       = pd.read_csv(f"{DATA_DIR}/transactions.csv")
    app      = pd.read_csv(f"{DATA_DIR}/application.csv")
    clients  = pd.read_csv(f"{DATA_DIR}/clients.csv")
    credit   = pd.read_csv(f"{DATA_DIR}/credit.csv")
    payments = pd.read_csv(f"{DATA_DIR}/payments.csv")
    scores   = pd.read_csv(f"{DATA_DIR}/credit_scores.csv")
    mcc_asgn = pd.read_csv(f"{DATA_DIR}/client_mcc_assignments.csv")

    tx["transaction_date"]  = pd.to_datetime(tx["transaction_date"])
    app["timestamp"]        = pd.to_datetime(app["timestamp"])
    credit["application_date"] = pd.to_datetime(credit["application_date"])

    print(f"  transactions : {len(tx):,} rows")
    print(f"  app sessions : {len(app):,} rows")
    print(f"  clients      : {len(clients):,} rows")
    print(f"  credits      : {len(credit):,} rows")
    print(f"  payments     : {len(payments):,} rows")
    print(f"  credit scores: {len(scores):,} rows")
    return dict(tx=tx, app=app, clients=clients, credit=credit,
                payments=payments, scores=scores, mcc_asgn=mcc_asgn)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 1 — BALANCE DYNAMICS
# ─────────────────────────────────────────────────────────────────────────────

def feat_balance_dynamics(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Captures how the balance evolves over the observation window.

    Features:
      balance_mean          — average balance across all transactions
      balance_min           — lowest balance seen (floor proximity)
      balance_max           — peak balance (capacity)
      balance_last          — most recent balance (current state)
      balance_trend         — linear regression slope of balance over time
                              positive = growing, negative = shrinking
      balance_drop_pct      — (max - last) / max: how far balance has fallen
                              from peak → high value = financial pressure
      balance_mean_recent   — average balance in last 3 months
      balance_change_3m     — balance_mean_recent - balance_mean_full
                              negative = recent deterioration
    """
    obs = tx[
        (tx["transaction_date"].dt.date >= OBS_START) &
        (tx["transaction_date"].dt.date <= OBS_END) &
        (tx["status"] == "SUCCESS")
    ].copy()

    obs["ts"] = obs["transaction_date"].astype(np.int64) // 10**9  # unix seconds

    def _slope(g):
        if len(g) < 2:
            return 0.0
        x = g["ts"].values.astype(float)
        y = g["balance"].values.astype(float)
        x -= x.mean()
        denom = (x ** 2).sum()
        return float((x * y).sum() / denom) if denom > 0 else 0.0

    trend = obs.groupby("client_id").apply(_slope).reset_index(name="balance_trend")

    agg = obs.groupby("client_id").agg(
        balance_mean  = ("balance", "mean"),
        balance_min   = ("balance", "min"),
        balance_max   = ("balance", "max"),
        balance_last  = ("balance", "last"),
    ).reset_index()

    recent = obs[obs["transaction_date"].dt.date >= RECENT_START]
    recent_agg = recent.groupby("client_id").agg(
        balance_mean_recent = ("balance", "mean")
    ).reset_index()

    out = (agg
           .merge(trend,       on="client_id", how="left")
           .merge(recent_agg,  on="client_id", how="left"))

    out["balance_drop_pct"]    = (
        (out["balance_max"] - out["balance_last"]) /
        out["balance_max"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)

    out["balance_change_3m"]   = out["balance_mean_recent"] - out["balance_mean"]

    return out[[
        "client_id", "balance_mean", "balance_min", "balance_max",
        "balance_last", "balance_trend", "balance_drop_pct",
        "balance_mean_recent", "balance_change_3m"
    ]].fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 2 — BURN RATE
# ─────────────────────────────────────────────────────────────────────────────

def feat_burn_rate(tx: pd.DataFrame, clients: pd.DataFrame) -> pd.DataFrame:
    """
    How fast is the client spending relative to their income?

    Features:
      monthly_spend_mean    — average monthly outflow (successful tx only)
      monthly_spend_recent  — average monthly outflow last 3 months
      burn_rate_ratio       — monthly_spend_mean / monthly_salary
                              > 1 means spending more than earning
      burn_acceleration     — monthly_spend_recent / monthly_spend_mean
                              > 1 means spending is accelerating
      tx_count_mean_monthly — average number of transactions per month
      tx_count_recent       — tx count in last 3 months
    """
    obs = tx[
        (tx["transaction_date"].dt.date >= OBS_START) &
        (tx["transaction_date"].dt.date <= OBS_END) &
        (tx["status"] == "SUCCESS")
    ].copy()
    obs["month"] = obs["transaction_date"].dt.to_period("M")

    monthly_spend = (
        obs.groupby(["client_id", "month"])["amount"].sum()
        .reset_index()
        .groupby("client_id")["amount"]
        .mean()
        .reset_index(name="monthly_spend_mean")
    )

    monthly_count = (
        obs.groupby(["client_id", "month"])["transaction_id"].count()
        .reset_index()
        .groupby("client_id")["transaction_id"]
        .mean()
        .reset_index(name="tx_count_mean_monthly")
    )

    recent = obs[obs["transaction_date"].dt.date >= RECENT_START].copy()
    recent["month"] = recent["transaction_date"].dt.to_period("M")
    n_recent_months = max(1, len(recent["month"].unique()))

    recent_spend = (
        recent.groupby("client_id")["amount"].sum()
        / n_recent_months
    ).reset_index(name="monthly_spend_recent")

    recent_count = (
        recent.groupby("client_id")["transaction_id"].count()
    ).reset_index(name="tx_count_recent")

    out = (monthly_spend
           .merge(monthly_count,  on="client_id", how="left")
           .merge(recent_spend,   on="client_id", how="left")
           .merge(recent_count,   on="client_id", how="left")
           .merge(clients[["client_id", "monthly_salary"]], on="client_id", how="left"))

    out["burn_rate_ratio"]   = (
        out["monthly_spend_mean"] / out["monthly_salary"].replace(0, np.nan)
    ).fillna(0)

    out["burn_acceleration"] = (
        out["monthly_spend_recent"] / out["monthly_spend_mean"].replace(0, np.nan)
    ).fillna(1).clip(0, 5)

    return out[[
        "client_id", "monthly_spend_mean", "monthly_spend_recent",
        "burn_rate_ratio", "burn_acceleration",
        "tx_count_mean_monthly", "tx_count_recent"
    ]].fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 3 — VOLATILITY
# ─────────────────────────────────────────────────────────────────────────────

def feat_volatility(tx: pd.DataFrame) -> pd.DataFrame:
    """
    How erratic is the client's financial behaviour?

    Features:
      balance_std           — standard deviation of balance readings
      spend_std             — std of individual transaction amounts
      monthly_spend_std     — std of monthly totals (month-to-month volatility)
      cv_balance            — coefficient of variation of balance (std/mean)
                              high = very volatile relative to level
      failed_tx_rate        — fraction of transactions that failed
      failed_tx_rate_recent — failed rate in last 3 months (worsening?)
    """
    obs = tx[
        (tx["transaction_date"].dt.date >= OBS_START) &
        (tx["transaction_date"].dt.date <= OBS_END)
    ].copy()
    obs["month"] = obs["transaction_date"].dt.to_period("M")

    bal_vol = obs.groupby("client_id").agg(
        balance_std  = ("balance", "std"),
        balance_mean = ("balance", "mean"),
        spend_std    = ("amount",  "std"),
    ).reset_index()
    bal_vol["cv_balance"] = (
        bal_vol["balance_std"] / bal_vol["balance_mean"].replace(0, np.nan)
    ).fillna(0)

    monthly_vol = (
        obs[obs["status"] == "SUCCESS"]
        .groupby(["client_id", "month"])["amount"].sum()
        .reset_index()
        .groupby("client_id")["amount"]
        .std()
        .reset_index(name="monthly_spend_std")
    )

    fail_all = obs.groupby("client_id").apply(
        lambda g: (g["status"] == "FAILED").mean()
    ).reset_index(name="failed_tx_rate")

    recent_obs = obs[obs["transaction_date"].dt.date >= RECENT_START]
    fail_recent = recent_obs.groupby("client_id").apply(
        lambda g: (g["status"] == "FAILED").mean()
    ).reset_index(name="failed_tx_rate_recent")

    out = (bal_vol[["client_id", "balance_std", "spend_std", "cv_balance"]]
           .merge(monthly_vol,  on="client_id", how="left")
           .merge(fail_all,     on="client_id", how="left")
           .merge(fail_recent,  on="client_id", how="left"))

    return out.fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 4 — CATEGORY SPIKES
# ─────────────────────────────────────────────────────────────────────────────

def feat_category_spikes(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Detects unusual concentration of spend in a trigger MCC category.
    The key insight: a spike in repair/electronics/travel/clothing spend
    relative to the client's normal pattern is the strongest loan signal.

    Features (per trigger MCC: repair, electronics, clothing, travel):
      spend_{cat}_total     — total spend in that category (obs window)
      spend_{cat}_share     — share of total spend in that category
      spend_{cat}_recent    — spend in last 3 months
      spike_{cat}           — recent share / baseline share (ratio > 1 = spike)
                              > 2 means recent spend in this category doubled
                              vs their historical norm

    Also:
      max_single_trigger_tx — largest single transaction in any trigger MCC
      dominant_trigger_mcc  — which trigger MCC has the highest total spend
    """
    obs = tx[
        (tx["transaction_date"].dt.date >= OBS_START) &
        (tx["transaction_date"].dt.date <= OBS_END) &
        (tx["status"] == "SUCCESS")
    ].copy()

    recent = obs[obs["transaction_date"].dt.date >= RECENT_START].copy()

    total_spend = obs.groupby("client_id")["amount"].sum().reset_index(name="total_spend_obs")

    rows = {}
    for mcc, name in [(5211, "repair"), (1021, "electronics"),
                      (5680, "clothing"), (3001, "travel")]:
        cat_obs    = obs[obs["mcc_code"] == mcc].groupby("client_id")["amount"].sum().reset_index(name=f"spend_{name}_total")
        cat_recent = recent[recent["mcc_code"] == mcc].groupby("client_id")["amount"].sum().reset_index(name=f"spend_{name}_recent")

        if name not in rows:
            rows[name] = cat_obs.merge(cat_recent, on="client_id", how="outer").fillna(0)
        rows[name] = rows[name].merge(total_spend, on="client_id", how="left")

        # Baseline share = total category / total obs window
        rows[name][f"spend_{name}_share"] = (
            rows[name][f"spend_{name}_total"] /
            rows[name]["total_spend_obs"].replace(0, np.nan)
        ).fillna(0)

        # Recent share = recent category spend / recent total spend
        recent_total = recent.groupby("client_id")["amount"].sum().reset_index(name="recent_total")
        rows[name] = rows[name].merge(recent_total, on="client_id", how="left").fillna(0)
        recent_share = (
            rows[name][f"spend_{name}_recent"] /
            rows[name]["recent_total"].replace(0, np.nan)
        ).fillna(0)

        # Spike ratio: how much more concentrated recently vs historically
        rows[name][f"spike_{name}"] = (
            recent_share / rows[name][f"spend_{name}_share"].replace(0, np.nan)
        ).fillna(1).clip(0, 10)

        rows[name] = rows[name].drop(columns=["total_spend_obs", "recent_total"])

    # Merge all category frames
    out = rows["repair"]
    for name in ["electronics", "clothing", "travel"]:
        out = out.merge(rows[name], on="client_id", how="outer")
    out = out.fillna(0)

    # Largest single trigger transaction
    trigger_obs = obs[obs["mcc_code"].isin(TRIGGER_MCCS)]
    max_single  = trigger_obs.groupby("client_id")["amount"].max().reset_index(name="max_single_trigger_tx")
    out = out.merge(max_single, on="client_id", how="left").fillna(0)

    # Which trigger MCC dominates (encoded as integer for tree models)
    spend_cols  = {5211: "spend_repair_total", 1021: "spend_electronics_total",
                   5680: "spend_clothing_total", 3001: "spend_travel_total"}
    def _dominant(row):
        best_mcc, best_val = 0, -1
        for mcc, col in spend_cols.items():
            if col in row and row[col] > best_val:
                best_val = row[col]
                best_mcc = mcc
        return best_mcc

    out["dominant_trigger_mcc"] = out.apply(_dominant, axis=1)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 5 — ZERO-DAY COUNT
# ─────────────────────────────────────────────────────────────────────────────

def feat_zero_day_count(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Counts how often the client's balance was at or near the minimum floor.
    A client repeatedly hitting the floor is under severe financial pressure.

    Features:
      zero_day_count        — number of transactions where balance <= MIN_BALANCE * 1.5
      zero_day_rate         — zero_day_count / total tx count
      zero_day_count_recent — same count but only in last 3 months
      consecutive_zero_max  — longest consecutive run of near-zero balance tx
                              signals sustained (not just occasional) distress
    """
    obs = tx[
        (tx["transaction_date"].dt.date >= OBS_START) &
        (tx["transaction_date"].dt.date <= OBS_END)
    ].copy().sort_values(["client_id", "transaction_date"])

    threshold = MIN_BALANCE * 1.5

    obs["near_zero"] = (obs["balance"] <= threshold).astype(int)

    total_counts = obs.groupby("client_id")["transaction_id"].count().reset_index(name="tx_total")
    zero_counts  = obs.groupby("client_id")["near_zero"].sum().reset_index(name="zero_day_count")

    recent = obs[obs["transaction_date"].dt.date >= RECENT_START]
    zero_recent = recent.groupby("client_id")["near_zero"].sum().reset_index(name="zero_day_count_recent")

    # Longest consecutive run of near-zero transactions per client
    def _max_run(group):
        vals = group["near_zero"].values
        max_run, current = 0, 0
        for v in vals:
            if v == 1:
                current += 1
                max_run  = max(max_run, current)
            else:
                current = 0
        return max_run

    max_runs = obs.groupby("client_id").apply(_max_run).reset_index(name="consecutive_zero_max")

    out = (total_counts
           .merge(zero_counts,   on="client_id", how="left")
           .merge(zero_recent,   on="client_id", how="left")
           .merge(max_runs,      on="client_id", how="left"))

    out["zero_day_rate"] = (
        out["zero_day_count"] / out["tx_total"].replace(0, np.nan)
    ).fillna(0)

    return out[[
        "client_id", "zero_day_count", "zero_day_rate",
        "zero_day_count_recent", "consecutive_zero_max"
    ]].fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 6 — URGENCY SCORE
# ─────────────────────────────────────────────────────────────────────────────

def feat_urgency_score(tx: pd.DataFrame, app: pd.DataFrame) -> pd.DataFrame:
    """
    A composite signal of financial stress and intent to borrow.
    Combines transaction behaviour + app behaviour into a single index.

    Components:
      balance_check_freq    — avg daily balance checks (app sessions)
      balance_check_spike   — ratio of recent checks to historical average
                              sharply higher = client is monitoring anxiously
      high_amount_tx_share  — share of tx that are unusually large
                              (> 2x the client's own median amount)
      failed_spike          — failed_rate_recent / failed_rate_overall
                              > 1 means things are getting worse
      urgency_score         — weighted composite [0, 1]
                              0.35 * balance_check_spike_norm
                            + 0.30 * high_amount_tx_share
                            + 0.20 * failed_spike_norm
                            + 0.15 * zero proximity (from balance_drop_pct)
    """
    obs = tx[
        (tx["transaction_date"].dt.date >= OBS_START) &
        (tx["transaction_date"].dt.date <= OBS_END)
    ].copy()

    # App-side: balance check frequency
    obs_app = app[
        (app["timestamp"].dt.date >= OBS_START) &
        (app["timestamp"].dt.date <= OBS_END)
    ].copy()

    n_obs_days = (OBS_END - OBS_START).days + 1

    check_all = (
        obs_app[obs_app["action"] == "check_balance"]
        .groupby("client_id")["session_id"].count()
        / n_obs_days
    ).reset_index(name="balance_check_freq")

    n_recent_days = (RECENT_END - RECENT_START).days + 1
    recent_app = obs_app[obs_app["timestamp"].dt.date >= RECENT_START]
    check_recent = (
        recent_app[recent_app["action"] == "check_balance"]
        .groupby("client_id")["session_id"].count()
        / n_recent_days
    ).reset_index(name="balance_check_freq_recent")

    app_feats = check_all.merge(check_recent, on="client_id", how="outer").fillna(0)
    app_feats["balance_check_spike"] = (
        app_feats["balance_check_freq_recent"] /
        app_feats["balance_check_freq"].replace(0, np.nan)
    ).fillna(1).clip(0, 10)

    # Transaction-side: unusually large transactions
    median_amount = obs[obs["status"] == "SUCCESS"].groupby("client_id")["amount"].median().reset_index(name="median_amount")
    obs_succ = obs[obs["status"] == "SUCCESS"].merge(median_amount, on="client_id", how="left")
    obs_succ["is_large"] = (obs_succ["amount"] > obs_succ["median_amount"] * 2).astype(int)

    large_share = obs_succ.groupby("client_id").apply(
        lambda g: g["is_large"].mean()
    ).reset_index(name="high_amount_tx_share")

    # Failed rate overall vs recent
    fail_all = obs.groupby("client_id").apply(
        lambda g: (g["status"] == "FAILED").mean()
    ).reset_index(name="failed_rate_all")

    recent_tx = obs[obs["transaction_date"].dt.date >= RECENT_START]
    fail_recent = recent_tx.groupby("client_id").apply(
        lambda g: (g["status"] == "FAILED").mean()
    ).reset_index(name="failed_rate_recent")

    fail_feats = fail_all.merge(fail_recent, on="client_id", how="left").fillna(0)
    fail_feats["failed_spike"] = (
        fail_feats["failed_rate_recent"] /
        fail_feats["failed_rate_all"].replace(0, np.nan)
    ).fillna(1).clip(0, 5)

    # Balance drop (recomputed here for urgency weighting)
    bal_agg = obs[obs["status"] == "SUCCESS"].groupby("client_id").agg(
        balance_max  = ("balance", "max"),
        balance_last = ("balance", "last"),
    ).reset_index()
    bal_agg["balance_drop_pct"] = (
        (bal_agg["balance_max"] - bal_agg["balance_last"]) /
        bal_agg["balance_max"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)

    # Combine
    out = (app_feats
           .merge(large_share, on="client_id", how="outer")
           .merge(fail_feats,  on="client_id", how="outer")
           .merge(bal_agg[["client_id", "balance_drop_pct"]], on="client_id", how="outer")
           .fillna(0))

    # Normalise each component to [0, 1] before combining
    def _norm(series):
        mn, mx = series.min(), series.max()
        return (series - mn) / (mx - mn + 1e-9)

    out["urgency_score"] = (
        0.35 * _norm(out["balance_check_spike"])
      + 0.30 * _norm(out["high_amount_tx_share"])
      + 0.20 * _norm(out["failed_spike"])
      + 0.15 * _norm(out["balance_drop_pct"])
    ).clip(0, 1)

    return out[[
        "client_id", "balance_check_freq", "balance_check_spike",
        "high_amount_tx_share", "failed_spike", "urgency_score"
    ]].fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 7 — CREDIT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def feat_credit_analysis(credit: pd.DataFrame,
                          payments: pd.DataFrame,
                          scores: pd.DataFrame) -> pd.DataFrame:
    """
    Repayment quality and credit profile from the existing credit history
    (the phase 2 credit taken in Oct 2024 and paid off by Sep 2025).

    Features:
      has_prior_credit         — 1 if client ever had a credit
      prior_credit_approved    — 1 if most recent credit was approved
      prior_credit_closed      — 1 if credit is fully paid off
      payments_on_time_rate    — on-time payments / total past payments
      payments_late_rate       — late payments / total past payments
      payments_missed_rate     — missed payments / total past payments
      avg_days_late            — average days late across all late payments
      max_days_late            — worst single late payment
      credit_utilization       — remaining balance / credit limit (from scores)
      credit_score             — final credit score (300–850)
      dti_ratio                — debt-to-income from credit score calc
      payment_history_pts      — component score (0–110)
    """
    # Only look at credits from the observation window
    obs_credit = credit[
        credit["application_date"].dt.date <= pd.Timestamp(OBS_END).date()
    ].copy()

    has_credit = obs_credit.groupby("client_id").size().reset_index(name="n_credits")
    has_credit["has_prior_credit"] = 1

    approved = obs_credit[obs_credit["status"] != "DECLINED"]
    approved_flag = (
        approved.groupby("client_id").size().reset_index(name="_")
        .assign(prior_credit_approved=1)[["client_id", "prior_credit_approved"]]
    )

    closed_flag = (
        obs_credit[obs_credit["status"] == "CLOSED"]
        .groupby("client_id").size().reset_index(name="_")
        .assign(prior_credit_closed=1)[["client_id", "prior_credit_closed"]]
    )

    # Payment quality (only past/resolved payments, not SCHEDULED)
    past_pay = payments[payments["status"] != "SCHEDULED"].copy()
    if "days_late" in past_pay.columns:
        past_pay["days_late"] = pd.to_numeric(past_pay["days_late"], errors="coerce")

    pay_agg = past_pay.groupby("client_id").apply(lambda g: pd.Series({
        "payments_on_time_rate":  (g["status"] == "PAID_ON_TIME").mean(),
        "payments_late_rate":     (g["status"].isin(["PAID_LATE", "OVERDUE"])).mean(),
        "payments_missed_rate":   (g["status"] == "MISSED").mean(),
        "avg_days_late":          g.loc[g["status"].isin(["PAID_LATE","OVERDUE"]), "days_late"].mean(),
        "max_days_late":          g["days_late"].max(),
    })).reset_index()

    pay_agg = pay_agg.fillna(0)

    # Credit score components
    score_cols = ["client_id", "credit_score", "credit_utilization",
                  "dti_ratio", "payment_history_pts", "dti_pts",
                  "payments_on_time", "payments_late", "payments_missed"]
    score_cols = [c for c in score_cols if c in scores.columns]
    score_feats = scores[score_cols].copy()

    # Merge everything
    all_clients = credit["client_id"].unique()
    base = pd.DataFrame({"client_id": all_clients})

    out = (base
           .merge(has_credit[["client_id", "has_prior_credit"]], on="client_id", how="left")
           .merge(approved_flag,  on="client_id", how="left")
           .merge(closed_flag,    on="client_id", how="left")
           .merge(pay_agg,        on="client_id", how="left")
           .merge(score_feats,    on="client_id", how="left"))

    out["has_prior_credit"]       = out["has_prior_credit"].fillna(0)
    out["prior_credit_approved"]  = out["prior_credit_approved"].fillna(0)
    out["prior_credit_closed"]    = out["prior_credit_closed"].fillna(0)

    return out.fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 8 — APP BEHAVIOUR
# ─────────────────────────────────────────────────────────────────────────────

def feat_app_behaviour(app: pd.DataFrame) -> pd.DataFrame:
    """
    App usage patterns are a strong leading indicator of financial anxiety.
    Phase 2 and phase 4 both show elevated balance-check frequency.

    Features:
      total_sessions            — total app opens in observation window
      total_balance_checks      — total 'check_balance' actions
      balance_check_ratio       — balance checks / total sessions
      avg_session_duration_sec  — mean session length
      sessions_per_month        — engagement level
      sessions_last_3m          — sessions in last 3 months
      balance_checks_last_3m    — balance checks in last 3 months
      check_ratio_last_3m       — balance check ratio in last 3 months
                                  (higher than overall = rising anxiety)
    """
    obs = app[
        (app["timestamp"].dt.date >= OBS_START) &
        (app["timestamp"].dt.date <= OBS_END)
    ].copy()

    n_obs_months = ((OBS_END.year - OBS_START.year) * 12 +
                    OBS_END.month - OBS_START.month + 1)

    agg = obs.groupby("client_id").agg(
        total_sessions           = ("session_id", "count"),
        total_balance_checks     = ("action", lambda x: (x == "check_balance").sum()),
        avg_session_duration_sec = ("duration_sec", "mean"),
    ).reset_index()

    agg["balance_check_ratio"]  = (
        agg["total_balance_checks"] / agg["total_sessions"].replace(0, np.nan)
    ).fillna(0)
    agg["sessions_per_month"]   = agg["total_sessions"] / n_obs_months

    recent = obs[obs["timestamp"].dt.date >= RECENT_START]
    recent_agg = recent.groupby("client_id").agg(
        sessions_last_3m       = ("session_id", "count"),
        balance_checks_last_3m = ("action", lambda x: (x == "check_balance").sum()),
    ).reset_index()
    recent_agg["check_ratio_last_3m"] = (
        recent_agg["balance_checks_last_3m"] /
        recent_agg["sessions_last_3m"].replace(0, np.nan)
    ).fillna(0)

    return agg.merge(recent_agg, on="client_id", how="left").fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 9 — CLIENT PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def feat_client_profile(clients: pd.DataFrame) -> pd.DataFrame:
    """
    Static demographic and relationship features.

    Features:
      age                   — client age
      monthly_salary        — declared monthly salary
      employment_type_enc   — employment encoded (EMPLOYED=0, SELF_EMPLOYED=1,
                              STUDENT=2, RETIRED=3)
      is_employed           — binary flag for employed
      dependants            — number of dependants (affects financial pressure)
      account_age_years     — how long they've been a client
      marital_encoded       — SINGLE=0, MARRIED=1, DIVORCED=2, WIDOWED=3
    """
    emp_map     = {"EMPLOYED": 0, "SELF_EMPLOYED": 1, "STUDENT": 2, "RETIRED": 3}
    marital_map = {"SINGLE": 0, "MARRIED": 1, "DIVORCED": 2, "WIDOWED": 3}

    out = clients[["client_id", "age", "monthly_salary", "employment_type",
                   "dependants", "account_open_date"]].copy()

    out["employment_type_enc"] = out["employment_type"].map(emp_map).fillna(0)
    out["is_employed"]         = (out["employment_type"] == "EMPLOYED").astype(int)

    if "marital_status" in clients.columns:
        out["marital_encoded"] = clients["marital_status"].map(marital_map).fillna(0)
    else:
        out["marital_encoded"] = 0

    ref_date = pd.Timestamp(OBS_END)
    out["account_age_years"] = (
        (ref_date - pd.to_datetime(out["account_open_date"])).dt.days / 365.25
    ).clip(lower=0)

    return out[[
        "client_id", "age", "monthly_salary", "employment_type_enc",
        "is_employed", "dependants", "account_age_years", "marital_encoded"
    ]].fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# TARGET LABEL
# ─────────────────────────────────────────────────────────────────────────────

"""
REPLACE the build_target() function in feature_engineering.py with this.

The ground truth label is now stored directly in client_mcc_assignments.csv
(column: will_seek_loan), generated by transactions_generate.py.
No more inferring the label from spend patterns — which was the source
of the near-constant target problem.
"""

def build_target(tx, mcc_asgn):
    """
    Label comes directly from client_mcc_assignments.csv (will_seek_loan column).
    This is the ground truth set at generation time — no inference needed.

    Also computes target_share_p4 and spike_ratio as informational columns
    (not used as model features — they would be leakage — but useful for
    verifying that the data generation worked correctly).
    """
    # Ground truth label — use this as the model target
    label = mcc_asgn[["client_id", "will_seek_loan"]].copy()

    # Informational: how concentrated was phase 4 spend in target_mcc?
    p4 = tx[
        (tx["transaction_date"].dt.date >= TARGET_START) &
        (tx["transaction_date"].dt.date <= TARGET_END) &
        (tx["status"] == "SUCCESS")
    ].copy()

    total_p4 = p4.groupby("client_id")["amount"].sum().reset_index(name="total_p4")
    p4m = p4.merge(mcc_asgn[["client_id", "target_mcc"]], on="client_id", how="left")
    target_spend = (
        p4m[p4m["mcc_code"] == p4m["target_mcc"]]
        .groupby("client_id")["amount"].sum()
        .reset_index(name="target_spend_p4")
    )
    p4_stats = total_p4.merge(target_spend, on="client_id", how="left").fillna(0)
    p4_stats["target_share_p4"] = (
        p4_stats["target_spend_p4"] / p4_stats["total_p4"].replace(0, np.nan)
    ).fillna(0)

    result = label.merge(p4_stats[["client_id", "target_share_p4"]],
                         on="client_id", how="left").fillna(0)

    print(f"[TARGET] Label distribution:")
    print(f"  will_seek_loan=1 : {result['will_seek_loan'].sum()}")
    print(f"  will_seek_loan=0 : {(result['will_seek_loan']==0).sum()}")
    print(f"[TARGET] Phase 4 target_share_p4 by label:")
    print(result.groupby("will_seek_loan")["target_share_p4"].mean().round(3))

    return result[["client_id", "target_share_p4", "will_seek_loan"]]

# ─────────────────────────────────────────────────────────────────────────────
# MAIN — ASSEMBLE FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  feature_engineering.py")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)
    data = load_data()

    tx       = data["tx"]
    app      = data["app"]
    clients  = data["clients"]
    credit   = data["credit"]
    payments = data["payments"]
    scores   = data["scores"]
    mcc_asgn = data["mcc_asgn"]

    print("\n[FEATURES] Computing feature groups ...")

    f1 = feat_balance_dynamics(tx)
    print(f"  1. Balance dynamics     : {len(f1.columns)-1} features")

    f2 = feat_burn_rate(tx, clients)
    print(f"  2. Burn rate            : {len(f2.columns)-1} features")

    f3 = feat_volatility(tx)
    print(f"  3. Volatility           : {len(f3.columns)-1} features")

    f4 = feat_category_spikes(tx)
    print(f"  4. Category spikes      : {len(f4.columns)-1} features")

    f5 = feat_zero_day_count(tx)
    print(f"  5. Zero-day count       : {len(f5.columns)-1} features")

    f6 = feat_urgency_score(tx, app)
    print(f"  6. Urgency score        : {len(f6.columns)-1} features")

    f7 = feat_credit_analysis(credit, payments, scores)
    print(f"  7. Credit analysis      : {len(f7.columns)-1} features")

    f8 = feat_app_behaviour(app)
    print(f"  8. App behaviour        : {len(f8.columns)-1} features")

    f9 = feat_client_profile(clients)
    print(f"  9. Client profile       : {len(f9.columns)-1} features")

    target = build_target(tx, mcc_asgn)
    print(f"\n[TARGET] will_take_loan distribution:")
    print(f"  1 (will take loan) : {target['will_seek_loan'].sum()}")
    print(f"  0 (will not)       : {(target['will_seek_loan'] == 0).sum()}")

    # Merge all feature groups on client_id
    print("\n[MERGE] Assembling final feature matrix ...")
    base = clients[["client_id"]].copy()
    for f in [f1, f2, f3, f4, f5, f6, f7, f8, f9, target]:
        base = base.merge(f, on="client_id", how="left")

    base = base.fillna(0)

    # Drop any intermediate helper columns not needed by the model
    drop_cols = []
    base = base.drop(columns=[c for c in drop_cols if c in base.columns])

    base.to_csv(OUT_PATH, index=False)

    print(f"\n  Final feature matrix : {len(base)} rows × {len(base.columns)} columns")
    print(f"  Feature columns      : {len(base.columns) - 2}")  # excl. client_id + target
    print(f"\n  Columns:")
    for col in base.columns:
        print(f"    {col}")
    print(f"\n  Saved -> {OUT_PATH}")
    print("=" * 60)

    return base


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN — ASSEMBLE FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  feature_engineering.py")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)
    data = load_data()

    tx       = data["tx"]
    app      = data["app"]
    clients  = data["clients"]
    credit   = data["credit"]
    payments = data["payments"]
    scores   = data["scores"]
    mcc_asgn = data["mcc_asgn"]

    print("\n[FEATURES] Computing feature groups ...")

    f1 = feat_balance_dynamics(tx)
    print(f"  1. Balance dynamics     : {len(f1.columns)-1} features")

    f2 = feat_burn_rate(tx, clients)
    print(f"  2. Burn rate            : {len(f2.columns)-1} features")

    f3 = feat_volatility(tx)
    print(f"  3. Volatility           : {len(f3.columns)-1} features")

    f4 = feat_category_spikes(tx)
    print(f"  4. Category spikes      : {len(f4.columns)-1} features")

    f5 = feat_zero_day_count(tx)
    print(f"  5. Zero-day count       : {len(f5.columns)-1} features")

    f6 = feat_urgency_score(tx, app)
    print(f"  6. Urgency score        : {len(f6.columns)-1} features")

    f7 = feat_credit_analysis(credit, payments, scores)
    print(f"  7. Credit analysis      : {len(f7.columns)-1} features")

    f8 = feat_app_behaviour(app)
    print(f"  8. App behaviour        : {len(f8.columns)-1} features")

    f9 = feat_client_profile(clients)
    print(f"  9. Client profile       : {len(f9.columns)-1} features")

    target = build_target(tx, mcc_asgn)
    print(f"\n[TARGET] will_take_loan distribution:")
    print(f"  1 (will take loan) : {target['will_seek_loan'].sum()}")
    print(f"  0 (will not)       : {(target['will_seek_loan'] == 0).sum()}")

    # Merge all feature groups on client_id
    print("\n[MERGE] Assembling final feature matrix ...")
    base = clients[["client_id"]].copy()
    for f in [f1, f2, f3, f4, f5, f6, f7, f8, f9, target]:
        base = base.merge(f, on="client_id", how="left")

    base = base.fillna(0)

    # Drop any intermediate helper columns not needed by the model
    drop_cols = []
    base = base.drop(columns=[c for c in drop_cols if c in base.columns])

    base.to_csv(OUT_PATH, index=False)

    print(f"\n  Final feature matrix : {len(base)} rows × {len(base.columns)} columns")
    print(f"  Feature columns      : {len(base.columns) - 2}")  # excl. client_id + target
    print(f"\n  Columns:")
    for col in base.columns:
        print(f"    {col}")
    print(f"\n  Saved -> {OUT_PATH}")
    print("=" * 60)

    return base


if __name__ == "__main__":
    main()