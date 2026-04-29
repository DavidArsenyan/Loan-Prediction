"""
credit_score.py
================
Calculates a credit score (300–850) for every client using 5 components,
each worth 20% of the total score. All values are derived from actual data.

Components
----------
  1. Payment History      (20%) — on-time/late/missed payments
  2. Credit Utilization   (20%) — outstanding balance vs credit limit
  3. Credit Inquiries     (20%) — too many applications in a short window
  4. Debt-to-Income Ratio (20%) — monthly payment burden vs salary
  5. Relationship Data    (20%) — account age, credit history length

Scoring math
------------
  Each component contributes 0–110 points.
  Total max = 5 × 110 = 550.
  Final score = 300 (floor) + total_component_points (capped at 550 → max 850).

Reads  : ./data/clients.csv
         ./data/credit.csv
         ./data/payments.csv
Outputs: ./data/credit_scores.csv
"""

import pandas as pd
import numpy as np
import os
from datetime import date, datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR    = "../data"
IN_CLIENTS  = f"{DATA_DIR}/clients.csv"
IN_CREDIT   = f"{DATA_DIR}/credit.csv"
IN_PAYMENTS = f"{DATA_DIR}/payments.csv"
OUT_SCORES  = f"{DATA_DIR}/credit_scores.csv"

SCORE_FLOOR = 300
SCORE_CAP   = 850
MAX_PER_COMPONENT = 110   # 5 × 110 = 550 max additional points
SCORE_DATE  = date(2025, 12, 1)   # "today" — when scores are computed


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 1 — PAYMENT HISTORY  (20%)
# Base: 110 pts.  Deductions for late/missed payments.
# ─────────────────────────────────────────────────────────────────────────────

def score_payment_history(payments_df: pd.DataFrame, client_id: int) -> tuple[float, dict]:
    """
    Deductions:
      PAID_LATE (1-5 days)   : -5 pts each
      PAID_LATE (6-30 days)  : -12 pts each
      PAID_LATE (31-60 days) : -25 pts each
      MISSED / OVERDUE       : -35 pts each
    """
    client_pay = payments_df[
        (payments_df["client_id"] == client_id) &
        (payments_df["status"] != "SCHEDULED")
    ]

    if len(client_pay) == 0:
        # No payment history — neutral, not rewarded nor penalised
        return 70.0, {"total_payments": 0, "late": 0, "missed": 0}

    deduction = 0.0
    late_count   = 0
    missed_count = 0

    for _, row in client_pay.iterrows():
        days = row["days_late"]
        s    = row["status"]

        if s == "PAID_ON_TIME":
            pass
        elif s in ("PAID_LATE", "OVERDUE"):
            late_count += 1
            if days is None or days <= 5:
                deduction += 5
            elif days <= 30:
                deduction += 12
            else:
                deduction += 25
        elif s == "MISSED":
            missed_count += 1
            deduction += 35

    score = max(0.0, MAX_PER_COMPONENT - deduction)
    return score, {
        "total_payments": len(client_pay),
        "on_time":        int((client_pay["status"] == "PAID_ON_TIME").sum()),
        "late":           late_count,
        "missed":         missed_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 2 — CREDIT UTILIZATION  (20%)
# utilization = remaining_balance / credit_limit
# ─────────────────────────────────────────────────────────────────────────────

UTIL_TABLE = [
    (0.00, 0.10, 110),
    (0.10, 0.30, 95),
    (0.30, 0.50, 75),
    (0.50, 0.70, 50),
    (0.70, 0.90, 25),
    (0.90, 1.01, 5),
]

def score_credit_utilization(credit_df: pd.DataFrame,
                             payments_df: pd.DataFrame,
                             client_id: int) -> tuple[float, dict]:
    active = credit_df[
        (credit_df["client_id"] == client_id) &
        (credit_df["status"].isin(["ACTIVE", "CLOSED"]))
    ]
    if active.empty:
        # No credit → neutral
        return 80.0, {"utilization": None}

    # Latest remaining balance from payments
    client_pay = payments_df[payments_df["client_id"] == client_id]
    if not client_pay.empty:
        remaining = float(client_pay.sort_values("due_date").iloc[-1]["remaining_balance"])
    else:
        remaining = float(active.iloc[0]["credit_amount"])

    limit = float(active.iloc[0]["credit_limit"])
    util  = remaining / limit if limit > 0 else 0.0
    util  = max(0.0, min(1.0, util))

    pts = 5.0   # default
    for lo, hi, p in UTIL_TABLE:
        if lo <= util < hi:
            pts = float(p)
            break

    return pts, {"utilization": round(util, 3), "remaining_balance": round(remaining)}


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 3 — CREDIT INQUIRIES  (20%)
# Too many applications in a 12-month window reduce the score.
# ─────────────────────────────────────────────────────────────────────────────

INQUIRY_TABLE = [
    (0, 110),
    (1, 95),
    (2, 75),
    (3, 50),
    (4, 25),
]

def score_credit_inquiries(credit_df: pd.DataFrame, client_id: int) -> tuple[float, dict]:
    client_credits = credit_df[credit_df["client_id"] == client_id].copy()
    if client_credits.empty:
        return 110.0, {"inquiries_12m": 0}

    client_credits["application_date"] = pd.to_datetime(client_credits["application_date"])
    window_start = pd.Timestamp(SCORE_DATE) - pd.DateOffset(months=12)
    recent = client_credits[client_credits["application_date"] >= window_start]
    n = len(recent)

    pts = 0.0
    for threshold, score in INQUIRY_TABLE:
        if n <= threshold:
            pts = float(score)
            break
    if n >= 5:
        pts = 5.0

    return pts, {"inquiries_12m": n, "total_credits": len(client_credits)}


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 4 — DEBT-TO-INCOME RATIO  (20%)
# DTI = total monthly credit payments / monthly salary
# ─────────────────────────────────────────────────────────────────────────────

DTI_TABLE = [
    (0.00, 0.10, 110),
    (0.10, 0.20, 90),
    (0.20, 0.30, 70),
    (0.30, 0.40, 45),
    (0.40, 0.50, 20),
    (0.50, 2.00, 5),
]

def score_dti(credit_df: pd.DataFrame, client: pd.Series) -> tuple[float, dict]:
    salary = float(client["monthly_salary"])
    active = credit_df[
        (credit_df["client_id"] == int(client["client_id"])) &
        (credit_df["status"] == "ACTIVE")
    ]

    total_monthly = float(active["monthly_payment"].sum()) if not active.empty else 0.0
    dti = total_monthly / salary if salary > 0 else 0.0
    dti = max(0.0, min(2.0, dti))

    pts = 5.0
    for lo, hi, p in DTI_TABLE:
        if lo <= dti < hi:
            pts = float(p)
            break

    return pts, {
        "monthly_debt_payment": round(total_monthly),
        "monthly_salary":       round(salary),
        "dti_ratio":            round(dti, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 5 — RELATIONSHIP DATA  (20%)
# Account age, credit history, diversity of products.
# ─────────────────────────────────────────────────────────────────────────────

def score_relationship(client: pd.Series, credit_df: pd.DataFrame) -> tuple[float, dict]:
    """
    Points breakdown:
      Account age
        < 1 year  : 10 pts
        1–3 years : 30 pts
        3–5 years : 50 pts
        > 5 years : 70 pts
      Credit history
        Closed credit (good repayment) : +25 pts
        Active credit                  : +15 pts
        No credit history              : +0 pts
      Declined only                    : -10 pts (penalise rejected applications)
    """
    today     = SCORE_DATE
    open_date = datetime.strptime(str(client["account_open_date"]), "%Y-%m-%d").date()
    years     = (today - open_date).days / 365.25

    if years < 1:
        age_pts = 10.0
    elif years < 3:
        age_pts = 30.0
    elif years < 5:
        age_pts = 50.0
    else:
        age_pts = 70.0

    client_credits = credit_df[credit_df["client_id"] == int(client["client_id"])]
    has_closed  = (client_credits["status"] == "CLOSED").any()
    has_active  = (client_credits["status"] == "ACTIVE").any()
    all_declined = len(client_credits) > 0 and (client_credits["status"] == "DECLINED").all()

    credit_pts = 0.0
    if has_closed:
        credit_pts += 25.0
    if has_active:
        credit_pts += 15.0
    if all_declined:
        credit_pts -= 10.0

    total = min(age_pts + credit_pts, MAX_PER_COMPONENT)

    return total, {
        "account_age_years": round(years, 1),
        "has_closed_credit": bool(has_closed),
        "has_active_credit": bool(has_active),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SCORE ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def rating_label(score: int) -> str:
    if score >= 750: return "Excellent"
    if score >= 670: return "Good"
    if score >= 580: return "Fair"
    if score >= 500: return "Poor"
    return "Very Poor"


def compute_score(client: pd.Series,
                  credit_df: pd.DataFrame,
                  payments_df: pd.DataFrame) -> dict:
    cid = int(client["client_id"])

    p1, d1 = score_payment_history(payments_df, cid)
    p2, d2 = score_credit_utilization(credit_df, payments_df, cid)
    p3, d3 = score_credit_inquiries(credit_df, cid)
    p4, d4 = score_dti(credit_df, client)
    p5, d5 = score_relationship(client, credit_df)

    total_pts = p1 + p2 + p3 + p4 + p5
    total_pts = max(0.0, min(total_pts, 550.0))
    final     = int(SCORE_FLOOR + total_pts)
    final     = max(SCORE_FLOOR, min(final, SCORE_CAP))

    return {
        "client_id":                    cid,
        "score_date":                   SCORE_DATE.isoformat(),
        "credit_score":                 final,
        "rating":                       rating_label(final),
        # Component scores (raw points out of 110)
        "payment_history_pts":          round(p1, 1),
        "credit_utilization_pts":       round(p2, 1),
        "credit_inquiries_pts":         round(p3, 1),
        "dti_pts":                      round(p4, 1),
        "relationship_pts":             round(p5, 1),
        # Detail fields
        "payments_on_time":             d1.get("on_time"),
        "payments_late":                d1.get("late"),
        "payments_missed":              d1.get("missed"),
        "credit_utilization":           d2.get("utilization"),
        "remaining_balance":            d2.get("remaining_balance"),
        "inquiries_last_12m":           d3.get("inquiries_12m"),
        "monthly_debt_payment":         d4.get("monthly_debt_payment"),
        "monthly_salary":               d4.get("monthly_salary"),
        "dti_ratio":                    d4.get("dti_ratio"),
        "account_age_years":            d5.get("account_age_years"),
        "has_closed_credit":            d5.get("has_closed_credit"),
        "has_active_credit":            d5.get("has_active_credit"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  credit_score.py")
    print("=" * 55)

    clients_df  = pd.read_csv(IN_CLIENTS)
    credit_df   = pd.read_csv(IN_CREDIT)
    try:
        payments_df = pd.read_csv(IN_PAYMENTS)
    except Exception:
        payments_df = pd.DataFrame(columns=["client_id","credit_id","payment_number",
            "due_date","amount_due","amount_paid","payment_date","days_late",
            "interest_paid","principal_paid","remaining_balance","status"])

    records = []
    for _, client in clients_df.iterrows():
        records.append(compute_score(client, credit_df, payments_df))

    df = pd.DataFrame(records)
    df.to_csv(OUT_SCORES, index=False)

    print(f"  Clients scored : {len(df)}")
    print(f"  Score range    : {df['credit_score'].min()} – {df['credit_score'].max()}")
    print(f"  Score mean     : {df['credit_score'].mean():.0f}")
    print(f"\n  Rating distribution:")
    for rating, count in df["rating"].value_counts().items():
        bar = "█" * (count // 2)
        print(f"    {rating:<12} {count:>4}  {bar}")
    print(f"\n  Component averages (out of 110 pts each):")
    for col, label in [
        ("payment_history_pts",    "Payment History  "),
        ("credit_utilization_pts", "Credit Utilization"),
        ("credit_inquiries_pts",   "Credit Inquiries "),
        ("dti_pts",                "Debt-to-Income   "),
        ("relationship_pts",       "Relationship Data"),
    ]:
        print(f"    {label}: {df[col].mean():.1f} avg")
    print(f"\n  Saved → {OUT_SCORES}")
    print("=" * 55)

    return df


if __name__ == "__main__":
    main()