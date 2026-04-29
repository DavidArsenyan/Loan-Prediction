"""
credit_generate.py
===================
Reads transactions and client data to generate realistic credit applications.

Logic
-----
• Detects phase 2 spike (Jul–Sep 2024) per client from transactions.csv
• Clients whose trigger_mcc share > 65% in phase 2 get an application generated
• Approval depends on: DTI, salary, employment type, failed_tx rate
• Approved clients get a 12-month credit starting ~Oct 2024 → paid off Sep 2025
• Phase 4 clients (Oct–Nov 2025 spike) do NOT get a credit → prediction target
• Some clients may have been declined → no payments generated for them

Reads  : ./data/transactions.csv
         ./data/clients.csv
         ./data/client_mcc_assignments.csv
Outputs: ./data/credit.csv
         ./data/payments.csv
"""

import pandas as pd
import numpy as np
import random
import os
import calendar
from datetime import date, datetime, timedelta

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = "../data"
IN_TX      = f"{DATA_DIR}/transactions.csv"
IN_CLIENTS = f"{DATA_DIR}/clients.csv"
IN_MCC     = f"{DATA_DIR}/client_mcc_assignments.csv"
OUT_CREDIT = f"{DATA_DIR}/credit.csv"
OUT_PAY    = f"{DATA_DIR}/payments.csv"

# Phase 2 window for spike detection
P2_START = date(2024, 7, 1)
P2_END   = date(2024, 9, 30)
P3_START = date(2024, 10, 1)   # credit starts here
TODAY    = date(2025, 12, 1)

# Spike threshold to trigger an application
SPIKE_SHARE_THRESHOLD = 0.65

# DTI cap for approval (monthly_payment / salary)
DTI_MAX = 0.40

# Candidate terms — shortest that keeps payment within DTI cap
CREDIT_TERMS = [12, 18, 24]

# Annual rate by employment type and salary level
def get_rate(employment: str, salary: int, failed_rate: float) -> float:
    base = {
        "EMPLOYED":      random.uniform(10.0, 14.0),
        "SELF_EMPLOYED": random.uniform(13.0, 18.0),
        "STUDENT":       random.uniform(18.0, 24.0),
        "RETIRED":       random.uniform(12.0, 16.0),
    }.get(employment, 14.0)
    # Higher failed_rate → higher risk → higher rate
    penalty = min(failed_rate * 20, 8.0)   # up to +8% for very risky clients
    return round(min(base + penalty, 30.0), 2)

MCC_PURPOSE = {
    5211: "Home Repair / Renovation",
    1021: "Electronics / Gadgets",
    5680: "Fashion / Clothing",
    3001: "Travel / Vacation",
}

EMPLOYMENT_APPROVAL_BASE = {
    "EMPLOYED":      0.85,
    "SELF_EMPLOYED": 0.70,
    "STUDENT":       0.45,
    "RETIRED":       0.75,
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def add_months(d: date, n: int) -> date:
    month    = d.month - 1 + n
    year     = d.year + month // 12
    month    = month % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(d.day, last_day))

def annuity_payment(principal: float, annual_rate: float, months: int) -> float:
    r = annual_rate / 100 / 12
    if r == 0:
        return principal / months
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)

def round100(x: float) -> int:
    return max(100, round(x / 100) * 100)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DETECT SPIKE IN PHASE 2 TRANSACTIONS
# ─────────────────────────────────────────────────────────────────────────────

def detect_phase2_spikes(tx_df: pd.DataFrame,
                          mcc_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each client, compute what fraction of phase 2 spend went to trigger_mcc.
    Returns a DataFrame with spike_share and total_spike_spend per client.
    """
    p2 = tx_df[
        (pd.to_datetime(tx_df["transaction_date"]).dt.date >= P2_START) &
        (pd.to_datetime(tx_df["transaction_date"]).dt.date <= P2_END) &
        (tx_df["status"] == "SUCCESS")
    ].copy()

    spend_total = p2.groupby("client_id")["amount"].sum().reset_index(name="total_spend_p2")
    p2_merged   = p2.merge(mcc_df[["client_id", "trigger_mcc"]], on="client_id")
    spend_trigger = (
        p2_merged[p2_merged["mcc_code"] == p2_merged["trigger_mcc"]]
        .groupby("client_id")["amount"].sum()
        .reset_index(name="trigger_spend_p2")
    )

    result = spend_total.merge(spend_trigger, on="client_id", how="left").fillna(0)
    result["spike_share"] = result["trigger_spend_p2"] / result["total_spend_p2"].replace(0, np.nan)
    result["spike_share"] = result["spike_share"].fillna(0)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — DECIDE CREDIT AMOUNT FROM SPIKE SPEND
# ─────────────────────────────────────────────────────────────────────────────

def derive_credit_amount(trigger_spend: float, salary: float,
                         rate: float) -> tuple[int, int]:
    """
    Returns (credit_amount, term_months) ensuring the monthly payment
    stays within DTI_MAX × salary.

    Strategy:
      1. Desired need = trigger_spend × 0.5–1.0 (the shortfall they want covered)
      2. For each candidate term (12, 18, 24 months), compute the monthly payment
      3. Use the shortest term where payment ≤ DTI_MAX × salary
      4. If no term works, shrink the amount until it fits in 24 months
    """
    max_monthly = salary * DTI_MAX
    # Desired amount: 50–100% of what they spent in spike
    need = trigger_spend * random.uniform(0.5, 1.0)
    need = max(100_000, min(need, salary * 6))   # floor / ceiling

    r = rate / 100 / 12

    def monthly(principal, months):
        if r == 0: return principal / months
        return principal * r * (1+r)**months / ((1+r)**months - 1)

    # Try each candidate term from shortest to longest
    for term in CREDIT_TERMS:
        pmt = monthly(need, term)
        if pmt <= max_monthly:
            return round(need / 50_000) * 50_000, term

    # Need is too large even for 24 months → shrink to what 24m can afford
    if r == 0:
        max_principal = max_monthly * 24
    else:
        max_principal = max_monthly * ((1+r)**24 - 1) / (r * (1+r)**24)

    affordable = max(50_000, min(max_principal * 0.90, need))
    return round(affordable / 50_000) * 50_000, 24


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — APPROVAL DECISION
# ─────────────────────────────────────────────────────────────────────────────

def decide_approval(client: pd.Series, monthly_payment: float,
                    failed_rate: float) -> tuple[bool, str]:
    """
    Returns (approved: bool, decline_reason: str).
    """
    salary     = float(client["monthly_salary"])
    employment = str(client["employment_type"])

    dti = monthly_payment / salary if salary > 0 else 1.0

    # Hard rules
    if dti > DTI_MAX:
        return False, f"DTI too high ({dti:.1%} > {DTI_MAX:.0%})"
    if employment == "STUDENT" and monthly_payment > salary * 0.25:
        return False, "Student income too low"
    if failed_rate > 0.45:
        return False, "High failed transaction rate"

    # Probabilistic approval based on employment type
    base_prob = EMPLOYMENT_APPROVAL_BASE.get(employment, 0.7)
    # Better behaviour → higher chance
    behaviour_bonus = max(0, (0.40 - DTI_MAX) * 0.5)
    prob = min(base_prob + behaviour_bonus - failed_rate * 0.3, 0.97)

    approved = random.random() < prob
    reason   = "" if approved else "Risk assessment declined"
    return approved, reason


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PAYMENT BEHAVIOUR (driven by failed_rate from Phase 1+2)
# ─────────────────────────────────────────────────────────────────────────────

def payment_weights(failed_rate: float, balance_pressure: float) -> list[float]:
    """
    Stress index from transaction history → payment probability weights.
    Buckets: [ON_TIME, LATE_1_5, LATE_6_30, LATE_31_60, MISSED]
    """
    stress = min(failed_rate * 1.5 + balance_pressure * 0.5, 1.0)
    ideal  = [0.94, 0.03, 0.015, 0.010, 0.005]
    worst  = [0.42, 0.25, 0.18,  0.09,  0.06 ]
    w = [ideal[i] + stress * (worst[i] - ideal[i]) for i in range(5)]
    total = sum(w)
    return [x / total for x in w]

from typing import List, Tuple, Optional, Union

def simulate_payment(
    due_date: date,
    weights: List[float]
) -> Tuple[str, Optional[date], Optional[int], float]:
    """
    Returns (status, payment_date, days_late, fraction_paid).
    """
    buckets = ["ON_TIME", "LATE_1_5", "LATE_6_30", "LATE_31_60", "MISSED"]
    bucket  = random.choices(buckets, weights=weights)[0]

    if bucket == "ON_TIME":
        return "PAID_ON_TIME", due_date - timedelta(days=random.randint(0, 2)), 0, 1.0

    if bucket == "LATE_1_5":
        d = random.randint(1, 5)
        return "PAID_LATE", due_date + timedelta(days=d), d, 1.0

    if bucket == "LATE_6_30":
        d = random.randint(6, 30)
        return "PAID_LATE", due_date + timedelta(days=d), d, 1.0

    if bucket == "LATE_31_60":
        d = random.randint(31, 60)
        frac = random.uniform(0.5, 1.0)
        return "PAID_LATE", due_date + timedelta(days=d), d, frac

    # MISSED
    return "MISSED", None, None, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — BUILD CREDIT + PAYMENTS
# ─────────────────────────────────────────────────────────────────────────────

def generate_credit_and_payments(
    client: pd.Series,
    trigger_mcc: int,
    trigger_spend: float,
    failed_rate: float,
    balance_pressure: float,
    credit_id: int,
) -> tuple[dict, list[dict]]:
    """Generate one credit record and its full payment schedule."""
    salary = float(client["monthly_salary"])

    rate     = get_rate(str(client["employment_type"]), int(salary), failed_rate)
    amount, term = derive_credit_amount(trigger_spend, salary, rate)
    monthly  = round100(annuity_payment(amount, rate, term))

    approved, decline_reason = decide_approval(client, monthly, failed_rate)

    # Application date: first week of Oct 2024
    app_date = date(2024, 10, random.randint(1, 10))

    if not approved:
        credit = {
            "credit_id":         credit_id,
            "client_id":         int(client["client_id"]),
            "application_date":  app_date.isoformat(),
            "approved_date":     None,
            "credit_amount":     amount,
            "credit_limit":      amount,
            "annual_rate_pct":   rate,
            "term_months":       term,

            "monthly_payment":   monthly,
            "purpose":           MCC_PURPOSE.get(trigger_mcc, "General"),
            "status":            "DECLINED",
            "decline_reason":    decline_reason,
            "start_date":        None,
            "end_date":          None,
        }
        return credit, []

    # Approved — credit starts a few days after application
    start_date = app_date + timedelta(days=random.randint(2, 7))
    end_date   = add_months(start_date, term)

    credit = {
        "credit_id":         credit_id,
        "client_id":         int(client["client_id"]),
        "application_date":  app_date.isoformat(),
        "approved_date":     (app_date + timedelta(days=random.randint(1, 3))).isoformat(),
        "credit_amount":     amount,
        "credit_limit":      amount,
        "annual_rate_pct":   rate,
        "term_months":       term,
        "monthly_payment":   monthly,
        "purpose":           MCC_PURPOSE.get(trigger_mcc, "General"),
        "status":           "CLOSED" if end_date <= TODAY else "ACTIVE",
        "decline_reason":    None,
        "start_date":        start_date.isoformat(),
        "end_date":          end_date.isoformat(),
    }

    # Generate payment schedule
    weights   = payment_weights(failed_rate, balance_pressure)
    r         = rate / 100 / 12
    balance   = float(amount)
    payments  = []
    pay_id    = credit_id * 1000 + 1

    for i in range(1, term + 1):
        due_date  = add_months(start_date, i)
        interest  = round(balance * r, 2)
        principal = round(monthly - interest, 2)
        balance   = round(max(0.0, balance - principal), 2)

        status, paid_date, days_late, fraction = simulate_payment(due_date, weights)

        amount_paid = round100(monthly * fraction) if fraction > 0 else None

        # If paid_date would be past TODAY → still overdue
        if paid_date and paid_date > TODAY:
            status      = "OVERDUE"
            days_late   = (TODAY - due_date).days
            paid_date   = None
            amount_paid = None

        payments.append({
            "payment_id":        pay_id,
            "credit_id":         credit_id,
            "client_id":         int(client["client_id"]),
            "payment_number":    i,
            "due_date":          due_date.isoformat(),
            "amount_due":        monthly,
            "amount_paid":       amount_paid,
            "payment_date":      paid_date.isoformat() if paid_date else None,
            "days_late":         days_late,
            "interest_paid":     interest,
            "principal_paid":    principal,
            "remaining_balance": balance,
            "status":            status,
        })
        pay_id += 1

    return credit, payments


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  credit_generate.py")
    print("=" * 55)

    clients_df = pd.read_csv(IN_CLIENTS)
    tx_df      = pd.read_csv(IN_TX)
    mcc_df     = pd.read_csv(IN_MCC)

    # Compute per-client transaction behaviour features
    tx_df["transaction_date"] = pd.to_datetime(tx_df["transaction_date"])

    # Failed rate in phase 1+2 (before credit) — reflects pre-credit behaviour
    pre_credit = tx_df[tx_df["transaction_date"].dt.date <= P2_END]
    fail_rates = (
        pre_credit.groupby("client_id")
        .apply(lambda g: (g["status"] == "FAILED").mean())
        .reset_index(name="failed_rate")
    )

    # Balance pressure: how much did balance drop from max to min in phase 2
    p2_bal = tx_df[
        (tx_df["transaction_date"].dt.date >= P2_START) &
        (tx_df["transaction_date"].dt.date <= P2_END)
    ].groupby("client_id").agg(
        bal_max=("balance", "max"),
        bal_min=("balance", "min")
    ).reset_index()
    p2_bal["balance_pressure"] = (
        (p2_bal["bal_max"] - p2_bal["bal_min"]) /
        p2_bal["bal_max"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)

    # Detect phase 2 spikes
    spikes = detect_phase2_spikes(tx_df.assign(
        transaction_date=tx_df["transaction_date"].dt.strftime("%Y-%m-%d %H:%M")
    ), mcc_df)

    # Merge all
    analysis = (
        mcc_df
        .merge(clients_df[["client_id", "monthly_salary",
                            "employment_type", "first_name", "last_name"]], on="client_id")
        .merge(spikes,    on="client_id", how="left")
        .merge(fail_rates, on="client_id", how="left")
        .merge(p2_bal[["client_id", "balance_pressure"]], on="client_id", how="left")
        .fillna(0)
    )

    # Only clients whose phase 2 showed a real spike get an application
    applicants = analysis[analysis["spike_share"] >= SPIKE_SHARE_THRESHOLD].copy()
    print(f"  Clients with phase 2 spike : {len(applicants)}")

    all_credits  = []
    all_payments = []
    credit_id    = 5001

    for _, row in applicants.iterrows():
        client = clients_df[clients_df["client_id"] == row["client_id"]].iloc[0]

        credit, payments = generate_credit_and_payments(
            client        = client,
            trigger_mcc   = int(row["trigger_mcc"]),
            trigger_spend  = float(row["trigger_spend_p2"]),
            failed_rate   = float(row["failed_rate"]),
            balance_pressure = float(row["balance_pressure"]),
            credit_id     = credit_id,
        )
        all_credits.append(credit)
        all_payments.extend(payments)
        credit_id += 1

    df_credit   = pd.DataFrame(all_credits)
    df_payments = pd.DataFrame(all_payments)

    df_credit.to_csv(OUT_CREDIT,  index=False)
    df_payments.to_csv(OUT_PAY, index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    status_counts = df_credit["status"].value_counts()
    print(f"\n  Credit applications : {len(df_credit)}")
    for s, n in status_counts.items():
        print(f"    {s:<12} {n:>4}")

    approved = df_credit[df_credit["status"] != "DECLINED"]
    print(f"\n  Approved loans:")
    print(f"    Avg amount      : {approved['credit_amount'].mean():,.0f} AMD")
    print(f"    Avg rate        : {approved['annual_rate_pct'].mean():.1f}%")
    print(f"    Avg term        : {approved['term_months'].mean():.0f} months")

    if len(df_payments) > 0:
        pay_status = df_payments["status"].value_counts()
        print(f"\n  Payments generated  : {len(df_payments)}")
        for s, n in pay_status.items():
            print(f"    {s:<22} {n:>4}  ({n/len(df_payments)*100:.1f}%)")

    print(f"\n  Saved → {OUT_CREDIT}")
    print(f"  Saved → {OUT_PAY}")
    print("=" * 55)

    return df_credit, df_payments


if __name__ == "__main__":
    main()