"""
transaction_generate.py
========================
Generates transactions.csv and application.csv (app sessions).

Timeline per client (2 years):
  Phase 1  Jan–Jun 2024  (6 mo)  Normal diverse spending
  Phase 2  Jul–Sep 2024  (3 mo)  >75% spend in trigger_mcc → financial stress
                                  → this client had a PAST loan (Oct 2024)
  Phase 3  Oct 2024–Sep 2025 (12 mo)  Normal + repaying credit
  Phase 4  Oct–Nov 2025  (2 mo)  PREDICTION TARGET
                                  label=1 (~50%): spike in target_mcc
                                                  + elevated balance checks
                                                  + larger amounts
                                  label=0 (~50%): normal spending, no spike

Key change vs previous version
--------------------------------
Previously ALL clients had a strong phase 4 spike (78% prob on target_mcc),
making the label trivially 1 for everyone. Now only ~50% of clients show
the spike pattern — the rest have normal phase 4 behaviour. This creates a
genuinely balanced binary classification target so the model has something
to learn.

The two groups also differ subtly in phase 3 behaviour (the observation
window), planting learnable signal:
  Will-seek-loan clients (label=1):
    · Slightly higher balance-check frequency in phase 3 (building anxiety)
    · Slightly higher spend multiplier on their target_mcc in phase 3
    · Slightly more failed transactions near end of phase 3 (balance pressure)
  Won't-seek-loan clients (label=0):
    · Fully normal phase 3 behaviour

This means the model can learn to distinguish them from phases 1-3 alone,
which is the correct framing: predict intent BEFORE phase 4 happens.

Reads  : ./data/clients.csv
Outputs: ./data/transactions.csv
         ./data/application.csv
         ./data/client_mcc_assignments.csv  (includes will_seek_loan column)
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, date, timedelta

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR     = "../data"
IN_CLIENTS   = f"{DATA_DIR}/clients.csv"
OUT_TX       = f"{DATA_DIR}/transactions.csv"
OUT_APP      = f"{DATA_DIR}/application.csv"

PHASE1_START = date(2024,  1,  1)
PHASE1_END   = date(2024,  6, 30)
PHASE2_START = date(2024,  7,  1)
PHASE2_END   = date(2024,  9, 30)
PHASE3_START = date(2024, 10,  1)
PHASE3_END   = date(2025,  9, 30)
PHASE4_START = date(2025, 10,  1)
PHASE4_END   = date(2025, 11, 30)

TX_PER_MONTH = 10
MIN_BALANCE  = 10_000

# Fraction of clients who will seek a loan in phase 4
# Remaining (1 - LOAN_SEEKER_RATE) will have normal phase 4 behaviour
LOAN_SEEKER_RATE = 0.50

# ── MCC categories ────────────────────────────────────────────────────────────
TRIGGER_MCCS = [5211, 1021, 5680, 3001]

MCC_INFO = {
    5211: ("Repair / Hardware",  35_000, 15_000),
    1021: ("Electronics",        40_000, 20_000),
    5680: ("Clothing",           25_000, 10_000),
    3001: ("Travel",             50_000, 25_000),
    5411: ("Supermarket",        12_000,  4_000),
    5812: ("Restaurant",          8_000,  3_000),
    5912: ("Pharmacy",            7_000,  2_500),
    6011: ("ATM / Cash",         20_000,  8_000),
}

BACKGROUND_MCCS = [5411, 5812, 5912, 6011]

MERCHANTS = {
    5211: ["Fix Price Build", "ArmBuild", "StroiMag", "HomePro", "MasterBuild"],
    1021: ["iSpace", "TechZone", "Samsung Store", "MediaMarkt", "DigiTech"],
    5680: ["Zara Armenia", "H&M Yerevan", "LC Waikiki", "Mango", "Reserved"],
    3001: ["ArmAvia", "FlyDubai", "Booking.com", "S7 Airlines", "Hotels.am"],
    5411: ["SAS Supermarket", "Yerevan City", "Unimart", "Rossia", "Carrefour"],
    5812: ["Lavash Cafe", "Pizza di Roma", "Dolmama", "Mer Taghe", "Green Bean"],
    5912: ["ArmPharm", "Nor Pharma", "Vita", "Zdravitsa", "PharmAm"],
    6011: ["Ameriabank ATM", "Ardshinbank ATM", "Inecobank ATM", "ACBA ATM"],
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def random_datetime(start: date, end: date) -> datetime:
    delta_days = (end - start).days
    d = start + timedelta(days=random.randint(0, delta_days))
    return datetime(d.year, d.month, d.day,
                    random.randint(7, 22), random.randint(0, 59))


def round100(x: float) -> int:
    return max(100, round(x / 100) * 100)


def spend_amount(mcc: int, multiplier: float = 1.0) -> int:
    mean, std = MCC_INFO[mcc][1], MCC_INFO[mcc][2]
    raw = np.random.normal(mean * multiplier, std)
    return round100(max(500, raw))


def make_phase_dates(start: date, end: date) -> list:
    """Spread ~TX_PER_MONTH random datetimes across each month in [start, end]."""
    import calendar as _cal
    dates   = []
    current = start
    while current <= end:
        month_end = date(current.year, current.month,
                         _cal.monthrange(current.year, current.month)[1])
        month_end = min(month_end, end)
        n = max(1, int(np.random.normal(TX_PER_MONTH, 2)))
        for _ in range(n):
            dates.append(random_datetime(current, month_end))
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return sorted(dates)


def choose_mcc_normal() -> int:
    """Normal spending: mostly background with occasional JTBD."""
    pool = BACKGROUND_MCCS * 4 + TRIGGER_MCCS
    return random.choice(pool)


def choose_mcc_spike(trigger_mcc: int) -> int:
    """Strong spike: ~78% on trigger_mcc."""
    if random.random() < 0.78:
        return trigger_mcc
    return random.choice(BACKGROUND_MCCS)


def choose_mcc_weak_signal(target_mcc: int) -> int:
    """
    Weak pre-loan signal for will-seek-loan clients during phase 3.
    Slightly elevated chance on target_mcc vs fully normal spending.
    Not a spike — just a gentle lean toward the category.
    """
    pool = BACKGROUND_MCCS * 4 + TRIGGER_MCCS + [target_mcc]  # slight overweight
    return random.choice(pool)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_transactions_for_client(
    client: pd.Series,
    trigger_mcc: int,
    target_mcc: int,
    will_seek_loan: bool,
    tx_id_start: int,
) -> tuple[list[dict], list[dict]]:
    """
    Generate all transactions + app sessions for one client across all 4 phases.

    will_seek_loan=True  → phase 4 has a strong spike in target_mcc
                           phase 3 has subtle early signals (slightly more
                           balance checks, slightly higher target_mcc spend)
    will_seek_loan=False → phase 4 is normal spending, no spike at all
                           phase 3 is fully normal

    Returns (tx_records, app_records).
    """
    salary    = float(client["monthly_salary"])
    client_id = int(client["client_id"])
    bank_acct = int(client["bank_account"])

    balance         = round100(salary * random.uniform(2, 5))
    tx_records      = []
    app_records     = []
    tx_id           = tx_id_start
    sess_id         = client_id * 100_000
    credited_months = set()

    def credit_salary(dt: datetime):
        nonlocal balance
        key = (dt.year, dt.month)
        if key not in credited_months:
            balance += round100(salary + random.uniform(-20_000, 20_000))
            credited_months.add(key)

    def make_tx(dt: datetime, mcc: int, multiplier: float = 1.0,
                extra_fail_prob: float = 0.0) -> dict:
        nonlocal balance, tx_id
        credit_salary(dt)
        amount = spend_amount(mcc, multiplier)

        insufficient = (balance - amount) < MIN_BALANCE
        random_fail  = random.random() < (0.04 + extra_fail_prob)

        if insufficient or random_fail:
            status = "FAILED"
            resp   = 51 if insufficient else 57
        else:
            status = "SUCCESS"
            resp   = 0
            balance -= amount

        noise    = int(balance * random.uniform(-0.03, 0.03))
        disp_bal = max(MIN_BALANCE, balance + noise)

        rec = {
            "transaction_id":   tx_id,
            "client_id":        client_id,
            "transaction_date": dt.strftime("%Y-%m-%d %H:%M"),
            "amount":           amount,
            "mcc_code":         mcc,
            "category":         MCC_INFO[mcc][0],
            "merchant_name":    random.choice(MERCHANTS[mcc]),
            "bank_account":     bank_acct,
            "status":           status,
            "response_code":    resp,
            "balance":          disp_bal,
        }
        tx_id += 1
        return rec

    def make_app_session(dt: datetime, action: str,
                         duration_range: tuple = (10, 300)) -> dict:
        nonlocal sess_id
        rec = {
            "session_id":   sess_id,
            "client_id":    client_id,
            "timestamp":    dt.strftime("%Y-%m-%d %H:%M"),
            "duration_sec": random.randint(*duration_range),
            "action":       action,
            "device_os":    random.choice(["iOS", "Android"]),
        }
        sess_id += 1
        return rec

    def add_balance_checks(dt: datetime, n_min: int, n_max: int):
        for _ in range(random.randint(n_min, n_max)):
            chk = datetime(dt.year, dt.month, dt.day,
                           random.randint(8, 22), random.randint(0, 59))
            app_records.append(make_app_session(chk, "check_balance", (10, 60)))

    # ── Phase 1: Normal (Jan–Jun 2024) ───────────────────────────────────
    for dt in make_phase_dates(PHASE1_START, PHASE1_END):
        mcc = choose_mcc_normal()
        tx_records.append(make_tx(dt, mcc))
        pre = dt - timedelta(minutes=random.randint(1, 5))
        app_records.append(make_app_session(pre, "payment_authorization", (30, 180)))
        if random.random() < 0.30:
            add_balance_checks(dt, 1, 1)

    # ── Phase 2: Spike in trigger_mcc (Jul–Sep 2024) ─────────────────────
    for dt in make_phase_dates(PHASE2_START, PHASE2_END):
        mcc  = choose_mcc_spike(trigger_mcc)
        mult = random.uniform(1.5, 3.0) if mcc == trigger_mcc else 1.0
        tx_records.append(make_tx(dt, mcc, mult))
        pre = dt - timedelta(minutes=random.randint(1, 5))
        app_records.append(make_app_session(pre, "payment_authorization", (30, 180)))
        add_balance_checks(dt, 2, 4)  # financial anxiety

    # ── Phase 3: Normal + repaying credit (Oct 2024–Sep 2025) ────────────
    # Split into early phase 3 (calm) and late phase 3 (building signal
    # for will-seek-loan clients)
    phase3_dates = make_phase_dates(PHASE3_START, PHASE3_END)
    phase3_midpoint = date(2025, 4, 1)  # first 6 months calm, last 6 building

    for dt in phase3_dates:
        is_late_p3    = dt.date() >= phase3_midpoint
        is_loan_seeker = will_seek_loan and is_late_p3

        if is_loan_seeker:
            # Subtle pre-loan signals building in late phase 3:
            # slight lean toward target_mcc, more balance checks,
            # slightly elevated fail rate (balance under pressure)
            mcc          = choose_mcc_weak_signal(target_mcc)
            mult         = random.uniform(1.1, 1.4) if mcc == target_mcc else 1.0
            extra_fail   = 0.04  # additional 4% fail chance on top of base
            check_prob   = 0.45  # more frequent balance checks (vs 0.25 normal)
        else:
            mcc          = choose_mcc_normal()
            mult         = 1.0
            extra_fail   = 0.0
            check_prob   = 0.25

        tx_records.append(make_tx(dt, mcc, mult, extra_fail))
        pre = dt - timedelta(minutes=random.randint(1, 5))
        app_records.append(make_app_session(pre, "payment_authorization", (30, 180)))
        if random.random() < check_prob:
            add_balance_checks(dt, 1, 1)

    # ── Phase 4: Target (Oct–Nov 2025) — THE PREDICTION TARGET ───────────
    for dt in make_phase_dates(PHASE4_START, PHASE4_END):
        if will_seek_loan:
            # Strong spike: high concentration + large amounts + high anxiety
            mcc  = choose_mcc_spike(target_mcc)
            mult = random.uniform(1.5, 3.0) if mcc == target_mcc else 1.0
            tx_records.append(make_tx(dt, mcc, mult))
            pre = dt - timedelta(minutes=random.randint(1, 5))
            app_records.append(make_app_session(pre, "payment_authorization", (30, 180)))
            add_balance_checks(dt, 2, 4)  # peak anxiety
        else:
            # Fully normal spending — no loan intent signal
            mcc = choose_mcc_normal()
            tx_records.append(make_tx(dt, mcc))
            pre = dt - timedelta(minutes=random.randint(1, 5))
            app_records.append(make_app_session(pre, "payment_authorization", (30, 180)))
            if random.random() < 0.25:
                add_balance_checks(dt, 1, 1)

    return tx_records, app_records


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    clients_df = pd.read_csv(IN_CLIENTS)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 55)
    print("  transaction_generate.py — building timeline")
    print("=" * 55)
    print(f"  Clients loaded     : {len(clients_df)}")
    print(f"  Timeline           : {PHASE1_START} -> {PHASE4_END}")
    print(f"  Loan seekers (~{LOAN_SEEKER_RATE:.0%}): label=1 in phase 4")
    print(f"  Non-seekers  (~{1-LOAN_SEEKER_RATE:.0%}): label=0 in phase 4")
    print()

    all_tx          = []
    all_app         = []
    mcc_assignments = []
    tx_id           = 100_001

    # Pre-assign which clients will seek a loan in phase 4
    # Using deterministic assignment (not random per-client) to guarantee balance
    n_clients    = len(clients_df)
    n_seekers    = round(n_clients * LOAN_SEEKER_RATE)
    seeker_flags = [True] * n_seekers + [False] * (n_clients - n_seekers)
    random.shuffle(seeker_flags)

    for (_, client), will_seek_loan in zip(clients_df.iterrows(), seeker_flags):
        trigger_mcc = random.choice(TRIGGER_MCCS)
        remaining   = [m for m in TRIGGER_MCCS if m != trigger_mcc]
        target_mcc  = random.choice(remaining)

        mcc_assignments.append({
            "client_id":      int(client["client_id"]),
            "trigger_mcc":    trigger_mcc,
            "target_mcc":     target_mcc,
            "will_seek_loan": int(will_seek_loan),  # ground truth label
        })

        tx_records, app_records = generate_transactions_for_client(
            client, trigger_mcc, target_mcc, will_seek_loan, tx_id
        )
        all_tx.extend(tx_records)
        all_app.extend(app_records)
        tx_id += len(tx_records) + 1

    # Save
    df_tx  = pd.DataFrame(all_tx).sort_values(["client_id", "transaction_date"])
    df_app = pd.DataFrame(all_app).sort_values(["client_id", "timestamp"])
    df_mcc = pd.DataFrame(mcc_assignments)

    df_tx.to_csv(OUT_TX,  index=False)
    df_app.to_csv(OUT_APP, index=False)
    df_mcc.to_csv(f"{DATA_DIR}/client_mcc_assignments.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    import calendar as _cal
    df_tx["phase"] = pd.cut(
        pd.to_datetime(df_tx["transaction_date"]).dt.date.apply(lambda d: d.toordinal()),
        bins=[date(2023, 12, 31).toordinal(),
              PHASE1_END.toordinal(),
              PHASE2_END.toordinal(),
              PHASE3_END.toordinal(),
              PHASE4_END.toordinal() + 1],
        labels=["Phase1_Normal", "Phase2_Spike", "Phase3_Repaying", "Phase4_Target"]
    )

    print(f"  Transactions total : {len(df_tx):,}")
    print(f"  App sessions total : {len(df_app):,}")
    print()
    print("  Transactions by phase:")
    for phase, grp in df_tx.groupby("phase", observed=True):
        success_pct = (grp["status"] == "SUCCESS").mean() * 100
        print(f"    {str(phase):<24} {len(grp):>6,} tx  ({success_pct:.0f}% success)")

    print()
    print("  Label distribution in client_mcc_assignments:")
    label_counts = df_mcc["will_seek_loan"].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"    will_seek_loan={label}  {count:>4} clients  "
              f"({count/len(df_mcc)*100:.0f}%)")

    print()
    print("  Phase 2 spike verification (trigger_mcc share, should be >75%):")
    p2      = df_tx[df_tx["phase"] == "Phase2_Spike"]
    merged  = p2.merge(df_mcc[["client_id", "trigger_mcc"]], on="client_id")
    trigger_share = (
        merged.groupby("client_id").apply(
            lambda g: (g[g["mcc_code"] == g["trigger_mcc"].iloc[0]]["amount"].sum()
                       / g["amount"].sum()),
            include_groups=False
        ).mean()
    )
    print(f"    avg trigger_mcc share in Phase 2: {trigger_share:.1%}")

    print()
    print("  Phase 4 spike verification (target_mcc share by label):")
    p4     = df_tx[df_tx["phase"] == "Phase4_Target"]
    p4m    = p4.merge(df_mcc[["client_id", "target_mcc", "will_seek_loan"]], on="client_id")
    for label in [1, 0]:
        grp   = p4m[p4m["will_seek_loan"] == label]
        share = (
            grp.groupby("client_id").apply(
                lambda g: (g[g["mcc_code"] == g["target_mcc"].iloc[0]]["amount"].sum()
                           / g["amount"].sum()),
                include_groups=False
            ).mean()
        )
        print(f"    will_seek_loan={label}: avg target_mcc share = {share:.1%}")

    print()
    print(f"  Saved -> {OUT_TX}")
    print(f"  Saved -> {OUT_APP}")
    print(f"  Saved -> {DATA_DIR}/client_mcc_assignments.csv")
    print("=" * 55)

    return df_tx, df_app, df_mcc


if __name__ == "__main__":
    main()