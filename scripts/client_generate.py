"""
client_info_generate.py
=======================
Generates clients.csv — the master client table.
All other scripts read from this file.

Output: ./data/clients.csv
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import date, timedelta

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR  = "../data"
OUTPUT_PATH = f"{OUTPUT_DIR}/clients.csv"
N_CLIENTS   = 120

# Armenian first names split by gender
MALE_NAMES   = ["Davit", "Armen", "Hayk", "Tigran", "Gor", "Vardan", "Artur",
                "Samvel", "Vahan", "Narek", "Suren", "Gegham", "Raffi", "Erik",
                "Levon", "Ashot", "Ruben", "Vahagn", "Grigor", "Edgar"]

FEMALE_NAMES = ["Anna", "Maria", "Nare", "Lena", "Ani", "Sona", "Lilit",
                "Karine", "Meri", "Tatevik", "Lusine", "Mariam", "Gohar",
                "Anahit", "Kristine", "Narine", "Suzanna", "Marine", "Armine", "Gayane"]

SURNAMES     = ["Hakobyan", "Petrosyan", "Sargsyan", "Grigoryan", "Mkrtchyan",
                "Hovhannisyan", "Gevorgyan", "Karapetyan", "Abrahamyan",
                "Stepanyan", "Avagyan", "Manukyan", "Danielyan", "Asatryan",
                "Galstyan", "Martirosyan", "Simonyan", "Poghosyan", "Arsenyan",
                "Khachatryan", "Muradyan", "Ghazaryan", "Baghdasaryan"]

CITIES       = ["Yerevan", "Gyumri", "Vanadzor", "Vagharshapat", "Abovyan",
                "Hrazdan", "Kapan", "Gavar", "Sevan", "Dilijan"]

EMPLOYMENT   = ["EMPLOYED", "SELF_EMPLOYED", "STUDENT", "RETIRED"]
# Weights: most clients are employed
EMP_WEIGHTS  = [0.65, 0.20, 0.08, 0.07]

# Monthly salary ranges by employment type (AMD)
SALARY_RANGES = {
    "EMPLOYED":      (200_000,  800_000),
    "SELF_EMPLOYED": (150_000,  600_000),
    "STUDENT":       (50_000,   150_000),
    "RETIRED":       (80_000,   200_000),
}

MARITAL      = ["SINGLE", "MARRIED", "DIVORCED", "WIDOWED"]
MAR_WEIGHTS  = [0.35, 0.50, 0.10, 0.05]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def rand_date(start: date, end: date) -> date:
    return start + timedelta(days=random.randint(0, (end - start).days))

def generate_phone() -> str:
    return f"+374 {random.randint(10,99)} {random.randint(100000,999999)}"

def generate_email(first: str, last: str, uid: int) -> str:
    domains = ["gmail.com", "mail.ru", "yahoo.com", "yandex.ru"]
    return f"{first.lower()}.{last.lower()}{uid}@{random.choice(domains)}"

def generate_bank_account() -> int:
    return random.randint(10_000_000, 99_999_999)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    records = []

    for i in range(N_CLIENTS):
        client_id = i + 1
        gender    = random.choice(["M", "F"])

        first_name = random.choice(MALE_NAMES if gender == "M" else FEMALE_NAMES)
        last_name  = random.choice(SURNAMES)

        # Age: 22–65 for employed/self-employed, 18–25 for students, 60–75 for retired
        employment = random.choices(EMPLOYMENT, weights=EMP_WEIGHTS)[0]
        if employment == "STUDENT":
            age = random.randint(18, 25)
        elif employment == "RETIRED":
            age = random.randint(60, 75)
        else:
            age = random.randint(22, 62)

        birth_year  = date.today().year - age
        birth_date  = rand_date(date(birth_year, 1, 1), date(birth_year, 12, 31))

        # Salary: round to nearest 10,000
        lo, hi  = SALARY_RANGES[employment]
        salary  = round(random.randint(lo, hi) / 10_000) * 10_000

        # Account opened 1–6 years before the data starts (Jan 2024)
        acct_start      = date(2018, 1, 1)
        acct_end        = date(2023, 12, 31)
        account_open    = rand_date(acct_start, acct_end)

        # Number of dependants (affects DTI context)
        if employment == "STUDENT":
            dependants = 0
        elif employment == "RETIRED":
            dependants = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2])[0]
        else:
            dependants = random.choices([0, 1, 2, 3], weights=[0.3, 0.35, 0.25, 0.1])[0]

        records.append({
            "client_id":        client_id,
            "first_name":       first_name,
            "last_name":        last_name,
            "gender":           gender,
            "birth_date":       birth_date.isoformat(),
            "age":              age,
            "city":             random.choice(CITIES),
            "marital_status":   random.choices(MARITAL, weights=MAR_WEIGHTS)[0],
            "dependants":       dependants,
            "employment_type":  employment,
            "monthly_salary":   salary,
            "bank_account":     generate_bank_account(),
            "account_open_date":account_open.isoformat(),
            "phone":            generate_phone(),
            "email":            generate_email(first_name, last_name, client_id),
        })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 55)
    print("  client_info_generate.py — done")
    print("=" * 55)
    print(f"  Clients generated  : {len(df)}")
    print(f"  Employment split   :")
    for emp, cnt in df["employment_type"].value_counts().items():
        print(f"    {emp:<18} {cnt:>4} ({cnt/len(df)*100:.0f}%)")
    print(f"  Avg monthly salary : {df['monthly_salary'].mean():,.0f} AMD")
    print(f"  Salary range       : {df['monthly_salary'].min():,.0f} – {df['monthly_salary'].max():,.0f} AMD")
    print(f"  Saved → {OUTPUT_PATH}")
    print("=" * 55)

    return df


if __name__ == "__main__":
    main()