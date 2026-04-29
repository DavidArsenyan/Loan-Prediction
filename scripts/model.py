"""
model.py
========
Loan propensity model: predicts whether a client will take a loan
in the next 30 days, based on behavioural and financial features.

Model   : LightGBM (primary)
Input   : ../data/features.csv  (output of feature_engineering.py)
Output  : ../data/model_scores.csv   — predicted probability per client
          ../data/model.pkl          — saved model + SHAP artifact
          ../data/feature_importance.csv

Feature selection rationale
----------------------------
From the 75+ available features we keep ~30 that carry the most
signal and the least redundancy, grouped as:

  Balance dynamics    balance_trend, balance_drop_pct, balance_change_3m
  Burn rate           burn_acceleration, burn_rate_ratio, monthly_spend_recent
  Volatility          cv_balance, monthly_spend_std, failed_tx_rate_recent
  Category spikes     spike_{repair,electronics,clothing,travel},
                      max_single_trigger_tx, dominant_trigger_mcc
  Zero-day            zero_day_rate, consecutive_zero_max, zero_day_count_recent
  Urgency             urgency_score, balance_check_spike, high_amount_tx_share
  Credit history      credit_score, payments_on_time_rate, payments_missed_rate,
                      avg_days_late, prior_credit_closed, credit_utilization
  App behaviour       check_ratio_last_3m, sessions_last_3m
  Client profile      monthly_salary, age, is_employed, account_age_years

Excluded intentionally:
  Raw totals (spend_*_total) — replaced by spike ratios and shares
  Redundant counts (payments_on_time, payments_late as raw numbers)
    → rates are scale-invariant and more informative
  balance_mean/min/max — summarised by balance_trend + balance_drop_pct
  Sessions totals — replaced by normalised ratios
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report,
)
import lightgbm as lgb
import shap

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR        = "../data"
FEATURES_PATH   = f"{DATA_DIR}/features.csv"
OUT_SCORES      = f"{DATA_DIR}/model_scores.csv"
OUT_MODEL       = f"{DATA_DIR}/model.pkl"
OUT_IMPORTANCE  = f"{DATA_DIR}/feature_importance.csv"

TARGET_COL      = "will_seek_loan"
ID_COL          = "client_id"

RANDOM_STATE    = 42
N_FOLDS         = 5


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = [

    # ── Balance dynamics ─────────────────────────────────────────────────
    "balance_trend",
    "balance_drop_pct",
    "balance_change_3m",

    # ── Burn rate ─────────────────────────────────────────────────────────
    "burn_acceleration",
    "burn_rate_ratio",
    "monthly_spend_recent",

    # ── Volatility ────────────────────────────────────────────────────────
    "cv_balance",
    "monthly_spend_std",
    "failed_tx_rate",
    "failed_tx_rate_recent",

    # ── Category spikes ───────────────────────────────────────────────────
    "spike_repair",
    "spike_electronics",
    "spike_clothing",
    "spike_travel",
    "max_single_trigger_tx",
    "dominant_trigger_mcc",

    # ── Zero-day count ────────────────────────────────────────────────────
    "zero_day_rate",
    "consecutive_zero_max",
    "zero_day_count_recent",

    # ── Urgency score ──────────────────────────────────────────────────────
    "urgency_score",
    "balance_check_spike",
    "high_amount_tx_share",
    "failed_spike",

    # ── Credit history ────────────────────────────────────────────────────
    "credit_score",
    "payments_on_time_rate",
    "payments_missed_rate",
    "avg_days_late",
    "prior_credit_closed",
    "credit_utilization",

    # ── App behaviour ─────────────────────────────────────────────────────
    "check_ratio_last_3m",
    "sessions_last_3m",

    # ── Client profile ────────────────────────────────────────────────────
    "monthly_salary",
    "age",
    "is_employed",
    "account_age_years",
]

# Categorical features (LightGBM handles natively)
CAT_FEATURES = ["dominant_trigger_mcc", "is_employed"]


# ─────────────────────────────────────────────────────────────────────────────
# LIGHTGBM PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "boosting_type":    "gbdt",
    "n_estimators":     500,
    "learning_rate":    0.05,
    "num_leaves":       31,           # conservative: dataset is small (~120 clients)
    "max_depth":        5,
    "min_child_samples":5,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "class_weight":     "balanced",   # handles label imbalance automatically
    "random_state":     RANDOM_STATE,
    "verbose":          -1,
    "n_jobs":           -1,
}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD & VALIDATE
# ─────────────────────────────────────────────────────────────────────────────

def load_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = pd.read_csv(FEATURES_PATH)
    print(f"[DATA] Loaded {len(df)} clients, {len(df.columns)} columns")

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"[WARN] Missing features (will be zero-filled): {missing}")
        for col in missing:
            df[col] = 0.0

    X = df[FEATURES].copy()
    y = df[TARGET_COL].copy()

    # Cast categoricals to int for LightGBM
    for col in CAT_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype(int)

    print(f"[DATA] Target balance: {int(y.sum())} positive / {int((y==0).sum())} negative")
    print(f"[DATA] Feature matrix: {X.shape[0]} rows × {X.shape[1]} columns")
    return df, X, y


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-VALIDATED EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_cv(X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, float, float]:
    """Manual CV loop — returns (oof_probs, oof_auc, oof_ap)."""
    print(f"\n[CV] Running {N_FOLDS}-fold stratified cross-validation ...")

    cv        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if y_tr.nunique() < 2:
            print(f"  Fold {fold}: skipped (only one class in train)")
            oof_probs[val_idx] = y_tr.mean()
            continue

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(period=-1)])

        oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, oof_probs[val_idx]) if y_val.nunique() > 1 else float("nan")
        print(f"  Fold {fold}: AUC = {fold_auc:.4f}  "
              f"({int(y_val.sum())} pos / {int((y_val==0).sum())} neg in val)")

    oof_auc, oof_ap = 0.0, 0.0
    if y.nunique() > 1:
        oof_auc = roc_auc_score(y, oof_probs)
        oof_ap  = average_precision_score(y, oof_probs)
        print(f"\n  OOF ROC-AUC  : {oof_auc:.4f}")
        print(f"  OOF Avg Prec : {oof_ap:.4f}")
    else:
        print("\n  [WARN] Only one class in target — check build_target()")

    preds = (oof_probs >= 0.5).astype(int)
    print("\n[CV] Classification report (threshold = 0.50):")
    print(classification_report(y, preds,
                                 target_names=["No Loan", "Will Take Loan"], digits=3))
    return oof_probs, oof_auc, oof_ap


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN FINAL MODEL ON ALL DATA
# ─────────────────────────────────────────────────────────────────────────────

def train_final(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
    """Train on the full dataset to get the final deployable model."""
    print("\n[TRAIN] Training final model on full dataset ...")
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X, y)
    print("  Done.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap(model: lgb.LGBMClassifier, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Compute SHAP values for all clients using TreeExplainer.

    Returns
    -------
    shap_values   : np.ndarray, shape (n_clients, n_features)
                    SHAP contribution of each feature to the log-odds of loan.
    expected_value: float
                    Base log-odds (model prior). sigmoid(expected_value) = base probability.
    """
    print("\n[SHAP] Computing SHAP values via TreeExplainer ...")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)

    # LightGBM binary returns either a single array or [neg_class, pos_class]
    if isinstance(sv, list):
        sv = sv[1]   # positive class (loan=1)

    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = float(ev[1]) if len(ev) > 1 else float(ev[0])
    else:
        ev = float(ev)

    base_prob = 1.0 / (1.0 + np.exp(-ev))
    print(f"  SHAP matrix : {sv.shape}")
    print(f"  Base rate   : {base_prob:.4f}  (expected_value={ev:.4f})")
    return sv, ev


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def show_feature_importance(model: lgb.LGBMClassifier, feature_names: list) -> pd.DataFrame:
    importance = pd.DataFrame({
        "feature":    feature_names,
        "importance": model.booster_.feature_importance(importance_type="gain"),
        "split":      model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance", ascending=False)

    print("\n[IMPORTANCE] Top 20 features by gain:")
    print(f"  {'Feature':<35} {'Gain':>10}  {'Splits':>8}")
    print(f"  {'-'*35} {'-'*10}  {'-'*8}")
    for _, row in importance.head(20).iterrows():
        bar = "█" * min(int(row["importance"] / (importance["importance"].max() or 1) * 20), 20)
        print(f"  {row['feature']:<35} {row['importance']:>10.1f}  {int(row['split']):>8}  {bar}")

    return importance


# ─────────────────────────────────────────────────────────────────────────────
# SCORING OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def save_scores(df: pd.DataFrame, oof_probs: np.ndarray,
                model: lgb.LGBMClassifier, X: pd.DataFrame) -> None:
    """
    Save per-client scores. Column names match what crm_app2.py expects:
      oof_probability   — OOF probability (unbiased cross-validation estimate)
      final_probability — probability from the final model (production)
      predicted_label   — binary prediction at 0.5 threshold
      risk_tier         — human-readable tier based on final_probability
      outcome           — TP / TN / FP / FN
      correct           — bool: prediction matches actual
    """
    final_probs = model.predict_proba(X)[:, 1]

    scores = pd.DataFrame({
        "client_id":         df[ID_COL].values,
        "actual_label":      df[TARGET_COL].values,
        "oof_probability":   oof_probs.round(4),
        "final_probability": final_probs.round(4),
        "predicted_label":   (oof_probs >= 0.5).astype(int),
    })

    def _tier(p):
        if p >= 0.75: return "High"
        if p >= 0.50: return "Medium"
        if p >= 0.25: return "Low"
        return "Very Low"

    # scores["risk_tier"] = scores["final_probability"].apply(_tier)
    scores["risk_tier"] = scores["oof_probability"].apply(_tier)

    def _outcome(row):
        p, a = int(row["predicted_label"]), int(row["actual_label"])
        if p == 1 and a == 1: return "TP"
        if p == 0 and a == 0: return "TN"
        if p == 1 and a == 0: return "FP"
        return "FN"

    scores["outcome"] = scores.apply(_outcome, axis=1)
    scores["correct"] = scores["outcome"].isin(["TP", "TN"])

    scores.to_csv(OUT_SCORES, index=False)

    print(f"\n[SCORES] Risk tier distribution:")
    for tier, count in scores["risk_tier"].value_counts().items():
        bar = "█" * count
        print(f"  {tier:<10} {count:>4}  {bar}")

    acc = scores["correct"].mean()
    print(f"\n[SCORES] Threshold-0.5 accuracy: {acc*100:.1f}%")
    for oc, cnt in scores["outcome"].value_counts().items():
        print(f"  {oc}: {cnt}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  model.py — Loan Propensity Model (LightGBM + SHAP)")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────
    df, X, y = load_features()

    # ── 2. Cross-validated evaluation (honest OOF estimate) ───────────────
    oof_probs, oof_auc, oof_ap = evaluate_cv(X, y)

    # ── 3. Train final model on all data ─────────────────────────────────
    model = train_final(X, y)

    # ── 4. Feature importance ─────────────────────────────────────────────
    importance = show_feature_importance(model, FEATURES)
    importance[["feature", "importance"]].to_csv(OUT_IMPORTANCE, index=False)
    import shap

    print("\n[SHAP] Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)  # shape: (120, 35)
    expected_val = float(explainer.expected_value)
    print(f"  SHAP values shape : {shap_values.shape}")
    print(f"  Expected value    : {expected_val:.4f}")
    # ── 5. SHAP values ────────────────────────────────────────────────────
    # shap_values, expected_value = compute_shap(model, X)

    # ── 6. Save scores ────────────────────────────────────────────────────
    save_scores(df, oof_probs, model, X)

    # ── 7. Save model + SHAP artifact ────────────────────────────────────
    artifact = {
        "model":          model,
        "features":       FEATURES,
        "oof_auc":        oof_auc,
        "oof_ap":         oof_ap,
        # SHAP — used by CRM for per-client explanations and analytics charts
        "shap_values":    shap_values,      # np.ndarray (n_clients, n_features)
        "expected_value": expected_val,   # float, log-odds base rate
    }
    # with open(OUT_MODEL, "wb") as f:
    #     pickle.dump({
    #         "model": model,
    #         "features": FEATURES,
    #         "shap_values": shap_values,  # ← add this
    #         "expected_value": expected_val,  # ← add this
    #         "oof_auc": auc,
    #         "oof_ap": ap,
    #     }, f)

    with open(OUT_MODEL, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\n[SAVED] Scores     -> {OUT_SCORES}")
    print(f"[SAVED] Importance -> {OUT_IMPORTANCE}")
    print(f"[SAVED] Model+SHAP -> {OUT_MODEL}")
    print("=" * 60)

    return model, oof_probs, importance, shap_values


if __name__ == "__main__":
    main()