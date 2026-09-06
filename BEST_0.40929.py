"""Public 0.40929 — same LightGBM 10x10 recipe as 0.40967, plus 5 features.

WHAT CHANGED vs previous best (0.40967)
---------------------------------------
ONLY five extra row-wise features. Nothing else.

Unchanged from submission_lgbm_extfeat_teammate_proc.csv (0.40967):
  - LightGBM objective binary / metric binary_logloss
  - learning_rate=0.1, num_leaves=31
  - 10 repeats x 10 stratified folds, seeds 1000..1009
  - early stopping patience 100 on the scored fold, max 2000 rounds
  - average 100 models' test probabilities
  - all previous extended features (pay ratios, utilisation, delinquency,
    bill deltas, pay-to-limit, zero-payment months, severe_delinq_count)

Added (the only delta):
  n_revolve         count of PAY_* ==  0   (revolving)
  n_duly            count of PAY_* == -1   (paid duly)
  n_inactive        count of PAY_* == -2   (no consumption)
  log_limit         log1p(LIMIT_BAL)
  remaining_credit  LIMIT_BAL - BILL_AMT1

Why those PAY counts: codes -2, -1, 0 are not a clean risk ladder
(revolving defaults less than paid-duly). Numeric pay_mean / pay_slope
treat them as ordinal; type counts do not.

NOT used in this file (those public-failed): extra_trees, bagging,
CatBoost blends, heavy regularisation, calibration, inner-ES refit.

    python BEST_0.40929.py

Needs train.csv / test.csv / sample_submission.csv in
inter-uni-datathon-stream-1-credit-card-clients/ next to this script
or in the parent folder.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent


def find_data_dir() -> Path:
    for candidate in (
        ROOT,
        ROOT / "inter-uni-datathon-stream-1-credit-card-clients",
        ROOT.parent / "inter-uni-datathon-stream-1-credit-card-clients",
    ):
        if (candidate / "train.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Place train.csv next to this script, or in "
        "inter-uni-datathon-stream-1-credit-card-clients/"
    )


DATA_DIR = find_data_dir()
OUT_DIR = ROOT / "outputs"
SEED = 42
N_REPEAT = 10
N_SPLITS = 10
EARLY_STOPPING_ROUNDS = 100
MAX_ROUNDS = 2000
CLIP_EPS = 1e-6

ID, TARGET = "client_id", "default"
PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

BASE_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "verbose": -1,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": -1,
}


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def clip_proba(p: np.ndarray) -> np.ndarray:
    return np.clip(p, CLIP_EPS, 1.0 - CLIP_EPS)


def safe_ratio(num, den):
    return (num / den.where(den > 0)).replace([np.inf, -np.inf], np.nan)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """0.40967 features + the five 0.40929 columns at the bottom."""
    X = df.copy()

    ratio_cols = []
    for i in range(1, 6):
        c = f"payratio_{i}"
        X[c] = safe_ratio(X[f"PAY_AMT{i}"], X[f"BILL_AMT{i + 1}"])
        ratio_cols.append(c)
    R = X[ratio_cols]
    X["payratio_mean"] = R.mean(axis=1)
    X["payratio_min"] = R.min(axis=1)
    X["payratio_last"] = X[ratio_cols[0]]
    X["n_months_paid_full"] = (R >= 1.0).sum(axis=1)
    X["n_months_paid_none"] = (R <= 0.01).sum(axis=1)

    for i in range(1, 7):
        X[f"util_{i}"] = X[f"BILL_AMT{i}"] / X["LIMIT_BAL"]
    U = X[[f"util_{i}" for i in range(1, 7)]]
    X["util_mean"] = U.mean(axis=1)
    X["util_max"] = U.max(axis=1)
    X["util_last"] = X["util_1"]
    X["util_trend"] = X["util_1"] - X["util_6"]

    P = X[PAY_STATUS[::-1]].to_numpy(dtype=float)
    m = np.arange(P.shape[1], dtype=float)
    mc = m - m.mean()
    X["pay_slope"] = (P * mc).sum(axis=1) / (mc ** 2).sum()
    X["pay_recent_minus_oldest"] = X["PAY_0"] - X["PAY_6"]
    X["pay_mean"] = P.mean(axis=1)

    D = np.clip(P, 0, None)
    X["delinq_max"] = D.max(axis=1)
    X["delinq_sum"] = D.sum(axis=1)
    X["delinq_count"] = (P > 0).sum(axis=1)
    newest_first = X[PAY_STATUS].to_numpy(dtype=float) > 0
    X["delinq_streak_now"] = newest_first.cumprod(axis=1).sum(axis=1)

    for i in range(1, 6):
        X[f"bill_delta_{i}"] = X[f"BILL_AMT{i}"] - X[f"BILL_AMT{i + 1}"]
    Dl = X[[f"bill_delta_{i}" for i in range(1, 6)]]
    X["bill_delta_mean"] = Dl.mean(axis=1)
    X["bill_delta_last"] = X["bill_delta_1"]
    X["bill_total_change"] = X["BILL_AMT1"] - X["BILL_AMT6"]

    pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    for i in range(1, 7):
        X[f"pay_to_limit_{i}"] = X[f"PAY_AMT{i}"] / X["LIMIT_BAL"]
    PL = X[[f"pay_to_limit_{i}" for i in range(1, 7)]]
    X["pay_to_limit_mean"] = PL.mean(axis=1)
    X["pay_to_limit_last"] = X["pay_to_limit_1"]
    X["zero_payment_months"] = (X[pay_amt_cols] == 0).sum(axis=1)
    X["pay_amt_recent_vs_older"] = X["PAY_AMT1"] - X[["PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]].mean(
        axis=1
    )
    X["pay_amt_trend"] = X["PAY_AMT1"] - X["PAY_AMT6"]
    X["severe_delinq_count"] = (X[PAY_STATUS] >= 2).sum(axis=1)

    # --- ONLY change vs 0.40967 ---
    X["n_revolve"] = (X[PAY_STATUS] == 0).sum(axis=1)
    X["n_duly"] = (X[PAY_STATUS] == -1).sum(axis=1)
    X["n_inactive"] = (X[PAY_STATUS] == -2).sum(axis=1)
    X["log_limit"] = np.log1p(X["LIMIT_BAL"].astype(float))
    X["remaining_credit"] = X["LIMIT_BAL"] - X["BILL_AMT1"]
    return X


def validate_and_write(out: pd.DataFrame, sample: pd.DataFrame, path: Path) -> None:
    assert len(out) == 6000
    assert list(out.columns) == [ID, "default_probability"]
    assert out[ID].equals(sample[ID])
    assert out.notna().all().all()
    assert np.isfinite(out["default_probability"]).all()
    assert ((out["default_probability"] > 0) & (out["default_probability"] < 1)).all()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def main() -> None:
    set_seed()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    train_fe, test_fe = build_features(train), build_features(test)
    feats = [c for c in train_fe.columns if c not in (ID, TARGET)]
    X = train_fe[feats]
    y = train_fe[TARGET].to_numpy()
    X_test = test_fe[feats]
    print(f"{len(feats)} features (70 from 0.40967 + 5 new), NaNs={int(X.isna().sum().sum()):,}")

    test_acc = np.zeros(len(X_test))
    n_models = 0
    rep_oof, rep_ll = [], []

    for r in range(N_REPEAT):
        seed = 1000 + r
        params = dict(BASE_PARAMS)
        params.update(
            seed=seed,
            bagging_seed=seed,
            feature_fraction_seed=seed,
            data_random_seed=seed,
        )
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        oof = np.zeros(len(X))

        for tr_idx, va_idx in skf.split(X, y):
            model = lgb.train(
                params,
                lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx]),
                num_boost_round=MAX_ROUNDS,
                valid_sets=[lgb.Dataset(X.iloc[va_idx], label=y[va_idx])],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            bi = int(model.best_iteration)
            oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=bi)
            test_acc += model.predict(X_test, num_iteration=bi)
            n_models += 1

        oof = clip_proba(oof)
        rep_oof.append(oof)
        rep_ll.append(float(log_loss(y, oof)))
        print(
            f"  repeat {r}: OOF={rep_ll[-1]:.5f}  (running mean {np.mean(rep_ll):.5f})",
            flush=True,
        )

    test_pred = clip_proba(test_acc / n_models)
    oof_avg = clip_proba(np.mean(rep_oof, axis=0))
    print(f"\nmodels averaged                 : {n_models}")
    print(f"mean repeat OOF                 : {np.mean(rep_ll):.5f} (sd {np.std(rep_ll):.5f})")
    print(f"OOF of averaged predictor       : {log_loss(y, oof_avg):.5f}")
    print(f"OOF AUC                         : {roc_auc_score(y, oof_avg):.5f}")
    print("OOF is optimistic (early stopping uses the scored fold).")

    pred = pd.DataFrame({ID: test[ID].values, "default_probability": test_pred})
    out = sample[[ID]].merge(pred, on=ID, how="left", validate="one_to_one")
    sub_path = OUT_DIR / "submission_lgbm_pay_types_0.40929.csv"
    validate_and_write(out, sample, sub_path)
    print(f"\nwrote {sub_path}")


if __name__ == "__main__":
    main()
