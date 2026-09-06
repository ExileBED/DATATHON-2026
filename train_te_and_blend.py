"""Train nested PAY target-encoding LightGBM (10x10) and blend 50/50 with PAY-types.

This is the script that produced the leaderboard file
`submission_blend_te_paytypes.csv` (public log loss 0.40900).

The PAY-types half used the already-trained 0.40929 artefacts in `artifacts/`.
The TE half is trained here. Blend weight is 0.5, selected on local OOF.

    python train_te_and_blend.py

A full re-run writes `outputs/` and may differ by tiny float amounts across
machines. The file at the folder root is the exact leaderboard upload.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
BEST_PY = HERE / "BEST_0.40929.py"
OUT = HERE / "outputs"
ART = HERE / "artifacts"
PAY_TYPES_LOCAL = 0.422449736062935
PAY_TYPES_OOF = ART / "oof_pay_types.csv"
PAY_TYPES_TEST = ART / "test_pay_types.csv"
BLEND_WEIGHT_PAY_TYPES = 0.5

M = 50.0
ID, TARGET = "client_id", "default"
CLIP_EPS = 1e-6
N_REPEAT, N_SPLITS = 10, 10

BASE = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "verbose": -1,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": -1,
}

SPECS = [
    ("te_pay0", ["PAY_0"]),
    ("te_pay2", ["PAY_2"]),
    ("te_pat2", ["PAY_0", "PAY_2"]),
    ("te_pat3", ["PAY_0", "PAY_2", "PAY_3"]),
]


def load_best():
    spec = importlib.util.spec_from_file_location("best_040929", BEST_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["best_040929"] = mod
    spec.loader.exec_module(mod)
    return mod


def clip_proba(p):
    return np.clip(np.asarray(p, dtype=float), CLIP_EPS, 1.0 - CLIP_EPS)


def te_fit(keys, y):
    df = pd.DataFrame({"k": keys, "y": y})
    g = df.groupby("k")["y"].agg(["sum", "count"])
    prior = float(y.mean())
    return ((g["sum"] + M * prior) / (g["count"] + M)).to_dict(), prior


def te_apply(keys, table, prior):
    return pd.Series(keys).map(table).fillna(prior).to_numpy(dtype=float)


def key(df, cols):
    return df[cols].astype(int).astype(str).agg("|".join, axis=1).to_numpy()


def encode(Xa, ya, Xb_list, seed):
    outs_a = {n: np.zeros(len(Xa)) for n, _ in SPECS}
    outs_b = [{n: None for n, _ in SPECS} for _ in Xb_list]
    for name, cols in SPECS:
        ka = key(Xa, cols)
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for i, j in inner.split(Xa, ya):
            t, p = te_fit(ka[i], ya[i])
            outs_a[name][j] = te_apply(ka[j], t, p)
        t, p = te_fit(ka, ya)
        for bi, Xb in enumerate(Xb_list):
            outs_b[bi][name] = te_apply(key(Xb, cols), t, p)
    return outs_a, outs_b


def make_submission(sample, test_ids, proba, path: Path):
    pred = pd.DataFrame({ID: test_ids.values, "default_probability": clip_proba(proba)})
    out = sample[[ID]].merge(pred, on=ID, how="left", validate="one_to_one")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def main() -> None:
    os.environ["PYTHONHASHSEED"] = "42"
    random.seed(42)
    np.random.seed(42)
    OUT.mkdir(parents=True, exist_ok=True)

    best = load_best()
    train = pd.read_csv(best.DATA_DIR / "train.csv")
    test = pd.read_csv(best.DATA_DIR / "test.csv")
    sample = pd.read_csv(best.DATA_DIR / "sample_submission.csv")
    y = train[TARGET].to_numpy()

    Xtr = best.build_features(train).copy()
    Xte = best.build_features(test).copy()
    base_feats = [c for c in Xtr.columns if c not in (ID, TARGET)]
    feats = base_feats + [n for n, _ in SPECS]
    print(f"nested TE on {len(base_feats)} base feats + {len(SPECS)} encodings")

    acc = np.zeros(len(Xte))
    n_models = 0
    rep_oof, rep_ll = [], []
    t0 = time.time()
    for r in range(N_REPEAT):
        s = 1000 + r
        p = dict(BASE)
        p.update(seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=s)
        oof = np.zeros(len(Xtr))
        for a, b in skf.split(Xtr, y):
            A, B = Xtr.iloc[a].copy(), Xtr.iloc[b].copy()
            ea, (eb, et) = encode(A, y[a], [B, Xte], s)
            for nme, _ in SPECS:
                A[nme] = ea[nme]
                B[nme] = eb[nme]
            T = Xte.copy()
            for nme, _ in SPECS:
                T[nme] = et[nme]
            m = lgb.train(
                p,
                lgb.Dataset(A[feats], label=y[a]),
                num_boost_round=2000,
                valid_sets=[lgb.Dataset(B[feats], label=y[b])],
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            bi = int(m.best_iteration)
            oof[b] = m.predict(B[feats], num_iteration=bi)
            acc += m.predict(T[feats], num_iteration=bi)
            n_models += 1
        oof = clip_proba(oof)
        rep_oof.append(oof)
        rep_ll.append(float(log_loss(y, oof)))
        print(
            f"  repeat {r}: OOF={rep_ll[-1]:.6f}  running={np.mean(rep_ll):.6f}  "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    te_oof = clip_proba(np.mean(rep_oof, axis=0))
    te_test = clip_proba(acc / n_models)
    te_avg = float(log_loss(y, te_oof))
    te_mean = float(np.mean(rep_ll))
    print(
        f"\nTE mean-repeat OOF={te_mean:.6f}  averaged OOF={te_avg:.6f}  "
        f"AUC={roc_auc_score(y, te_oof):.5f}"
    )

    pd.DataFrame({ID: train[ID], "oof_default_probability": te_oof}).to_csv(
        OUT / "oof_push4_te.csv", index=False
    )
    make_submission(sample, test[ID], te_test, OUT / "submission_push4_te.csv")

    lgbm_oof = pd.read_csv(PAY_TYPES_OOF)["oof_default_probability"].to_numpy()
    lgbm_test = pd.read_csv(PAY_TYPES_TEST)["default_probability"].to_numpy()
    w = BLEND_WEIGHT_PAY_TYPES
    mix_oof = clip_proba(w * lgbm_oof + (1.0 - w) * te_oof)
    blend_ll = float(log_loss(y, mix_oof))
    blend_test = clip_proba(w * lgbm_test + (1.0 - w) * te_test)
    make_submission(sample, test[ID], blend_test, OUT / "submission_blend_te_paytypes.csv")

    grid = []
    for gw in np.linspace(0, 1, 11):
        ll = float(log_loss(y, clip_proba(gw * lgbm_oof + (1.0 - gw) * te_oof)))
        grid.append({"w_pay_types": float(gw), "averaged_oof_ll": ll})
        print(f"  w_pay_types={gw:.1f}  ll={ll:.6f}")

    summary = {
        "te_mean_repeat_ll": te_mean,
        "te_averaged_oof_ll": te_avg,
        "te_averaged_oof_auc": float(roc_auc_score(y, te_oof)),
        "repeat_ll": rep_ll,
        "pay_types_local_oof": PAY_TYPES_LOCAL,
        "submitted_w_pay_types": w,
        "submitted_blend_oof_ll": blend_ll,
        "blend_grid": grid,
        "note": "Re-run output is in outputs/. Root CSV is the original upload.",
    }
    with open(OUT / "push4_blend_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsubmitted mix: {w:.2f} PAY-types + {1-w:.2f} TE   local OOF={blend_ll:.6f}")
    print(f"wrote {OUT / 'submission_blend_te_paytypes.csv'}")


if __name__ == "__main__":
    main()
