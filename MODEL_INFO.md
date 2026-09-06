# Final model information

**Submitted file:** `submission_blend_te_paytypes.csv`  
**Public log loss:** 0.40900  
**Type:** probability ensemble of two LightGBM families (not a pretrained net, not AutoML)

## Ensemble

| Component | Weight | Models averaged | Features |
|---|---|---|---|
| PAY-types LightGBM (`BEST_0.40929.py`) | **0.50** | 100 (10 seeds × 10 folds) | 75 row-wise columns |
| Nested TE LightGBM (`train_te_and_blend.py`) | **0.50** | 100 (10 seeds × 10 folds) | 75 columns + 4 target-encoded PAY columns |

Seeds: `1000` … `1009`. Each seed also sets `bagging_seed`, `feature_fraction_seed`, and `data_random_seed`.

## Shared LightGBM hyperparameters

| Setting | Value |
|---|---|
| Package | `lightgbm==4.7.0` |
| `objective` | `binary` |
| `metric` | `binary_logloss` |
| `learning_rate` | 0.1 |
| `num_leaves` | 31 |
| `deterministic` | True |
| `force_row_wise` | True |
| `num_boost_round` | 2000 |
| Early stopping | 100 rounds on the scored fold |
| Probability clip | `[1e-6, 1-1e-6]` |

## Target encoding (TE family only)

| Setting | Value |
|---|---|
| Columns | `te_pay0`, `te_pay2`, `te_pat2`, `te_pat3` |
| Groups | `PAY_0`; `PAY_2`; `PAY_0\|PAY_2`; `PAY_0\|PAY_2\|PAY_3` |
| Smoothing `M` | 50 |
| Train-row encoding | inner 5-fold, same outer seed |
| Val/test encoding | table fit on the outer training partition |
| Shrinkage | flat (toward fold prior), not hierarchical |

## Not used in this file

CatBoost, extra_trees, bagging as a different booster, Platt/isotonic calibration, public-leaderboard blend-weight search, manual row edits, external data, pretrained checkpoints.

Saved booster files are not attached: retraining 200 LightGBM models is minutes to tens of minutes, not days. Saved **probability** artefacts for both families are in `artifacts/`.
