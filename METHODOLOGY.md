# Methodology report — credit-card default probabilities

**Task:** estimate `P(default = 1)` for 6,000 hidden-test customers.  
**Metric:** binary log loss (lower is better).  
**File under review:** `submission_blend_te_paytypes.csv`  
**Public score:** 0.40900  
**Code:** `BEST_0.40929.py` (75 features + PAY-types 10×10) and `train_te_and_blend.py` (nested target encoding 10×10 + 50/50 blend)

## 1. Approach

The submitted prediction is an **unweighted average of two LightGBM probability vectors**:

1. **PAY-types model (public 0.40929).** Seed- and split-averaged LightGBM on 75 row-wise features. No target encoding.
2. **Nested TE model.** Same 75 features, plus four fold-wise target-encoded repayment columns. Same LightGBM hyperparameters and 10×10 averaging.

Final test probability:

```
0.5 * P_PAY-types + 0.5 * P_nested-TE
```

clipped to `(1e-6, 1-1e-6)`.

This continues the team’s public records:

| Public log loss | What it was |
|---|---|
| 0.40981 | Teammate LightGBM 10×10, original engineered features |
| 0.40967 | Same loop + extra utilisation / pay-to-limit / delinquency features |
| 0.40929 | Same loop + five PAY-type / limit columns |
| **0.40900** | **0.40929 model averaged 50/50 with nested PAY target encoding (100 models each)** |

Log loss penalises confident mistakes, so the pipeline predicts probabilities, averages many similar models, and avoids class weighting / SMOTE.

## 2. Data cleaning and preprocessing

Training: 24,000 labelled rows. Test: 6,000 rows. No missing values. No `client_id` overlap.

We **did not** drop undocumented category codes (`EDUCATION` 0/5/6, `MARRIAGE` 0), negative bill amounts, or duplicate feature rows. Those values are left as-is for LightGBM. `client_id` is never used as a predictor.

No scaling and no imputation. Pay-ratio columns are `PAY_AMT{i} / BILL_AMT{i+1}` with NaN where the denominator is ≤ 0 (LightGBM handles NaN natively).

Demographic fields `SEX`, `EDUCATION`, `MARRIAGE`, and `AGE` are used as provided. They are source-coded categories; undocumented extra codes are not relabelled. They should not be read as a production lending policy.

## 3. Feature engineering

### 3.1 Row-wise features (both halves)

Nothing in this block is fitted on the training set and then applied to test.

**Carried from 0.40967:** payment / previous-bill ratios; limit utilisation; repayment-status slope / mean / recent-minus-oldest; delinquency max/sum/count/streak; `severe_delinq_count`; bill deltas; pay-to-limit ratios; zero-payment months.

**Added for 0.40929 (still used):**

| Feature | Definition |
|---|---|
| `n_revolve` | count of `PAY_0, PAY_2, …, PAY_6` equal to `0` (revolving) |
| `n_duly` | count equal to `-1` (paid duly) |
| `n_inactive` | count equal to `-2` (no consumption) |
| `log_limit` | `log1p(LIMIT_BAL)` |
| `remaining_credit` | `LIMIT_BAL - BILL_AMT1` |

Reason: default rates are not monotone in the raw PAY codes. Revolving (`0`) defaults less than paid-duly (`-1`). Numeric `pay_mean` treats `-2 < -1 < 0` as an ordered risk ladder; type counts do not.

**75 predictors** on the PAY-types half.

### 3.2 Nested target encoding (TE half only)

Four extra columns, fitted **inside each outer training fold only**:

| Column | Group |
|---|---|
| `te_pay0` | `PAY_0` |
| `te_pay2` | `PAY_2` |
| `te_pat2` | `PAY_0` and `PAY_2` |
| `te_pat3` | `PAY_0`, `PAY_2`, and `PAY_3` |

For a group, the encoded value is the smoothed default rate

```
(sum(y) + M * prior) / (count + M)     with M = 50
```

`prior` is the mean default rate of the rows used to fit that table. Unseen keys fall back to `prior`.

Training rows receive an **inner 5-fold** encoding so a row never uses its own label. Outer validation and test rows use a table fit on the full outer-training partition.

This is **flat** shrinkage (toward the global fold prior), not hierarchical shrinkage toward a parent pattern.

**79 predictors** on the TE half (75 + 4).

## 4. Validation strategy

Inside each of 10 repeats, `StratifiedKFold` with 10 folds (`shuffle=True`, `random_state=1000+r`). Early stopping uses that fold’s validation set. Out-of-fold probabilities are therefore **optimistic** relative to a fair holdout. We still report them because they use the same protocol as the 0.40981 / 0.40967 / 0.40929 records.

Blend weight `0.5` was chosen by an 11-point grid on **averaged OOF** of the two 10×10 predictors (weight on PAY-types from 0.0 to 1.0). Best local weight was 0.50 (OOF 0.42217).

| Model | Mean-repeat OOF | Averaged-predictor OOF | OOF AUC |
|---|---|---|---|
| PAY-types only (0.40929) | 0.42474 | 0.42245 | 0.79001 |
| Nested TE only | 0.42456 | 0.42240 | 0.79025 |
| **50/50 blend (submitted)** | — | **0.42217** | — |

The public metric is the organiser’s hidden-test log loss (**0.40900**).

## 5. Models tested and final selection

The **submitted** model is the 50/50 blend above.

Other internal experiments (not this file):

| Attempt | Public log loss | Outcome |
|---|---|---|
| Extended LGBM 10×10 | 0.40967 | Previous record |
| PAY-types 10×10 (75 features) | 0.40929 | Parent of this blend; still 2nd on public without TE |
| Nested TE 10×10 only | ~0.40901 (teammate recipe) | TE half; not submitted alone |
| 50/50 PAY-types + 100-model TE | **0.40900** | **Selected** |
| extra_trees / bagging blends | 0.41037–0.41211 | Training-loop change; local OOF lied; discarded |
| Rare EDUCATION/MARRIAGE grouping | 0.40983 | Worse |
| Fold-wise PAY_0 rate with self-label leak | 0.41010 | Worse; discarded |
| Later public-weight TE mixes (10-model TE at 50% / 75%) | 0.40869 / 0.40851 | Better public, **worse local**; weight tuned on the public sample. Not this packet. |

Selection rule for **this** file: keep the established 10×10 LightGBM loop; add leak-controlled TE; mix with the 0.40929 parent using **local OOF**, not the public leaderboard.

We did **not** put the 0.40851 (75% single-seed TE) file in this packet. That mix improved public log loss while local OOF got worse, so it is a different, more public-sample-dependent experiment.

## 6. Ensembling and post-processing

- **Inner ensemble:** unweighted mean of 100 LightGBM models per family (10 repeats × 10 folds).
- **Outer ensemble:** unweighted 50/50 of the two family means.
- **Post-processing:** clip to `[1e-6, 1-1e-6]`. No Platt/isotonic calibration, no temperature scaling, no manual row edits.
- **Submission construction:** merge onto `sample_submission.csv` by `client_id`.

`reconstruct_submission.py` checks that the uploaded CSV equals that 50/50 mix of the two saved test vectors (max absolute error ~1e-16).

## 7. Final model hyperparameters

Identical for both LightGBM families:

```
objective: binary
metric: binary_logloss
learning_rate: 0.1
num_leaves: 31
verbose: -1
deterministic: True
force_row_wise: True
num_boost_round: 2000
early_stopping_rounds: 100
n_repeat: 10
n_splits: 10
```

TE-only extras: smoothing `M=50`; inner 5-fold encoding; four encoded columns listed above.

Ensemble weights: 1/100 inside each family; **0.5 / 0.5** across families.

LightGBM 4.7.0. No pretrained checkpoint.

## 8. Key results, observations, limitations

- `PAY_0` is the strongest raw signal (code `2` defaults at ~69%, revolving `0` at ~13%).
- Type counts help because PAY codes are not a risk ladder; nested TE goes further by replacing codes with empirical default rates, including pairs and triples trees would need several splits to isolate.
- Local OOF and public log loss both improved from 0.40929 → 0.40900 (local 0.42245 → 0.42217).
- Optimistic 10×10 OOF is **not** a reliable screen for training-procedure changes (extra_trees looked better locally and scored 0.41211 publicly).
- TE features are built from labels, so they can overfit rare PAY patterns. Mitigations: inner-fold encoding, `M=50`, 100-model averaging, 50% weight on the non-TE parent.
- Limitations: early stopping peeks at the scored fold; 200 correlated trees; demographics used as competition features, not as a fair lending design; no causal claim about education/sex/marriage; public/private split can still move a 0.00029 gap.

## 9. Reproducibility

See `README.md`. Intermediate **feature** tables are not saved; they are built in memory. Intermediate **prediction** vectors used in the uploaded mix are in `artifacts/`.
