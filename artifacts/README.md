# Intermediate artefacts

These files were used to build the uploaded mix. They can be regenerated from the official CSVs plus `BEST_0.40929.py` and `train_te_and_blend.py`. They are included so reviewers can check the last step without retraining.

| File | What it is |
|---|---|
| `submission_lgbm_pay_types_0.40929.csv` | PAY-types 10×10 test probabilities (public 0.40929). Same scores as `test_pay_types.csv`. |
| `test_pay_types.csv` | Same PAY-types test vector as used in the 50/50 mix (`client_id`, `default_probability`). |
| `oof_pay_types.csv` | PAY-types 10×10 averaged OOF (`client_id`, `oof_default_probability`). |
| `meta_pay_types.json` | Local OOF summary for the PAY-types half. |
| `submission_push4_te.csv` | Nested-TE 10×10 test probabilities. |
| `oof_push4_te.csv` | Nested-TE 10×10 averaged OOF. |
| `push4_blend_summary.json` | Local blend grid and the selected weight 0.5. |
| `submission_blend_te_paytypes.csv` | Copy of the uploaded file (also at folder root). |

Row-wise feature tables are **not** stored. `build_features()` rebuilds them from `train.csv` / `test.csv`.
