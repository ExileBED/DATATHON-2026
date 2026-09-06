# Required disclosure

Short provenance statement for the **0.40900** leaderboard file `submission_blend_te_paytypes.csv`.

## External datasets

None. Only the competition files `train.csv`, `test.csv`, and `sample_submission.csv` were used.

## External code, notebooks, repositories, or public solutions

The 10×10 LightGBM loop (`learning_rate=0.1`, 31 leaves, early stopping on the scored fold) was adapted from a teammate baseline that previously scored **0.40981**, then from the team’s **0.40967** / **0.40929** scripts.

The nested target-encoding design (four PAY-code groups, smoothing `M=50`, inner 5-fold encoding on train rows, outer-train encoding for validation/test) follows a teammate script `push4.py`. That script imported `verify_75` / `build_75`, which were not supplied; this packet uses `BEST_0.40929.build_features` as the 75-column builder. No public Kaggle kernel was copied as the final pipeline.

## Pretrained models

None. All LightGBM models are trained from scratch on the competition training set.

## AutoML / external modelling systems

None. Training uses the `lightgbm` Python package (`lgb.train`) and `scikit-learn` `StratifiedKFold`.

## AI tools or coding agents

Yes. Cursor (coding agent) was used to inspect the competition brief, compare internal experiments, implement nested target encoding on the 75-feature set, run training, blend OOF/test probabilities, and draft this submission packet. The submitted prediction file was produced by executing the LightGBM scripts on the competition data, not by an AI model emitting probabilities directly.

## Manual modification or post-processing of predictions

None. Test probabilities are `0.5 * PAY-types + 0.5 * nested TE`, clipped to `(1e-6, 1-1e-6)` only to keep log loss defined. No hand edits and no spreadsheet changes.

## Information beyond the competition-provided files

None for the fitted model. Organiser documentation (task description, data dictionary, evaluation as binary log loss) was used to choose the metric and to interpret `PAY_*` codes. No extra customer data were added.
