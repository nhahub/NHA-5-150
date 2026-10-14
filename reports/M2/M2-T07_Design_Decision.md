# M2-T07: Design decision, risk input for Track 3B

*Text for the Feature Engineering Summary. Evidence: `M2-T07.ipynb` (run on the verified file, SHA-256 `de94302b...`).*

## Decision

Track 3B uses **Track 3A's out-of-fold predicted spoilage probability** as its risk input. It does **not** use the `spoilage_risk` column.

- For training batches the input is the out-of-fold (OOF) probability.
- For a new batch (Simulator Mode) the input is `predict_proba` of the final tuned Track 3A classifier.

## Why

1. **`spoilage_risk` does not exist for a new batch.** It is a generated score tied to the process that produces `was_spoiled` (kept as an evaluation benchmark since Milestone 1). A user who enters a batch in Simulator Mode cannot supply it. A classifier probability can be computed from information known when the batch arrives.
2. **Nothing is lost by replacing it.** On the real file `spoilage_risk` alone has an AUC of 0.594, and 99.6% of its variance can be rebuilt from the input columns (out-of-fold R2 = 0.996); the part the inputs cannot rebuild has an AUC of 0.515 against `was_spoiled`. The prototype classifier reaches a holdout AUC of 0.587 against 0.597 for `spoilage_risk` (95% range of the holdout AUC: 0.578 to 0.597), and its probabilities correlate 0.91 with `spoilage_risk` on the holdout.
3. **Consistency.** The same risk input is used to train Track 3B, in the M3-T05 confounding controls and in Simulator Mode.

## How the out-of-fold predictions are produced

Predictions made by a model on rows it was trained on are over-optimistic (prototype in-sample AUC 0.610 against 0.586 out-of-fold; for a very flexible random forest the in-sample AUC is 1.00 and the holdout AUC 0.55). Out-of-fold prediction removes this:

1. Take the final tuned Track 3A classifier from M3-T02 (M3-T04 follows M3-T02).
2. Split the training rows into 5 folds, stratified on `was_spoiled` and shuffled (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
3. For each fold, train a copy of the classifier on the other 4 folds and predict the spoilage probability of the held-out fold.
4. Combine the held-out predictions. Every training row now has exactly one probability, produced by a model that never saw that row.
5. These probabilities are the risk input used to train Track 3B.

**New batches.** The risk input is `predict_proba` of the final classifier trained on all training rows. The batch was never part of the training data, so it is treated the same way as a held-out fold.

## Scheme choice (K and fold type)

Five schemes were compared on the real data (training rows up to 2024-08-07, the M1-T04 split; holdout after):

| Scheme | Rows with a probability | OOF AUC | Gap to holdout AUC (0.587) |
|---|---|---|---|
| Stratified 5-fold, shuffled | 100% | 0.586 | 0.001 |
| Stratified 10-fold, shuffled | 100% | 0.587 | 0.001 |
| Time-blocked 5-fold | 100% | 0.587 | 0.000 |
| Expanding window (past only) | 80% | 0.583 | 0.005 |
| Grouped 5-fold (product x store) | 100% | 0.585 | 0.002 |

All schemes that cover every row agree within 0.002, much less than the uncertainty of the holdout AUC (about plus or minus 0.01), so stratified shuffled 5-fold is chosen as the simplest. The expanding window leaves the first 20% of rows without a probability and is rejected. Grouping by product and store changes nothing, in line with the Milestone 1 finding that a product-store pair carries no information of its own.

## Validation of the procedure

| Check | Result |
|---|---|
| Every training row predicted exactly once | yes (0 rows missing, 0 predicted twice) |
| Hand-written loop equals scikit-learn `cross_val_predict` | yes |
| OOF AUC against holdout AUC | 0.586 against 0.587 |
| OOF against holdout probabilities | mean 0.1935 against 0.1939, standard deviation 0.046 against 0.045 |
| Calibration | predicted and observed spoilage rates agree across 10 groups |

## Rules for Milestone 3

- When M3-T04 produces the probabilities with the tuned classifier, repeat the scheme comparison. If the OOF AUC and the holdout AUC differ by more than 0.01, switch to time-blocked folds.
- The notebook prototype is a demonstration. Only probabilities from the final tuned classifier go to Track 3B.
- Keep `spoilage_risk`, the seven result columns (`units_sold`, `units_wasted`, `waste_pct`, `waste_cost`, `revenue`, `profit`, `profit_margin_pct`) and the discount treatments (`markdown_applied`, `discount_pct`, `selling_price`) out of the Track 3A inputs.
