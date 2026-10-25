# M2-T12: Track 3A OOF Evaluation, Support Functions & Decision Checks

**Project:** Smart Shelf: Perishable Food Waste Reduction and Dynamic Markdown Optimizer  
**Task:** M2-T12 — Out-of-Fold Spoilage Classifier Generator, Shared Low-Confidence Support, and Empirical Decision Checks  
**Status:** Complete  
**Date:** 2026-10-25  
**Artifacts Generated:**

- `src/m3_oof.py` (Reusable OOF prediction and evaluation library)
- `src/support.py` (Shared low-confidence support calculation and thresholding)
- `scripts/M2_T12_decision_checks.py` (Validation runner and empirical decision analysis)
- `data/processed/track_3a_oof_train.csv` (Baseline OOF predictions on 80,042 training rows; gitignored)

---

## 1. Executive Summary

Task M2-T12 resolves key prerequisites and settles empirical design decisions agreed prior to Milestone 3:

1. **Reusable OOF Generator:** Established `src/m3_oof.py` for strictly time-ordered, leak-free cross-validation with date-boundary block splitting. Generated baseline OOF predictions on all 80,042 training rows using LightGBM on `TRACK_3A_FEATURES_FULL` and logged the run to MLflow (`track_3a_oof_baseline`).
2. **Feature List Normalization:** Corrected one-hot column names in `src/m2_feature_pipeline.py` to match exact dataset naming (`store_id_STORE_*`, `supplier_id_SUPPLIER_*`, `region_Southwest`). Added `validate_feature_columns_exist` for programmatic validation.
3. **Shared Low-Confidence Support:** Implemented `src/support.py` with `compute_support` and `flag_low_confidence`, validated by unit tests covering unseen product–store cells and training partition isolation.
4. **Decision 4 Settled (Profit Reconstructability):** Proven mathematically and empirically that profit can be derived **exactly** (maximum error = 0.000000) from sell-through ($s = \text{units\_sold} / \text{initial\_quantity}$) and price/cost attributes. Strongly recommend **modeling sell-through and deriving profit** rather than direct profit regression.
5. **Decision 5 Settled (Support Keys & Threshold):** Adopted Option B: support cells defined at the granular `product_name x store_id` level (3,150 cells) with a threshold of **20**, flagging **14.36% (2,866 rows)** of holdout test rows. The original proposal's `category x region x store` flag never triggers (0.00% flagged across all 500 cells at thresholds 10 to 100).
6. **Reliability & Calibration Verification:** Rigorous empirical evaluation across holdout test rows and training OOF rows shows no measurable difference in calibration or accuracy; the bootstrap intervals are wide (about +/-0.009 on the test period), so moderate differences cannot be ruled out (test Brier score difference: -0.00343, 95% bootstrap CI [-0.01187, +0.00568]). On the test period, actual calibration gaps show that flagged batches over-predict by +0.35 points (19.78% vs 19.43%), while unflagged batches under-predict by -0.57 points (19.30% vs 19.87%). The flag is sensitive to the threshold (20 flags 14.4% of test rows, 25 flags 44.6%, 30 flags 74.3%). The flag reliably measures **data sparsity** (sample size in training), not demonstrated model error; the dashboard should word the flag as "limited history for this product at this store", not as low model confidence.

---

## 2. Track 3A Out-of-Fold (OOF) Scheme Evaluation

A small fixed LightGBM classifier (`random_state=42`, `n_estimators=200`, `learning_rate=0.05`, `num_leaves=15`, `min_child_samples=100`, no tuning) was evaluated across three cross-validation schemes on `TRACK_3A_FEATURES_FULL` (80,042 training rows). All time blocks are cut strictly at date boundaries so rows with the same transaction date are never split across blocks.

### 2.1 Scheme Comparison Results

| CV Scheme | Evaluated Rows | Brier Score | Log Loss | ROC-AUC | Gap to Benchmark AUC | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Time-Blocked 5-Fold** | 80,042 (100.0%) | **0.15397** | **0.48452** | **0.5842** | -0.0089 | **Selected Baseline.** Date-boundary contiguous time blocks; each block predicted by model trained on all other blocks. |
| **Stratified 5-Fold** | 80,042 (100.0%) | 0.15390 | 0.48427 | 0.5857 | -0.0073 | Reference standard (M2-T07). Shuffled stratified partitions across training period. |
| **Expanding Window** | 63,961 (80.0%) | 0.15453 | 0.48621 | 0.5783 | -0.0147 | Strictly past-only. Leaves Block 0 (16,081 rows) unpredicted (NaN); rejected per Decision #1. |
| **`spoilage_risk` Benchmark** | 80,042 (100.0%) | 0.15343 | 0.48279 | 0.5931 | 0.0000 | Synthetic generation risk benchmark (cannot be used as a model feature). |

### 2.2 Baseline OOF Dataset Construction

- **File Written:** `data/processed/track_3a_oof_train.csv` (80,042 rows).
- **Columns:** `record_id`, `oof_proba`, `block`, `scheme`, `feature_set_name`, `model_type`, `shelf_life_version`.
- **Integrity Assertions:**
  - Every training row appears exactly once with a valid probability in [0.0, 1.0] and zero NaNs.
  - Zero test rows (19,958 holdout rows) were present or predicted in the OOF dataset.
  - For each fold, training and validation partitions were mutually disjoint with no overlapping transaction dates.
- **MLflow Tracking Run:** Logged to Track 3A experiment (`smart-shelf-3a-spoilage-classifier`), tagged `purpose=oof_baseline`, `feature_set_name=track_3a_full`, `shelf_life_version=expiry_remaining_ratio`.

---

## 3. Shared Low-Confidence Support Module (`src/support.py`)

### 3.1 Function Specifications & Constants

- **Default Grouping Keys:** `DEFAULT_SUPPORT_KEYS = ("product_name", "store_id")`
- **Default Threshold:** `DEFAULT_SUPPORT_THRESHOLD = 20`
- `compute_support(df_train, keys=DEFAULT_SUPPORT_KEYS)`: Groups the training partition by specified keys and computes cell density.
- `flag_low_confidence(df, support_table, threshold=20, keys=DEFAULT_SUPPORT_KEYS)`: Merges target DataFrame against the training support table. Cells unseen in training receive count = 0 and are automatically flagged `True`.

### 3.2 Unit Test Verifications

Unit tests executed in `src/support.py` and `scripts/M2_T12_decision_checks.py` verified:

- Accurate aggregation across composite tuples (`product_name`, `store_id`).
- Preservation of target DataFrame row order and index during key joins.
- Default threshold 20 properly flags all observations below 20 observations.
- Completely unseen product x store pairs in the target set receive count 0 and are flagged.
- Key and threshold parameter overrides work seamlessly.
- Strictly training-only data ingestion.

---

## 4. Decision 4: Profit Reconstructability & Modeling Recommendation

### 4.1 Accounting Identities & Mathematical Proof

We evaluated the empirical relationship between financial outcomes and pre-sale features across all 100,000 transactions:

1. **Revenue Identity:** $\text{revenue} = \text{units\_sold} \times \text{selling\_price}$ holds with **maximum error = 0.000000**.
2. **Waste Cost Identity:** $\text{waste\_cost} = \text{units\_wasted} \times \text{cost\_price}$ holds with **maximum error = 0.000000**.
3. **Inventory Conservation:** $\text{initial\_quantity} = \text{units\_sold} + \text{units\_wasted}$ holds in **100.0% of rows** (max error = 0.000000).
4. **Pricing Equation:** $\text{selling\_price} = \text{base\_price} \times (1 - \text{discount\_pct})$ holds within retail 2-decimal rounding precision (max error = 0.005000).

### 4.2 Exact Profit Formulas

Total procurement cost equals:
$$\text{total\_cost} = \text{initial\_quantity} \times \text{cost\_price} = (\text{units\_sold} \times \text{cost\_price}) + \text{waste\_cost}$$

Therefore, profit satisfies:
$$\text{profit} = \text{revenue} - \text{waste\_cost} - (\text{units\_sold} \times \text{cost\_price}) = \text{revenue} - (\text{initial\_quantity} \times \text{cost\_price})$$

- **Observed Maximum Error:** **0.000000**.

Defining sell-through ratio $s = \frac{\text{units\_sold}}{\text{initial\_quantity}}$ ($0 \le s \le 1$), profit is reconstructed from sell-through as:
$$\text{profit} = \text{initial\_quantity} \times \left[ s \times \text{selling\_price} - \text{cost\_price} \right]$$

- **Exact Sell-Through Reconstruction Error:** **0.000000** (exact accounting equivalence).
- **Discount-Expanded Formula Error:** 2.500000 (due strictly to 2-decimal rounding of selling_price: $\le 0.005 \times 500 = 2.50$).

### 4.3 Distributional Metrics & Outlier Characteristics

- **Negative Profit Share:** **28.70%** (28,703 batches operated at a financial loss).
- **Profit Percentiles ($):**
  - 1st Percentile: $-39,476.84
  - 5th Percentile: $-2,026.09
  - 50th Percentile (Median): $276.70
  - 95th Percentile: $6,242.81
  - 99th Percentile: $44,919.37
- **Profit Margin Percentiles (%):**
  - 1st Percentile: -151.4%
  - 5th Percentile: -70.9%
  - 50th Percentile (Median): 26.3%
  - 95th Percentile: 55.2%
  - 99th Percentile: 59.0%

### 4.4 Final Recommendation for Decision 4

**RECOMMENDATION: Model Sell-Through and Derive Profit.**

- Profit has extreme heavy tails spanning from -\$39,476 to +\$44,919, driven by high-volume batches. Directly fitting a regressor on raw profit introduces severe variance and violates inventory conservation boundaries.
- Sell-through $s \in [0, 1]$ is strictly bounded and well-behaved. Deriving profit guarantees 100% adherence to physical accounting identities, eliminates outlier distortions, and provides consistent inputs for Track 3C.

---

## 5. Decision 5: Support Cell Distribution & Low-Confidence Threshold Calibration

### 5.1 Primary Analysis: `product_name x store_id` (Option B)

The support space consists of 63 unique products distributed across 50 stores, yielding **3,150 distinct cells** in the training period (80,042 rows).

Distribution of training counts per `product_name x store_id` cell:

- **Total Cells:** 3,150
- **Minimum:** 9 rows
- **5th Percentile:** 17 rows
- **25th Percentile:** 21 rows
- **Median:** 25 rows
- **Mean:** 25.41 rows
- **75th Percentile:** 29 rows
- **95th Percentile:** 36 rows
- **Maximum:** 49 rows

#### Threshold Calibration on Holdout Test Set (19,958 rows)

| Threshold | Test Rows Flagged | Share Flagged (%) | Spoilage Rate (Flagged) | Spoilage Rate (Unflagged) | Operational Assessment |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **10** | 9 | 0.05% | 22.22% | 19.81% | Inactive; captures only extreme bottom tails (9 rows). |
| **15** | 305 | 1.53% | 19.02% | 19.82% | Very conservative (305 rows, 1.53%). |
| **20** | **2,866** | **14.36%** | **19.43%** | **19.87%** | **Selected Option B Baseline. Within the target 5-15% range.** |
| **25** | 8,904 | 44.61% | 19.59% | 19.99% | Overly broad (8,904 rows, 44.61% of test set). |
| **30** | 14,821 | 74.26% | 19.86% | 19.66% | Dominates test set (14,821 rows, 74.26%). |

- **Unseen Product x Store Pairs:** Exactly **0** test rows belonged to product–store combinations unseen in the training partition. Every product x store combination in the test set had historical presence in training.

### 5.2 Evidence for Deviation: `category x region x store` Cell Distribution

The project proposal originally suggested defining support cells at `category x region x store`. Because each store operates within a single unique region, this grouping contains exactly **500 distinct cells**.

Distribution of training counts across the 500 `category x region x store` cells:

- **Total Cells:** 500
- **Minimum Cell Size:** 129 rows
- **Median Cell Size:** 159 rows
- **Maximum Cell Size:** 203 rows
- **Test Rows Flagged at Threshold 30:** **0 (0.00%)**
- **Test Rows Flagged at Threshold 100:** **0 (0.00%)**

*Finding:* Because every single category–store cell has at least 129 observations in the training data, the proposal's flag is completely inert at reasonable thresholds ($\le 100$). It requires artificial thresholds above 140 simply to trigger, whereas `product_name x store_id` reflects genuine item-level operational sparsity.

---

## 6. Prediction Reliability & Calibration Check (Step 3)

To test whether the low-confidence flag identifies genuinely less reliable model predictions, the final baseline LightGBM model (`TRACK_3A_FEATURES_FULL`, 80,042 training rows) was evaluated using `fit_final_and_predict_test` on holdout test rows and out-of-fold predictions on training rows.

### 6.1 Holdout Test Set Evaluation (19,958 rows)

| Prediction Cohort | Row Count | Actual Spoilage Rate | Mean Predicted Prob | Brier Score | Log Loss | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Flagged (Low Confidence, < 20)** | **2,866** | **19.43%** | **19.78%** | **0.15370** | **0.48327** | **0.5991** |
| **Unflagged (Normal Support, $\ge$ 20)** | **17,092** | **19.87%** | **19.30%** | **0.15714** | **0.49183** | **0.5834** |
| **All Test Rows** | 19,958 | 19.81% | 19.37% | 0.15665 | 0.49060 | 0.5855 |

- **Brier Score Difference (Flagged minus Unflagged):** **-0.00343**
- **Bootstrap 95% Confidence Interval (1,000 resamples, seed 42):** **[-0.01187, +0.00568]**

### 6.2 Training Out-of-Fold (OOF) Evaluation (80,042 rows)

| Prediction Cohort | Row Count | Actual Spoilage Rate | Mean Predicted Prob | Brier Score | Log Loss | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Flagged (Low Confidence, < 20)** | **8,324** | **19.25%** | **19.50%** | **0.15340** | **0.48304** | **0.5847** |
| **Unflagged (Normal Support, $\ge$ 20)** | **71,718** | **19.36%** | **19.20%** | **0.15404** | **0.48470** | **0.5842** |
| **All Training OOF Rows** | 80,042 | 19.35% | 19.23% | 0.15397 | 0.48452 | 0.5842 |

- **Brier Score Difference (Flagged minus Unflagged):** **-0.00064**
- **Bootstrap 95% Confidence Interval (1,000 resamples, seed 42):** **[-0.00600, +0.00487]**

### 6.3 Plain-Language Reading & Interpretation

- **Calibration and Accuracy:** In both the test set and training OOF partitions, the 95% bootstrap confidence interval for the difference in Brier score spans zero ([-0.01187, +0.00568] on test; [-0.00600, +0.00487] on train OOF). Actual test calibration gaps show that flagged batches over-predict by +0.35 points (mean predicted 19.78% vs actual rate 19.43%), while unflagged batches under-predict by -0.57 points (mean predicted 19.30% vs actual rate 19.87%). AUC differs by 0.016 on the test period, but no interval was computed and the flagged group has only 2,866 rows, so this is not interpreted.
- **Core Conclusion:** The difference in model accuracy and calibration between flagged and unflagged batches shows **no measurable difference; the bootstrap intervals are wide (about +/-0.009 on the test period), so moderate differences cannot be ruled out**. The low-confidence flag measures **data sparsity** (limited historical sample size for a specific product at a specific store), **not demonstrated model error**.
- **Threshold Sensitivity:** The flag is sensitive to the threshold (20 flags 14.4% of test rows, 25 flags 44.6%, 30 flags 74.3%).
- **Dashboard Presentation:** The dashboard should word the flag as "limited history for this product at this store", not as low model confidence. This ensures decision-makers understand the flag signals empirical sparsity rather than faulty probabilities.

---

## 7. Deviations from Plan

1. **Low-Confidence Flag Granularity and Threshold (Option B):**
   - **Deviation:** Replaced the proposal's `category x region x store` grouping with `product_name x store_id` cells with a threshold of **20** (flagged when training count < 20; unseen cells count as 0 and are flagged).
   - **Reason for Deviation:** The original proposal's flag never triggers under realistic conditions. All 500 `category x region x store` cells have at least 129 training rows (median 159, max 203), resulting in exactly **0.00% (0 rows)** flagged at threshold 30 and threshold 100. Product x store is the only granularity where empirical support varies across stores.
   - **Test Rows Flagged:** At threshold 20, the flag identifies **14.36% (2,866 rows)** of holdout test rows, satisfying the target 5%–15% operational coverage.
   - **Reliability Verification Result:** Model calibration and error show no measurable difference; the bootstrap intervals are wide (about +/-0.009 on the test period), so moderate differences cannot be ruled out (test Brier difference = -0.00343, 95% CI [-0.01187, +0.00568]; actual test gaps: flagged +0.35 points over-prediction, unflagged -0.57 points under-prediction). The flag is sensitive to the threshold (20 flags 14.4% of test rows, 25 flags 44.6%, 30 flags 74.3%). The flag measures historical data sparsity, not demonstrated model error, and the dashboard should word the flag as "limited history for this product at this store", not as low model confidence.

2. **Feature-name correction in the M2-T11 lists:** `TRACK_3A_FEATURES_FULL` and `TRACK_3B_FEATURES` in `src/m2_feature_pipeline.py` contained hand-typed one-hot names that do not exist in `model_ready_m2_dataset.csv` (`store_id_S001`..., `supplier_id_SUP01`..., `region_South`). They were corrected to the real column names (`store_id_STORE_001`..., `supplier_id_SUPPLIER_01`..., `region_Southwest`). List sizes are unchanged (180 / 90 / 43). A new function, `validate_feature_columns_exist`, checks that every listed name exists in the dataset.
