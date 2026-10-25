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
3. **Shared Low-Confidence Support:** Implemented `src/support.py` with `compute_support` and `flag_low_confidence`, validated by unit tests covering unseen cells and training partition isolation.
4. **Decision 4 Settled (Profit Reconstructability):** Proven mathematically and empirically that profit can be derived **exactly** (maximum error = 0.000000) from sell-through ($s = \text{units\_sold} / \text{initial\_quantity}$) and price/cost attributes. Strongly recommend **modeling sell-through and deriving profit** rather than direct profit regression.
5. **Decision 5 Settled (Support Threshold):** Evaluated the empirical training cell density of `category x region x store` (500 cells, minimum cell size = 129). Proved that initial thresholds of 10 to 100 flag 0.00% of test rows. Determined that a threshold of **140 to 145** flags **6.37% to 11.09%** of test rows, satisfying the target 5–15% operational range.

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

### 3.1 Function Specifications
- `compute_support(df_train, keys=("category", "region", "store_id"))`: Groups the training partition by specified keys and computes cell density.
- `flag_low_confidence(df, support_table, threshold=30)`: Merges target DataFrame against the training support table. Cells unseen in training receive count = 0 and are automatically flagged `True`.

### 3.2 Unit Test Verifications
Unit tests executed in `scripts/M2_T12_decision_checks.py` verified:
- Accurate aggregation across composite tuples (`category`, `region`, `store_id`).
- Preservation of target DataFrame row order and index during key joins.
- Unseen cells in training partition are imputed with count 0 and flagged as low confidence.
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
- *Rationale:* Profit has extreme heavy tails spanning from -\$39,476 to +\$44,919, driven by high-volume batches (e.g. pharmaceuticals and bulk meats). Directly fitting a regressor on raw profit introduces severe variance, distortion from extreme outliers, and violations of the physical inventory boundary.
- Conversely, sell-through $s \in [0, 1]$ is strictly bounded, well-behaved, and has a direct behavioral interpretation (demand response).
- Deriving profit from predicted sell-through $\hat{s}$ guarantees 100% adherence to physical accounting identities, eliminates outlier distortions without loss of economic fidelity, and provides consistent inputs for the Track 3C discount optimizer.

---

## 5. Decision 5: Support Cell Distribution & Low-Confidence Threshold Calibration

### 5.1 Training Cell Density (`category x region x store`)
There are 10 categories, 5 regions, and 50 stores. Because each store operates within a single unique region, the support space contains exactly **500 distinct cells**.

Distribution of training counts across the 500 cells (80,042 training rows):
- **Minimum:** 129 rows
- **1st Percentile:** 133.0 rows
- **5th Percentile:** 138.9 rows
- **25th Percentile:** 151.0 rows
- **Median:** 159.0 rows
- **Mean:** 160.1 rows
- **75th Percentile:** 169.0 rows
- **95th Percentile:** 182.0 rows
- **Maximum:** 203 rows

### 5.2 Threshold Calibration on Holdout Test Set (19,958 rows)

| Threshold | Test Rows Flagged | Share Flagged (%) | Spoilage Rate (Flagged) | Spoilage Rate (Unflagged) | Operational Assessment |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **10** | 0 | 0.00% | N/A | 19.81% | Inactive (min training cell = 129). |
| **20** | 0 | 0.00% | N/A | 19.81% | Inactive. |
| **30** | 0 | 0.00% | N/A | 19.81% | Inactive (starting proposal threshold). |
| **50** | 0 | 0.00% | N/A | 19.81% | Inactive. |
| **100** | 0 | 0.00% | N/A | 19.81% | Inactive. |
| **135** | 438 | 2.19% | 19.41% | 19.82% | Low coverage (bottom 2.2% tails). |
| **140** | **1,271** | **6.37%** | **20.54%** | **19.76%** | **Optimal lower bound (flags 6.4%).** |
| **145** | **2,213** | **11.09%** | **19.43%** | **19.86%** | **Optimal upper bound (flags 11.1%).** |
| **150** | 4,453 | 22.31% | 19.45% | 19.92% | Too broad (flags 22.3%, above 15% target). |

### 5.3 Final Recommendation for Decision 5
**RECOMMENDATION: Calibrate Support Threshold to 140 (or 145) for `category x region x store`.**
- *Empirical Finding:* Because the synthetic dataset is uniformly generated across the 50 stores, every category–store cell has at least 129 observations in the training set. Thresholds $\le 100$ flag 0% of rows.
- Setting the threshold to **140** flags **6.37% (1,271 test rows)**; setting it to **145** flags **11.09% (2,213 test rows)**. Both comfortably satisfy the team's target of flagging roughly 5% to 15% of holdout rows.
- *Granular Alternative:* If model owners define support cells at the finer `product_name x store_id` level (3,150 cells, mean = 25.4 rows), a threshold of **20** flags **14.36%** of test rows.

---

## 6. Deviations from Plan

None.
