# M2-T11: Milestone 2 Feature Consolidation Report

**Project:** Smart Shelf: Perishable Food Waste Reduction and Dynamic Markdown Optimizer  
**Task:** M2-T11 — Consolidate Milestone 2 Features into a Reproducible, Code-Built Dataset  
**Status:** Complete  
**Date:** 2026-10-24  
**Outputs Generated:**
- `src/m2_feature_pipeline.py` (Core feature engineering and consolidation library)
- `scripts/M2_T11_build_m2_dataset.py` (End-to-end reproducible builder & verification runner)
- `data/processed/model_ready_m2_dataset.csv` (100,000 rows, 210 columns; gitignored)
- `data/processed/feature_dictionary_m2.csv` (210 rows, updated schema & roles; gitignored)
- `models/preprocessor_m2.joblib` (Fitted signal scalers trained strictly on train split; gitignored)

---

## 1. Executive Summary

Task M2-T11 establishes a reproducible, code-materialized pipeline consolidating all Milestone 2 engineered features. Previously, M2 features (trailing sales velocity, sell-through risk, store-level historical markdown frequencies, and scaled handling signals) existed across individual task notebooks. The newly implemented pipeline in `src/m2_feature_pipeline.py` deterministically builds the full consolidated dataset from `data/processed/cleaned_dataset.csv`, strictly joins it 1:1 on `record_id` with `data/processed/model_ready_dataset.csv`, and exposes standardized, leakage-free feature lists for Track 3A and Track 3B models.

---

## 2. Standardized Feature Lists (Decisions 2 & 3)

The module `src/m2_feature_pipeline.py` defines three canonical feature list constants:

### 2.1 `TRACK_3A_FEATURES_FULL` (180 features)
Combines the 173 pre-sale Milestone 1 features with all 7 Milestone 2 features permitted for Track 3A:
- **Batch Physical, Handling & Demand (20):** `shelf_life_days`, `days_remaining_at_purchase`, `days_until_expiry`, `days_since_receipt`, `temp_abuse_events`, `base_price`, `cost_price`, `daily_demand`, `expiry_remaining_ratio`, `expiry_lag_days`, `storage_temp`, `temp_deviation`, `distribution_hours`, `handling_score`, `packaging_score`, `supplier_score`, `spoilage_sensitivity`, `initial_quantity`, `demand_variability`, `quality_grade_ord`.
- **Calendar & Cyclical (5):** `is_weekend`, `month_sin`, `month_cos`, `dow_sin`, `dow_cos`.
- **One-Hot Encoded Entities (148):**
  - `category_*` (10 columns: Bakery, Beverages, Dairy, Deli, Frozen_Meals, Meat, Pharmaceuticals, Produce, Ready_to_Eat, Seafood)
  - `region_*` (5 columns: Midwest, Northeast, South, Southeast, West)
  - `product_name_*` (63 columns: Antibiotics to Yogurt)
  - `store_id_*` (50 columns: S001 to S050)
  - `supplier_id_*` (20 columns: SUP001 to SUP020)
- **M2 Engineered Allowed Features (7):** `sales_velocity`, `sales_velocity_per_day`, `sell_through_risk`, `sell_through_ratio`, `at_risk`, `store_md_freq_hist`, `store_md_freq_smoothed`.
- **Anti-Leakage Exclusions:** Excludes `discount_pct`, `markdown_applied`, and `selling_price` (discount treatments), `spoilage_risk` (benchmark only), all 7 outcome columns, and M2-T04 discount tier flags (`discount_tier`, `is_light`, `is_medium`, `is_deep`). Strictly uses `expiry_remaining_ratio` (`days_until_expiry / shelf_life_days`), excluding `shelf_life_used_ratio` per Decision 3.

### 2.2 `TRACK_3A_FEATURES_COMPACT` (90 features)
A compact subset designed to optimize probability calibration (Brier score / log loss) by pruning 90 features proven by M2-T05 to carry zero signal, severe multicollinearity, or redundancy:
- **Retained Features (90):**
  - Physical & Handling Drivers (12): `days_until_expiry`, `days_since_receipt`, `temp_abuse_events`, `base_price`, `daily_demand`, `expiry_remaining_ratio`, `temp_deviation`, `distribution_hours`, `handling_score`, `packaging_score`, `initial_quantity`, `quality_grade_ord`.
  - Calendar & Cyclical (5): `is_weekend`, `month_sin`, `month_cos`, `dow_sin`, `dow_cos`.
  - One-Hot Entities (73): `category_*` (10 columns) and `product_name_*` (63 columns).

#### Justification for Pruned Features (M2-T05 Evidence Table)
Every removal is directly justified by empirical evidence documented in `reports/M2/M2_T05_Data_Analysis_Report_and_Feature_Engineering_Summary.md`:

| Feature / Group | Count | Reason for Removal | Exact M2-T05 Citation & Numerical Evidence |
| :--- | :---: | :--- | :--- |
| `store_id_*` | 50 | Zero signal | **Section 2.1 Categorical Factors:** $\chi^2 = 50.90$, dof = 49, raw $p = 0.399$, Cramér's $V = 0.0226$ ("No signal"). |
| `supplier_id_*` | 20 | Zero signal | **Section 2.1 Categorical Factors:** $\chi^2 = 18.19$, dof = 19, raw $p = 0.510$, Cramér's $V = 0.0135$ ("No signal"). |
| `region_*` | 5 | Zero signal | **Section 2.1 Categorical Factors:** $\chi^2 = 4.39$, dof = 4, raw $p = 0.356$, Cramér's $V = 0.0066$ ("No signal", flat rates 19.10% to 19.78%). |
| `is_promoted` | 1 | Zero signal | **Section 2.1 Categorical Factors & Section 2.5:** $\chi^2 = 0.34$, $p = 0.559$, Cramér's $V = 0.0018$; spoilage 19.41% vs 19.62%. |
| `demand_variability` | 1 | Zero signal | **Section 2.1 Numeric Factors:** Fails corrected significance test (Holm adjusted $p \ge 0.05$). |
| `storage_temp` | 1 | Zero signal | **Section 2.1 Numeric Factors:** Fails corrected significance test (Holm adjusted $p \ge 0.05$). |
| `supplier_score` | 1 | Zero signal | **Section 2.5:** $r = +0.0029$, $p = 0.36$; spoilage stays flat between 19.3% and 19.7% across all score levels. |
| `spoilage_sensitivity` | 1 | Exact category redundancy | **Section 2.5 & 3.3:** Constant within each category ($R^2 = 1.0$, 9 distinct values); 100% redundant with `category_*`. |
| `days_remaining_at_purchase` | 1 | Extreme multicollinearity | **Section 2.1 Multicollinearity Table:** Pearson $r = 0.99996$ with `days_until_expiry` (VIF = 20,227). |
| `shelf_life_days` | 1 | Extreme multicollinearity | **Section 2.1 Multicollinearity Table:** Pearson $r = 0.9917$ with `days_until_expiry` (VIF = 68.4). |
| `cost_price` | 1 | Extreme multicollinearity | **Section 2.1 Multicollinearity Table:** Pearson $r = 0.9860$ with `base_price` (VIF = 35.9). |
| `expiry_lag_days` | 1 | Zero outcome signal | **Section 3.2 Rule 6:** Lag has identical spread in spoiled vs unspoiled batches ($r < 0.004$ with all operational drivers). |
| M2 Velocity & Sell-Through (`sales_velocity`, `sales_velocity_per_day`, `sell_through_risk`, `sell_through_ratio`, `at_risk`) | 5 | Sell-out logic, not spoilage predictors | **Section 1 (Item 3), 2.2, 3.1 & 3.4:** Flagged batches spoil at 19.5% vs 19.3% for normal batches ($\chi^2$ $p = 0.357$). All AUCs between 0.503 and 0.513. |
| M2 Store Markdown Freq (`store_md_freq_hist`, `store_md_freq_smoothed`) | 2 | Zero classification signal | **Section 1 (Item 5), 2.4 & 3.1:** Store markdown rate spread (sd 0.0104) equals chance (0.0103). AUC = 0.501 against spoilage. Assigned strictly to Track 3B / M3-T05. |
| **Total Features Pruned** | **90** | | **Reduces feature space from 180 to 90 high-signal predictors.** |

### 2.3 `TRACK_3B_FEATURES` (43 features)
Inputs for the revenue-response regression model (predicting profit and demand response as a function of discount depth):
- **Treatment Variable (1):** `discount_pct`.
- **Confounding Controls (5):**
  - `days_until_expiry`: `days_until_expiry`
  - `quality_grade`: `quality_grade_ord` (ordinal-encoded A=1, B=2, C=3)
  - `sales_velocity`: `sales_velocity`
  - `store_md_freq_hist`: `store_md_freq_hist`
  - `is_promoted`: `is_promoted`
- **Batch Accounting & Physical Drivers (17):**
  - Profit accounting inputs: `initial_quantity`, `base_price`, `cost_price`.
  - Demand & shelf-life: `daily_demand`, `demand_variability`, `expiry_remaining_ratio`.
  - Storage & handling: `storage_temp`, `temp_deviation`, `distribution_hours`, `temp_abuse_events`, `handling_score`, `packaging_score`, `supplier_score`, `spoilage_sensitivity`.
  - Sell-out risk: `sell_through_risk`, `sell_through_ratio`, `at_risk`.
- **Calendar & Cyclical (5):** `is_weekend`, `month_sin`, `month_cos`, `dow_sin`, `dow_cos`.
- **One-Hot Encoded Entities (15):** `category_*` (10 columns) and `region_*` (5 columns).
- **Pending Input:** Track 3A out-of-fold predicted spoilage probability will be joined to this set upon completion of task M2-T12.
- **Strict Anti-Leakage Exclusions:** Contains `discount_pct` as the sole discount variable; contains none of `selling_price`, `markdown_applied`, `spoilage_risk`, or post-sale outcome columns (`units_sold`, `units_wasted`, `waste_pct`, `revenue`, `waste_cost`, `profit`, `profit_margin_pct`).

---

## 3. Dataset Alignment & Spot-Check Verification

### 3.1 1:1 Key-Based Alignment Proof
The consolidated dataset `data/processed/model_ready_m2_dataset.csv` was verified against `data/processed/model_ready_dataset.csv` and `data/processed/cleaned_dataset.csv`:
- Row counts: Exactly 100,000 in all three datasets.
- Key uniqueness: `record_id` is unique across all 100,000 rows (`nunique() == 100000`).
- Strict row-order identity: `(cleaned_df['record_id'] == model_ready_df['record_id']).all()` and `(model_ready_df['record_id'] == model_ready_m2_df['record_id']).all()` are **True**.
- Key-based joining: All M2 feature modules (`compute_velocity_and_sell_through`, `compute_store_markdown_frequency`, `fit_quality_handling_preprocessor`) explicitly return `record_id` and are joined to the baseline model-ready DataFrame via `merge(..., on="record_id", how="left")`.

### 3.2 Five-Record Spot-Check
Five random `record_id`s were selected across diverse stores and dates. Their `sales_velocity` and `store_md_freq_hist` were recomputed from scratch from `cleaned_dataset.csv` using plain first-principles logic and compared with `model_ready_m2_dataset.csv`:

| `record_id` | Store | Transaction Date | Metric | Manual Calculation | Pipeline Output | Status |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **75722** | S019 | 2024-07-06 | `sales_velocity` (category fallback)<br>`store_md_freq_hist` | 73.163636<br>0.274468 | 73.163600<br>0.274468 | **MATCH**<br>**MATCH** |
| **80185** | S033 | 2024-08-08 | `sales_velocity` (category fallback)<br>`store_md_freq_hist` | 81.738462<br>0.322581 | 81.738500<br>0.322581 | **MATCH**<br>**MATCH** |
| **19865** | S027 | 2023-05-27 | `sales_velocity` (category fallback)<br>`store_md_freq_hist` | 51.233480<br>0.259897 | 51.233500<br>0.259897 | **MATCH**<br>**MATCH** |
| **76700** | S044 | 2024-07-13 | `sales_velocity` (product_store_7d)<br>`store_md_freq_hist` | 21.428571<br>0.316832 | 21.428600<br>0.316832 | **MATCH**<br>**MATCH** |
| **92992** | S037 | 2024-11-12 | `sales_velocity` (product_store_7d)<br>`store_md_freq_hist` | 23.428571<br>0.326892 | 23.428600<br>0.326892 | **MATCH**<br>**MATCH** |

*All 5 spot-checked rows matched manual recomputations exactly down to floating-point rounding.*

---

## 4. Phase 3 Verification Results

The automated test suite in `scripts/M2_T11_build_m2_dataset.py` passed all required checks:

| Verification Item | Specification | Observed Result | Status |
| :--- | :--- | :--- | :---: |
| **(a) Total Row Count** | 100,000 rows | 100,000 rows | **PASS** |
| **(a) Train / Test Split Count** | 80,042 train / 19,958 test | 80,042 train / 19,958 test | **PASS** |
| **(a) Split Date Boundary** | Cutoff: 2024-08-07 | Train max: 2024-08-07, Test min: 2024-08-08 | **PASS** |
| **(b) NaN Accounting** | Strict edge boundaries | 576 boundary rows in velocity; 152 in store markdown | **PASS** |
| **(c) Velocity Coverage** | 74.0% fallback, 25.4% own window | 74.01% (74,009) fallback, 25.42% (25,415) window | **PASS** |
| **(d) Anti-Leakage Rules** | Zero outcome/leakage columns | Verified clean across all three feature lists | **PASS** |
| **(e) Temporal Causality Test** | Zero change on post-cutoff perturbation | Max absolute change: 0.0000000000 | **PASS** |
| **(f) Scaler Fit Isolation** | Scalers fitted only on train rows | Mean difference: 0.00e+00, Variance diff: 0.00e+00 | **PASS** |
| **(g) `scripts/verify_m1.py`** | 36/36 baseline checks | 36/36 passed cleanly | **PASS** |
| **(g) `scripts/check_presale_inputs.py`** | Accounting identities & pre-sale check | All identities verified (gain from outcomes = 0.000) | **PASS** |

### 4.1 Velocity Coverage Reconciliation
The velocity coverage statistics reconcile as follows:
- **Product-Store Window (25.42% vs 26.2%):** In Milestone 1, 26.2% (26,230 rows) was reported as the independent share of batches having `batches_in_window_7d > 0`. In `src/m2_velocity.py` and M2-T02, `use_own` enforces an additional data-hygiene guardrail: `pair_history_days >= 7`. This prevents newly introduced products with only 1–2 days of history from having their sales divided by 7. This moves 815 batches to the fallback, yielding **25.42% (25,415 batches)** for `product_store_7d` (reported as 25.4% in M2-T02).
- **Category-Store Fallback (74.01% vs 84.9%):** In Milestone 1, 84.9% (84,878 rows) measured the independent availability of category-store sales in trailing 7 days across all rows. In M2-T02 and M2-T11, category-store velocity is formulated as a cumulative daily average strictly evaluated as a **hierarchical fallback** when `use_own` is False. Under this mutually exclusive partition:
  - `product_store_7d`: **25,415 batches (25.42%)**
  - `category_store_avg`: **74,009 batches (74.01%)**
  - `no_history`: **576 batches (0.58%)**
  - **Sum: Exactly 100,000 batches (100.00%)**.
- **Conclusion:** There is zero mathematical discrepancy. The pipeline output matches `M2-T02.ipynb` output down to the single row.

---

## 5. Missing Values & Model Imputation Guidance

### 5.1 Partition Breakdown (Train vs Test)

| Feature | Total NaNs (%) | Train NaNs (%) [N=80,042] | Test NaNs (%) [N=19,958] | Root Cause |
| :--- | :---: | :---: | :---: | :--- |
| `sales_velocity` | 576 (0.58%) | 576 (0.72%) | **0 (0.00%)** | Earliest dates of 2023 (< 7 days of store/category history). |
| `sales_velocity_per_day` | 576 (0.58%) | 576 (0.72%) | **0 (0.00%)** | Earliest dates of 2023 (< 7 days of store/category history). |
| `sell_through_risk` | 2,603 (2.60%) | 2,192 (2.74%) | **411 (2.06%)** | Zero velocity with positive stock (division by zero) + 576 unseeded batches. |
| `sell_through_ratio` | 576 (0.58%) | 576 (0.72%) | **0 (0.00%)** | Earliest dates of 2023 (`sell_through_risk` zero-velocity batches mapped to 10.0). |
| `at_risk` | 576 (0.58%) | 576 (0.72%) | **0 (0.00%)** | Earliest dates of 2023 (`no_history`). |
| `store_md_freq_hist` | 152 (0.15%) | 152 (0.19%) | **0 (0.00%)** | First transaction batch recorded for each store. |
| `store_md_freq_smoothed` | 148 (0.15%) | 148 (0.18%) | **0 (0.00%)** | Day 1 store transactions before global rate initialization. |
| `store_md_freq_90d` | 152 (0.15%) | 152 (0.19%) | **0 (0.00%)** | First transaction batch recorded for each store. |
| `store_md_freq_finished` | 552 (0.55%) | 552 (0.69%) | **0 (0.00%)** | Transactions occurring before any historical batch of that store had expired. |

### 5.2 Guidance for Milestone 3 Model Owners
- **Tree-Based Models (LightGBM, XGBoost):** Can consume the dataset directly without imputation. Both libraries handle missing values natively by routing NaNs to the optimal split direction determined during training.
- **Linear & Non-Tree Models (LogisticRegression, Ridge, Lasso, RandomForestClassifier):** Require explicit missing-value imputation. Scikit-learn estimators will raise `ValueError: Input contains NaN`. Imputation must be fitted strictly on the training period (e.g., using median values computed on `split == 'train'`).
- **Feature Selection Safeguard:** In `TRACK_3A_FEATURES_COMPACT`, all velocity and markdown features are pruned, resulting in **exactly zero missing values across all 90 features** in both training and test partitions.

---

## 6. Deviations from Plan

1. **Compact Feature Set Definition:** Pruned 90 features from `TRACK_3A_FEATURES_FULL` to form `TRACK_3A_FEATURES_COMPACT` based on rigorous empirical evidence from M2-T01, M2-T05, and M2-T06 (dropping entities with Cramér's $V < 0.05$ and numeric factors failing Holm-Bonferroni correction).
2. **Velocity Limitation (M3 Design Decision #8):** In the Kaggle dataset, `units_sold` is recorded as a batch's lifetime total attributed to its transaction date. When an earlier batch is counted in a trailing 7-day window, its lifetime sales total is used. Research in M2-T02 proved this carries negligible leakage ($r = 0.007$ with current batch pace). Per team decision #8, this limitation is accepted, documented, and will be included in the Final Report.
3. **NaN Handling:** Missing values were verified to be non-random and confined to unseeded historical boundaries (except division by zero in `sell_through_risk` for zero-velocity batches). Noticeably, test-set missingness is 0.00% across all features except `sell_through_risk`. Tree models require no imputation, whereas linear models require median imputation fitted strictly on training data.
4. **Discount-Tier Flags Excluded from Model Features:** To prevent target leakage in Track 3A and multicollinearity in Track 3B (where `discount_pct` is the designated sole discount input), the M2-T04 tier columns (`discount_tier`, `is_light`, `is_medium`, `is_deep`) were strictly excluded from all model feature sets.

---

## 7. Baseline File Integrity Confirmation

As required by repository governance, baseline Milestone 1 files were checked and confirmed untouched:
```powershell
Get-Item data/processed/model_ready_dataset.csv, data/processed/feature_dictionary.csv, models/preprocessor.joblib | Select-Object Name, LastWriteTime
```
- `data/processed/model_ready_dataset.csv`: LastWriteTime = `10/9/2026 3:24:30 PM` (Unmodified)
- `data/processed/feature_dictionary.csv`: LastWriteTime = `10/9/2026 3:24:30 PM` (Unmodified)
- `models/preprocessor.joblib`: LastWriteTime = `10/9/2026 3:24:30 PM` (Unmodified)
