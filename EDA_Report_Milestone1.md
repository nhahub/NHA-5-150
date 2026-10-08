# Smart Shelf: Milestone 1 EDA Report

**Dataset:** Perishable Goods Management (Kaggle, `likithagedipudi/perishable-goods-management`). The dataset card states that it was synthetically generated and may not reflect real-world data.
**File verified:** `perishable_goods_management.csv`, SHA-256 `de94302b867c9debedfd45c431306623fdfc038f5ed8ca17736339b4460a6674`
**Source tasks:** M1-T01 (verification), M1-T02 (markdown vs discount), M1-T03 (EDA), M1-T04 (preprocessing), M1-T06 (spoilage_sensitivity)

---

## 1. Summary

- The dataset has 100,000 rows and 42 columns, with no missing values, no duplicate rows and no impossible values.
- 19.4 % of batches spoiled (19,442 of 100,000). The classes are moderately imbalanced.
- `markdown_applied` and `discount_pct` are fully consistent. The contradiction reported in the v4 proposal was checked and not found (0 rows).
- `spoilage_sensitivity` is a legitimate input feature. It is a fixed property of the product category and is not leaking the target.
- Spoilage is only weakly linked to any single input. The clearest signals are `quality_grade`, `packaging_score`, `handling_score`, `temp_deviation` and the share of shelf life remaining. Region, promotion status, month and weekday show no meaningful effect.
- Waste is not the same thing as spoilage. Waste is highest for Frozen_Meals and Pharmaceuticals, which are not the highest-spoilage categories.
- A cleaned dataset and a model-ready dataset (178 features) were exported for Milestone 2.

---

## 2. Dataset overview

| Item | Value |
|---|---|
| Rows × columns | 100,000 × 42 |
| Period | 2023-01-01 to 2024-12-31 |
| Product categories | 10 |
| Stores / regions / suppliers | 50 / 5 / 20 |
| Product names | 63 |
| `product_id` | 45,301 unique values (identifier-like, not used as a feature) |
| Target | `was_spoiled` (0/1) |

**Units.** `discount_pct` is a fraction (0.25 = 25 %). `waste_pct` is a percent (25.0 = 25 %).

### Column-name corrections (M1-T01)

The project plan used some column names that differ from the file. The names in the file are the correct ones.

| Name in the plan | Name in the file |
|---|---|
| `markdown` | `markdown_applied` |
| `demand_volatility` | `demand_variability` |
| `days_until` | `days_until_expiry` |
| `days_remaining` | `days_remaining_at_purchase` |
| `profit_margin` | `profit_margin_pct` |

---

## 3. Data quality and integrity

### 3.1 Basic checks (M1-T01, M1-T03)

- Shape is exactly (100,000, 42) and the total number of missing values is 0. There are no blank text cells.
- `record_id` is unique, and there are 0 duplicate rows when it is ignored.
- All of these checks returned 0 rows: expiry before transaction date, negative `units_sold` or `units_wasted`, `discount_pct` outside 0–1, `waste_pct` outside 0–100, `was_spoiled` not 0/1, `selling_price` above `base_price`, and `units_sold + units_wasted` above `initial_quantity`.

### 3.2 Date columns

`days_remaining_at_purchase` equals `expiration_date − transaction_date` in all 100,000 rows. `days_until_expiry` is 0 to 3 days lower (mean 1.4 days) in 74,191 rows and is never higher. The gap is not explained by `distribution_hours`, so its origin is unknown and is recorded as a **data limitation**. `day_of_week`, `is_weekend` and `month` were checked against `transaction_date` and have 0 mismatches.

### 3.3 `markdown_applied` vs `discount_pct` (M1-T02)

The v4 proposal reported rows with `markdown_applied = 1` but `discount_pct = 0` (example: a Salami batch). On the downloaded file this does not occur.

| Check | Result |
|---|---|
| Rows with `markdown_applied = 1` | 30,651 |
| Rows with `markdown_applied = 1` and `discount_pct = 0` | **0** (0.00 % of all rows, 0.00 % of markdown rows) |
| Rows with `discount_pct > 0` and `markdown_applied = 0` | 0 |
| Rows where `selling_price` differs from `base_price` | 100 % of markdown rows, 0 % of non-markdown rows |
| Largest gap between implied discount (1 − selling/base) and `discount_pct` | 0.0049 (cent rounding), 0 rows outside tolerance |
| Salami rows with `markdown_applied = 1` | 465 rows, all with a discount between 0.10 and 0.75 |

**Interpretation.** The two fields are consistent: `markdown_applied` is exactly the indicator `discount_pct > 0`. This is not a contradiction, and no rows are excluded or flagged.

**Depth of markdowns.** The discount is never between 0 and 0.10. It ranges from 0.10 to 0.75 and takes 66 distinct values.

| Discount band | Rows | Share of markdown rows |
|---|---|---|
| 10–20 % | 9,248 | 30.2 % |
| 20–30 % | 9,155 | 29.9 % |
| 30–40 % | 2,274 | 7.4 % |
| 40–50 % | 3,531 | 11.5 % |
| 50–60 % | 3,185 | 10.4 % |
| 60–75 % | 3,258 | 10.6 % |

60.0 % of markdown rows have a discount of 30 % or less, and 6,443 rows (21.0 %) are above 50 %.

**Note for the plan.** The v4 proposal text should be corrected to say the quirk was checked and not found. For Track 3B and 3C, the candidate-discount grid should be 0 plus values within [0.10, 0.75].

---

## 4. Target and outcomes

![Target balance](../figures/01_target_balance.png)

- **Spoilage:** 19.4 % of batches spoiled (19,442 of 100,000). Use PR-AUC and recall, and class weights, rather than accuracy.
- **Waste and profit:** 42.6 % of batches have zero waste, and 28.7 % have negative profit.
- **Skew:** Many numeric columns are right-skewed (see `figures/02_numeric_histograms.png`).

### Outliers

The IQR rule was used (1.5 × IQR beyond the quartiles).

| Column | Outliers |
|---|---|
| `units_sold` | none |
| `units_wasted` | 5.2 % |
| `waste_pct` | 8.0 % |
| `revenue` | 9.8 % |
| `waste_cost` | 12.1 % |
| `profit` | 15.8 % |

Outliers are **kept**. They are real batches (total losses), not errors. The 100 worst batches (lowest profit −169,495) are all Pharmaceuticals.

![Outlier boxplots](../figures/04_outlier_boxplots.png)

---

## 5. What is linked to spoilage and waste

### 5.1 Category

![Spoilage by category](../figures/05_spoilage_by_category.png)

| Category | Spoilage rate | Average `waste_pct` |
|---|---|---|
| Frozen_Meals | 16.9 % | 50.0 % |
| Beverages | 17.7 % | 20.7 % |
| Bakery | 18.3 % | 15.3 % |
| Produce | 18.7 % | 18.5 % |
| Dairy | 18.9 % | 16.9 % |
| Deli | 19.4 % | 16.9 % |
| Seafood | 20.7 % | 17.4 % |
| Pharmaceuticals | 20.7 % | 37.5 % |
| Meat | 21.0 % | 18.2 % |
| Ready_to_Eat | 22.1 % | 14.6 % |

Frozen_Meals has the lowest spoilage rate but the highest waste. So waste is driven by something other than spoilage, mainly shelf life, storage temperature and demand.

### 5.2 Strength of the links (chi-square and Cramér's V)

With 100,000 rows almost every p-value is tiny, so Cramér's V (strength of the link) matters more than p.

| Variable | p-value | Cramér's V |
|---|---|---|
| `quality_grade` | < 0.001 | 0.091 |
| `markdown_applied` | < 0.001 | 0.058 |
| `category` | < 0.001 | 0.039 |
| `region` | 0.36 | 0.007 |
| `is_promoted` | 0.56 | 0.002 |

All are below 0.1, which counts as very weak. `region` and `is_promoted` are not significant.

**Quality grade.** The spoilage rate rises from grade A (15.3 %) to B (20.6 %) to C (25.9 %). According to the M1-T03 heatmaps the same order holds within every category. Region barely changes the rate in any category.

![Heatmaps](../figures/05_heatmaps.png)

### 5.3 Numeric columns

Linear correlations with `was_spoiled` are small:

| Column | Correlation |
|---|---|
| `packaging_score` | −0.088 |
| `markdown_applied` | +0.058 |
| `handling_score` | −0.056 |
| `discount_pct` | +0.047 |
| `temp_deviation` | +0.043 |
| `spoilage_risk` (itself) | +0.131 |

Splitting the batches into groups (deciles where possible) shows small effects that are consistent in direction:

- The spoilage rate falls from about 24 % to 14 % as `packaging_score` rises.
- It falls from about 22 % to 16 % as `handling_score` rises.
- It rises from about 18 % to 23 % as `temp_deviation` rises.
- It falls from about 22 % to 18 % as the share of shelf life left (`days_until_expiry / shelf_life_days`) rises. This trend and the `temp_deviation` trend are noisier than the other two: the group-by-group line is not strictly monotone. The raw `days_until_expiry` has correlation −0.004 with spoilage, so the ratio is the right feature.

![Spoilage vs numeric columns](../figures/05_spoilage_vs_numeric.png)

For `waste_pct`, the strongest links are `shelf_life_days` (+0.31), `storage_temp` (−0.27) and `daily_demand` (−0.23).

### 5.4 Markdowns and spoilage

Marked-down batches spoil more often. This is consistent with discounts being given to batches that already look risky, and it does not show that discounts cause spoilage. It is a **hypothesis** for the Track 3B confounding check, not a result.

---

## 6. Correlations and redundancy

![Correlation heatmap](../figures/06_correlation_heatmap.png)

`spoilage_risk` and the outcome columns (`units_sold`, `units_wasted`, `waste_pct`, `waste_cost`, `revenue`, `profit`, `profit_margin_pct`) come from the same process as `was_spoiled` or happen after it. They show high correlations but cannot be model inputs.

Near-duplicate pairs (|r| > 0.85):

- `days_remaining_at_purchase` / `days_until_expiry` (1.000)
- `base_price` / `selling_price` (0.996)
- `shelf_life_days` with both day columns (0.992)
- `base_price` / `cost_price` (0.986)
- `cost_price` / `selling_price` (0.982)

---

## 7. Patterns over time

![Time patterns](../figures/07_time_patterns.png)

The spoilage rate stays between about 18 % and 20 % in every month from 2023-01 to 2024-12 and is flat across weekdays (about 19–20 %). The number of rows per month is steady (about 3,800 to 4,400). There is **no visible trend or seasonality**.

---

## 8. `spoilage_sensitivity` verification (M1-T06)

**Verdict: LEGITIMATE input feature (keep).**

- **What it is.** The column has 9 distinct values (0.30 to 0.95) and is constant inside every category (Frozen_Meals 0.30 … Seafood 0.95; Meat and Ready_to_Eat both 0.90). It is a fixed, pre-sale property of the category, not a per-batch measurement.
- **Correlations.** With `was_spoiled`, r = 0.037 (95 % CI 0.031 to 0.043), which explains about 0.1 % of the variance. With `spoilage_risk`, r = 0.254. It is an ingredient of `spoilage_risk`: adding it raises the R² of the risk score from 0.85 to 0.88.
- **Leakage tests.** It is known before the sale and is not derived from the target. Single-column test AUC is 0.530 (category alone 0.528, `spoilage_risk` 0.586). Adding it to a gradient-boosting model that already has `category` changes test AUC by +0.0005. None of the four leakage flags fired.
- **Collinearity.** It is 100 % explained by the `category` one-hot columns, so unregularised linear models should drop one of the two.
- **For M3-T08 (elasticity).** Across the 10 categories the column shows a descriptive association with markdown frequency (rho 0.64) and depth (rho 0.56), but no clear relation with demand response to discount (rho −0.38). With only 10 categories this is not a test. Use it only as a category-level segment or interaction term.

`spoilage_risk` stays reserved as an evaluation benchmark and is not an input to either track.

---

## 9. Preprocessing and cleaned dataset (M1-T04)

- **Dates** were parsed with the explicit format `%Y-%m-%d`. All 100,000 values parsed, and no expiration date precedes its transaction date.
- **New temporal features:** `days_since_receipt`, `shelf_life_used_ratio`, `expiry_lag_days` (the gap between `days_remaining_at_purchase` and `days_until_expiry`), `quarter`, `week_of_year`, `day_of_month`, `days_since_data_start`, and sin/cos encodings of `month` and `day_of_week`. The four calendar features are kept in the cleaned dataset but not used as model features, because EDA found no trend or seasonality and `days_since_data_start` cannot extrapolate to later dates.
- **Encoding.** One-hot for `category`, `region`, `product_name`, `store_id` and `supplier_id` (148 columns). Ordinal for `quality_grade` (C = 1, B = 2, A = 3).
- **Scaling.** 11 right-skewed columns use log1p then standardisation, and 12 use standardisation. Binary and sin/cos columns are unchanged.
- **Split.** Time-ordered chronological split by transaction_date: Train <= 2024-08-07 (80,042 rows, 19.35% spoilage), Test > 2024-08-07 (19,958 rows, 19.81% spoilage) to eliminate lookahead bias. Scalers and encoders were fitted on the training rows only.
- **Leakage control.** `was_spoiled`, `spoilage_risk` and the seven outcome columns are excluded from the 178-column feature matrix and kept as separate unscaled columns.
- **Flags for Track 3A.** `markdown_applied`, `discount_pct` and `selling_price` are discount treatments. They stay in the matrix for Tracks 3B and 3C and are flagged `exclude (treatment)` for the Track 3A classifier.

### Exported files (`processed/`)

| File | Content |
|---|---|
| `cleaned_dataset.csv` | 100,000 rows × 56 columns, 0 missing values, includes the train/test `split` label |
| `model_ready_dataset.csv` | 100,000 rows, 178 scaled and encoded features plus the unscaled target and outcome columns |
| `feature_dictionary.csv` | 189 rows describing each column's role, transformation and Track 3A use |
| `preprocessor.joblib` | fitted preprocessing pipeline |
| `m1_t04_preprocessing_report.json`, `m1_t06_verdict.json` | machine-readable summaries |

---

## 10. Interactive visualisations

The notebook **`M1-T05.ipynb`** builds nine interactive Plotly charts (hover, zoom, click-to-hide legend). Each chart checks its numbers against this report with `assert` statements, so the report and the charts cannot silently disagree. Charts are saved as small HTML files (Plotly loaded from a CDN) in `figures_interactive/`:

1. Spoilage rate by category.
2. Category × quality grade.
3. Monthly spoilage rate by category.
4. Spoilage across groups of four drivers (shelf life left, temperature deviation, handling, packaging).
5. Spoilage vs average waste by category.
6. Markdown depth and spoilage.
7. Profit distribution by category.
8. Category × region heatmap.
9. Correlation of each input with `was_spoiled`.

The M1-T03 notebook also contains three earlier interactive charts, saved as HTML in `figures/`:

- `interactive_spoilage_by_category.html`
- `interactive_monthly_by_category.html`
- `interactive_profit_distribution.html` (7 MB, because it holds all 100,000 rows; do not commit it)

---

## 11. Notes for Milestone 2 and limitations

**Modelling notes**

- Do not use `spoilage_risk` or the outcome columns as inputs.
- Use ratios such as `days_until_expiry / shelf_life_days`, and drop one column from each near-duplicate pair (days, prices).
- Expect **modest predictive power** from Track 3A, since every single input has a weak link to spoilage. Report this as a limitation.
- Treat `markdown_applied` and `discount_pct` as treatments in Tracks 3B and 3C, and check for confounding because marked-down batches already spoil more.

**Limitations**

- The dataset is synthetic and may not reflect real-world behaviour.
- The 0–3 day gap between `days_until_expiry` and `days_remaining_at_purchase` has an unknown origin.
- Linear correlations are small, so conclusions about drivers are descriptive and not causal.

---

## 12. Proposal Reconciliation & Key Empirical Findings

This section reconciles initial proposal assumptions (v4) against empirical findings verified by `verify_m1.py` (36/36 checks passed), `check_presale_inputs.py`, and the Milestone 1 notebooks.

### 12.1 Proposal Reconciliations

| # | Proposal Assumption | Verified Finding | Resolution / Recommendation |
|---|---|---|---|
| 1 | Potential contradiction: `markdown_applied = 1` with `discount_pct = 0` (e.g., Salami quirk) | **0 contradictory rows** across all 100,000 transactions. `markdown_applied` equals `discount_pct > 0` everywhere. | The two fields are fully consistent; no rows excluded or flagged. |
| 2 | Candidate discount grid searched across 0% to 50% | Discounts in the data take 66 distinct values strictly within **[0.10, 0.75]** (median 0.25). 21.0% (6,443 rows) are deeper than 50%. No discounts exist between 0 and 0.10. | Track 3C candidate grid must be $\{0\}$ plus discrete steps in $[0.10, 0.75]$. Note that support is thin above 70% (170 rows in (0.70, 0.75) with clipping at 0.75). |
| 3 | Remaining shelf-life ratio defined as `days_until_expiry / shelf_life_days` | `days_remaining_at_purchase` exactly equals the transaction-to-expiration date gap. `days_until_expiry` is **0 to 3 days lower in 74,191 rows (74.2%)**. | Exported both `shelf_life_used_ratio` (date gap, inverted) and `expiry_remaining_ratio` (proposal formula). They correlate at only 0.45 (mean abs difference 0.40 on items with $\le 5$ days shelf life). |
| 4 | Trailing 7-day sales velocity per product-store pair | Only **26.2%** of batches have an earlier sale of the same product in the same store in the trailing 7 days. | In Milestone 2, use the category × store fallback, which covers **84.9%** of batches. |
| 5 | Track 3A tuned primarily for high AUC / Recall | `was_spoiled` behaves as a random draw from `spoilage_risk` (observed rate rises smoothly from 11.1% to 28.9% across deciles). Honest features achieve test AUC **~0.587**; the leaky `spoilage_risk` benchmark reaches **0.593**. | Prioritize probability calibration (Brier score, reliability curve, log loss) alongside recall. Acknowledge the ~0.59 AUC ceiling. |
| 6 | Train/test data split | Saved split is **strictly time-ordered by transaction date**: Train $\le$ 2024-08-07 (80,042 rows, 19.35% spoilage), Test > 2024-08-07 (19,958 rows, 19.81% spoilage). | Eliminates temporal lookahead. Track 3B must generate out-of-fold Track 3A probabilities via time-blocked cross-validation inside the training window. |
| 7 | Operational point-of-sale data assumption | Dataset is synthetically generated (100,000 rows, 0% missing values, pre-computed risk score). | Results demonstrate the algorithmic methodology. Spoilage following risk scores reflects the generative process rather than organic retail noise. |

### 12.2 Pre-Sale Integrity & Accounting Identities

Tested via `check_presale_inputs.py` across candidate inputs (`daily_demand`, `demand_variability`, `initial_quantity`, `temp_abuse_events`, `distribution_hours`):
- **Independence from Outcomes:** Post-sale outcome columns (`units_sold`, `units_wasted`, `revenue`, `profit`) add zero explanatory power ($R^2$ gain $= 0.000$) on top of pre-sale features, proving candidate features do not embed post-sale results.
- **Accounting Identity:** $\text{initial\_quantity} = \text{units\_sold} + \text{units\_wasted}$ holds in **100.0% of rows**. $\text{initial\_quantity}$ is the known starting batch size, not a leak. Consequently, sales and waste share a single degree of freedom; Milestone 3 should model sell-through and derive waste rather than fitting them as independent targets.
- **Signal Strength:** `daily_demand` is 99% reconstructible from other pre-sale inputs ($R^2 = 0.990$), whereas `demand_variability` and `distribution_hours` exhibit virtually zero correlation with targets or outcomes.

### 12.3 High Collinearity & Redundancy Summary

The following 6 feature pairs exhibit $|r| > 0.90$ on the scaled training rows and are flagged bidirectionally in `feature_dictionary.csv`:
1. `shelf_life_days` $\sim$ `days_remaining_at_purchase` ($r = 0.996$)
2. `base_price` $\sim$ `cost_price` ($r = 0.991$)
3. `days_remaining_at_purchase` $\sim$ `days_until_expiry` ($r = 0.986$)
4. `shelf_life_days` $\sim$ `days_until_expiry` ($r = 0.982$)
5. `base_price` $\sim$ `selling_price` ($r = 0.979$)
6. `cost_price` $\sim$ `selling_price` ($r = 0.972$)

Additionally:
- `spoilage_sensitivity` is $100\%$ collinear with the 10 category one-hot columns ($R^2 = 1.0$).
- `shelf_life_used_ratio` is an exact linear complement of $\text{days\_remaining\_at\_purchase} / \text{shelf\_life\_days}$.
- `quality_grade_ord` is a deterministic bucket of `handling_score` and `packaging_score`.

