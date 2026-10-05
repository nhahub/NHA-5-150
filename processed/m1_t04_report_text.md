**Preprocessing & Feature Encoding (M1-T04)**

**Dates.** `transaction_date` and `expiration_date` were stored as `YYYY-MM-DD` text and parsed with an explicit
`%Y-%m-%d` format. All 100,000 values parsed; none have a time component; no expiration date precedes its transaction date.

**Existing temporal columns were verified, not rebuilt.** Against `transaction_date`: `day_of_week` has
0 mismatches (Monday = 0), `is_weekend` has
0 (Saturday and Sunday), `month` has
0. All three were kept as delivered.
`days_remaining_at_purchase` equals the date gap in every row, whereas `days_until_expiry` is 0-3 days lower in
74,191 rows and is undocumented; it was kept unchanged and its gap exposed as `expiry_lag_days`.

**New temporal features.** `days_since_receipt` (= transaction_date - (expiration_date - shelf_life_days); range
0-243 days), `shelf_life_used_ratio`, `expiry_remaining_ratio` (the proposal's remaining-shelf-life ratio, built from `days_until_expiry`), `expiry_lag_days`,
`quarter`, `week_of_year`, `day_of_month`, `days_since_data_start`, and sin/cos encodings of `month` and `day_of_week`.
The four calendar features (`quarter`, `week_of_year`, `day_of_month`, `days_since_data_start`) are kept in the cleaned
dataset but not used as model features: EDA found no trend or seasonality, and `days_since_data_start` cannot extrapolate to later dates.

**Encoding.** One-hot: category, region, product_name, store_id, supplier_id (148 columns).
Ordinal: quality_grade (C=1, B=2, A=3). `product_id` (45,301 unique values) is not used as a feature.

**Scaling.** 11 right-skewed numeric columns use log1p then standardisation, 12 use standardisation;
binary and sin/cos columns are unchanged. Scalers and encoders were fitted on the 80,042
training rows only (time-ordered 80/20 split: train up to 2024-08-07, test after).

**Leakage control.** `was_spoiled`, `spoilage_risk` and the seven outcome columns are excluded from the
178-column feature matrix and kept as separate unscaled columns.

**Flags for Track 3A.** `markdown_applied`, `discount_pct` and `selling_price` are discount treatments: they stay in the matrix
and are flagged `exclude (treatment)` for the Track 3A classifier. For Track 3B and 3C only `discount_pct` is an input
(the treatment); `selling_price` and `markdown_applied` are deterministic functions of it, so the dictionary flags them
`never` in `track_3b_use`.

**Redundancy.** The dictionary's `redundant_with` column lists features that duplicate each other: exact relations (for example `shelf_life_used_ratio`, `selling_price`, `quality_grade_ord`) and 6 pairs with |r| above 0.9 on the scaled training rows. Nothing is dropped; tree models are unaffected, linear models should keep one feature per group. `spoilage_sensitivity` was held for the M1-T06 leakage check, which cleared it as a legitimate
input feature (flagged `allowed`).
