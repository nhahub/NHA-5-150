# M3-T14 Report: Track 3 Inference Pipeline & Inventory Snapshot

**Task:** M3-T14 — End-to-end Track 3 inference pipeline and Manager View inventory snapshot  
**Branch:** `M3-T14-inference-pipeline`  
**MLflow Experiment:** `smart-shelf-3c-discount-optimizer` (run `M3-T14-inference-pipeline`)  
**Code & Artifacts:**
- Module: [src/m3_t14_pipeline.py](file:///d:/DEPI/Project/Github%20Repo/NHA-5-150/src/m3_t14_pipeline.py)
- Verification script: [scripts/M3_T14_build_and_verify.py](file:///d:/DEPI/Project/Github%20Repo/NHA-5-150/scripts/M3_T14_build_and_verify.py)
- Training-only defaults: [models/track_3_inference_defaults.json](file:///d:/DEPI/Project/Github%20Repo/NHA-5-150/models/track_3_inference_defaults.json) (`583,751` bytes / `570.1 KB`)
- Inventory snapshot: [data/processed/inventory_snapshot.csv](file:///d:/DEPI/Project/Github%20Repo/NHA-5-150/data/processed/inventory_snapshot.csv) (`145` rows, `45` pre-decision columns)
- Summary JSON: [reports/M3/m3_t14_results_summary.json](file:///d:/DEPI/Project/Github%20Repo/NHA-5-150/reports/M3/m3_t14_results_summary.json)

---

## 1. Pre-Registered Design Decisions (P1–P5)

- **P1 (Input Contract & Feature Lineage):** Raw batch inputs require 5 mandatory fields (`category`, `days_until_expiry`, `initial_quantity`, `base_price`, `cost_price`). 19 optional operational/identity fields fall back to training-only category medians/modes when omitted or `NaN`. Derived ratios and M2 history features are computed without refitting any frozen transformer.
  - In `mode="real"`, precomputed causal M2 history is looked up by `record_id` from `data/processed/model_ready_m2_dataset.csv` when present, falling back to training medians otherwise.
  - In `mode="hypothetical"`, `sales_velocity`, `sales_velocity_per_day`, and `store_md_freq_hist` come from training-only medians (`product_name × store_id` -> `category × store_id` -> `category`), while `sell_through_risk`, `sell_through_ratio`, and `at_risk` are recomputed dynamically via `src.m2_velocity.sell_through()` so simulator overrides (`initial_quantity`, `days_until_expiry`) propagate end-to-end.
- **P2 (Training-Only Defaults Artifact):** `build_and_save_defaults()` computes all category medians, categorical modes, store-to-region mappings, and historical velocity/markdown medians strictly from `split == 'train'` (`transaction_date <= 2024-08-07`, `N = 80,042`) and saves them to `models/track_3_inference_defaults.json`. Contains aggregated medians/modes only (no row-level data, no usernames, no local paths). Regenerated via `python scripts/M3_T14_build_and_verify.py` (or `python -c "from src.m3_t14_pipeline import build_and_save_defaults; build_and_save_defaults()"`).
- **P3 (Validation & Defaulted-Input Low Confidence Rule):**
  - Empty DataFrames, missing/NaN required fields, negative/impossible numeric values (`days_until_expiry < 0`, `initial_quantity <= 0`, `base_price <= 0`, `cost_price < 0`, `discount_pct` outside `[0, 1)`), invalid `quality_grade` (not in `{A, B, C}`), or unknown `category` raise `ValueError`.
  - Unknown `product_name` or `store_id` values do not raise; they receive category defaults, set `unknown_entity = True`, and force `confidence_level = "Low"` (`"unknown_entity"`) on R1 batches.
  - `defaults_dominate = True` triggers when `(n_optional_defaulted / 19 > 0.50)` **OR** when any key sensitive field (`packaging_score`, `handling_score`, `days_until_expiry`, `days_since_receipt`, `temp_deviation`) was defaulted, forcing `confidence_level = "Low"` (`"defaults_dominate"`) on R1 batches and never raising a level (`R0` remains `"No recommendation"`).
- **P4 (Inventory Snapshot):** Latest test date with `>= 100` batches is **`2024-12-31`** (`N = 145` batches: `38` R1, `107` R0), exported to `data/processed/inventory_snapshot.csv` with all 12 post-decision/outcome leakage columns removed (`was_spoiled`, `spoilage_risk`, `units_sold`, `units_wasted`, `waste_pct`, `revenue`, `waste_cost`, `profit`, `profit_margin_pct`, `markdown_applied`, `discount_pct`, `selling_price`).
- **P5 (Global Caveat):** Every pipeline output and Manager View summary strip carries `GLOBAL_CAVEAT`: *"Advisory estimate. The observed-data backtest was inconclusive; see the validation report."*

---

## 2. Lineage Audit Table

| Raw Field | Model Feature(s) | Contract Tier | Hypothetical Single-Row Computable? |
|---|---|---|---|
| `category`, `days_until_expiry`, `initial_quantity`, `base_price`, `cost_price` | Scaled/one-hot features in `preprocessor.joblib` + profit formula | **Required (5)** | Yes |
| `shelf_life_days`, `days_remaining_at_purchase`, `days_since_receipt`, `storage_temp`, `temp_deviation`, `temp_abuse_events`, `distribution_hours`, `handling_score`, `packaging_score`, `supplier_score`, `spoilage_sensitivity`, `daily_demand`, `demand_variability` | Scaled numeric features in `preprocessor.joblib` | **Optional-with-default (13)** | Yes (falls back to training category median) |
| `product_name`, `store_id`, `region`, `supplier_id`, `quality_grade`, `transaction_date` | One-hot features + `quality_grade_ord` + `is_weekend` + `month_sin/cos`, `dow_sin/cos` | **Optional-with-default (6)** | Yes (falls back to training category mode / `2024-08-07`) |
| `expiry_remaining_ratio`, `shelf_life_used_ratio`, `expiry_lag_days`, `selling_price` | Scaled numeric features + R0/R1 scenario rule | **Derived** | Yes |
| `sales_velocity`, `sales_velocity_per_day`, `store_md_freq_hist` | Track 3B history features | **History-derived (M2)** | **No** (cannot be computed from 1 isolated row; looked up from training medians in `track_3_inference_defaults.json`) |
| `sell_through_risk`, `sell_through_ratio`, `at_risk` | Track 3B sell-through features | **History + Batch derived** | Yes (computed from looked-up `sales_velocity_per_day` + row `initial_quantity` & `days_until_expiry`) |

---

## 3. Parity Verification & Documented Tolerance Deviation

`run_pipeline(..., mode="real")` was evaluated against `data/processed/track_3a_proba_test.csv` and `data/processed/optimizer_recs_with_confidence_test.csv`:

| Check | `test_random_1000` (313 R1 / 687 R0) | `snapshot_2024-12-31` (38 R1 / 107 R0) | In-Memory M2 Direct (all `19,958` test rows) |
|---|---|---|---|
| `spoil_proba` max abs diff | `3.61e-07` (`< 1e-5`) | `2.94e-07` (`< 1e-5`) | **`1.11e-16`** (`< 1e-9`) |
| `spoil_proba` mean abs diff | `7.18e-08` | `6.06e-08` | — |
| `region`, `status`, `tier`, `confidence_level`, `confidence_reasons` match | **100.0%** (`1.0`) | **100.0%** (`1.0`) | — |
| R1 `recommended_depth` exact match | **100.0%** (`1.0`) | **100.0%** (`1.0`) | — |
| R1 `predicted_profit_at_rec` max abs diff | `1.82e-12` | `2.27e-13` | — |
| Wall-clock runtime | `2.622 s` | `0.150 s` | — |

> [!NOTE]
> **Documented Deviation (Probability Tolerance `3.61e-07` vs `1e-9`):**  
> `model_ready_m2_dataset.csv` was serialized during M1/M2 with 6-significant-digit float formatting (e.g. `base_price = -1.23313` vs full float64 `-1.2331269340527025` from `preprocessor.joblib.transform(cleaned_dataset.csv)`). Across all `19,958` test rows, 27 continuous/cyclical features differ by up to `5.000e-06` (`base_price: 5.000e-06`, `cost_price: 4.999e-06`, `days_remaining_at_purchase: 4.998e-06`, `distribution_hours: 4.997e-06`, `selling_price: 4.996e-06`), propagating to `3.61e-07` on `test_random_1000` (`2.94e-07` on snapshot, `5.20e-07` across all `19,958` test rows). When scoring the stored `model_ready_m2_dataset.csv` feature matrix directly without re-transforming against the 6-sig-fig CSV, max diff against `track_3a_proba_test.csv` is **`1.11e-16`**.

---

## 4. Hypothetical Examples & Counterfactual Sanity Check (`mode="hypothetical"`)

- **H1 (`H1_R0_fresh`):** `region=R0`, `status=no_markdown_historical_practice`, `spoil_chance_pct=19.74%`, `tier=None`, `recommended_depth=null`, `predicted_profit_at_rec=null`, `confidence_level=No recommendation`, `confidence_reasons=["no_discount_region"]`, `defaults_dominate=True`, `unknown_entity=True`.
- **H2 (`H2_R1_short_expiry`):** `region=R1`, `status=recommended`, `spoil_chance_pct=19.15%`, `tier=Medium`, `recommended_depth=0.25`, `predicted_profit_at_rec=10.3031`, `confidence_level=Low`, `confidence_reasons=["low_history_3a", "support_edge"]|unknown_entity|defaults_dominate`.
- **H3 (`H3_Grade_C_long_shelf_life`):** `region=R1`, `status=recommended`, `spoil_chance_pct=17.67%`, `tier=Low`, `recommended_depth=0.15`, `predicted_profit_at_rec=93.7471`, `confidence_level=Low`, `confidence_reasons=["low_history_3a", "support_edge"]|unknown_entity|defaults_dominate`.
- **H4 (`H4_all_optional_defaulted`):** `region=R1`, `status=recommended`, `spoil_chance_pct=17.21%`, `tier=Low`, `recommended_depth=0.10`, `predicted_profit_at_rec=24.5764`, `confidence_level=Low`, `confidence_reasons=["support_edge"]|defaults_dominate`.
- **H5 (`H5_unseen_product_store`):** `region=R1`, `status=recommended`, `spoil_chance_pct=21.28%`, `tier=Medium`, `recommended_depth=0.25`, `predicted_profit_at_rec=13.7722`, `confidence_level=Low`, `confidence_reasons=["low_history_3a", "limited_depth_category", "support_edge"]|unknown_entity|defaults_dominate`.
- **Counterfactual sanity check (`handling_score`):** Frozen Track 3A coefficient is `-0.1408`. Increasing `handling_score` from `0.2` to `0.9` via `simulate()` lowers `spoil_proba` from `0.2764` to `0.2667` (`delta = -0.0097`), matching the model coefficient direction (`chosen_depth=0.25` at `0.9`: `supported=True`, `s_hat=0.7445`, `expected_units_sold=22.33`, `batch_profit=6.01`).

---

## 5. Snapshot Statistics, Manager View Gaps & Robustness

- **Snapshot (`2024-12-31`, `N = 145`):**
  - `R0` ("No recommendation"): `107` rows (**`73.79%`**, ~74% of the snapshot).
  - `R1` (recommendations available): `38` rows (`26.21%`).
  - **R1 Tier distribution:** `Low: 24`, `Medium: 12`, `High: 2`.
  - **R1 Confidence distribution:** `Low: 29`, `Moderate: 9`.
- **Section 5.4 Wireframe Fields Not Supplied by Pipeline (`manager_view_table`):**
  - Live `current_stock` depleted mid-day (only arrival `initial_quantity` / `stock_units` is available pre-sale).
  - Store-level financial `baseline_profit_at_0pct` comparison in `R1` (not estimated because `0%` is outside R1's `[10%, 75%]` support grid).
  - Realized post-action metrics (`units_sold`, `waste_cost`, `actual_profit`) — intentionally excluded to prevent leakage.
- **Robustness Checks (all passed):** `empty_df` (`ValueError`), `missing_required_col` (`ValueError`), `negative_days_until_expiry` (`ValueError`), `unknown_category` (`ValueError`), `invalid_quality_grade` (`ValueError`), `single_row` (`n_out=1`), `all_nan_optional_fields` (`Low`, `defaults_dominate=True`), `p3_key_field_default_packaging_score` (`Low`, `[]|defaults_dominate`), `duplicated_record_id` (`n_out=2`), `repeated_call_determinism` (`True`, exact `0.0` diff).

---

## 6. Milestone 4 Runtime Files & Deployment Options

| Runtime File Needed by Dashboard | Gitignored? | Options to Obtain (Teammate / Deployed App) |
|---|---|---|
| `models/track_3a_classifier.joblib` & `.json` | Yes (`models/*.joblib`) | (A) Run `scripts/M3_T01_...` training pipeline; (B) Download from shared object storage / MLflow artifacts; (C) Force-commit small model artifacts if repo policy allows |
| `models/track_3b_regressor.joblib` & `.json` | Yes (`models/*.joblib`) | Same as above (run M3-T06 script or pull from artifact store) |
| `models/preprocessor.joblib` & `preprocessor_m2.joblib` | Yes (`models/*.joblib`) | Run M1/M2 preprocessing scripts or pull from artifact store |
| `models/track_3b_region_support_ref.json`, `models/track_3c_tier_edges.json` | No (`*.json` not gitignored) | Already trackable in git (or regenerated by M3-T05 / M3-T10 scripts) |
| `models/track_3_inference_defaults.json` (`583,751` bytes) | No (`*.json` not gitignored) | Commit to git (only 570 KB, aggregated medians/modes only) **or** regenerate via `python scripts/M3_T14_build_and_verify.py` |
| `data/processed/inventory_snapshot.csv` | Yes (`data/processed/*.csv`) | Regenerate via `python scripts/M3_T14_build_and_verify.py` (requires `cleaned_dataset.csv`) or pull from storage |
| `data/processed/cleaned_dataset.csv` & `model_ready_m2_dataset.csv` | Yes (`data/processed/*.csv`) | Needed at runtime by `m3_t03_explain.py`, `m3_t12_confidence.py` (training support tables), and `"real"` mode M2 lookup; regenerate from M1/M2 pipeline or pre-export compact training support tables |
