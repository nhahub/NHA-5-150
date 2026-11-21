# M3-T13: Milestone 3 Model & Code Manifest

**Task:** M3-T13 — Model, Artifact, Script & Runtime Manifest  
**Branch:** `M3-T13-m3-deliverable-reports`

---

## 1. Saved Models, Preprocessors & JSON Metadata Artifacts

| Artifact Path | Produced By | Features / Schema | Training Period | Tracked in Git? |
| :--- | :--- | :--- | :--- | :--- |
| `models/preprocessor.joblib` | `notebooks/M1-T04.ipynb` | `ColumnTransformer` (`35` raw -> `178` M1 features) | `<= 2024-08-07` (`80,042`) | No (`*.joblib`) |
| `models/preprocessor_m2.joblib` | `scripts/M2_T11_build_m2_dataset.py` | Dict of `5` z-score pipelines (`temp_abuse_events`, `handling_score`, `packaging_score`, `supplier_score`, `spoilage_sensitivity`) | `<= 2024-08-07` (`80,042`) | No (`*.joblib`) |
| `models/track_3a_classifier.joblib` & `.json` | `scripts/M3_T01_train_track3a.py` | `TRACK_3A_FEATURES_COMPACT` (`90` features) | `<= 2024-08-07` (`80,042`) | No (`*.joblib`; `.json` trackable) |
| `models/track_3b_region_support_ref.json` | `scripts/M3_T05_addendum_r1.py` | `R0`/`R1` rules, `10` `(grade_C × R1_expiry_tercile × depth_bin)` support counts, discount scaler params | `<= 2024-08-07` (`24,475` R1) | Yes (`*.json` not ignored) |
| `models/track_3b_regressor.joblib` & `.json` | `scripts/M3_T06_train_track3b.py` | `TRACK_3B_FEATURES` + `oof_proba` (`44` features) | `<= 2024-08-07` (`24,475` R1) | No (`*.joblib`; `.json` trackable) |
| `models/track_3c_tier_edges.json` | `scripts/M3_T10_tier_mapping.py` | `edges` (`0.15, 0.25, 0.35`), CDF, train counts & SHA-256 | `<= 2024-08-07` (`24,475` R1) | Yes (`*.json` not ignored) |
| `models/track_3_inference_defaults.json` | `scripts/M3_T14_build_and_verify.py` | Category medians/modes, store-region map, `product×store` & `category×store` history medians (`583,751` bytes) | `<= 2024-08-07` (`80,042`) | Yes (`*.json` not ignored) |

---

## 2. Exact Script Run Order (From Raw Data `data/raw/perishable_goods_management.csv`)

1. `notebooks/M1-T04.ipynb` (execute all cells) -> `cleaned_dataset.csv`, `model_ready_dataset.csv`, `feature_dictionary.csv`, `models/preprocessor.joblib`
2. `python scripts/M2_T11_build_m2_dataset.py` -> `model_ready_m2_dataset.csv`, `feature_dictionary_m2.csv`, `models/preprocessor_m2.joblib`
3. `python scripts/M2_T12_decision_checks.py` -> `track_3a_oof_train.csv` (v1)
4. `python scripts/M3_T01_train_track3a.py` -> `models/track_3a_classifier.joblib` & `.json`
5. `python scripts/M3_T02_closeout_track3a.py` -> `track_3a_oof_train_v2.csv`, `track_3a_proba_test.csv`
6. `python scripts/M3_T03_explain_demo.py` -> explainability validation & figures
7. `python scripts/M3_T04_build_and_diagnose.py` -> `model_ready_3b_dataset.csv`
8. `python scripts/M3_T05_diagnose_confounding.py` -> Pass 1 confounding diagnostics
9. `python scripts/M3_T05_addendum_r1.py` -> `models/track_3b_region_support_ref.json`
10. `python scripts/M3_T06_train_track3b.py` -> `models/track_3b_regressor.joblib` & `.json`
11. `python scripts/M3_T07_evaluate_track3b.py` -> single test evaluation (`lgbm_ce_monotone`)
12. `python scripts/M3_T08_elasticity_view.py` -> `reports/tables/M3_T08_category_elasticity.csv`
13. `python scripts/M3_T09_run_optimizer.py` -> `optimizer_recs_train.csv`, `optimizer_recs_test.csv`
14. `python scripts/M3_T10_tier_mapping.py` -> `models/track_3c_tier_edges.json`
15. `python scripts/M3_T11_validate_optimizer.py` -> Lane 1 & Lane 2 validation results
16. `python scripts/M3_T12_confidence_propagation.py` -> `optimizer_recs_with_confidence_train.csv`, `optimizer_recs_with_confidence_test.csv`
17. `python scripts/M3_T14_build_and_verify.py` -> `models/track_3_inference_defaults.json`, `data/processed/inventory_snapshot.csv`, parity & robustness verification

---

## 3. Canonical Entry Points (`src/`)

| Entry Point | Module | Role |
| :--- | :--- | :--- |
| `score_batches(df)` | `src/m3_t01_track3a.py` | Frozen Track 3A calibrated spoilage probability |
| `explain_batch(df, top_k=5)` | `src/m3_t03_explain.py` | Exact linear log-odds/probability-point drivers + `<20` limited-history flag |
| `score_whatif(batch_df)` | `src/m3_t09_optimizer.py` | 14-depth supported profit curve (`0.10..0.75`) via `predict_profit_curve()` |
| `optimize_batches(batch_df)` | `src/m3_t09_optimizer.py` | Profit-maximizing supported depth (shallowest tie-break) + `R0` guard + `flat_curve` |
| `assign_tier(df)` | `src/m3_t10_tiers.py` | Maps `recommended_depth` to `Low`/`Medium`/`High`/`Critical` (`R0` -> `None`) |
| `attach_confidence(recs_df, batch_df)` | `src/m3_t12_confidence.py` | Weakest-link confidence (`No recommendation`, `Low`, `Moderate`) + `GLOBAL_CAVEAT` |
| `run_pipeline(raw_df, mode)` | `src/m3_t14_pipeline.py` | End-to-end `validate_input` -> `preprocessor` -> 3A -> 3B/3C -> tiers -> confidence + P3 |
| `simulate(base_batch, overrides, chosen_depth)` | `src/m3_t14_pipeline.py` | Single-batch `"hypothetical"` Simulator mode + depth evaluation |
| `manager_view_table(snapshot_df)` | `src/m3_t14_pipeline.py` | Formats Manager View summary strip + sorted inventory table |

---

## 4. Streamlit Runtime Dependencies (Milestone 4)

- **Gitignored binaries (`models/*.joblib`):** `preprocessor.joblib`, `track_3a_classifier.joblib`, `track_3b_regressor.joblib` (plus `preprocessor_m2.joblib` if M2 regeneration is run). Obtained by running the regeneration pipeline above or pulling from an external artifact store / MLflow.
- **Gitignored data CSVs (`data/processed/*.csv`):** `inventory_snapshot.csv` (Manager View default input), `cleaned_dataset.csv`, and `model_ready_m2_dataset.csv` (used at runtime by `m3_t03_explain.py`, `m3_t12_confidence.py` training support tables, and `"real"` mode M2 history lookup). Obtained by running M1/M2/T14 scripts or pre-exporting compact support tables.
- **JSON metadata & tables:** `models/track_3b_region_support_ref.json`, `models/track_3c_tier_edges.json`, `models/track_3_inference_defaults.json`, `models/track_3a_classifier.json`, `models/track_3b_regressor.json`, and `reports/tables/M3_T08_category_elasticity.csv` (all trackable in git or regenerated by scripts).

---

## 5. Open Known Issues (Carried from `README.md`)

1. **Hardcoded Paths in Earlier M2 Notebooks:** `M2-T02.ipynb`, `M2-T03__Sell_Through_Risk.ipynb`, `M2_T04_discount_flags.ipynb`, `M2-T07.ipynb`, and `M2_T08_store_markdown_frequency.ipynb` still use relative paths instead of `src.config`.
2. **Gitignore Scope for `reports/tables/*.csv`:** `.gitignore` ignores `*.csv` except explicitly whitelisted files (`!reports/tables/M2_T09_mlflow_runs_*.csv`, `!reports/tables/M3_T08_category_elasticity.csv`).
3. **Unexported M2-T10 Auxiliary Tables:** Four auxiliary tables noted in `reports/M2/M2_T10_figure_notes.md` were never exported to `reports/tables/`.
4. **Top-to-Bottom Notebook Execution:** Earlier exploratory notebooks have not been systematically re-executed end-to-end since Milestone 1.

---

## 6. Fresh Clone Quick Start

1. **Environment & Raw Data:** Create the Python 3.11 environment (`pip install -r requirements.txt`) and place the Kaggle *Perishable Goods Management* CSV at `data/raw/perishable_goods_management.csv` (`100,000 × 42`, SHA-256 `de94302b867c9debedfd45c431306623fdfc038f5ed8ca17736339b4460a6674`).
2. **Manual Step (`M1-T04`):** Open `notebooks/M1-T04.ipynb` in Jupyter and run all cells to generate `data/processed/cleaned_dataset.csv`, `model_ready_dataset.csv`, `feature_dictionary.csv`, and `models/preprocessor.joblib` (discard any notebook metadata diff if run via `nbconvert`).
3. **CLI Pipeline Sequence (in exact order):**
   ```bash
   python scripts/M2_T11_build_m2_dataset.py
   python scripts/M2_T12_decision_checks.py
   python scripts/M3_T01_train_track3a.py
   python scripts/M3_T02_closeout_track3a.py
   python scripts/M3_T03_explain_demo.py
   python scripts/M3_T04_build_and_diagnose.py
   python scripts/M3_T05_diagnose_confounding.py
   python scripts/M3_T05_addendum_r1.py
   python scripts/M3_T06_train_track3b.py
   python scripts/M3_T07_evaluate_track3b.py
   python scripts/M3_T08_elasticity_view.py
   python scripts/M3_T09_run_optimizer.py
   python scripts/M3_T10_tier_mapping.py
   python scripts/M3_T11_validate_optimizer.py
   python scripts/M3_T12_confidence_propagation.py
   python scripts/M3_T14_build_and_verify.py
   ```
4. **What a Fresh Clone Cannot Do Until the Artifacts Decision Is Made:**
   - Cannot import/run `score_batches()`, `explain_batch()`, `optimize_batches()`, `run_pipeline()`, `simulate()`, `manager_view_table()`, or the Streamlit app immediately after `git clone` because `models/*.joblib` (`preprocessor.joblib`, `track_3a_classifier.joblib`, `track_3b_regressor.joblib`) and `data/processed/*.csv` (`cleaned_dataset.csv`, `model_ready_m2_dataset.csv`, `inventory_snapshot.csv`) are gitignored.
   - Cannot run `scripts/verify_m1.py` or `scripts/check_presale_inputs.py` until Step 1–2 (`M1-T04.ipynb`) completes.
