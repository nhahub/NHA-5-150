# Smart Shelf: Perishable Food Waste Reduction and Dynamic Markdown Optimizer

[![Milestone 1 Status](https://img.shields.io/badge/Milestone%201-Completed%20(36%2F36%20Checks%20Passed)-brightgreen)](#2-milestone-deliverables--task-map)
[![Milestone 2 Status](https://img.shields.io/badge/Milestone%202-Completed%20(T01--T13%20Verified%20%26%20Merged)-brightgreen)](#2-milestone-deliverables--task-map)
[![Milestone 3 Status](https://img.shields.io/badge/Milestone%203-Planned%20(Track%203A%2F3B%2F3C%20Modeling)-blue)](#2-milestone-deliverables--task-map)
[![Python Version](https://img.shields.io/badge/Python-3.11-blue)](requirements.txt)
[![Code Architecture](https://img.shields.io/badge/Architecture-End--to--End%20Pipeline%20%26%20Decision%20Engine-orange)](#4-repository-structure)

An end-to-end machine learning decision support system and interactive Streamlit application designed to reduce perishable food waste in retail grocery operations and maximize recovered revenue through calibrated spoilage risk estimation and dynamic markdown optimization.

---

## 1. Project Overview & System Architecture

Clearance decisions in retail food chains are frequently driven by manual staff inspection or rigid static markdown rules. This leads to two standard operational failure modes:

1. **Premature markdowns:** Items discounted unnecessarily early, sacrificing gross margin on products that would have sold at full price.
2. **Late markdowns:** Items discounted too close to expiration, failing to stimulate demand before expiration and resulting in 100% inventory write-off.

### Dataset Foundation

The project utilizes the Kaggle **Perishable Goods Management** dataset:

- **Dimensions:** 100,000 transaction rows × 42 columns, spanning January 2023 to December 2024 across 50 stores, 5 regions, 20 suppliers, 10 categories, and 63 distinct products.
- **Nature of Data:** The dataset is synthetic. Consequently, all reported business-impact figures, revenue gains, and waste reductions are simulated estimates derived from this synthetic data.

### System Architecture

Smart Shelf links two predictive machine learning models into a dynamic discount optimization engine:

```
                                      [Raw Batch Inputs]
                                               │
                                 ┌─────────────┴─────────────┐
                                 ▼                           ▼
                     [Pre-Sale & Handling Features]     [Candidate Discount Grid]
                     (temp_abuse, handling, lag, etc.)    ({0} ∪ [0.10, 0.75])
                                 │                           │
                                 ▼                           │
                        ┌──────────────────┐                 │
                        │ Track 3A Model   │                 │
                        │ Spoilage Risk    │                 │
                        └────────┬─────────┘                 │
                                 │                           │
                   Calibrated Spoilage Probability           │
                   (OOF probability in training)             │
                                 │                           │
                                 └─────────────┬─────────────┘
                                               ▼
                                      ┌──────────────────┐
                                      │ Track 3B Model   │
                                      │ Revenue Response │
                                      └────────┬─────────┘
                                               │
                                     Predicted Profit Curve
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │ Track 3C Engine  │
                                      │ Optimal Discount │
                                      └────────┬─────────┘
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │   Milestone 4    │
                                      │ Streamlit App &  │
                                      │ What-If Sim      │
                                      └──────────────────┘
```

1. **Track 3A — Spoilage Risk Classifier:** Predicts whether an inventory batch will spoil (`was_spoiled = 1`) before selling out. It relies strictly on pre-sale attributes and concrete handling quality signals (`temp_abuse_events`, `handling_score`, `packaging_score`, `supplier_score`, `quality_grade`). Because individual pre-sale features have low linear correlation with spoilage (test AUC ceiling ~0.59), Track 3A's headline performance metric is **probability calibration** (Brier score, log loss, reliability curve).
2. **Track 3B — Revenue-Response Regressor:** Models demand elasticity as a function of markdown depth (`discount_pct`), estimating the bounded sell-through rate $s \in [0, 1]$ or profit curve for candidate discounts. It adjusts for discounting bias and reverse causality by conditioning on Track 3A out-of-fold spoilage probabilities (added to inputs in Milestone 3), store historical markdown frequency, and sales velocity. Profit is derived from sell-through via the exact accounting identity:
   $$\text{profit} = \text{initial\_quantity} \times [s \times \text{selling\_price} - \text{cost\_price}]$$
3. **Track 3C — Dynamic Discount Optimizer:** Evaluates candidate discount depths on the empirical discrete grid ($\{0\} \cup [0.10, 0.75]$) to select the profit-maximizing discount tier for each batch. Audited via a two-lane evaluation framework:
   - **Lane 1 — Matched Evaluation (real evidence):** Evaluates historical batches where the optimizer's recommended discount happens to closely match the discount that was actually applied historically. Observed profit is compared with observed profit, providing real ground-truth evidence on a smaller, non-random subset of batches.
   - **Lane 2 — Counterfactual Scoring (simulated, self-evaluated):** Evaluates batches where the optimizer recommends a discount that was never actually tried historically. The outcome is evaluated by Track 3B's own prediction scoring its own recommendation, which is expected to look favorable almost by construction. It is clearly labeled as simulated / model-estimated, never presented as measured evidence, and never blended with Lane 1.
4. **Milestone 4 — Deployment & Simulator:** A Streamlit web application providing a real-time **Manager View** for operational store clearance and an interactive **Simulator Mode** for testing hypothetical pricing scenarios, supported by MLflow experiment tracking and model monitoring.
5. **Milestone 5 — Documentation & Presentation:** Final project report with separate matched and counterfactual business cases, worked example walkthrough, and stakeholder presentation.

---

## 2. Milestone Deliverables & Task Map

### Milestone 1: Data Audit, Leakage Boundary & Preprocessing Foundation

| Task ID | Component / Deliverable | Status | Description & Key Outcomes |
| :--- | :--- | :---: | :--- |
| **M1-T01** | [`notebooks/M1-T01.ipynb`](notebooks/M1-T01.ipynb) | **Done** | **Dataset Ingestion & Schema Audit:** Verified 100,000 transactions, 42 columns, and 0 missing values. Documented schema deviations and established strict data-leakage boundaries (`spoilage_risk` reserved for benchmark only; post-sale outcome columns excluded). |
| **M1-T02** | [`notebooks/M1-T02.ipynb`](notebooks/M1-T02.ipynb)<br>[`reports/M1/m1_t02_resolution.json`](reports/M1/m1_t02_resolution.json) | **Done** | **Discount & Markdown Integrity Check:** Verified 30,651 markdown rows (30.7%). Proved `markdown_applied == (discount_pct > 0)` in 100% of rows (disproving the 0-discount contradiction hypothesis). Confirmed `selling_price` matches `base_price * (1 - discount_pct)`. |
| **M1-T03** | [`notebooks/M1-T03.ipynb`](notebooks/M1-T03.ipynb) | **Done** | **Exploratory Data Analysis (EDA):** Generated static distributions, boxplots, heatmaps, and temporal charts in [`reports/figures/m1/`](reports/figures/m1/). Confirmed baseline spoilage rate of 19.44% and absence of multi-year drift or seasonality. |
| **M1-T04** | [`notebooks/M1-T04.ipynb`](notebooks/M1-T04.ipynb)<br>[`reports/M1/m1_t04_preprocessing_report.json`](reports/M1/m1_t04_preprocessing_report.json)<br>[`reports/M1/m1_t04_report_text.md`](reports/M1/m1_t04_report_text.md) | **Done** | **Preprocessing & Feature Encoding:** Built a scikit-learn `ColumnTransformer` (11 `log(1+x)` scaled, 12 standardized, 148 one-hot, 7 binary/cyclical features = 178 total features). Enforced a strict chronological split (Train: `<= 2024-08-07`, 80,042 rows; Test: `> 2024-08-07`, 19,958 rows) with zero lookahead bias. Scalers fit exclusively on training data. |
| **M1-T05** | [`notebooks/M1-T05.ipynb`](notebooks/M1-T05.ipynb)<br>[`reports/M1/EDA_Report_Milestone1.md`](reports/M1/EDA_Report_Milestone1.md) | **Done** | **Interactive Visualizations & Synthesis Report:** Created 9 CDN-backed interactive Plotly HTML charts in [`reports/figures/interactive/`](reports/figures/interactive/) with assertion-guarded metrics. Compiled the comprehensive Milestone 1 EDA & Preprocessing Report. |
| **M1-T06** | [`notebooks/M1-T06.ipynb`](notebooks/M1-T06.ipynb)<br>[`reports/M1/m1_t06_report_text.md`](reports/M1/m1_t06_report_text.md)<br>[`reports/M1/m1_t06_verdict.json`](reports/M1/m1_t06_verdict.json) | **Done** | **Spoilage-Sensitivity Verification:** Proved `spoilage_sensitivity` is a static category-level attribute (9 distinct values, $R^2 = 1.0$ against category one-hot). Cleared it as a valid pre-sale feature while documenting exact collinearity constraints. |

### Milestone 2: Statistical Analysis & Advanced Feature Engineering

| Task ID | Component / Deliverable | Status | Description & Key Outcomes |
| :--- | :--- | :---: | :--- |
| **M2-T01** | [`notebooks/M2_T01_statistical_analysis.ipynb`](notebooks/M2_T01_statistical_analysis.ipynb)<br>[`src/m2_t01_analysis.py`](src/m2_t01_analysis.py)<br>[`reports/M2/M2_T01_Spoilage_Drivers_Report.md`](reports/M2/M2_T01_Spoilage_Drivers_Report.md) | **Done** | **Statistical Analysis of Spoilage Drivers:** Completed Chi-Square tests of independence, ANOVA $F$, Welch's $t$, Mann-Whitney $U$, Cohen's $d$, and multiple-testing adjustments (Holm & Bonferroni) across 24 factors. Confirmed `quality_grade` is meaningful ($\chi^2 = 821.74, V = 0.091$), `category` has weak effect ($V = 0.039$), and `region` is independent ($p_{\text{adj}} = 1.0$). |
| **M2-T02** | [`notebooks/M2-T02.ipynb`](notebooks/M2-T02.ipynb)<br>[`src/m2_velocity.py`](src/m2_velocity.py)<br>[`reports/M2/M2-T02_Feature_Summary.md`](reports/M2/M2-T02_Feature_Summary.md) | **Done** | **Sales Velocity Feature Engineering:** Built trailing 7-day sales velocity on strictly prior transactions (`transaction_date - 7d <= t < transaction_date`). Established hierarchical fallback: product-store window covers 25.42% of rows, category-store fallback covers 74.01%, and 0.58% (576 rows) have no sales history (assigned NaN velocity). |
| **M2-T03** | [`notebooks/M2-T03__Sell_Through_Risk.ipynb`](notebooks/M2-T03__Sell_Through_Risk.ipynb)<br>[`src/m2_velocity.py`](src/m2_velocity.py) | **Done** | **Sell-Through Risk & Depletion Ratios:** Engineered `sell_through_risk` (days to sell out) and `sell_through_ratio` (`days_to_sell / days_until_expiry`). Code computes `days_to_sell = current_stock / v.where(v > 0)`, yielding NaN when velocity is zero or NaN. Validated pre-sale arrival stock proxy (`current_stock = initial_quantity`) with zero outcome leakage. |
| **M2-T04** | [`notebooks/M2_T04_discount_flags.ipynb`](notebooks/M2_T04_discount_flags.ipynb) | **Done** | **Discount Tiers & Flag Verification:** Constructed per-batch discount flags and depth tiers: None (0%), Light (10–30%, 18.4%), Medium (30–50%, 5.8%), Deep (>50–75%, 6.4%) based on M1-T02 cut points. M3 Design Decision 2 excludes discrete tier indicator columns from modeling features to prevent leakage and redundancy with continuous `discount_pct`. |
| **M2-T05** | [`reports/M2/M2_T05_Data_Analysis_Report_and_Feature_Engineering_Summary.md`](reports/M2/M2_T05_Data_Analysis_Report_and_Feature_Engineering_Summary.md) | **Done** | **Interim Analysis Report & Feature Engineering Summary:** Reconciled feature definitions, leak-prevention rules, and statistical rankings into a consolidated milestone deliverable. |
| **M2-T06** | [`notebooks/M2_T06_quality_handling_features.ipynb`](notebooks/M2_T06_quality_handling_features.ipynb)<br>[`reports/M2/m2_t06_resolution.json`](reports/M2/m2_t06_resolution.json) | **Done** | **Quality & Handling Signal Transforms:** Fitted $z$-score and `log1p` scalers strictly on training rows for `temp_abuse_events`, `handling_score`, `packaging_score`, `supplier_score`, and `spoilage_sensitivity`. |
| **M2-T07** | [`notebooks/M2-T07.ipynb`](notebooks/M2-T07.ipynb)<br>[`reports/M2/M2-T07_Design_Decision.md`](reports/M2/M2-T07_Design_Decision.md) | **Done** | **Design Decision on Track 3B Risk Input:** Formally decided that Track 3B uses Track 3A's out-of-fold predicted spoilage probability rather than the synthetic benchmark `spoilage_risk` column. Prototype stratified 5-fold CV showed OOF AUC 0.586 closely tracks holdout AUC 0.587. |
| **M2-T08** | [`notebooks/M2_T08_store_markdown_frequency.ipynb`](notebooks/M2_T08_store_markdown_frequency.ipynb) | **Done** | **Store Markdown Frequency:** Computed store-level historical markdown frequencies on strictly earlier dates (`store_md_freq_hist`, Empirical Bayes smoothed, 90-day rolling, and finished batches) as confounding controls for Track 3B. |
| **M2-T09** | [`src/mlflow_utils.py`](src/mlflow_utils.py)<br>[`scripts/M2_T09_test_mlflow_run.py`](scripts/M2_T09_test_mlflow_run.py)<br>[`reports/M2/M2-T09_MLflow_Setup.md`](reports/M2/M2-T09_MLflow_Setup.md) | **Done** | **MLflow Infrastructure Setup:** Configured local tracking directory (`models/mlruns`), experiment namespaces (`smart-shelf-3a/3b/3c`), leakage enforcement, standard metric logging contracts, and author attribution. |
| **M2-T10** | [`notebooks/M2_T10_enhanced_visualizations.ipynb`](notebooks/M2_T10_enhanced_visualizations.ipynb)<br>[`reports/M2/M2_T10_figure_notes.md`](reports/M2/M2_T10_figure_notes.md) | **Done** | **Enhanced Visualizations:** Generated high-resolution diagnostic figures for feature distributions, interaction heatmaps, and markdown response curves. |
| **M2-T11** | [`src/m2_feature_pipeline.py`](src/m2_feature_pipeline.py)<br>[`scripts/M2_T11_build_m2_dataset.py`](scripts/M2_T11_build_m2_dataset.py)<br>[`reports/M2/M2_T11_Consolidation_Report.md`](reports/M2/M2_T11_Consolidation_Report.md)<br>[`reports/M2/M2_T11_M3_Design_Decisions.md`](reports/M2/M2_T11_M3_Design_Decisions.md) | **Done** | **Feature Consolidation & M2 Dataset Build:** Consolidated all M2 features into `data/processed/model_ready_m2_dataset.csv` (100,000 rows × 210 columns) and `models/preprocessor_m2.joblib` via deterministic code pipeline. Canonicalized feature lists (Full 180, Compact 90, Track 3B 43). |
| **M2-T12**<br>**M2-T12b** | [`src/m3_oof.py`](src/m3_oof.py)<br>[`src/support.py`](src/support.py)<br>[`scripts/M2_T12_decision_checks.py`](scripts/M2_T12_decision_checks.py)<br>[`reports/M2/M2_T12_Decision_Checks.md`](reports/M2/M2_T12_Decision_Checks.md) | **Done** | **OOF Generator, Support Flag & Decision Checks:** Built reusable time-blocked OOF generator for Track 3A (`data/processed/track_3a_oof_train.csv`) settling Decision 1. Formally settled Decision 4 (sell-through modeling and exact profit identity) and Decision 5 (low-confidence support flag on `product_name x store_id`, threshold 20 rows, flagging 14.36% of test set). |
| **M2-T13** | [`src/mlflow_utils.py`](src/mlflow_utils.py)<br>[`src/config.py`](src/config.py)<br>[`scripts/M2_T13_test_helpers.py`](scripts/M2_T13_test_helpers.py)<br>[`reports/M2/M2-T09_MLflow_Setup.md`](reports/M2/M2-T09_MLflow_Setup.md) | **Done** | **Milestone 3 MLflow Helpers:** Implemented reusable helpers: model serialization with JSON metadata (`save_model_with_metadata`, `load_model_with_metadata`), calibration curves, confusion matrix, SHAP summary plots, OOF prediction logging, and CSV summary export/merge. |

### Milestone 3: Machine Learning Model Development, Revenue Optimization, and Explainability (Planned)

*Track definitions and scopes per project proposal (`Smart_Shelf_v5.md`):*

| Track / Component | Status | Description & Planned Methodology |
| :--- | :---: | :--- |
| **Track 3A — Spoilage Risk Classifier** | Planned | Select from Random Forest, XGBoost, or LightGBM to predict binary `was_spoiled`. Train on `TRACK_3A_FEATURES_FULL` (180 features) and `COMPACT` (90 features) using time-blocked cross-validation inside the training period. Calibration (Brier score, log loss, reliability curve) is the headline metric. Benchmark against `spoilage_risk` column, compute confusion matrices, SHAP explanations, and attach low-confidence prediction flags. |
| **Track 3B — Revenue-Response Regressor** | Planned | Predict revenue outcome / bounded sell-through ($s \in [0, 1]$) as a function of markdown depth (`discount_pct`) learned from historical discount variation. Uses Track 3A out-of-fold predicted spoilage probabilities (`oof_proba` from `track_3a_oof_train.csv`) and confounding controls. Derive batch profit curves deterministically via the physical accounting identity. |
| **Track 3C — Discount Optimizer** | Planned | Evaluate what-if scoring across candidate discount grid $\{0\} \cup [0.10, 0.75]$. Recommend profit-maximizing discount tier for each batch, mapped to explainable display tiers (Low / Medium / High / Critical). Backtested via two-lane validation framework (**Lane 1: Matched Historical Evaluation**, **Lane 2: Counterfactual Scoring**), with confidence propagation from Tracks 3A and 3B. |

*Milestone 3 Key Deliverables:* Model Evaluation Report, Optimizer Validation Report, Model Code, Final Models.

### Milestone 4: MLOps, Deployment, and Monitoring (Planned)

- **MLOps Implementation:** Experiment tracking, run parameters, metrics, and artifact management using MLflow.
- **Model Deployment:** Interactive Streamlit dashboard featuring a **Manager View** (inventory items flagged by spoilage risk and recommended discount tiers with confidence flags) and a **Simulator Mode** (interactive what-if pricing scenario testing).
- **Model Monitoring & Retraining Strategy:** Tracking prediction drift, estimated spoilage reduction, and estimated revenue recovery (clearly labeled as backtested/simulated estimates), with defined alerting thresholds and periodic retraining plans.
- *Milestone 4 Key Deliverables:* Deployed Model (Streamlit dashboard), MLOps Report, Monitoring Setup.

### Milestone 5: Final Documentation and Presentation (Planned)

- **Final Project Report:** Comprehensive project synthesis including problem framing, dataset description, exploration, model development, deployment, before/after business case (strictly separating Lane 1 matched and Lane 2 simulated evidence), and explicit discussion of synthetic-data limitations.
- **Worked Example:** End-to-end walkthrough of a single representative batch: raw features → classifier risk prediction & confidence flag → SHAP explanation → revenue-response what-if scoring → optimizer recommendation & tier → backtested outcome.
- **Final Presentation:** Polished stakeholder presentation demonstrating system methodology, results, and live Simulator Mode.
- *Milestone 5 Key Deliverables:* Final Project Report (incl. before/after business case), Final Presentation.

---

## 3. Key Verified Data Facts & Design Decisions

All findings below have been audited, cross-reconciled, and programmatically verified across Milestone 1 and Milestone 2:

1. **Chronological Data Split:**
   - **Cutoff Date:** `2024-08-07`
   - **Training Set:** Batches arriving $\le$ `2024-08-07` (80,042 rows, 80.04%).
   - **Test Set:** Batches arriving $>$ `2024-08-07` (19,958 rows, 19.96%).
   - Strict chronological ordering guarantees zero lookahead bias; all feature scalers and support tables are fitted exclusively on training rows.
2. **Physical Accounting Identity:**
   - $\text{initial\_quantity} = \text{units\_sold} + \text{units\_wasted}$ holds exactly in **100.0% of rows** (0 violations across 100,000 transactions).
   - Arrival batch stock is strictly represented pre-sale by `initial_quantity`.
3. **Discount Grid & Markdown Consistency:**
   - Markdown indicator identity: `markdown_applied == (discount_pct > 0)` holds in **100.0% of rows** (30,651 markdown rows).
   - No discounts exist in the range $(0, 0.10)$. Historical discounts take 66 distinct values strictly within $[0.10, 0.75]$.
   - The empirical candidate discount search space for Track 3C is $\{0\} \cup [0.10, 0.75]$.
4. **Dual Shelf-Life Ratios (Decision 3):**
   - `days_remaining_at_purchase` exactly tracks the physical calendar date difference (`expiration_date - transaction_date`).
   - `days_until_expiry` is 0 to 3 days shorter than the calendar gap in 74,191 rows (74.2%).
   - **Decision:** `expiry_remaining_ratio` (`days_until_expiry / shelf_life_days`) is the standardized feature across all modeling tracks. `shelf_life_used_ratio` is reserved strictly for ablation experiments and must never be included in the same model.
5. **Sales Velocity Attribution:**
   - Trailing 7-day velocity coverage across all 100,000 rows:
     - **25.42%** from own product-store 7-day window.
     - **74.01%** from category $\times$ store fallback average.
     - **0.58% (576 rows)** have no historical data (assigned NaN velocity).
   - `sell_through_risk` is calculated as `current_stock / v.where(v > 0)`. When velocity `v` is zero or NaN, `sell_through_risk` evaluates to NaN (with `risk_status` assigned `"no_history"` for NaN velocity or `"at_risk_zero_velocity"` for zero velocity with stock).
6. **Canonical Feature Lists (`src/m2_feature_pipeline.py`):**
   - `TRACK_3A_FEATURES_FULL`: 180 features (M1 pre-sale numeric + cyclical + one-hot categories, products, stores, suppliers, regions + M2 velocity, store markdown frequency, quality/handling signals). Excludes all discount fields and post-sale outcomes.
   - `TRACK_3A_FEATURES_COMPACT`: 90 features. Prunes 90 features from FULL based on empirical findings in `reports/M2/M2_T11_Consolidation_Report.md`:
     - Region one-hots (5: `region_Midwest`, `region_Northeast`, `region_Southeast`, `region_Southwest`, `region_West`)
     - Store one-hots (50: `store_id_STORE_001` to `050`)
     - Supplier one-hots (20: `supplier_id_SUPPLIER_01` to `020`)
     - Pre-sale features with zero signal or severe collinearity (8: `demand_variability`, `storage_temp`, `supplier_score`, `spoilage_sensitivity`, `shelf_life_days`, `days_remaining_at_purchase`, `cost_price`, `expiry_lag_days`; note that `is_promoted` is excluded from Track 3A altogether)
     - Engineered features (7: `sales_velocity`, `sales_velocity_per_day`, `sell_through_risk`, `sell_through_ratio`, `at_risk`, `store_md_freq_hist`, `store_md_freq_smoothed`).
   - `TRACK_3B_FEATURES`: 43 features (controls + treatment `discount_pct`). Does **not** include `oof_proba`; Track 3A out-of-fold probability from `track_3a_oof_train.csv` is merged into Track 3B inputs in Milestone 3.
7. **Low-Confidence Support Flag (Decision 5):**
   - **Specification:** Measures data sparsity at the `product_name x store_id` level with a threshold of 20 historical training observations (`compute_support` and `flag_low_confidence` in [`src/support.py`](src/support.py)).
   - **Coverage:** Flags **14.36% (2,866 rows)** of the test partition.
   - **Documented Deviation:** Deviates from the initial proposal's `category x region x store` key, which never triggered because all 500 cells had $\ge 129$ training rows.
   - **Model Error Analysis:** The flag reflects training data density, not demonstrated model miscalibration. On the test partition, Brier score difference between flagged and unflagged rows is $-0.00343$ (95% bootstrap CI $[-0.01187, +0.00568]$), showing no measurable calibration difference.
   - **UI Guidance:** The dashboard must present this flag as **"limited history for this product at this store"**, not as low model confidence.
8. **Exact Profit Derivation (Decision 4):**
   - Profit is deterministically derived from sell-through:
     $$\text{profit} = \text{initial\_quantity} \times [s \times \text{selling\_price} - \text{cost\_price}]$$
     where $\text{selling\_price} = \text{round}(\text{base\_price} \times (1 - \text{discount\_pct}), 2)$. Holds with zero error across all 100,000 rows.
   - Track 3B models bounded sell-through $s \in [0, 1]$, and Track 3C optimizes derived profit.
9. **Baseline Track 3A OOF Performance:**
   - Fixed baseline LightGBM model on `TRACK_3A_FEATURES_FULL` using 5-fold time-blocked cross-validation (`track_3a_oof_train.csv`):
     - **Brier Score:** `0.15397`
     - **Log Loss:** `0.48452`
     - **ROC-AUC:** `0.5842` (synthetic benchmark `spoilage_risk` AUC: `0.5931`).

---

## 4. Repository Structure

```
├── data/
│   ├── raw/                            # Raw dataset (perishable_goods_management.csv) [untracked]
│   └── processed/                      # Cleaned, split & scaled datasets [untracked]
│       ├── cleaned_dataset.csv         # Date-parsed dataset with chronological split flags
│       ├── model_ready_dataset.csv     # Milestone 1 model-ready dataset (178 features)
│       ├── feature_dictionary.csv      # M1 feature metadata & leakage flags
│       ├── model_ready_m2_dataset.csv  # Consolidated Milestone 2 dataset (210 columns)
│       ├── feature_dictionary_m2.csv   # M2 feature metadata (210 rows)
│       └── track_3a_oof_train.csv      # Track 3A time-blocked out-of-fold predictions
├── models/                             # Serialized model pipelines & MLflow tracking [untracked]
│   ├── preprocessor.joblib             # Milestone 1 fitted ColumnTransformer
│   ├── preprocessor_m2.joblib          # Milestone 2 quality/handling scalers
│   ├── mlruns/                         # Local MLflow file-based experiment tracking store
│   │   └── M2_T09_mlflow_runs_master.csv # Default destination for merged MLflow summaries [untracked]
│   └── mlartifacts/                    # Local MLflow artifact repository
├── notebooks/                          # Interactive exploratory & analytical notebooks
│   ├── M1-T01.ipynb to M1-T06.ipynb    # Milestone 1 data audit, EDA & preprocessing notebooks
│   ├── M2_T01_statistical_analysis.ipynb
│   ├── M2-T02.ipynb                    # Trailing 7-day sales velocity
│   ├── M2-T03__Sell_Through_Risk.ipynb # Sell-through risk & days to sell
│   ├── M2_T04_discount_flags.ipynb     # Discount tier analysis
│   ├── M2_T06_quality_handling_features.ipynb
│   ├── M2-T07.ipynb                    # Risk input formulation analysis
│   ├── M2_T08_store_markdown_frequency.ipynb
│   └── M2_T10_enhanced_visualizations.ipynb
├── reports/
│   ├── M1/                             # Milestone 1 EDA and preprocessing audit reports
│   │   ├── EDA_Report_Milestone1.md
│   │   └── m1_t04_report_text.md
│   ├── M2/                             # Milestone 2 statistical, design, and consolidation reports
│   │   ├── M2_T01_Spoilage_Drivers_Report.md
│   │   ├── M2-T02_Feature_Summary.md
│   │   ├── M2_T05_Data_Analysis_Report_and_Feature_Engineering_Summary.md
│   │   ├── M2-T07_Design_Decision.md
│   │   ├── M2-T09_MLflow_Setup.md
│   │   ├── M2_T10_figure_notes.md
│   │   ├── M2_T11_Consolidation_Report.md
│   │   ├── M2_T11_M3_Design_Decisions.md
│   │   └── M2_T12_Decision_Checks.md
│   ├── figures/                        # Generated figures and charts
│   │   ├── m1/                         # Static figures from Milestone 1
│   │   ├── m2/                         # Static figures from Milestone 2
│   │   └── interactive/                # Standalone Plotly HTML interactive charts
│   └── tables/                         # Exported CSV tables and MLflow run summaries
│       └── M2_T09_mlflow_runs_*.csv    # Per-author MLflow run summaries (tracked)
├── scripts/                            # Standalone execution and verification scripts
│   ├── verify_m1.py                    # Milestone 1 assertion verification suite (36 checks)
│   ├── check_presale_inputs.py         # Presale feature independence & leakage test
│   ├── M2_T09_test_mlflow_run.py       # MLflow logging verification script
│   ├── M2_T11_build_m2_dataset.py      # Milestone 2 dataset consolidation builder
│   ├── M2_T12_decision_checks.py       # Decision checks & baseline OOF generator
│   └── M2_T13_test_helpers.py          # MLflow helpers verification suite
├── src/                                # Reusable production library modules
│   ├── __init__.py
│   ├── config.py                       # Central path management and tracking constants
│   ├── data_loader.py                  # Milestone 1 dataset loader (175 features)
│   ├── m2_feature_pipeline.py          # Consolidated M2 pipeline & feature lists (FULL, COMPACT, 3B)
│   ├── m2_t01_analysis.py              # Statistical testing execution module
│   ├── m2_velocity.py                  # Trailing sales velocity & sell-through calculation
│   ├── m3_oof.py                       # Reusable Track 3A out-of-fold generator
│   ├── mlflow_utils.py                 # MLflow tracking, leakage enforcement & logging helpers
│   ├── pipeline.py                     # Milestone 1 preprocessing pipeline
│   └── support.py                      # Low-confidence cell support computation & flagging
├── .gitignore                          # Governance rules excluding data and model binaries
├── pyrightconfig.json                  # Python language server typing configuration
├── requirements.txt                    # Pinned environment dependencies
└── README.md                           # Master project guide
```

---

## 5. Getting Started & Execution Guide

### 1. Environment Setup

Clone the repository and set up a Python 3.11 virtual environment (e.g., using Conda):

```bash
# Create and activate environment
conda create -n smart-shelf python=3.11 -y
conda activate smart-shelf

# Install dependencies
pip install -r requirements.txt
```

### 2. Place Raw Dataset

Download the Kaggle **Perishable Goods Management** dataset CSV and place it in `data/raw/`:

```bash
data/raw/perishable_goods_management.csv
```

*Expected file properties: 100,000 rows × 42 columns; SHA-256: `de94302b867c9debedfd45c431306623fdfc038f5ed8ca17736339b4460a6674`.*

### 3. Dataset Regeneration Pipeline

To regenerate all processed datasets, feature dictionaries, and serialized preprocessors from scratch, execute the following steps in order:

```bash
# Step 1: Generate Milestone 1 processed data and fitted ColumnTransformer
# Open notebooks/M1-T04.ipynb in your Jupyter GUI or notebook editor and execute all cells.
# (If executed via headless nbconvert, discard notebook metadata diffs using git checkout -- notebooks/M1-T04.ipynb)
# Produces:
#   - data/processed/cleaned_dataset.csv
#   - data/processed/model_ready_dataset.csv
#   - data/processed/feature_dictionary.csv
#   - models/preprocessor.joblib

# Step 2: Build consolidated Milestone 2 dataset and fitted M2 signal scalers
python scripts/M2_T11_build_m2_dataset.py
# Produces:
#   - data/processed/model_ready_m2_dataset.csv (210 columns)
#   - data/processed/feature_dictionary_m2.csv (210 rows)
#   - models/preprocessor_m2.joblib

# Step 3: Run Decision 4 & 5 verification and generate Track 3A baseline OOF predictions
python scripts/M2_T12_decision_checks.py
# Produces:
#   - data/processed/track_3a_oof_train.csv
#   - reports/M2/M2_T12_Decision_Checks.md
```

### 4. Integrity Verification Suites (Read-Only)

Run the automated verification scripts at any time to validate dataset integrity, leakage boundaries, and MLflow logging:

```bash
# Verify Milestone 1 data foundation (36 assertion checks)
python scripts/verify_m1.py

# Verify pre-sale feature independence and physical accounting identity
python scripts/check_presale_inputs.py

# Verify MLflow tracking infrastructure
python scripts/M2_T09_test_mlflow_run.py

# Verify Milestone 3 MLflow helper utilities
python scripts/M2_T13_test_helpers.py
```

### 5. Safe Data Loading Note

- The consolidated Milestone 2 canonical feature sets live in [`src/m2_feature_pipeline.py`](src/m2_feature_pipeline.py): `TRACK_3A_FEATURES_FULL` (180 features), `TRACK_3A_FEATURES_COMPACT` (90 features), and `TRACK_3B_FEATURES` (43 features).
- [`src/data_loader.py`](src/data_loader.py) is the earlier Milestone 1 loader, which loads features from `data/processed/model_ready_dataset.csv` (175 pre-sale features for Track 3A and pre-sale features + `discount_pct` for Track 3B) using `data/processed/feature_dictionary.csv`.

---

## 6. MLflow Experiment Tracking

Experiment tracking is fully standardized through [`src/mlflow_utils.py`](src/mlflow_utils.py) using a local filesystem tracking store located under `models/mlruns/`.

### Experiments

Three standard experiment namespaces are configured:

- `smart-shelf-3a-spoilage-classifier`: Spoilage probability classifier runs.
- `smart-shelf-3b-revenue-response`: Demand elasticity and revenue-response regression runs.
- `smart-shelf-3c-discount-optimizer`: Policy optimization and discount selection runs.

### Author Attribution & Reproducibility

- Runs are tagged with the author's identity using the `SMART_SHELF_AUTHOR` environment variable (falling back to `git config user.name`).
- Machine usernames, local file paths, and email addresses are filtered to ensure sanitized tracking artifacts.
- Fixed random state: `RANDOM_STATE = 42`.

### Reusable Milestone 3 Helpers

The following helper functions in `src/mlflow_utils.py` support Milestone 3 modeling:

- `save_model_with_metadata(model, track, feature_list, shelf_life_version, train_period, mlflow_run_id=None, models_dir=None, name=None, overwrite=False, log_as_artifact=True)`: Saves model pipeline to `.joblib` alongside a standardized `.json` metadata file (feature list, shelf-life version, training period, library versions, git commit, and run ID) and optionally logs both as MLflow artifacts.
- `load_model_with_metadata(path)`: Loads `.joblib` model and accompanying `.json` metadata together with strict schema and feature-list validation.
- `log_calibration_curve(y_true, proba, n_bins=10, name="reliability_curve")`: Computes Brier score, logs reliability curve figure artifact to MLflow, and returns calculated Brier score.
- `log_confusion_matrix(y_true, y_pred, name="confusion_matrix")`: Computes confusion matrix, logs formatted heatmap plot as an artifact, and returns confusion matrix array.
- `log_shap_summary(model, X_sample, max_rows=2000, name="shap_summary")`: Samples at most `max_rows` with seed 42, logs a TreeExplainer SHAP summary plot artifact, and returns mean absolute SHAP value per feature.
- `log_oof_predictions(oof_df, name="track_3a_oof")`: Validates presence of `record_id` and `oof_proba`, and logs the out-of-fold predictions DataFrame as a CSV artifact.
- `export_runs_summary(author=None, include_setup_tests=False, output_dir=None)`: Exports author runs across the three experiments to `reports/tables/M2_T09_mlflow_runs_<author>.csv`.
- `merge_runs_summaries(tables_dir=None, output_path=None)`: Merges all per-author run summaries. By default writes to `models/mlruns/M2_T09_mlflow_runs_master.csv` (gitignored under `models/mlruns/`). Only per-author CSV files are kept in `reports/tables/`.

### Windows Application Control Troubleshooting

On Windows systems with Application Control or AppLocker policies enabled, importing `shap` or compiled C-extensions may trigger `WinError 4551`. To unblock all Conda DLLs, run this command in PowerShell:

```powershell
Get-ChildItem -Path $env:CONDA_PREFIX -Recurse -Include *.dll,*.pyd | Unblock-File
```

---

## 7. Team Workflow & Contribution Rules

To ensure code quality, strict leakage prevention, and seamless team collaboration:

1. **Branching Model:**
   - Never commit directly to `main`.
   - Create task-specific feature branches from `main` using the task identifier:

     ```bash
     git checkout -b <Task-ID>-<description>
     ```

2. **Commit Conventions:**
   - Prefix commit messages with the Task ID:

     ```bash
     git commit -m "<Task-ID>: <description>"
     ```

3. **Data Governance & Clean Commits:**
   - Under no circumstances should raw datasets, processed CSV files, or binary model artifacts (`.joblib`, `.pkl`) be committed to Git.
   - All paths must be imported from [`src/config.py`](src/config.py) rather than hardcoded.
4. **Reproducibility:**
   - Always set `random_state=42` on models, splits, and samplers.
   - Preprocessors, scalers, and support tables must be fit strictly on the training partition (`transaction_date <= 2024-08-07`).

---

## 8. Known Issues & Technical Debt

The following known issues are documented for resolution during Milestone 3:

1. **Hardcoded Paths in Earlier Notebooks:** Notebooks `M2-T02.ipynb`, `M2-T03__Sell_Through_Risk.ipynb`, `M2_T04_discount_flags.ipynb`, `M2-T07.ipynb`, and `M2_T08_store_markdown_frequency.ipynb` load data from relative file paths rather than importing canonical path constants from `src.config`.
2. **Gitignore Scope for Report Tables:** `.gitignore` excludes `*.csv`, with only `!reports/tables/M2_T09_mlflow_runs_*.csv` whitelisted. Consequently, all other CSV tables generated under `reports/tables/` are untracked by Git.
3. **Unexported Figure Tables:** The four auxiliary tables referenced in `reports/M2/M2_T10_figure_notes.md` (`M2_T10_segment_summary.csv`, `M2_T10_category_segments.csv`, `M2_T10_ratio_bands.csv`, `M2_T10_grade_by_segment.csv`) were never exported to `reports/tables/`.
4. **Top-to-Bottom Execution Verification:** End-to-end top-to-bottom re-execution of all exploratory notebooks has not been systematically re-tested since Milestone 1.

---

## 9. Next Steps: Milestone 3 Execution

1. **Track 3A Modeling:** Train candidate classifiers (Random Forest, XGBoost, LightGBM) on `TRACK_3A_FEATURES_FULL` and `COMPACT` using time-blocked cross-validation. Calibrate probabilities to optimize Brier score and log loss.
2. **Track 3B Modeling:** Fit bounded sell-through models against `data/processed/track_3a_oof_train.csv` and evaluate profit curve derivation.
3. **Track 3C Optimization:** Formulate the discrete markdown optimization algorithm over $\{0\} \cup [0.10, 0.75]$ and benchmark via Lane 1 matched and Lane 2 counterfactual evaluation.

---

## 10. Project Documentation

- [Project documentation](Smart_Shelf_Project_Documentation.md): planning, stakeholder analysis, data design and UI/UX design.
- [Project proposal (v5)](Smart_Shelf_v5.md).
