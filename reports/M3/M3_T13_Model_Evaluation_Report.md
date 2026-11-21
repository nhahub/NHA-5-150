# M3-T13: Milestone 3 Model Evaluation Report (Tracks 3A & 3B)

**Task:** M3-T13 — Compiled Milestone 3 Model Evaluation Report  
**Branch:** `M3-T13-m3-deliverable-reports`  
**Scope:** Strictly compiled from existing M3 task reports (`M3-T01` through `M3-T08`), `reports/M3/*.json` summaries, and MLflow runs. No retraining, no re-evaluation, and no modified thresholds or rules.

---

## 1. Project & Data Summary

- **Dataset:** Kaggle *Perishable Goods Management* (`data/raw/perishable_goods_management.csv`), `100,000` transactions × `42` raw columns spanning January 2023 to December 2024 across `50` stores, `5` regions, `20` suppliers, `10` categories, and `63` distinct products (`m1_t04`, `README.md`).
- **Synthetic-Data Limitation:** The dataset is synthetic (`was_spoiled` behaves like a Bernoulli draw from the latent `spoilage_risk` probability, and discounting follows an exact deterministic rule). All reported business-impact estimates, demand elasticities, and calibration metrics reflect synthetic data dynamics.
- **Chronological Split:** Cutoff `2024-08-07` — Training: `80,042` rows (`80.04%`, `<= 2024-08-07`); Test: `19,958` rows (`19.96%`, `> 2024-08-07`), with zero lookahead leakage (`M1-T04`, `M2-T11`).
- **Physical Accounting Identity:** `initial_quantity = units_sold + units_wasted` holds with `0` violations across all `100,000` rows; `profit = initial_quantity * (s * selling_price - cost_price)` holds to `3.27e-11` (`m3_t04_diagnostics_summary.json`).

---

## 2. Track 3A: Spoilage Risk Classifier (`M3-T01` – `M3-T03`)

*Sources: `reports/M3/M3_T01_Track3A_Report.md`, `m3_t01_results_summary.json`, `m3_t02_closeout_summary.json`, `M3_T03_Explainability_Report.md`, MLflow `smart-shelf-3a-spoilage-classifier` (final run `1c6393d91cbd4396b3baee6c91045aa2`, closeout `adccda93ff0f415aa4fe4989a4a69257`, explainability `b5b3d48246284a5985602266859d60c4`).*

### 2.1 Reference Lines & Candidate Comparison (5-Fold Time-Blocked CV, `N = 80,042`)

| Evaluator / Candidate | Feature Set | CV Brier | CV Log Loss | CV ROC-AUC | CV ECE | Source |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Constant Base-Rate (`0.19350` fold mean) | — | `0.15606` | `0.49128` | `0.4943` | `1.03e-06` | `m3_t01_results_summary.json` |
| M2 LightGBM OOF Baseline (`v1`) | `FULL` (180) | `0.15397` | `0.48452` | `0.5842` | `0.00337` | `m3_t01_results_summary.json` |
| `spoilage_risk` Benchmark (Train) | Latent ref | `0.15343` | `0.48279` | `0.5931` | `0.00249` | `m3_t01_results_summary.json` |
| **XGBoost** (Optuna 8 trials) | `FULL` (180) | `0.15373` | `0.48372` | `0.5885` | `0.00114` | `m3_t01_results_summary.json` |
| **Logistic Regression** (`C=0.005`, L2) | `COMPACT` (90) | `0.15374` | `0.48374` | `0.5885` | `0.00167` | `m3_t01_results_summary.json` |
| **LightGBM** | `FULL` (180) | `0.15373` | `0.48376` | `0.5882` | `0.00318` | `m3_t01_results_summary.json` |
| **Logistic Regression** (`C=0.001`, L2) | `FULL` (180) | `0.15374` | `0.48377` | `0.5884` | `0.00254` | `m3_t01_results_summary.json` |
| **XGBoost** | `COMPACT` (90) | `0.15377` | `0.48386` | `0.5880` | `0.00179` | `m3_t01_results_summary.json` |
| **LightGBM** | `COMPACT` (90) | `0.15382` | `0.48404` | `0.5864` | `0.00164` | `m3_t01_results_summary.json` |
| **Random Forest** | `FULL` (180) | `0.15389` | `0.48430` | `0.5852` | `0.00280` | `m3_t01_results_summary.json` |
| **Random Forest** | `COMPACT` (90) | `0.15393` | `0.48442` | `0.5849` | `0.00295` | `m3_t01_results_summary.json` |

### 2.2 Calibration & Chosen Model (Why Logistic Regression)
- **Calibration comparison (inner 3-fold CV):** For `logreg` (`COMPACT`), uncalibrated (`0.15374` Brier, `0.48374` log loss, `0.00167` ECE) vs. Platt `sigmoid` (`0.15373` / `0.48372` / `0.00218`) vs. isotonic (`0.15376` / `0.48397` / `0.00086`). For `xgboost` (`FULL`), uncalibrated (`0.15373` / `0.48372` / `0.00114`) vs. `sigmoid` (`0.15376` / `0.48386` / `0.00445`) vs. isotonic (`0.15377` / `0.48385` / `0.00260`).
- **Selected Model:** Uncalibrated Regularized Logistic Regression (`C=0.005`, L2) on `TRACK_3A_FEATURES_COMPACT` (`90` features), frozen at `models/track_3a_classifier.joblib`. Under the pre-registered parsimony rule, its CV Brier (`0.15374`) and log loss (`0.48374`) are within `0.00002` (noise) of tuned XGBoost (`0.15373` / `0.48372`), while using `90` zero-NaN features (no imputation needed in production) and native log-odds linearity.
- **`v2` OOF vs. `v1` OOF (`m3_t02_closeout_summary.json`):** `v2` (`track_3a_oof_train_v2.csv`) achieves `0.15374` Brier / `0.48374` log loss / `0.5885` AUC (`r = 0.8704` with M2 `v1`). Original `track_3a_oof_train.csv` was kept untouched for backwards compatibility, while `v2` (`oof_proba`) was wired into Track 3B (`M3-T04`).

### 2.3 Test Evaluation, AUC Ceiling & Confusion Matrices (`N = 19,958`)

| Evaluator (Test, `N = 19,958`) | Brier | Log Loss | ROC-AUC | ECE (10-bin) |
| :--- | :---: | :---: | :---: | :---: |
| **Frozen Track 3A (`logreg` COMPACT)** | **`0.15641`** | **`0.48988`** | **`0.5899`** | **`0.00579`** |
| Constant Base-Rate (`0.19350`) | `0.15889` | `0.49785` | `0.5000` | `0.00462` |
| `spoilage_risk` Benchmark (Test) | `0.15601` | `0.48853` | `0.5967` | `0.00392` |

- **AUC Ceiling Statement:** `was_spoiled` behaves like a Bernoulli draw from `spoilage_risk`, whose own test ROC-AUC is `0.5967`. Thus `~0.59–0.60` is the structural ROC-AUC ceiling; Track 3A recovers `98.9%` of the benchmark's discrimination (`0.5899` vs `0.5967`) with tight calibration (`ECE = 0.00579`).
- **Confusion Matrices (`m3_t01_results_summary.json`):**
  - `t = 0.20` (near base rate): `pred_pos = 41.15%`, precision `0.2444`, recall `0.5076`, F1 `0.3299` (`TN=9,799, FP=6,205, FN=1,947, TP=2,007`).
  - `t = 0.30`: `pred_pos = 2.46%`, precision `0.3184`, recall `0.0395`, F1 `0.0702` (`TN=15,670, FP=334, FN=3,798, TP=156`).
  - `t = 0.50`: `pred_pos = 0.00%`, precision `0.0000`, recall `0.0000`, F1 `0.0000` (`TN=16,004, FP=0, FN=3,954, TP=0`).
- **Illustrative Cost-Asymmetric Sweep (`track_3a_oof_train_v2.csv`, Train Only):** For `k = cost(FN) / cost(FP) in {1, 2, 3, 5}`, optimal thresholds are `0.390` (`cost/batch 0.1935`), `0.365` (`0.3868`), `0.245` (`0.5677`), and `0.170` (`0.7531`, recall `0.7399`, `pred_pos = 64.46%`). Illustrative only; unit-economic pricing is handled in Track 3C.

### 2.4 Explanations & Limited-History Flag (`M3-T03`)
- **Exact Linear Explanations (`src/m3_t03_explain.py`):** Computes exact log-odds contributions $\phi_j = \beta_j (x_j - \mu_j)$ relative to training mean (`max |sum - logit_diff| = 4.441e-16`), groups one-hots/cyclicals, and allocates probability points (`max sum error = 1.260e-13` pts). Top grouped drivers: `packaging_score` (`0.1919`), `handling_score` (`0.1203`), `days_since_receipt` (`0.0840`), `days_until_expiry` (`0.0818`), `temp_deviation` (`0.0702`).
- **Limited-History Support Flag & Support-Key Deviation:** Uses `product_name × store_id` `< 20` training rows (`src/support.py`), flagging `2,866 / 19,958` (`14.36%`) test rows. This deviates from the proposal's `category × region × store` key, which never triggered because all `500` cells had `>= 129` training rows. Flagged test rows show no calibration degradation (Brier `0.15310` vs `0.15696` unflagged); the flag communicates training data density (*"Limited history for this product at this store"*).

---

## 3. Track 3B: Confounding, Regressor & Elasticity View (`M3-T04` – `M3-T08`)

*Sources: `M3_T04`, `M3_T05`, `M3_T06`, `M3_T07`, `M3_T08` reports and JSON summaries; MLflow `smart-shelf-3b-revenue-response` (`486dd40b...`, `4f3e3191...`, `7b9dd2dc...`, `ce191bff...`, `d3ed3411...`, `72c8f748...`, `c770f14d...`).*

### 3.1 Headline Finding: Discount Assignment Is Rule-Based (`M3-T05`)
- Across all `80,042` training rows (and `0` violations across `19,958` test rows), `discount_pct > 0` holds **100.00% deterministically** iff:
  $$\text{raw\_expiry\_remaining\_ratio} \le 0.40 \;\lor\; \text{quality\_grade} == \text{'C'}$$
- **Disjoint Regions (`R0` vs. `R1`):**
  - **`R0` (`ratio > 0.40` and `grade != 'C'`):** `55,567` train rows (`69.42%`), `13,782` test rows (`69.06%`). **Zero discounted batches exist in `R0`** (train `s` mean `0.7679`, profit/unit `+$4.57`, loss share `16.8%`).
  - **`R1` (`ratio <= 0.40` or `grade == 'C'`):** `24,475` train rows (`30.58%`), `6,176` test rows (`30.94%`). **Zero undiscounted batches exist in `R1`** (train `s` mean `0.7986`, `s=0`: `3.43%`, `s=1`: `43.41%`, profit/unit `-$0.7849`, loss share `54.04%`; `m3_t05_addendum_summary.json`).
- **Meaning:** No `0%` vs `>0%` overlap exists anywhere. Any `0%` vs `30%` comparison (`+0.0436` in Pass 1) was model extrapolation across `R0`/`R1`. Only **depth variation inside `R1` (`10%–75%`)** is empirically identifiable.

### 3.2 Scenario Construction, Overlap, Propensity & Why DML Was Not Pursued (`M3-T04`, `M3-T05`)
- **Scenario Dataset (`data/processed/model_ready_3b_dataset.csv`):** `100,000 × 59` (`TRACK_3B_FEATURES` = `43` + `oof_proba` `v2`, target `s = units_sold / initial_quantity` in `[0, 1]`, `NaN`s preserved for LightGBM). Counterfactual-consistency check passed (`0` treatment-derived features).
- **Propensity & Why DML Was Not Pursued:** A propensity classifier for `discount_pct > 0` achieved **AUC `1.0000`**, Brier `2.61e-14`, and **`100.00%` extreme propensities** (`< 0.02` or `> 0.98`), violating positivity completely. Under pre-registered **Rule D**, DML/EconML was **not pursued** (`overlap_adequate_lt_5pct = False`).
- **Rule E (R1 Quasi-Random Depth Addendum):** Inside `R1` (`61` distinct depths in Grade C and `61` in Grade A/B), `depth ~ controls` yields OOF $R^2 = 0.8738$ with **residual SD = `6.10` discount points** (`>= 3.0` pts) and `residual ~ other features` OOF $R^2 = -0.0704$ (`< 0.05`). **Rule E PASSED**, establishing usable quasi-random depth variation inside `R1`.

### 3.3 R1 Regressor, Extreme Outcomes & Test Evaluation (`M3-T06`, `M3-T07`)
- **Extreme-Outcome Handling (`M3-T06`):** Modeling bounded $s \in [0, 1]$ with `cross_entropy` and deriving profit via the exact identity eliminates heavy dollar-scale tails (`profit/unit` quantiles on `R1` train: `min = -$316.80, p01 = -$67.92, p05 = -$7.07, p25 = -$1.98, p50 = -$0.23, p75 = +$1.31, p95 = +$6.07, p99 = +$66.86, max = +$200.56`; `54.04%` lose money; excluding the top/bottom `1%` profit tail drops batch profit RMSE from `$1,976` to `$1,298`; `m3_t06_results_summary.json`).
- **5-Fold Time-Blocked CV on Train `R1` (`N = 24,475`, 40 Optuna trials):** Selected **`lgbm_ce_monotone`** (`discount_pct >= 0`, `s` RMSE `0.12967`, MAE `0.10937`, $R^2$ `0.7094`, profit/unit MAE `$1.8337`, RMSE `$6.68`), beating `GroupMeanBaseline` (`0.22527` RMSE, `0.1229` $R^2$) and matching unconstrained `lgbm_ce` (`0.12975` RMSE) and `xgboost_logistic` (`0.12983` RMSE).
- **Final Test `R1` Evaluation (`N = 6,176`, Evaluated Once, `M3-T07`):**
  - Frozen `lgbm_ce_monotone`: `s` RMSE **`0.12781`**, MAE **`0.10810`**, $R^2$ **`0.7419`**, profit/unit MAE **`$1.73`**, RMSE **`$6.37`**, batch profit RMSE `$1,793` (`$1,204` excluding 1% tail) vs. `GroupMeanBaseline` (`0.23502` RMSE, `0.1274` $R^2$).
  - **Rule T1 PASSED** (`0.12781 <= 0.14912` and `0.7419 >= 0.6594`).
  - **Rule T2 (Unconstrained Shape Check):** Across `15` `(expiry × baseline_s × depth_band)` cells with `>= 200` training rows, **`100%` (`15/15`)** had positive unconstrained partial slopes and `0` had significantly negative slopes (**Rule T2 PASSED**).

### 3.4 Slope Estimates from Three Sources & Category Elasticity View (`M3-T05`, `T06`, `T07`, `T08`)
- **Three Independent Slope Estimates ($\Delta s / \Delta d$ across `R1`):**
  1. `M3-T05` (Train `R1` cross-fitted partialling-out): **`+0.2614`** (`95% CI [+0.2342, +0.2880]`; `10-30%`: `+0.5051`, `30-50%`: `+0.2616`, `50-75%`: `+0.0739`).
  2. `M3-T06` (Frozen model-implied partial slope on Train `R1`): **`+0.2231`** (`10-30%`: `+0.3800`, `30-50%`: `+0.1926`, `50-75%`: `+0.0684`).
  3. `M3-T07` (Test `R1` empirical residual slope from train-fit models): **`+0.2911`** (`95% CI [+0.2369, +0.3522]`).
- **Category Elasticity View (`M3-T08`, Train `R1` only):**
  - Cross-category comparisons must be restricted to the **common `10-30%` depth band** (`+1.84` pts per `+10%` discount in `Frozen_Meals` to `+4.29` in `Meat`), because overall slopes depend on each category's depth coverage (`Dairy` `5/13`, `Beverages` `3/13`, `Pharmaceuticals` `3/13`, `Frozen_Meals` `3/13` are flagged `limited_depth_range = True` and enter `R1` mostly/exclusively via `quality_grade == 'C'`).
  - Model overall slope falls inside the model-free 95% CI for `5/10` categories (`Dairy`, `Beverages`, `Pharmaceuticals`, `Frozen_Meals`, `Ready_to_Eat`).
  - **`spoilage_sensitivity` Check (Rule C3):** Spearman $\rho = -0.158$ vs. baseline $\hat{s}$ and `+0.103` vs. model slope across `n = 10` categories (no validation in either direction).

---

## 4. Consolidated Pre-Registered Rules & Verdicts

| Task | Rule | Definition | Verdict |
| :--- | :--- | :--- | :--- |
| `M3-T05` | **Rule S** *(Superseded)* | Coarse 5×5 quintile support (`discount_support_coarse`) | **Superseded** (crossed `R0`/`R1` boundary; replaced by `discount_support()`) |
| `M3-T05` | **Rule U** *(Pass 1)* | Profit/unit falls monotonically across discounts in `>= 80%` strata | **Invalidated in Pass 1** (`15/15` relied on `R0`/`R1` extrapolation) |
| `M3-T05` | **Rule M** *(Pass 1)* | Sell-through `s` slope `>= 0` in `>= 80%` strata | **Invalidated in Pass 1** (`15/15` relied on `R0`/`R1` extrapolation) |
| `M3-T05` | **Rule D** | Pursue DML only if Rule U/M fails AND extreme propensity `< 5%` | **Not Pursued** (`100%` extreme propensities; total positivity violation) |
| `M3-T05` | **Rule E** | `R1` residual depth SD `>= 3.0` pts AND `resid R^2 < 0.05` | **PASSED** (`SD = 6.10` pts, `resid R^2 = -0.0704`) |
| `M3-T07` | **Rule T1** | Test `R1` `s` RMSE `<= 0.14912` AND $R^2 >= 0.6594$ | **PASSED** (`RMSE = 0.12781`, $R^2 = 0.7419$) |
| `M3-T07` | **Rule T2** | Unconstrained `lgbm_ce` positive slope in `>= 80%` cells & `0` sig. neg. | **PASSED** (`15/15 = 100%` positive, `0` sig. neg.) |
| `M3-T08` | **Rule C1** | Category low confidence if `< 300` R1 rows OR `< 3/13` bins `>= 30` | **PASSED (`10/10`)** — *Post-hoc observation: too lenient on depth coverage; 4 categories `< 8/13` flagged `limited_depth_range`* |
| `M3-T08` | **Rule C2** | Plot/include `(category, depth)` cells iff `>= 30` train `R1` rows | **Applied** (solid vs dashed/excluded) |
| `M3-T08` | **Rule C3** | `spoilage_sensitivity` ranking check (`n = 10` categories) | **Descriptive only / No relationship** ($\rho = +0.103$ vs slope) |

---

## 5. Limitations & Consolidated Deviations from Plan

- **Key Limitations:** (1) Synthetic dataset with Bernoulli spoilage draws (`~0.59–0.60` AUC ceiling) and deterministic `0%` vs `>0%` discounting (`R0` vs `R1`); (2) zero `0%` vs `>0%` overlap (`69.4%` train / `69.1%` test in `R0` have zero discount evidence); (3) observational `R1` depth variation (unobserved confounding cannot be ruled out); (4) heavy right-censoring at `s = 1` (`>54%` at `>= 50%` discounts); (5) `4/10` categories restricted to shallow Grade C markdowns (`< 8/13` bins).
- **Consolidated Deviations (`M3-T01`–`M3-T08`):**
  1. `M3-T01`: Uncalibrated L2 Logistic Regression (`C=0.005`) on `COMPACT` (`90` features) selected over tree models (`FULL` `180`) by parsimony; `shap.LinearExplainer` used instead of `TreeExplainer`.
  2. `M3-T01`/`T03`: Support flag key changed from `category × region × store` (never triggered) to `product_name × store_id` `< 20` (`14.36%` test flagged).
  3. `M3-T05`: Region-based `discount_support()` replaced coarse 5×5 support; DML dropped due to `100%` extreme propensity.
  4. `M3-T06`: Track 3B trained on `R1` only (`24,475` rows) predicting bounded `s` with `oof_proba` `v2` rather than full-dataset direct profit regression with `v1`.
  5. `M3-T07`: Shape check evaluated across `[0.10, 0.75]` supported `R1` depths rather than against `0%`.
  6. `M3-T08`: `units_sold` derived deterministically (`s_hat * initial_quantity`); post-hoc `limited_depth_range` (`< 8/13`) flag added alongside Rule C1.

---

## 6. Source Discrepancies Found & Resolved (Correction Notes Added)

1. **`M3_T06_Regressor_Training_Report.md` vs. `m3_t06_results_summary.json` (`profit_per_unit` quantiles):** Report line 26 listed `min = -$359.73, p05 = -$19.93, p25 = -$2.89, p75 = +$0.88, p95 = +$17.46`. Recomputing from `data/processed/model_ready_3b_dataset.csv` proved `m3_t06_results_summary.json` is exact (`min = -$316.80, p05 = -$7.07, p25 = -$1.98, p75 = +$1.31, p95 = +$6.07`). A dated correction note was added to `M3_T06_Regressor_Training_Report.md`.
2. **`M3_T05_Confounding_Report.md` vs. `m3_t05_addendum_summary.json` (`R1` `s=0`/`s=1` shares):** Report line 23 listed `s=0: 2.21%, s=1: 41.58%`. Recomputing from `data/processed/model_ready_3b_dataset.csv` proved `m3_t05_addendum_summary.json` is exact (`s=0: 3.4321%, s=1: 43.4076%`). A dated correction note was added to `M3_T05_Confounding_Report.md`.
