# M2-T09: MLflow Experiment Tracking Setup & Teammate Guide

**Task ID:** `M2-T09`  
**Milestone:** 2 (Advanced Data Analysis & Feature Engineering)  
**Status:** Complete & Verified  
**Verified Test Run ID:** `63608781a6c14719985f324d6721d6a5`  
**Date:** 2026-10-17  

---

## 1. Overview & Architecture

To guarantee full traceability, reproducibility, and prevent post-hoc metric reconstruction, MLflow experiment tracking is initialized before any Milestone 3 model is trained.

The tracking architecture uses a **local-first MLflow tracking store** located inside the repository at `models/mlruns/` (ignored by Git), paired with **version-controlled CSV run summaries** committed under `reports/tables/`. This design ensures:
1. **Zero Git Bloat:** Raw SQLite databases, metric logs, and binary artifact dumps remain strictly ignored by Git.
2. **Conflict-Free Collaboration:** Each teammate exports their runs to their own author-specific CSV file (`reports/tables/M2_T09_mlflow_runs_<author>.csv`).
3. **Auditability on GitHub:** All teammate runs, parameters, metrics, and artifact references are committed and viewable directly in GitHub PRs and tables.
4. **Seamless Remote Upgradability:** By setting the environment variable `MLFLOW_TRACKING_URI`, any teammate or CI pipeline can transparently switch to a remote tracking server with zero code modifications.

---

## 2. Standardized Experiments

Three persistent experiments are pre-configured to segregate model development across tracks:

| Track | Experiment Name | Model Focus | Primary Metric / Evaluation Objective |
| :--- | :--- | :--- | :--- |
| **Track 3A** | `smart-shelf-3a-spoilage-classifier` | Spoilage Risk Classifier (`was_spoiled`) | **Probability Calibration:** Brier Score (headline), Log Loss, ROC-AUC vs. `spoilage_risk` benchmark (0.593 ceiling). |
| **Track 3B** | `smart-shelf-3b-revenue-response` | Revenue-Response Regressor (`profit`, `waste_pct`, `units_sold`) | **Profit Prediction:** MAE, RMSE, $R^2$ (accounting consistency: $\text{initial\_quantity} = \text{sold} + \text{wasted}$). |
| **Track 3C** | `smart-shelf-3c-discount-optimizer` | Dynamic Markdown Optimizer ($\{0\} \cup [0.10, 0.75]$) | **Two-Lane Validation:** Lane 1 (Matched historical) vs. Lane 2 (Counterfactual simulated), strictly unblended. |

---

## 3. How to Launch the Local MLflow UI

Teammates can launch and inspect the interactive MLflow UI locally in their browser:

```bash
# Activate the smart-shelf conda environment
conda activate smart-shelf

# Launch the MLflow UI pointing to models/mlruns (POSIX or Windows path)
mlflow ui --backend-store-uri models/mlruns --port 5000
```
Open your browser at `http://127.0.0.1:5000` to review runs, compare parameters, inspect metric curves, and view logged artifact plots.

---

## 4. Reusable Helper: `src.mlflow_utils`

All Milestone 3 training scripts and notebooks should use `src/mlflow_utils.py` rather than calling raw MLflow methods directly.

### 4.1 Author Attribution
The system identifies the author via:
1. `SMART_SHELF_AUTHOR` environment variable (e.g. `set SMART_SHELF_AUTHOR=Amira`).
2. Fallback to `git config user.name`.
*Note: Local usernames, emails, or personal filesystem paths are never recorded.*

### 4.2 Anti-Leakage Governance
The `start_tracked_run` context manager enforces strict project governance and **raises a `ValueError` immediately** if:
1. `spoilage_risk` or any post-sale outcome column (`units_sold`, `units_wasted`, `waste_pct`, `revenue`, `waste_cost`, `profit`, `profit_margin_pct`) is passed in `features`.
2. Any discount field (`discount_pct`, `markdown_applied`, `selling_price`) is included in Track 3A features.
3. The shelf-life version (`shelf_life_version`) is not explicitly stated (`expiry_remaining_ratio` or `shelf_life_used_ratio`).

---

## 5. Logging Templates by Track

### 5.1 Track 3A — Spoilage Risk Classifier Template
```python
from src.data_loader import load_dataset, load_track_3a_benchmark, get_feature_names
from src.mlflow_utils import start_tracked_run, log_track_3a_metrics, log_feature_list_artifact, log_figure_artifact
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# 1. Load data
X_train, y_train, X_test, y_test = load_dataset(track="3A")
benchmark_risk, benchmark_target = load_track_3a_benchmark()
benchmark_auc = roc_auc_score(benchmark_target, benchmark_risk)
features = get_feature_names(track="3A")

# 2. Train model
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

# 3. Compute metrics
brier = brier_score_loss(y_test, y_pred_proba)
logloss = log_loss(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

# 4. Tracked Run
with start_tracked_run(
    track="3A",
    model_type="LightGBM_Classifier",
    feature_set_name="M2_engineered_features",
    shelf_life_version="expiry_remaining_ratio",
    features=features,
    run_name="3a_lightgbm_tuned",
    params={"learning_rate": 0.05, "num_leaves": 31},
) as run:
    log_track_3a_metrics(
        brier_score=brier,
        log_loss_val=logloss,
        roc_auc=auc,
        benchmark_auc=benchmark_auc,
        threshold=0.5,
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
    )
    log_feature_list_artifact(features)
```

### 5.2 Track 3B — Revenue-Response Regressor Template
```python
from src.mlflow_utils import start_tracked_run, log_track_3b_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Outcome metrics (profit primary, sell-through secondary)
profit_mae = mean_absolute_error(y_test_profit, y_pred_profit)
profit_rmse = np.sqrt(mean_squared_error(y_test_profit, y_pred_profit))
profit_r2 = r2_score(y_test_profit, y_pred_profit)

with start_tracked_run(
    track="3B",
    model_type="GradientBoostingRegressor",
    feature_set_name="presale_plus_oof_risk_and_discount",
    shelf_life_version="expiry_remaining_ratio",
    features=features_3b,
    run_name="3b_gbm_confounding_controlled",
    params={"n_estimators": 100, "max_depth": 5},
) as run:
    log_track_3b_metrics(
        profit_mae=profit_mae,
        profit_rmse=profit_rmse,
        profit_r2=profit_r2,
        sell_through_mae=st_mae,
        sell_through_rmse=st_rmse,
        sell_through_r2=st_r2,
    )
```

### 5.3 Track 3C — Discount Optimizer Template
```python
from src.mlflow_utils import start_tracked_run, log_track_3c_metrics

with start_tracked_run(
    track="3C",
    model_type="GridSearchOptimizer",
    feature_set_name="optimizer_grid_0_to_75",
    shelf_life_version="expiry_remaining_ratio",
    run_name="3c_optimizer_backtest_v1",
    params={"candidate_grid": "[0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75]"},
) as run:
    # Strictly separated lanes - never blended into one headline metric
    log_track_3c_metrics(
        lane1_metrics={
            "matched_profit_gain_pct": 0.082,
            "matched_batches_evaluated": 1845,
            "observed_actual_profit": 52140.0,
        },
        lane2_metrics={
            "simulated_profit_gain_pct": 0.145,
            "simulated_batches_evaluated": 18113,
            "simulated_recovered_revenue": 142050.0,
        },
    )
```

---

## 6. Milestone Routine: Exporting & Sharing Run Summaries

Because `models/mlruns/` is ignored by Git, teammates share run results via the export utility:

1. **At the end of a modeling session**, run:
   ```python
   from src.mlflow_utils import export_runs_summary
   # Exports all your personal production runs (excluding setup tests)
   export_runs_summary()
   ```
   This generates `reports/tables/M2_T09_mlflow_runs_<your_name>.csv`.
2. **Commit and push only your personal CSV table:**
   ```bash
   git add reports/tables/M2_T09_mlflow_runs_<your_name>.csv
   git commit -m "M3-T02: log Track 3A LightGBM hyperparameter runs"
   ```
3. **To generate the consolidated master report** for Milestone reports and instructor reviews:
   ```python
   from src.mlflow_utils import merge_runs_summaries
   merge_runs_summaries()
   ```
   This combines all individual teammate CSVs into gitignored `models/mlruns/M2_T09_mlflow_runs_master.csv` (or any custom output path passed) with automated deduplication by `run_id`.

---

## 7. Verified Setup Test Run Evidence

The automated verification suite in [`scripts/M2_T09_test_mlflow_run.py`](file:///d:/DEPI/Project/Github%20Repo/NHA-5-150/scripts/M2_T09_test_mlflow_run.py) executed and validated all components:

- **Run ID:** `63608781a6c14719985f324d6721d6a5`
- **Experiment:** `smart-shelf-3a-spoilage-classifier`
- **Model Type:** `LogisticRegression_baseline` (C=0.01, solver='lbfgs')
- **Feature Set:** `M1_presale_175` (175 features, zero discount variables, zero outcomes)
- **Shelf-Life Version:** `expiry_remaining_ratio`
- **Split:** Train $\le$ 2024-08-07 (80,042 rows), Test $>$ 2024-08-07 (19,958 rows)
- **Results:**
  - `test_brier_score`: `0.156391`
  - `test_log_loss`: `0.489784`
  - `test_roc_auc`: `0.590856`
  - `benchmark_spoilage_risk_auc`: `0.596737`
  - `auc_gap_to_benchmark`: `-0.005881`
- **Artifacts Verified:**
  - `reliability_curve_test.png`
  - `track_3a_features.json`
- **Leakage Guard Tests:**
  - Target leakage (`spoilage_risk`): Blocked with `ValueError` [PASS]
  - Treatment confounding (`discount_pct` in 3A): Blocked with `ValueError` [PASS]
  - Missing shelf-life version: Blocked with `ValueError` [PASS]
- **Export & Merge Tests:**
  - Exported to [`reports/tables/M2_T09_mlflow_runs_Mohamed_Boshra.csv`](file:///d:/DEPI/Project/Github%20Repo/NHA-5-150/reports/tables/M2_T09_mlflow_runs_Mohamed_Boshra.csv) [PASS]
  - Merged to [`reports/tables/M2_T09_mlflow_runs_master.csv`](file:///d:/DEPI/Project/Github%20Repo/NHA-5-150/reports/tables/M2_T09_mlflow_runs_master.csv) [PASS]

---

## 8. Switching to a Remote Server Later

If the team or instructor later chooses to host a shared remote MLflow server (e.g. AWS EC2, GCP, or DagsHub), **zero code changes** are required:
Simply set the environment variable before executing:
```bash
export MLFLOW_TRACKING_URI="https://your-remote-mlflow-server.com"
```
`src/config.py` automatically checks `os.environ.get("MLFLOW_TRACKING_URI")` and points all tracking, logging, and exports to the remote server.

---

## 9. Deviations from Plan

1. **Local Tracking vs. Remote Server:** The task card suggested a "shared location". Per user decision and instructor approval, the project utilizes local file tracking (`models/mlruns/`) combined with version-controlled, per-author CSV run summaries (`reports/tables/M2_T09_mlflow_runs_<author>.csv`). This completely eliminates external hosting costs, firewall/credential friction, and authentication failures while maintaining full traceability and reviewability on GitHub.
2. **Environment Python Version:** Python 3.11 was selected for the dedicated `smart-shelf` conda environment because base Python 3.13.9 encounters wheel build failures with `shap` and `lightgbm` on Windows. All base scientific package versions (`scikit-learn==1.7.2`, `pandas==2.3.3`, `numpy==2.3.5`, `joblib==1.5.2`, `scipy==1.16.3`) were matched exactly, and `models/preprocessor.joblib` and `verify_m1.py` were verified to load and pass 36/36 checks with zero regressions.
3. **MLflow 3.x Compatibility:** In MLflow 3.x, local filesystem tracking backends display a maintenance notice. `os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"` was configured in `src/config.py` to enable local file storage without warnings or migration friction.
