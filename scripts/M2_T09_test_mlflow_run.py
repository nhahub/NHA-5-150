"""
scripts/M2_T09_test_mlflow_run.py - Automated verification and test run for M2-T09.

Executes an end-to-end test of the Smart Shelf MLflow tracking infrastructure:
1. Verifies anti-leakage governance (asserts ValueError on leakage attempt).
2. Fits an honest baseline classifier on the canonical time-ordered split (train <= 2024-08-07).
3. Logs run parameters, tags, Track 3A metrics (Brier, log loss, ROC-AUC vs benchmark),
   and artifacts (reliability curve, feature list).
4. Verifies run retrieval directly via MLflowClient.
5. Verifies run summary export and master merge functions.
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, log_loss, precision_score, recall_score, roc_auc_score

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANDOM_STATE, TABLES_DIR
from src.data_loader import get_feature_names, load_dataset, load_track_3a_benchmark
from src.mlflow_utils import (
    export_runs_summary,
    get_or_create_experiment,
    get_run_author,
    log_feature_list_artifact,
    log_figure_artifact,
    log_track_3a_metrics,
    merge_runs_summaries,
    setup_tracking_environment,
    start_tracked_run,
)
from mlflow.tracking import MlflowClient


def run_test():
    print("==================================================================")
    print("M2-T09: MLflow Experiment Tracking Verification & Test Run")
    print("==================================================================\n")

    # 1. Setup Tracking Environment
    tracking_uri = setup_tracking_environment()
    print(f"[1/6] Tracking URI configured: {tracking_uri}")
    author = get_run_author()
    print(f"      Run Author attributed: {author}")

    # 2. Verify Anti-Leakage Governance Enforcement
    print("\n[2/6] Verifying anti-leakage governance enforcement...")
    features_3a = get_feature_names(track="3A")
    print(f"      Track 3A allowed features count: {len(features_3a)}")

    # Leakage Test A: Target outcome injection
    leakage_passed_a = False
    try:
        with start_tracked_run(
            track="3A",
            model_type="LeakageTest",
            feature_set_name="test_leakage",
            shelf_life_version="expiry_remaining_ratio",
            features=features_3a + ["spoilage_risk"],
            purpose="leakage_verification",
        ):
            pass
    except ValueError as e:
        leakage_passed_a = True
        print(f"      [PASS] Target leakage blocked successfully: {e}")

    assert leakage_passed_a, "FAIL: start_tracked_run failed to block target leakage (spoilage_risk)!"

    # Leakage Test B: Discount treatment injection into Track 3A
    leakage_passed_b = False
    try:
        with start_tracked_run(
            track="3A",
            model_type="LeakageTest",
            feature_set_name="test_leakage",
            shelf_life_version="expiry_remaining_ratio",
            features=features_3a + ["discount_pct"],
            purpose="leakage_verification",
        ):
            pass
    except ValueError as e:
        leakage_passed_b = True
        print(f"      [PASS] Treatment confounding blocked successfully: {e}")

    assert leakage_passed_b, "FAIL: start_tracked_run failed to block discount treatment in Track 3A!"

    # Leakage Test C: Missing shelf-life version
    leakage_passed_c = False
    try:
        with start_tracked_run(
            track="3A",
            model_type="LeakageTest",
            feature_set_name="test_leakage",
            shelf_life_version="invalid_ratio_name",
            features=features_3a,
            purpose="leakage_verification",
        ):
            pass
    except ValueError as e:
        leakage_passed_c = True
        print(f"      [PASS] Invalid shelf-life version blocked successfully: {e}")

    assert leakage_passed_c, "FAIL: start_tracked_run failed to enforce shelf_life_version requirement!"

    # 3. Load Data for Honest Baseline
    print("\n[3/6] Loading data on official time-ordered split (train <= 2024-08-07)...")
    X_train, y_train, X_test, y_test = load_dataset(track="3A")
    assert X_train is not None and y_train is not None
    assert X_test is not None and y_test is not None
    print(f"      Train rows: {len(X_train)} (spoilage rate: {y_train.mean():.4f})")
    print(f"      Test rows:  {len(X_test)} (spoilage rate: {y_test.mean():.4f})")

    # Load benchmark for evaluation comparison
    benchmark_risk, benchmark_target = load_track_3a_benchmark()
    benchmark_auc = roc_auc_score(benchmark_target, benchmark_risk)
    print(f"      Benchmark (spoilage_risk) Test ROC-AUC: {benchmark_auc:.4f}")

    # 4. Train Honest Trivial Baseline Model
    print("\n[4/6] Training trivial baseline model (LogisticRegression, C=0.01)...")
    model = LogisticRegression(C=0.01, max_iter=200, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Compute Track 3A Headline Metrics
    brier = brier_score_loss(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"      Headline Metric - Brier Score: {brier:.4f}")
    print(f"      Headline Metric - Log Loss:    {logloss:.4f}")
    print(f"      Test ROC-AUC:                  {roc_auc:.4f} (Benchmark: {benchmark_auc:.4f})")

    # 5. Log Run to MLflow
    print("\n[5/6] Logging tracked run to MLflow...")
    shelf_life_choice = "expiry_remaining_ratio"
    run_id = None

    with start_tracked_run(
        track="3A",
        model_type="LogisticRegression_baseline",
        feature_set_name="M1_presale_175",
        shelf_life_version=shelf_life_choice,
        features=features_3a,
        run_name="M2_T09_test_baseline_run",
        purpose="setup_test",
        train_rows=len(X_train),
        test_rows=len(X_test),
        params={"C": 0.01, "solver": "lbfgs"},
    ) as run:
        run_id = run.info.run_id
        print(f"      Active Run ID: {run_id}")

        # Log Track 3A Metrics
        log_track_3a_metrics(
            brier_score=brier,
            log_loss_val=logloss,
            roc_auc=roc_auc,
            benchmark_auc=benchmark_auc,
            threshold=0.5,
            precision=prec,
            recall=rec,
            f1=f1,
        )

        # Log Feature List Artifact
        log_feature_list_artifact(features_3a, "track_3a_features.json")

        # Create and Log Reliability Curve Artifact
        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.plot(prob_pred, prob_true, "s-", color="#1f77b4", label="Baseline Classifier")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Reliability Curve (Test Run Baseline)")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle=":", alpha=0.6)
        log_figure_artifact(fig, "reliability_curve_test.png")
        plt.close(fig)

    print("      Run successfully logged and closed.")

    # 6. Verify Logged Run via MLflow Client
    print("\n[6/6] Verifying run retrieval via MLflowClient...")
    client = MlflowClient()
    retrieved_run = client.get_run(run_id)

    # Verify tags
    tags = retrieved_run.data.tags
    assert tags.get("track") == "3A", "Tag 'track' mismatch!"
    assert tags.get("purpose") == "setup_test", "Tag 'purpose' mismatch!"
    assert tags.get("no_spoilage_risk_leakage") == "true", "Tag 'no_spoilage_risk_leakage' mismatch!"
    assert tags.get("shelf_life_version") == shelf_life_choice, "Tag 'shelf_life_version' mismatch!"
    assert tags.get("run_author") == author, "Tag 'run_author' mismatch!"
    print(f"      [PASS] Tags verified (author='{author}', purpose='setup_test', track='3A')")

    # Verify metrics
    metrics = retrieved_run.data.metrics
    assert "test_brier_score" in metrics, "Metric 'test_brier_score' missing!"
    assert "test_log_loss" in metrics, "Metric 'test_log_loss' missing!"
    assert "test_roc_auc" in metrics, "Metric 'test_roc_auc' missing!"
    assert "benchmark_spoilage_risk_auc" in metrics, "Metric 'benchmark_spoilage_risk_auc' missing!"
    print(f"      [PASS] Metrics verified (Brier={metrics['test_brier_score']:.4f}, AUC={metrics['test_roc_auc']:.4f})")

    # Verify artifacts
    artifacts = client.list_artifacts(run_id)
    artifact_paths = [a.path for a in artifacts]
    assert "reliability_curve_test.png" in artifact_paths, "Artifact 'reliability_curve_test.png' missing!"
    assert "track_3a_features.json" in artifact_paths, "Artifact 'track_3a_features.json' missing!"
    print(f"      [PASS] Artifacts verified: {artifact_paths}")

    # Test Export and Merge Functions
    print("\nTesting summary export and merge functions...")
    author_csv = export_runs_summary(author=author, include_setup_tests=True)
    print(f"      [PASS] Author run summary exported: {author_csv.name}")
    assert author_csv.exists(), "Exported author CSV does not exist!"

    master_csv = merge_runs_summaries()
    print(f"      [PASS] Master run summary merged: {master_csv.name}")
    assert master_csv.exists(), "Merged master CSV does not exist!"

    df_exported = pd.read_csv(author_csv)
    print(f"      Exported rows: {len(df_exported)} | Run ID found: {run_id in df_exported['run_id'].values}")
    assert run_id in df_exported['run_id'].values, "Logged run ID missing from exported summary table!"

    print("\n==================================================================")
    print("M2-T09 MLflow Tracking Setup: ALL VERIFICATION CHECKS PASSED!")
    print(f"Logged Test Run ID: {run_id}")
    print("==================================================================")


if __name__ == "__main__":
    run_test()
