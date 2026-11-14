"""
scripts/M3_T02_closeout_track3a.py - Track 3A Evaluation Closeout, Final OOF & Frozen Scorer.

Executes M3-T02:
1. Computes test-period constant base-rate Brier and log loss.
2. Sweeps illustrative cost-asymmetric thresholds (k in {1, 2, 3, 5}) on CV OOF.
3. Regenerates time-blocked 5-fold OOF with the frozen final model -> track_3a_oof_train_v2.csv.
4. Scores test rows via score_batches() -> track_3a_proba_test.csv, verifies exact test Brier
   reproducibility and single-row simulator scoring.
5. Compares v1 OOF, v2 OOF, and test probability distributions + v1/v2 correlation.
6. Logs closeout run & v2 OOF artifact to MLflow smart-shelf-3a-spoilage-classifier.
"""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    REPORTS_DIR,
    TRACK_3A_OOF_TRAIN_PATH,
    TRACK_3A_OOF_TRAIN_V2_PATH,
    TRACK_3A_PROBA_TEST_PATH,
)
from src.m2_feature_pipeline import TRACK_3A_FEATURES_COMPACT
from src.m3_oof import generate_oof_predictions
from src.m3_t01_track3a import (
    evaluate_probs,
    load_track_3a_splits,
    make_candidate_factory,
    score_batches,
    sweep_cost_asymmetric_thresholds,
)
from src.mlflow_utils import (
    log_oof_predictions,
    log_track_3a_metrics,
    setup_tracking_environment,
    start_tracked_run,
)

REPORTS_M3_DIR = REPORTS_DIR / "M3"


def _dist_stats(arr: np.ndarray) -> dict:
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    setup_tracking_environment()
    REPORTS_M3_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("M3-T02: Track 3A Evaluation Closeout, Final OOF & Frozen Scorer")
    print("==================================================================")

    train_df, test_df = load_track_3a_splits()
    y_train = train_df["was_spoiled"].values
    y_test = test_df["was_spoiled"].values

    # 1. Test-period constant base-rate baseline
    train_base_rate = float(y_train.mean())
    test_const_proba = np.full(len(y_test), train_base_rate, dtype=float)
    test_const_metrics = evaluate_probs(y_test, test_const_proba)
    print(
        f"\n[1] Test Constant Base-Rate ({train_base_rate:.5f}): "
        f"Brier={test_const_metrics['brier']:.5f} | LogLoss={test_const_metrics['log_loss']:.5f}"
    )

    # 2. Regenerate time-blocked 5-fold OOF with frozen final config -> v2
    print("\n[2] Generating final time-blocked 5-fold OOF (v2) with frozen config...")
    frozen_factory = make_candidate_factory("logreg", {"C": 0.005}, needs_imputation=False)
    oof_v2_df = generate_oof_predictions(
        model_factory=frozen_factory,
        X=train_df[TRACK_3A_FEATURES_COMPACT],
        y=y_train,
        dates=train_df["transaction_date"].values,
        record_ids=train_df["record_id"].values,
        n_blocks=5,
        scheme="blocked",
        feature_set_name="track_3a_compact",
        model_type="logreg_uncalibrated",
        shelf_life_version="expiry_remaining_ratio",
    )
    oof_v2_df.to_csv(TRACK_3A_OOF_TRAIN_V2_PATH, index=False)
    v2_metrics = evaluate_probs(y_train, oof_v2_df["oof_proba"].values)
    print(
        f"    Saved {TRACK_3A_OOF_TRAIN_V2_PATH} ({len(oof_v2_df)} rows) | "
        f"Brier={v2_metrics['brier']:.5f} | LogLoss={v2_metrics['log_loss']:.5f} | AUC={v2_metrics['roc_auc']:.4f}"
    )

    # 3. Cost-asymmetric threshold sweep on CV OOF (training period only)
    print("\n[3] Illustrative cost-asymmetric threshold sweep on v2 CV OOF (train only):")
    cost_rows = sweep_cost_asymmetric_thresholds(y_train, oof_v2_df["oof_proba"].values, cost_ratios=(1, 2, 3, 5))
    for r in cost_rows:
        print(
            f"    k={r['cost_ratio_k']} | opt_thr={r['optimal_threshold']:.3f} | "
            f"exp_cost={r['expected_cost_per_batch']:.5f} | Recall={r['recall']:.4f} | "
            f"Prec={r['precision']:.4f} | PosRate={r['pred_pos_rate']:.4f}"
        )

    # 4. Frozen score_batches() verification + save track_3a_proba_test.csv
    print("\n[4] Testing score_batches() on test partition and single-row simulator input...")
    test_proba = score_batches(test_df)
    test_metrics = evaluate_probs(y_test, test_proba)

    # Verify exact match against stored M3-T01 test Brier
    m3_t01_summary = json.loads((REPORTS_M3_DIR / "m3_t01_results_summary.json").read_text())
    stored_test_brier = m3_t01_summary["test"]["model_metrics"]["brier"]
    brier_diff = abs(test_metrics["brier"] - stored_test_brier)
    assert brier_diff < 1e-12, f"Test Brier mismatch: {test_metrics['brier']} vs {stored_test_brier}"
    print(f"    [PASS] score_batches() reproduced stored test Brier={test_metrics['brier']:.8f} (diff={brier_diff:.2e})")

    single_proba = score_batches(test_df.iloc[[0]])
    assert single_proba.shape == (1,) and abs(float(single_proba[0]) - float(test_proba[0])) < 1e-12
    print(f"    [PASS] Single-row score_batches() returned {float(single_proba[0]):.6f} (exact match)")

    test_out_df = pd.DataFrame({
        "record_id": test_df["record_id"].values,
        "split": "test",
        "proba": test_proba,
    })
    test_out_df.to_csv(TRACK_3A_PROBA_TEST_PATH, index=False)
    print(f"    Saved {TRACK_3A_PROBA_TEST_PATH} ({len(test_out_df)} rows)")

    # 5. Distribution comparison: v1 OOF vs v2 OOF vs test proba
    oof_v1_df = pd.read_csv(TRACK_3A_OOF_TRAIN_PATH)
    aligned = oof_v1_df[["record_id", "oof_proba"]].merge(
        oof_v2_df[["record_id", "oof_proba"]],
        on="record_id",
        suffixes=("_v1", "_v2"),
    )
    corr_v1_v2 = float(aligned["oof_proba_v1"].corr(aligned["oof_proba_v2"]))
    v1_stats = _dist_stats(aligned["oof_proba_v1"].values)
    v2_stats = _dist_stats(aligned["oof_proba_v2"].values)
    test_stats = _dist_stats(test_proba)

    print(f"\n[5] Distribution Comparison (Pearson corr(v1, v2) = {corr_v1_v2:.4f}):")
    for label, st in [("OOF v1 (LGBM)", v1_stats), ("OOF v2 (LogReg)", v2_stats), ("Test (LogReg)", test_stats)]:
        print(
            f"    {label:16s} | mean={st['mean']:.4f} | std={st['std']:.4f} | "
            f"p05={st['p05']:.4f} | p50={st['p50']:.4f} | p95={st['p95']:.4f} | "
            f"min={st['min']:.4f} | max={st['max']:.4f}"
        )

    # 6. Log closeout run to MLflow
    bench_test_auc = float(roc_auc_score(y_test, test_df["spoilage_risk"].values))
    with start_tracked_run(
        track="3A",
        model_type="logreg_uncalibrated",
        feature_set_name="track_3a_compact",
        shelf_life_version="expiry_remaining_ratio",
        features=TRACK_3A_FEATURES_COMPACT,
        run_name="m3_t02_closeout_track_3a",
        purpose="evaluation_closeout",
        params={"C": 0.005, "penalty": "l2", "oof_version": "v2"},
    ):
        log_track_3a_metrics(
            brier_score=test_metrics["brier"],
            log_loss_val=test_metrics["log_loss"],
            roc_auc=test_metrics["roc_auc"],
            benchmark_auc=bench_test_auc,
            extra_metrics={
                "oof_v2_brier": v2_metrics["brier"],
                "oof_v2_log_loss": v2_metrics["log_loss"],
                "oof_v2_roc_auc": v2_metrics["roc_auc"],
                "test_const_brier": test_const_metrics["brier"],
                "test_const_log_loss": test_const_metrics["log_loss"],
                "oof_v1_v2_corr": corr_v1_v2,
            },
        )
        log_oof_predictions(oof_v2_df, name="track_3a_oof_train_v2")

    closeout = {
        "test_constant_base_rate": {
            "train_base_rate": train_base_rate,
            "metrics": test_const_metrics,
        },
        "oof_v2_metrics": v2_metrics,
        "cost_asymmetric_sweep_cv_oof_v2": cost_rows,
        "score_batches_check": {
            "test_brier": test_metrics["brier"],
            "stored_test_brier": stored_test_brier,
            "abs_diff": brier_diff,
            "single_row_proba": float(single_proba[0]),
        },
        "distributions": {
            "oof_v1_vs_v2_pearson_corr": corr_v1_v2,
            "oof_v1": v1_stats,
            "oof_v2": v2_stats,
            "test_proba": test_stats,
        },
    }
    out_json = REPORTS_M3_DIR / "m3_t02_closeout_summary.json"
    out_json.write_text(json.dumps(closeout, indent=2))
    print(f"\nSaved closeout summary: {out_json}")


if __name__ == "__main__":
    main()

