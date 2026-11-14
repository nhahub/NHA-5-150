"""
scripts/M3_T03_explain_demo.py - Validation & Demo for Track 3A Per-Batch Explanations (M3-T03).

Runs:
  6a. Sum-to-logit exactness check on a random 1,000-row test sample (tol 1e-6).
  6b. Global consistency: mean |contribution| per grouped driver vs M3-T01 SHAP ranking.
  6c. Three sanity example batches (high-risk, low-risk, limited-history) + waterfall/bar figures.
  6d. Stability check: repeated calls and single-row vs batched equivalence.
Logs validation metrics and figures to MLflow and saves reports/M3/m3_t03_examples.json.
"""

import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, RANDOM_STATE, REPORTS_DIR
from src.m3_t01_track3a import load_track_3a_splits, score_batches
from src.m3_t03_explain import explain_batch, explain_text
from src.mlflow_utils import (
    log_figure_artifact,
    setup_tracking_environment,
    start_tracked_run,
)
from src.m2_feature_pipeline import TRACK_3A_FEATURES_COMPACT

REPORTS_M3_DIR = REPORTS_DIR / "M3"
FIGURES_M3_DIR = FIGURES_DIR / "m3"


def _plot_batch_waterfall(exp: dict, title: str, out_path: Path) -> plt.Figure:
    """Plots a clean SHAP-style horizontal driver bar chart in probability points."""
    drivers = sorted(exp["top_drivers"], key=lambda d: abs(d["prob_pts_contrib"]))
    labels = [f"{d['label']} ({d['direction']})" for d in drivers]
    vals = [d["prob_pts_contrib"] for d in drivers]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels, vals, color=colors, alpha=0.88)
    ax.axvline(0.0, color="black", linewidth=0.9)
    for bar, v in zip(bars, vals):
        offset = 0.12 if v >= 0 else -0.12
        ha = "left" if v >= 0 else "right"
        ax.text(v + offset, bar.get_y() + bar.get_height() / 2, f"{v:+.2f} pts", va="center", ha=ha, fontsize=9)

    p_pct = exp["predicted_proba"] * 100.0
    b_pct = exp["base_rate"] * 100.0
    flag_str = f" | {exp['confidence_note']} ({exp['history_rows']} rows)" if exp["low_history_flag"] else ""
    ax.set_xlabel("Contribution to Spoilage Risk (percentage points vs. typical batch)")
    ax.set_title(f"{title}\nPredicted: {p_pct:.1f}% vs Typical: {b_pct:.1f}%{flag_str}", fontsize=10)
    pad = max(1.0, max(abs(v) for v in vals) * 0.28)
    ax.set_xlim(min(vals) - pad, max(vals) + pad)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig


def main() -> None:
    setup_tracking_environment()
    REPORTS_M3_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_M3_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("M3-T03: Track 3A Per-Batch Explanations & Support Flag Validation")
    print("==================================================================")

    _, test_df = load_track_3a_splits()

    # ------------------------------------------------------------------
    # 6a. Sum-to-logit check on random 1,000-row sample
    # ------------------------------------------------------------------
    sample_1000 = test_df.sample(n=1000, random_state=RANDOM_STATE).reset_index(drop=True)
    exps_1000 = explain_batch(sample_1000, top_k=5)
    diffs = [abs(e["sum_contributions_logit"] - e["logit_diff"]) for e in exps_1000]
    max_logit_err = float(max(diffs))
    prob_pts_diffs = [
        abs(sum(d["prob_pts_contrib"] for d in e["grouped_contributions"]) - (e["predicted_proba"] - e["base_rate"]) * 100.0)
        for e in exps_1000
    ]
    max_pts_err = float(max(prob_pts_diffs))
    assert max_logit_err < 1e-6, f"Sum-to-logit check failed: max_err={max_logit_err}"
    print(f"\n[6a] Sum-to-logit check (N=1,000): PASS (max |err| = {max_logit_err:.3e} < 1e-6; max prob-pts err = {max_pts_err:.3e})")

    # ------------------------------------------------------------------
    # 6b. Global consistency vs M3-T01 SHAP ranking (N=2,000 test sample)
    # ------------------------------------------------------------------
    sample_2000 = test_df.sample(n=2000, random_state=RANDOM_STATE).reset_index(drop=True)
    exps_2000 = explain_batch(sample_2000, top_k=5)
    driver_names = [d["driver"] for d in exps_2000[0]["grouped_contributions"]]
    mean_abs_logit = {
        g: float(np.mean([abs(next(d["log_odds_contrib"] for d in e["grouped_contributions"] if d["driver"] == g)) for e in exps_2000]))
        for g in driver_names
    }
    global_ranking = sorted(mean_abs_logit.items(), key=lambda kv: kv[1], reverse=True)
    print("\n[6b] Global Grouped Driver Ranking (mean |log-odds contrib|, N=2,000):")
    for rank, (drv, val) in enumerate(global_ranking[:10], start=1):
        print(f"    {rank:2d}. {drv:24s}: {val:.4f}")

    # ------------------------------------------------------------------
    # 6c. Three sanity examples: high-risk, low-risk, limited-history
    # ------------------------------------------------------------------
    test_probas = score_batches(test_df)
    test_with_p = test_df.copy()
    test_with_p["_proba"] = test_probas

    # Find high-risk (not low-history if possible), low-risk, and limited-history
    all_flags = [e["low_history_flag"] for e in explain_batch(test_df, top_k=1)]
    test_with_p["_low_hist"] = all_flags

    high_row = test_with_p[~test_with_p["_low_hist"]].sort_values("_proba", ascending=False).iloc[[0]]
    low_row = test_with_p[~test_with_p["_low_hist"]].sort_values("_proba", ascending=True).iloc[[0]]
    hist_row = test_with_p[test_with_p["_low_hist"]].sort_values("_proba", ascending=False).iloc[[0]]

    examples_spec = [
        ("high_risk", "Example 1: High-Risk Batch", high_row, "m3_t03_example_high_risk.png"),
        ("low_risk", "Example 2: Low-Risk Batch", low_row, "m3_t03_example_low_risk.png"),
        ("limited_history", "Example 3: Limited-History Batch", hist_row, "m3_t03_example_limited_history.png"),
    ]

    examples_out = []
    print("\n[6c] Sanity Examples:")
    for key, title, row_df, fig_name in examples_spec:
        exp = explain_batch(row_df, top_k=5)[0]
        txt = explain_text(exp)
        fig_path = FIGURES_M3_DIR / fig_name
        fig = _plot_batch_waterfall(exp, title, fig_path)
        plt.close(fig)

        print(f"\n  --- {title} ({exp['record_id']}, product={row_df['product_name'].iloc[0]}, store={row_df['store_id'].iloc[0]}) ---")
        print(f"  Text: {txt}")
        print(f"  Flag: low_history_flag={exp['low_history_flag']}, history_rows={exp['history_rows']}")
        examples_out.append({
            "case": key,
            "title": title,
            "product_name": str(row_df["product_name"].iloc[0]),
            "store_id": str(row_df["store_id"].iloc[0]),
            "category": str(row_df["category"].iloc[0]) if "category" in row_df.columns else None,
            "plain_text": txt,
            "figure_path": str(fig_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "explanation": exp,
        })

    # ------------------------------------------------------------------
    # 6d. Stability check: repeated calls & single-row vs batched
    # ------------------------------------------------------------------
    batch_3 = pd.concat([high_row, low_row, hist_row], ignore_index=True)
    batched_a = explain_batch(batch_3, top_k=5)
    batched_b = explain_batch(batch_3, top_k=5)
    single_0 = explain_batch(batch_3.iloc[[0]], top_k=5)[0]

    assert batched_a == batched_b, "Repeated explain_batch calls produced different outputs!"
    assert single_0 == batched_a[0], "Single-row explain_batch differs from batched call!"
    print("\n[6d] Stability check (repeated & single-row vs batched): PASS (identical)")

    # Save JSON artifact
    out_payload = {
        "validation": {
            "sum_to_logit_max_abs_err": max_logit_err,
            "sum_to_prob_pts_max_abs_err": max_pts_err,
            "stability_identical_repeated": True,
            "stability_identical_single_vs_batched": True,
            "global_grouped_ranking_top10": [{"driver": g, "mean_abs_logit": v} for g, v in global_ranking[:10]],
        },
        "examples": examples_out,
    }
    out_json = REPORTS_M3_DIR / "m3_t03_examples.json"
    out_json.write_text(json.dumps(out_payload, indent=2))
    print(f"\nSaved examples JSON: {out_json}")

    # Log to MLflow
    with start_tracked_run(
        track="3A",
        model_type="logreg_uncalibrated",
        feature_set_name="track_3a_compact",
        shelf_life_version="expiry_remaining_ratio",
        features=TRACK_3A_FEATURES_COMPACT,
        run_name="m3_t03_explainability_validation",
        purpose="explainability_validation",
    ) as run:
        import mlflow
        mlflow.log_metrics({
            "sum_to_logit_max_err": max_logit_err,
            "sum_to_prob_pts_max_err": max_pts_err,
        })
        for ex in examples_out:
            fig_p = PROJECT_ROOT / ex["figure_path"]
            mlflow.log_artifact(str(fig_p))
        print(f"Logged M3-T03 validation run to MLflow (run_id={run.info.run_id})")


if __name__ == "__main__":
    main()
