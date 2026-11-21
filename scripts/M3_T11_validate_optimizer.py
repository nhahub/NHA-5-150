"""
scripts/M3_T11_validate_optimizer.py - Track 3C Two-Lane Optimizer Validation (M3-T11).

Executes:
1. Pre-estimate integrity checks (region rule, R1=6,176, train-only tercile edges, zero R0 in lanes).
2. Lane 1 (observed evidence): followed vs not-followed within strata, week-grouped bootstrap 95% CI,
   200-permutation negative control, sensitivity at tol=0.05, and observed depth-band descriptives.
3. Lane 2 (simulated counterfactual): predicted gain on non-followed batches + model-error context.
4. Saves strictly separated JSON summary and lane-labeled figures; logs to MLflow.
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

from src.config import FIGURES_DIR, OPTIMIZER_RECS_TEST_PATH, REPORTS_DIR
from src.m2_feature_pipeline import TRACK_3B_FEATURES
from src.m3_t07_eval import load_train_test_3b_with_raw
from src.m3_t11_validation import (
    assign_r1_strata,
    derive_train_r1_strata_edges,
    lane1_followed_vs_not,
    lane2_counterfactual,
)
from src.mlflow_utils import (
    log_figure_artifact,
    log_track_3c_metrics,
    setup_tracking_environment,
    start_tracked_run,
)

REPORTS_M3_DIR = REPORTS_DIR / "M3"
FIGURES_M3_DIR = FIGURES_DIR / "m3"


def main() -> None:
    setup_tracking_environment()
    REPORTS_M3_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_M3_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("M3-T11: Track 3C Two-Lane Optimizer Validation (Test Partition)")
    print("==================================================================")

    tr, te = load_train_test_3b_with_raw()

    # ------------------------------------------------------------------
    # Integrity checks (before any estimate)
    # ------------------------------------------------------------------
    y_disc_te = te["raw_discount_pct"] > 0
    rule_te = (te["raw_expiry_remaining_ratio"] <= 0.40) | (te["quality_grade"] == "C")
    violations = int((y_disc_te != rule_te).sum())
    assert violations == 0, f"Test region rule violations: {violations}"

    tr_r1 = tr[tr["raw_discount_pct"] > 0].reset_index(drop=True)
    te_r1 = te[te["raw_discount_pct"] > 0].reset_index(drop=True)
    te_r0 = te[te["raw_discount_pct"] == 0].reset_index(drop=True)
    assert len(te_r1) == 6176, f"Expected 6,176 test R1 rows, got {len(te_r1)}"

    recs_te = pd.read_csv(OPTIMIZER_RECS_TEST_PATH)
    assert len(recs_te) == 6176 and (recs_te["region"] == "R1").all(), "R0 row or count mismatch in optimizer_recs_test.csv!"

    strata_edges = derive_train_r1_strata_edges(tr_r1)
    te_strata = assign_r1_strata(te_r1, strata_edges)
    te_r1_merged = pd.concat([te_r1, te_strata], axis=1).merge(recs_te, on="record_id", how="inner")
    assert len(te_r1_merged) == 6176 and (te_r1_merged["region"] == "R1").all()

    r0_share_pct = float(len(te_r0) / len(te) * 100.0)
    print(f"\n[Integrity] 0 rule violations | Test R1={len(te_r1)} ({100-r0_share_pct:.2f}%), Test R0={len(te_r0)} ({r0_share_pct:.2f}%, excluded)")

    # ------------------------------------------------------------------
    # Lane 1: Observed Evidence (Followed vs Not Followed)
    # ------------------------------------------------------------------
    print("\n[Lane 1 — Observed Evidence] Running tol=0.025 (primary) & tol=0.05 (sensitivity)...")
    l1_prim = lane1_followed_vs_not(te_r1_merged, tol=0.025, min_per_arm=20, n_boot=1000, n_perm=200, run_neg_control=True)
    l1_sens = lane1_followed_vs_not(te_r1_merged, tol=0.050, min_per_arm=20, n_boot=1000, n_perm=0, run_neg_control=False)

    print("  Strata counts (tol=0.025):")
    for st in l1_prim["strata_table"]:
        inc_lbl = "INC" if st["included"] else "EXC"
        print(f"    {st['stratum_id']:10s} | tot={st['n_total']:4d} | followed={st['n_followed']:3d} | not_followed={st['n_not_followed']:4d} | {inc_lbl}")

    print(
        f"  Primary (tol=0.025): {l1_prim['n_included_strata']}/{l1_prim['n_total_strata']} strata, "
        f"{l1_prim['n_batches_included']} batches ({l1_prim['coverage_share_pct']:.1f}% of test R1), "
        f"followed={l1_prim['n_followed_included']} -> VERDICT: {l1_prim['verdict']}"
    )
    for k in ["profit_per_unit", "s", "waste_share", "profit"]:
        est = l1_prim["estimates"][k]
        lo, hi = l1_prim["ci95"][k]
        print(f"    {k:16s}: diff={est:+.4f}, 95% CI=[{lo:+.4f}, {hi:+.4f}]")
    nc = l1_prim["negative_control"]
    print(
        f"  Negative Control (200 within-stratum permutations): null_mean={nc['null_mean']:+.4f}, "
        f"SD={nc['null_sd']:.4f}, 95%=[{nc['null_p025']:+.4f}, {nc['null_p975']:+.4f}], "
        f"real={nc['real_estimate']:+.4f}, p={nc['empirical_two_sided_p']:.3f}"
    )
    print(
        f"  Sensitivity (tol=0.050): {l1_sens['n_included_strata']}/{l1_sens['n_total_strata']} strata, "
        f"followed={l1_sens['n_followed_included']}, profit/u diff={l1_sens['estimates']['profit_per_unit']:+.4f} "
        f"[{l1_sens['ci95']['profit_per_unit'][0]:+.4f}, {l1_sens['ci95']['profit_per_unit'][1]:+.4f}] -> {l1_sens['verdict']}"
    )

    # Lane 1 Figure
    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    vals = [l1_prim["estimates"]["profit_per_unit"], l1_sens["estimates"]["profit_per_unit"], nc["null_mean"]]
    los = [
        l1_prim["estimates"]["profit_per_unit"] - l1_prim["ci95"]["profit_per_unit"][0],
        l1_sens["estimates"]["profit_per_unit"] - l1_sens["ci95"]["profit_per_unit"][0],
        nc["null_mean"] - nc["null_p025"],
    ]
    his = [
        l1_prim["ci95"]["profit_per_unit"][1] - l1_prim["estimates"]["profit_per_unit"],
        l1_sens["ci95"]["profit_per_unit"][1] - l1_sens["estimates"]["profit_per_unit"],
        nc["null_p975"] - nc["null_mean"],
    ]
    labels = ["Lane 1 Primary\n(tol=0.025)", "Lane 1 Sensitivity\n(tol=0.050)", "Permutation Null\n(200x within strata)"]
    ax1.errorbar(labels, vals, yerr=[los, his], fmt="o", capsize=6, markersize=8, color="#1f77b4", linewidth=2)
    ax1.axhline(0.0, color="red", linestyle="--", linewidth=1)
    ax1.set_ylabel("Observed Profit / Unit Diff (Followed - Not Followed, $)")
    ax1.set_title(f"Lane 1 (Observed Evidence, Not Model-Evaluated): Verdict = {l1_prim['verdict'].upper()}")
    ax1.grid(axis="y", alpha=0.25)
    fig1.tight_layout()
    fig1.savefig(FIGURES_M3_DIR / "m3_t11_lane1_observed_evidence.png", dpi=150)

    # ------------------------------------------------------------------
    # Lane 2: Counterfactual Simulated (Never Blended with Lane 1)
    # ------------------------------------------------------------------
    print("\n[Lane 2 — Simulated / Model-Estimated, Not Observed] Running counterfactual scoring...")
    l2 = lane2_counterfactual(te_r1_merged, tol=0.025, n_boot=1000)
    print(
        f"  N={l2['n_batches']} batches | mean gain/unit=${l2['mean_gain_unit']:+.4f} "
        f"[{l2['ci95_mean_gain_unit'][0]:+.4f}, {l2['ci95_mean_gain_unit'][1]:+.4f}] | "
        f"median=${l2['median_gain_unit']:+.4f}"
    )
    print(
        f"  mean gain/batch=${l2['mean_gain_batch']:+.2f} "
        f"[{l2['ci95_mean_gain_batch'][0]:+.2f}, {l2['ci95_mean_gain_batch'][1]:+.2f}] | "
        f"median=${l2['median_gain_batch']:+.2f}"
    )
    print(
        f"  positive_gain={l2['share_positive_gain_pct']:.1f}% | "
        f"below_CV_MAE(${l2['cv_profit_unit_mae']:.2f})={l2['share_below_cv_mae_pct']:.1f}%"
    )
    me = l2["model_error_at_actual_depth"]
    print(
        f"  Model error at actual depth: MAE=${me['mae_profit_per_unit']:.2f}/u, "
        f"RMSE=${me['rmse_profit_per_unit']:.2f}/u, mean_err(pred-obs)=${me['mean_error_pred_minus_obs']:+.2f}/u "
        f"(mean snap={me['mean_snap_dist']:.4f})"
    )

    # Lane 2 Figure
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    tiers_l2 = sorted(l2["by_tier"], key=lambda r: ["Low", "Medium", "High", "Critical"].index(r["tier"]))
    t_names = [f"{r['tier']}\n(n={r['n']})" for r in tiers_l2]
    t_means = [r["mean_gain_unit"] for r in tiers_l2]
    t_meds = [r["median_gain_unit"] for r in tiers_l2]
    x = np.arange(len(t_names))
    ax2.bar(x - 0.18, t_means, width=0.35, label="Mean simulated gain ($/unit)")
    ax2.bar(x + 0.18, t_meds, width=0.35, label="Median simulated gain ($/unit)")
    ax2.axhline(l2["cv_profit_unit_mae"], color="red", linestyle="--", linewidth=1.2, label=f"Regressor CV MAE (${l2['cv_profit_unit_mae']:.2f}/u)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(t_names)
    ax2.set_ylabel("Simulated Gain ($ / Initial Unit)")
    ax2.set_title("Lane 2 (Simulated / Model-Estimated, Not Observed): Gain by Tier vs. Model Error")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(FIGURES_M3_DIR / "m3_t11_lane2_counterfactual_simulated.png", dpi=150)

    # MLflow logging (strictly separated dicts)
    with start_tracked_run(
        track="3C",
        model_type="two_lane_optimizer_validation",
        feature_set_name="track_3b_r1_44",
        shelf_life_version="expiry_remaining_ratio",
        features=list(TRACK_3B_FEATURES),
        run_name="m3_t11_two_lane_validation",
        purpose="optimizer_validation",
        train_rows=len(tr_r1),
        test_rows=len(te_r1),
    ) as run:
        log_track_3c_metrics(
            lane1_metrics={
                "n_included_strata": l1_prim["n_included_strata"],
                "n_batches_included": l1_prim["n_batches_included"],
                "n_followed_included": l1_prim["n_followed_included"],
                "profit_per_unit_diff": l1_prim["estimates"]["profit_per_unit"],
                "s_diff": l1_prim["estimates"]["s"],
                "waste_share_diff": l1_prim["estimates"]["waste_share"],
                "batch_profit_diff": l1_prim["estimates"]["profit"],
                "neg_ctrl_null_mean": nc["null_mean"],
                "neg_ctrl_p": nc["empirical_two_sided_p"],
            },
            lane2_metrics={
                "n_batches": l2["n_batches"],
                "mean_gain_unit": l2["mean_gain_unit"],
                "median_gain_unit": l2["median_gain_unit"],
                "mean_gain_batch": l2["mean_gain_batch"],
                "median_gain_batch": l2["median_gain_batch"],
                "share_below_cv_mae_pct": l2["share_below_cv_mae_pct"],
                "actual_depth_mae_unit": me["mae_profit_per_unit"],
            },
            extra_metrics={"test_r0_excluded_share_pct": r0_share_pct},
        )
        log_figure_artifact(fig1, "m3_t11_lane1_observed_evidence.png")
        log_figure_artifact(fig2, "m3_t11_lane2_counterfactual_simulated.png")
        mlflow_run_id = run.info.run_id

    plt.close("all")

    payload = {
        "integrity": {
            "test_total": len(te),
            "test_r1": len(te_r1),
            "test_r0_excluded": len(te_r0),
            "test_r0_share_pct": r0_share_pct,
            "rule_violations": violations,
        },
        "lane1": {
            "primary_tol_025": l1_prim,
            "sensitivity_tol_050": {k: v for k, v in l1_sens.items() if k != "observed_depth_band_by_stratum"},
        },
        "lane2": l2,
        "mlflow_run_id": mlflow_run_id,
    }
    out_json = REPORTS_M3_DIR / "m3_t11_results_summary.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved summary JSON: {out_json}")
    print(f"Logged MLflow run: {mlflow_run_id}")


if __name__ == "__main__":
    main()
