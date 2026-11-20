"""
scripts/M3_T07_evaluate_track3b.py - Track 3B Single Final Test Evaluation & Shape Check.

Executes M3-T07:
1. Test-set integrity (region rule violations, R0/R1 sizes, depth support, distribution shift).
2. Single test evaluation of frozen Track 3B regressor on test R1 + Rule T1 check.
3. Descriptive stats & predict_profit_curve() guard verification on test R0.
4. Unconstrained diagnostic fit on train R1 + Rule T2 shape check + test residual slope + censoring.
5. Generates calibration & shape comparison figures and logs to MLflow.
"""

import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, REPORTS_DIR, TRACK_3B_MODEL_PATH
from src.m3_t04_scenario import DISCOUNT_BIN_LABELS, assign_discount_bin
from src.m3_t05_confounding import (
    DEFAULT_DISCOUNT_GRID,
    _build_region_support_reference,
    discount_support,
)
from src.m3_t06_regressor import GroupMeanBaseline, make_3b_model, predict_profit_curve
from src.m3_t07_eval import (
    compute_r1_metrics,
    compute_test_residual_slope,
    dist_summary,
    evaluate_rule_t2_shape,
    load_train_test_3b_with_raw,
)
from src.mlflow_utils import (
    load_model_with_metadata,
    log_figure_artifact,
    log_track_3b_metrics,
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
    print("M3-T07: Track 3B Final Test Evaluation & Shape Check (Single Use)")
    print("==================================================================")

    frozen_model, meta = load_model_with_metadata(TRACK_3B_MODEL_PATH)
    features_3b = list(meta["feature_list"])
    disc_slope = float(meta["disc_scale_slope"])
    disc_intercept = float(meta["disc_scale_intercept"])
    cv_rmse = float(meta["cv_scores"]["s_rmse"])
    cv_r2 = float(meta["cv_scores"]["s_r2"])

    tr, te = load_train_test_3b_with_raw()

    # ------------------------------------------------------------------
    # Step 1: Test-set integrity
    # ------------------------------------------------------------------
    y_disc_te = te["raw_discount_pct"] > 0
    rule_te = (te["raw_expiry_remaining_ratio"] <= 0.40) | (te["quality_grade"] == "C")
    violations = int((y_disc_te != rule_te).sum())
    assert violations == 0, f"STOP: {violations} region rule violations on test!"

    tr_r1 = tr[tr["raw_discount_pct"] > 0].reset_index(drop=True)
    te_r1 = te[te["raw_discount_pct"] > 0].reset_index(drop=True)
    te_r0 = te[te["raw_discount_pct"] == 0].reset_index(drop=True)

    # Depth distribution of test R1 + supported training bin share
    te_r1_bins = assign_discount_bin(te_r1["raw_discount_pct"]).astype(str)
    supp_te_grid, _ = discount_support(te_r1, grid=DEFAULT_DISCOUNT_GRID)
    grid_bins = [str(assign_discount_bin(pd.Series([d]))[0]) for d in DEFAULT_DISCOUNT_GRID]
    bin_to_col = {}
    for col, b in zip(supp_te_grid.columns, grid_bins):
        bin_to_col.setdefault(b, col)
    obs_supported = np.array([bool(supp_te_grid.iloc[i][bin_to_col[te_r1_bins[i]]]) for i in range(len(te_r1))])
    te_r1_supported_share = float(obs_supported.mean())

    depth_dist_te = pd.Series(te_r1_bins).value_counts().reindex(DISCOUNT_BIN_LABELS[1:], fill_value=0).to_dict()
    shift_check = {
        col: {"train_r1": dist_summary(tr_r1[col].to_numpy(dtype=float)), "test_r1": dist_summary(te_r1[col].to_numpy(dtype=float))}
        for col in ["oof_proba", "raw_expiry_remaining_ratio", "s"]
    }

    print(f"\n[1] Test Integrity: 0 rule violations | R0={len(te_r0)} ({len(te_r0)/len(te)*100:.2f}%), R1={len(te_r1)} ({len(te_r1)/len(te)*100:.2f}%)")
    print(f"    Test R1 observed depth in supported training bin: {te_r1_supported_share*100:.2f}%")
    for col, sd in shift_check.items():
        print(f"    Shift {col:26s}: train mean={sd['train_r1']['mean']:.4f} (sd={sd['train_r1']['std']:.4f}) | test mean={sd['test_r1']['mean']:.4f} (sd={sd['test_r1']['std']:.4f})")

    # ------------------------------------------------------------------
    # Step 2: Primary Evaluation on Test R1 + Rule T1
    # ------------------------------------------------------------------
    pred_s_te = np.clip(frozen_model.predict(te_r1[features_3b]), 0.0, 1.0)
    test_metrics = compute_r1_metrics(te_r1, pred_s_te)

    ref = _build_region_support_reference()
    base_model = GroupMeanBaseline().fit(
        tr_r1[features_3b], tr_r1["s"].to_numpy(dtype=float),
        disc_slope, disc_intercept, float(ref["grade_c_ord_cutoff"]),
    )
    pred_base_te = base_model.predict(te_r1[features_3b])
    base_test_metrics = compute_r1_metrics(te_r1, pred_base_te)

    rmse_limit = cv_rmse * 1.15
    r2_limit = cv_r2 - 0.05
    rule_t1_passed = bool((test_metrics["s_rmse"] <= rmse_limit) and (test_metrics["s_r2"] >= r2_limit))

    print(f"\n[2] Test R1 Primary Evaluation (N={len(te_r1)}):")
    print(f"    Frozen Model : s_RMSE={test_metrics['s_rmse']:.5f} (CV={cv_rmse:.5f}), s_MAE={test_metrics['s_mae']:.5f}, s_R2={test_metrics['s_r2']:.4f} (CV={cv_r2:.4f})")
    print(f"                   profit/u RMSE=${test_metrics['profit_unit_rmse']:.2f}, MAE=${test_metrics['profit_unit_mae']:.2f} | batch RMSE=${test_metrics['profit_rmse_all']:.0f} (no1%=${test_metrics['profit_rmse_no_ext1pct']:.0f})")
    print(f"    Group Baseline: s_RMSE={base_test_metrics['s_rmse']:.5f}, s_MAE={base_test_metrics['s_mae']:.5f}, s_R2={base_test_metrics['s_r2']:.4f}, profit/u RMSE=${base_test_metrics['profit_unit_rmse']:.2f}")
    print(f"    Rule T1 (RMSE <= {rmse_limit:.5f} AND R2 >= {r2_limit:.4f}): PASSED={rule_t1_passed}")

    # Calibration & subgroup errors
    te_eval = pd.DataFrame({
        "obs_s": te_r1["s"].to_numpy(dtype=float),
        "pred_s": pred_s_te,
        "bin": te_r1_bins,
        "grade_grp": np.where(te_r1["quality_grade"] == "C", "Grade_C", "Grade_AB"),
    })
    te_eval["decile"] = pd.qcut(te_eval["pred_s"].rank(method="first"), q=10, labels=False) + 1
    r1_edges = np.asarray(ref["r1_exp_tercile_edges"], dtype=float)
    te_eval["exp_terc"] = np.digitize(te_r1["raw_expiry_remaining_ratio"].to_numpy(dtype=float), r1_edges[1:-1], right=True) + 1

    cal_dec = te_eval.groupby("decile", observed=True).agg(n=("obs_s", "count"), mean_pred=("pred_s", "mean"), mean_obs=("obs_s", "mean")).reset_index().to_dict(orient="records")
    cal_bin = te_eval.groupby("bin", observed=True).agg(n=("obs_s", "count"), mean_pred=("pred_s", "mean"), mean_obs=("obs_s", "mean")).reindex(DISCOUNT_BIN_LABELS[1:]).reset_index().to_dict(orient="records")

    sub_err = {}
    for col in ["exp_terc", "grade_grp"]:
        sub_err[col] = {
            str(k): compute_r1_metrics(te_r1.loc[te_eval[col] == k], pred_s_te[te_eval[col] == k])
            for k in sorted(te_eval[col].unique())
        }

    # Plot test calibration
    fig_cal, axes_c = plt.subplots(1, 2, figsize=(12, 4.5))
    d_df = pd.DataFrame(cal_dec)
    axes_c[0].plot(d_df["mean_pred"], d_df["mean_obs"], marker="o", label="Test R1 Observed")
    axes_c[0].plot([0, 1], [0, 1], "k--", label="Perfect")
    axes_c[0].set_xlabel("Mean Predicted s")
    axes_c[0].set_ylabel("Mean Observed s")
    axes_c[0].set_title("Test R1 Calibration by Predicted-s Decile")
    axes_c[0].legend()
    axes_c[0].grid(alpha=0.25)

    b_df = pd.DataFrame(cal_bin)
    axes_c[1].plot(b_df["bin"], b_df["mean_pred"], marker="o", label="Mean Predicted s")
    axes_c[1].plot(b_df["bin"], b_df["mean_obs"], marker="s", label="Mean Observed s")
    axes_c[1].set_xlabel("Discount Depth Bin")
    axes_c[1].set_ylabel("Mean Sell-Through s")
    axes_c[1].set_title("Test R1 Calibration by 5-Point Depth Bin")
    axes_c[1].tick_params(axis="x", rotation=40)
    axes_c[1].legend()
    axes_c[1].grid(alpha=0.25)
    fig_cal.tight_layout()
    fig_cal.savefig(FIGURES_M3_DIR / "m3_t07_test_calibration.png", dpi=150)

    # ------------------------------------------------------------------
    # Step 3: Test R0 Descriptive Facts & Guard Verification
    # ------------------------------------------------------------------
    r0_prof_u = (te_r0["profit"] / te_r0["raw_initial_quantity"]).to_numpy(dtype=float)
    r0_desc = {
        "n_rows": int(len(te_r0)),
        "s_mean": float(te_r0["s"].mean()),
        "profit_per_unit_mean": float(r0_prof_u.mean()),
        "loss_share_pct": float((te_r0["profit"] < 0).mean() * 100.0),
    }
    r0_curves = predict_profit_curve(te_r0.head(10))
    assert (r0_curves["reason"] == "region_R0_no_discount_evidence").all() and r0_curves["s_hat"].isna().all()
    print(f"\n[3] Test R0 Descriptive (N={r0_desc['n_rows']}): s_mean={r0_desc['s_mean']:.4f}, profit/u=${r0_desc['profit_per_unit_mean']:.2f}, loss={r0_desc['loss_share_pct']:.1f}% | predict_profit_curve guard=PASS")

    # ------------------------------------------------------------------
    # Step 4: Shape Sanity Check (Unconstrained Diagnostic + Rule T2 + Test Slope + Censoring)
    # ------------------------------------------------------------------
    print("\n[4] Running Shape Sanity Check (unconstrained diagnostic on train R1)...")
    frozen_params = {
        "n_estimators": 500, "learning_rate": 0.08149619212376244, "num_leaves": 15,
        "min_child_samples": 93, "subsample": 0.985296571922335,
        "colsample_bytree": 0.9950553116457923, "reg_lambda": 7.480534388702453,
    }
    unc_model = make_3b_model("lgbm_ce", frozen_params, features_3b)
    unc_model.fit(tr_r1[features_3b], tr_r1["s"].to_numpy(dtype=float))

    t2_res = evaluate_rule_t2_shape(tr_r1, unc_model, frozen_model, features_3b, disc_slope, disc_intercept)
    print(
        f"    Rule T2 (unconstrained): {t2_res['n_t2_cells_ge_200']} cells >=200 rows | "
        f"positive_slope={t2_res['positive_slope_share']*100:.1f}% | any_sig_neg={t2_res['any_sig_negative']} -> PASSED={t2_res['rule_t2_passed']}"
    )
    for gs in t2_res["group_shapes"]:
        print(f"    Profit shape {gs['group']:16s}: {gs['shape_constrained']}")

    controls_only = [c for c in features_3b if c != "discount_pct"]
    te_slope = compute_test_residual_slope(tr_r1, te_r1, controls_only, n_boot=500)
    print(
        f"    Test-period empirical residual slope: {te_slope['slope']:+.4f} "
        f"(95% week-boot CI [{te_slope['ci95_low']:+.4f}, {te_slope['ci95_high']:+.4f}]) "
        f"vs M3-T05 (+0.2614) & M3-T06 (+0.2231)"
    )

    censor_te = []
    for b in DISCOUNT_BIN_LABELS[1:]:
        sub = te_r1[te_r1_bins == b]
        censor_te.append({
            "depth_bin": b,
            "n_rows": int(len(sub)),
            "s_eq_1_pct": float((sub["s"] == 1.0).mean() * 100.0) if len(sub) else 0.0,
        })

    # Plot constrained vs unconstrained shapes
    fig_sh, axes_sh = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for idx_g, gs in enumerate(t2_res["group_shapes"][:3]):  # 3 expiry terciles
        ax_s = axes_sh[0, idx_g]
        ax_p = axes_sh[1, idx_g]
        d = gs["depths"]
        ax_s.plot(d, gs["s_constrained"], "o-", label="Frozen (Monotone)")
        ax_s.plot(d, gs["s_unconstrained"], "s--", label="Unconstrained Diag")
        ax_s.set_title(f"{gs['group']} (n={gs['n_rows']})", fontsize=10)
        ax_p.plot(d, gs["profit_constrained"], "o-", label="Frozen (Monotone)")
        ax_p.plot(d, gs["profit_unconstrained"], "s--", label="Unconstrained Diag")
        ax_p.set_xlabel("Discount Depth")
        if idx_g == 0:
            ax_s.set_ylabel("Predicted Sell-Through s")
            ax_p.set_ylabel("Predicted Profit / Unit ($)")
            ax_s.legend(fontsize=8)
        ax_s.grid(alpha=0.25)
        ax_p.grid(alpha=0.25)
    fig_sh.suptitle("M3-T07 Shape Check: Frozen (Monotone) vs. Unconstrained Diagnostic (Supported Depths)", fontsize=12)
    fig_sh.tight_layout(rect=[0, 0, 1, 0.95])
    fig_sh.savefig(FIGURES_M3_DIR / "m3_t07_shape_constrained_vs_unconstrained.png", dpi=150)

    # ------------------------------------------------------------------
    # MLflow logging (Final Test Evaluation + Unconstrained Shape Diagnostic)
    # ------------------------------------------------------------------
    with start_tracked_run(
        track="3B",
        model_type="lgbm_ce_monotone",
        feature_set_name="track_3b_r1_44",
        shelf_life_version="expiry_remaining_ratio",
        features=features_3b,
        run_name="m3_t07_final_test_evaluation",
        purpose="final_test_evaluation",
        train_rows=len(tr_r1),
        test_rows=len(te_r1),
    ) as run_final:
        log_track_3b_metrics(
            profit_mae=test_metrics["profit_mae_all"],
            profit_rmse=test_metrics["profit_rmse_all"],
            profit_r2=test_metrics["s_r2"],
            sell_through_mae=test_metrics["s_mae"],
            sell_through_rmse=test_metrics["s_rmse"],
            sell_through_r2=test_metrics["s_r2"],
            extra_metrics={
                "profit_unit_mae": test_metrics["profit_unit_mae"],
                "profit_unit_rmse": test_metrics["profit_unit_rmse"],
                "test_residual_slope": te_slope["slope"],
                "rule_t1_passed": float(rule_t1_passed),
                "rule_t2_passed": float(t2_res["rule_t2_passed"]),
            },
        )
        log_figure_artifact(fig_cal, "m3_t07_test_calibration.png")
        log_figure_artifact(fig_sh, "m3_t07_shape_constrained_vs_unconstrained.png")
        final_run_id = run_final.info.run_id

    with start_tracked_run(
        track="3B",
        model_type="lgbm_ce_unconstrained_diag",
        feature_set_name="track_3b_r1_44",
        shelf_life_version="expiry_remaining_ratio",
        features=features_3b,
        run_name="m3_t07_shape_diagnostic_unconstrained",
        purpose="shape_diagnostic",
        train_rows=len(tr_r1),
        test_rows=len(te_r1),
        params=frozen_params,
    ) as run_diag:
        mlflow.log_metrics({
            "rule_t2_positive_slope_share": t2_res["positive_slope_share"],
            "rule_t2_n_cells": t2_res["n_t2_cells_ge_200"],
            "rule_t2_passed": float(t2_res["rule_t2_passed"]),
        })
        diag_run_id = run_diag.info.run_id

    plt.close("all")

    summary = {
        "integrity": {
            "test_total": len(te),
            "rule_violations": violations,
            "test_r0": len(te_r0),
            "test_r1": len(te_r1),
            "test_r1_supported_depth_share": te_r1_supported_share,
            "test_r1_depth_bin_counts": depth_dist_te,
            "distribution_shift": shift_check,
        },
        "test_r1_metrics": test_metrics,
        "baseline_r1_test_metrics": base_test_metrics,
        "rule_t1": {
            "cv_rmse": cv_rmse, "cv_r2": cv_r2,
            "rmse_limit": rmse_limit, "r2_limit": r2_limit,
            "passed": rule_t1_passed,
        },
        "calibration": {"by_pred_decile": cal_dec, "by_depth_bin": cal_bin},
        "subgroup_errors": sub_err,
        "test_r0_descriptive": r0_desc,
        "shape_check": {
            "rule_t2": {k: v for k, v in t2_res.items() if k != "group_shapes"},
            "group_shapes": t2_res["group_shapes"],
            "test_empirical_slope": te_slope,
            "test_censoring_s_eq_1": censor_te,
        },
        "mlflow_final_run_id": final_run_id,
        "mlflow_diag_run_id": diag_run_id,
    }
    out_json = REPORTS_M3_DIR / "m3_t07_results_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved M3-T07 summary JSON: {out_json}")


if __name__ == "__main__":
    main()
