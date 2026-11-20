"""
scripts/M3_T05_diagnose_confounding.py - Track 3B Confounding & Selection-Bias Diagnosis.

Executes M3-T05 strictly on training rows (split == "train", N = 80,042):
1. Counterfactual-consistency check on TRACK_3B_FEATURES + oof_proba.
2. Propensity diagnostics (5-fold CV AUC, Brier, extreme shares, discounted R^2, depth-3 tree).
3. Fixed LightGBM cross_entropy models: NAIVE, CONTROLLED, and 6 control ablations.
4. 5x5 binned diagnostic plots (s and profit/unit) with Rule S support marking + examples.
5. Mechanical Rule U, Rule M, Rule D verdicts for NAIVE vs CONTROLLED + ablation sensitivity.
6. Censoring check (s==0 and s==1 shares by discount bin).
7. Reusable discount_support() verification including shortest-expiry decile row.
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

from src.config import FIGURES_DIR, REPORTS_DIR
from src.m2_feature_pipeline import TRACK_3B_FEATURES
from src.m3_t04_scenario import DISCOUNT_BIN_LABELS, assign_discount_bin
from src.m3_t05_confounding import (
    ABLATION_GROUPS,
    DEFAULT_DISCOUNT_GRID,
    _assign_quintile_from_edges,
    _build_support_reference,
    discount_support,
    evaluate_pre_registered_rules,
    fit_cross_entropy_oof_and_grid,
    load_3b_train_only,
    run_propensity_diagnostics,
)
from src.mlflow_utils import (
    log_figure_artifact,
    setup_tracking_environment,
    start_tracked_run,
)

REPORTS_M3_DIR = REPORTS_DIR / "M3"
FIGURES_M3_DIR = FIGURES_DIR / "m3"


def _plot_5x5_diagnostic(
    tr: pd.DataFrame,
    naive_verdict: dict,
    ctrl_verdict: dict,
    metric: str,
    out_path: Path,
) -> plt.Figure:
    """Plots 5x5 small multiples of raw observed, NAIVE, and CONTROLLED curves with Rule S."""
    ref = _build_support_reference()
    oof_q = _assign_quintile_from_edges(tr["oof_proba"].to_numpy(dtype=float), ref["oof_edges"])
    exp_q = _assign_quintile_from_edges(tr["expiry_remaining_ratio"].to_numpy(dtype=float), ref["exp_edges"])
    tr_tmp = tr.copy()
    tr_tmp["oof_q"] = oof_q
    tr_tmp["exp_q"] = exp_q
    tr_tmp["bin"] = assign_discount_bin(tr_tmp["raw_discount_pct"]).astype(str)
    tr_tmp["profit_unit"] = tr_tmp["profit"] / tr_tmp["raw_initial_quantity"]

    grid = np.asarray(DEFAULT_DISCOUNT_GRID, dtype=float)
    grid_bins = [str(assign_discount_bin(pd.Series([d]))[0]) for d in grid]

    fig, axes = plt.subplots(5, 5, figsize=(16, 13), sharex=True, sharey=True)
    for oq in range(1, 6):
        for eq in range(1, 6):
            ax = axes[5 - oq, eq - 1]  # top row = highest risk (oq=5), left-to-right = shortest to longest expiry
            sub = tr_tmp[(tr_tmp["oof_q"] == oq) & (tr_tmp["exp_q"] == eq)]
            obs_col = "s" if metric == "s" else "profit_unit"
            obs_by_bin = sub.groupby("bin", observed=True)[obs_col].mean().to_dict()
            obs_cnt_by_bin = sub.groupby("bin", observed=True).size().to_dict()

            idx = (oq - 1) * 5 + (eq - 1)
            n_det = naive_verdict["strata_details"][idx]
            c_det = ctrl_verdict["strata_details"][idx]
            supp = np.asarray(c_det["supported_mask"], dtype=bool)

            key = "mean_pred_s" if metric == "s" else "mean_pred_profit_unit"
            y_naive = np.asarray(n_det[key], dtype=float)
            y_ctrl = np.asarray(c_det[key], dtype=float)

            # Plot unsupported dashed grey, supported solid colored
            ax.plot(grid, y_naive, color="#ff7f0e", linestyle="--", alpha=0.45, linewidth=1.0)
            ax.plot(grid, y_ctrl, color="#1f77b4", linestyle="--", alpha=0.45, linewidth=1.0)
            if supp.any():
                ax.plot(grid[supp], y_naive[supp], color="#ff7f0e", linewidth=1.8, label="NAIVE" if (oq == 5 and eq == 1) else None)
                ax.plot(grid[supp], y_ctrl[supp], color="#1f77b4", linewidth=2.0, label="CONTROLLED" if (oq == 5 and eq == 1) else None)

            # Observed supported points
            obs_x = [grid[j] for j, b in enumerate(grid_bins) if obs_cnt_by_bin.get(b, 0) >= 30 and b in obs_by_bin]
            obs_y = [obs_by_bin[grid_bins[j]] for j, b in enumerate(grid_bins) if obs_cnt_by_bin.get(b, 0) >= 30 and b in obs_by_bin]
            if obs_x:
                ax.scatter(obs_x, obs_y, color="black", s=14, zorder=4, label="Observed (>=30)" if (oq == 5 and eq == 1) else None)

            ax.set_title(f"Risk Q{oq} x Exp Q{eq} (n={len(sub)})", fontsize=8)
            ax.grid(alpha=0.2)
            if eq == 1:
                ax.set_ylabel("Sell-Through s" if metric == "s" else "Profit / Unit ($)", fontsize=8)
            if oq == 1:
                ax.set_xlabel("Discount", fontsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.995))
    title_str = "Sell-Through s vs. Discount" if metric == "s" else "Profit per Initial Unit vs. Discount"
    fig.suptitle(f"M3-T05 Binned Diagnostic (5x5 Strata, Solid = Rule S >=30, Dashed = Unsupported): {title_str}", y=0.97, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    return fig


def _plot_example_strata(
    tr: pd.DataFrame,
    naive_verdict: dict,
    ctrl_verdict: dict,
    out_path: Path,
) -> plt.Figure:
    """Plots 3 representative strata (Risk Q1/Exp Q5, Risk Q3/Exp Q3, Risk Q5/Exp Q1) for s and profit."""
    ref = _build_support_reference()
    oof_q = _assign_quintile_from_edges(tr["oof_proba"].to_numpy(dtype=float), ref["oof_edges"])
    exp_q = _assign_quintile_from_edges(tr["expiry_remaining_ratio"].to_numpy(dtype=float), ref["exp_edges"])
    tr_tmp = tr.copy()
    tr_tmp["oof_q"] = oof_q
    tr_tmp["exp_q"] = exp_q
    tr_tmp["bin"] = assign_discount_bin(tr_tmp["raw_discount_pct"]).astype(str)
    tr_tmp["profit_unit"] = tr_tmp["profit"] / tr_tmp["raw_initial_quantity"]

    grid = np.asarray(DEFAULT_DISCOUNT_GRID, dtype=float)
    grid_bins = [str(assign_discount_bin(pd.Series([d]))[0]) for d in grid]
    examples = [(1, 5, "Low Risk Q1 x Long Expiry Q5"), (3, 3, "Mid Risk Q3 x Mid Expiry Q3"), (5, 1, "High Risk Q5 x Short Expiry Q1")]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    for col_i, (oq, eq, title) in enumerate(examples):
        sub = tr_tmp[(tr_tmp["oof_q"] == oq) & (tr_tmp["exp_q"] == eq)]
        idx = (oq - 1) * 5 + (eq - 1)
        n_det = naive_verdict["strata_details"][idx]
        c_det = ctrl_verdict["strata_details"][idx]
        supp = np.asarray(c_det["supported_mask"], dtype=bool)
        obs_cnt = sub.groupby("bin", observed=True).size().to_dict()

        for row_i, (metric, obs_col, ylabel) in enumerate([
            ("mean_pred_s", "s", "Sell-Through s"),
            ("mean_pred_profit_unit", "profit_unit", "Profit / Initial Unit ($)"),
        ]):
            ax = axes[row_i, col_i]
            obs_mean = sub.groupby("bin", observed=True)[obs_col].mean().to_dict()
            y_n = np.asarray(n_det[metric], dtype=float)
            y_c = np.asarray(c_det[metric], dtype=float)
            ax.plot(grid, y_n, color="#ff7f0e", linestyle="--", alpha=0.45)
            ax.plot(grid, y_c, color="#1f77b4", linestyle="--", alpha=0.45)
            if supp.any():
                ax.plot(grid[supp], y_n[supp], color="#ff7f0e", linewidth=2.2, label="NAIVE (supported)")
                ax.plot(grid[supp], y_c[supp], color="#1f77b4", linewidth=2.5, label="CONTROLLED (supported)")
            ox = [grid[j] for j, b in enumerate(grid_bins) if obs_cnt.get(b, 0) >= 30 and b in obs_mean]
            oy = [obs_mean[grid_bins[j]] for j, b in enumerate(grid_bins) if obs_cnt.get(b, 0) >= 30 and b in obs_mean]
            if ox:
                ax.scatter(ox, oy, color="black", s=28, zorder=4, label="Observed (>=30)")
            if row_i == 0:
                ax.set_title(f"{title}\n(n={len(sub)})", fontsize=10)
            if col_i == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            if row_i == 1:
                ax.set_xlabel("Discount Depth", fontsize=10)
            ax.grid(alpha=0.25)
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=8)

    fig.suptitle("M3-T05 Representative Strata Examples (Solid = Rule S >=30, Dashed = Unsupported)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    return fig


def main() -> None:
    setup_tracking_environment()
    REPORTS_M3_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_M3_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("M3-T05: Track 3B Confounding & Selection-Bias Diagnosis (Train Only)")
    print("==================================================================")

    tr = load_3b_train_only()

    # 1. Counterfactual-consistency check
    controls_only = [f for f in TRACK_3B_FEATURES if f != "discount_pct"] + ["oof_proba"]
    bad_keywords = ["selling_price", "markdown", "discount", "tier", "revenue", "profit"]
    kw_hits = [f for f in controls_only if any(kw in f for kw in bad_keywords)]
    max_corr = float(tr[controls_only].corrwith(tr["raw_discount_pct"]).abs().max())
    assert len(kw_hits) == 0, f"Leakage/treatment-derived features found: {kw_hits}"
    print(f"\n[1] Counterfactual-consistency check: PASS (0 treatment-derived features, max |corr|={max_corr:.4f})")

    # 2. Propensity diagnostics
    print("\n[2] Running propensity diagnostics (5-fold time-blocked CV)...")
    prop = run_propensity_diagnostics(tr, n_blocks=5)
    print(
        f"    Propensity AUC={prop['propensity_auc']:.4f} | Brier={prop['propensity_brier']:.5f} | "
        f"Extreme (<0.02 or >0.98)={prop['extreme_total_share']*100:.2f}% "
        f"(<0.02: {prop['extreme_lt_002_share']*100:.2f}%, >0.98: {prop['extreme_gt_098_share']*100:.2f}%)"
    )
    print(f"    Discounted-only OOF R^2={prop['discounted_only_oof_r2']:.4f} (RMSE={prop['discounted_only_oof_rmse']:.4f}, n={prop['discounted_only_n']})")
    print(f"    Depth-3 tree accuracy={prop['depth3_tree_accuracy']:.4f}")
    print(prop["depth3_tree_rules"])

    # Propensity histogram
    fig_prop, ax_p = plt.subplots(figsize=(7.5, 4.2))
    p_hat = prop["propensity_oof"]
    y_t = prop["y_treat"]
    ax_p.hist(p_hat[y_t == 0], bins=40, alpha=0.6, density=True, label="Undiscounted (0%)")
    ax_p.hist(p_hat[y_t == 1], bins=40, alpha=0.6, density=True, label="Discounted (>0%)")
    ax_p.axvline(0.02, color="red", linestyle="--", linewidth=1)
    ax_p.axvline(0.98, color="red", linestyle="--", linewidth=1, label="Extreme cutoffs (0.02, 0.98)")
    ax_p.set_xlabel("Predicted Propensity P(discount > 0 | controls)")
    ax_p.set_ylabel("Density")
    ax_p.set_title(f"Track 3B Propensity Overlap (AUC={prop['propensity_auc']:.4f}, Extreme={prop['extreme_total_share']*100:.2f}%)")
    ax_p.legend()
    ax_p.grid(alpha=0.25)
    fig_prop.tight_layout()
    fig_prop.savefig(FIGURES_M3_DIR / "m3_t05_propensity_histogram.png", dpi=150)

    # 3. Fixed diagnostic models (NAIVE, CONTROLLED, + 6 ablations)
    print("\n[3] Fitting fixed LightGBM cross_entropy models (5-fold CV)...")
    cat_cols = [c for c in TRACK_3B_FEATURES if c.startswith("category_")]
    naive_cols = ["discount_pct"] + cat_cols
    ctrl_cols = list(TRACK_3B_FEATURES) + ["oof_proba"]

    naive_res = fit_cross_entropy_oof_and_grid(tr, naive_cols, compute_grid=True)
    ctrl_res = fit_cross_entropy_oof_and_grid(tr, ctrl_cols, compute_grid=True)

    idx_0 = DEFAULT_DISCOUNT_GRID.index(0.0)
    idx_30 = DEFAULT_DISCOUNT_GRID.index(0.30)
    naive_gain_0_30 = float((naive_res["grid_s"][:, idx_30] - naive_res["grid_s"][:, idx_0]).mean())
    ctrl_gain_0_30 = float((ctrl_res["grid_s"][:, idx_30] - ctrl_res["grid_s"][:, idx_0]).mean())

    print(f"    NAIVE      | OOF RMSE={naive_res['metrics']['oof_rmse']:.4f}, R2={naive_res['metrics']['oof_r2']:.4f} | mean s(0.30)-s(0)={naive_gain_0_30:+.4f}")
    print(f"    CONTROLLED | OOF RMSE={ctrl_res['metrics']['oof_rmse']:.4f}, R2={ctrl_res['metrics']['oof_r2']:.4f} | mean s(0.30)-s(0)={ctrl_gain_0_30:+.4f}")

    ablation_rows = []
    for abl_name, drop_cols in ABLATION_GROUPS.items():
        abl_cols = [c for c in ctrl_cols if c not in drop_cols]
        abl_res = fit_cross_entropy_oof_and_grid(tr, abl_cols, compute_grid=True)
        gain_0_30 = float((abl_res["grid_s"][:, idx_30] - abl_res["grid_s"][:, idx_0]).mean())
        shift = gain_0_30 - ctrl_gain_0_30
        ablation_rows.append({
            "ablation": abl_name,
            "dropped_cols": drop_cols,
            "oof_rmse": abl_res["metrics"]["oof_rmse"],
            "oof_r2": abl_res["metrics"]["oof_r2"],
            "mean_s_gain_0_to_30": gain_0_30,
            "shift_vs_controlled": float(shift),
        })
        print(f"    {abl_name:14s} | OOF R2={abl_res['metrics']['oof_r2']:.4f} | s(0.30)-s(0)={gain_0_30:+.4f} (shift={shift:+.4f})")

    # 4 & 5. Verdicts & Binned Diagnostic Plots
    naive_verdict = evaluate_pre_registered_rules(tr, naive_res["grid_s"], extreme_propensity_share=prop["extreme_total_share"])
    ctrl_verdict = evaluate_pre_registered_rules(tr, ctrl_res["grid_s"], extreme_propensity_share=prop["extreme_total_share"])

    print("\n[4-5] Pre-Registered Rule Verdicts:")
    for label, v in [("NAIVE", naive_verdict), ("CONTROLLED", ctrl_verdict)]:
        print(
            f"    {label:10s} | Rule U (harmful): {v['rule_u_triggered']} ({v['harmful_strata']}/{v['evaluable_strata']}={v['harmful_share']*100:.1f}%) "
            f"| Rule M (monotone s): {v['rule_m_passed']} ({v['monotone_demand_strata']}/{v['evaluable_strata']}={v['monotone_demand_share']*100:.1f}%) "
            f"| Rule D (DML): {v['rule_d_recommend_dml']}"
        )

    fig_s = _plot_5x5_diagnostic(tr, naive_verdict, ctrl_verdict, "s", FIGURES_M3_DIR / "m3_t05_binned_s_5x5.png")
    fig_p = _plot_5x5_diagnostic(tr, naive_verdict, ctrl_verdict, "profit", FIGURES_M3_DIR / "m3_t05_binned_profit_5x5.png")
    fig_ex = _plot_example_strata(tr, naive_verdict, ctrl_verdict, FIGURES_M3_DIR / "m3_t05_binned_examples.png")

    # 6. Censoring check (s == 0 and s == 1 by discount bin)
    tr_c = tr.copy()
    tr_c["discount_bin"] = assign_discount_bin(tr_c["raw_discount_pct"])
    censor_rows = []
    print("\n[6] Censoring Check (Train, N=80,042):")
    for b in DISCOUNT_BIN_LABELS:
        sub = tr_c[tr_c["discount_bin"] == b]
        s0 = float((sub["s"] == 0.0).mean() * 100.0)
        s1 = float((sub["s"] == 1.0).mean() * 100.0)
        censor_rows.append({"discount_bin": b, "n_rows": int(len(sub)), "share_s_eq_0_pct": s0, "share_s_eq_1_pct": s1})
        print(f"    {b:8s} | n={len(sub):5d} | s==0: {s0:5.2f}% | s==1: {s1:5.2f}%")

    # 7. Test reusable discount_support() including shortest-expiry decile row
    min_exp_row = tr.sort_values("expiry_remaining_ratio").iloc[[0]]
    max_exp_row = tr.sort_values("expiry_remaining_ratio", ascending=False).iloc[[0]]
    test_sample = pd.concat([min_exp_row, max_exp_row], ignore_index=True)
    supp_df, no_zero = discount_support(test_sample)
    print(
        f"\n[7] discount_support() check: shortest-expiry row no_undiscounted_history={bool(no_zero.iloc[0])}, "
        f"supported_pts={int(supp_df.iloc[0].sum())}/{len(DEFAULT_DISCOUNT_GRID)} | "
        f"longest-expiry row no_undiscounted_history={bool(no_zero.iloc[1])}, "
        f"supported_pts={int(supp_df.iloc[1].sum())}/{len(DEFAULT_DISCOUNT_GRID)}"
    )

    # 8. Log to MLflow smart-shelf-3b-revenue-response
    with start_tracked_run(
        track="3B",
        model_type="lgbm_cross_entropy_diagnostic",
        feature_set_name="track_3b_43_plus_oof",
        shelf_life_version="expiry_remaining_ratio",
        features=ctrl_cols,
        run_name="m3_t05_confounding_diagnosis",
        purpose="confounding_diagnosis",
        params={"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 400, "min_child_samples": 50},
    ) as run:
        mlflow.log_metrics({
            "propensity_oof_auc": prop["propensity_auc"],
            "propensity_extreme_share": prop["extreme_total_share"],
            "discounted_only_oof_r2": prop["discounted_only_oof_r2"],
            "naive_oof_r2": naive_res["metrics"]["oof_r2"],
            "controlled_oof_r2": ctrl_res["metrics"]["oof_r2"],
            "naive_mean_s_gain_0_30": naive_gain_0_30,
            "controlled_mean_s_gain_0_30": ctrl_gain_0_30,
            "controlled_harmful_share": ctrl_verdict["harmful_share"],
            "controlled_monotone_s_share": ctrl_verdict["monotone_demand_share"],
        })
        log_figure_artifact(fig_prop, "m3_t05_propensity_histogram.png")
        log_figure_artifact(fig_s, "m3_t05_binned_s_5x5.png")
        log_figure_artifact(fig_p, "m3_t05_binned_profit_5x5.png")
        log_figure_artifact(fig_ex, "m3_t05_binned_examples.png")
        mlflow_run_id = run.info.run_id

    plt.close("all")

    # Save summary JSON (drop large arrays from prop/verdicts)
    summary = {
        "counterfactual_consistency": {"passed": True, "keyword_hits": kw_hits, "max_abs_corr": max_corr},
        "propensity": {k: v for k, v in prop.items() if k not in ("propensity_oof", "y_treat")},
        "models": {
            "naive": {**naive_res["metrics"], "mean_s_gain_0_to_30": naive_gain_0_30},
            "controlled": {**ctrl_res["metrics"], "mean_s_gain_0_to_30": ctrl_gain_0_30},
            "ablations": ablation_rows,
        },
        "verdicts": {
            "naive": {k: v for k, v in naive_verdict.items() if k != "strata_details"},
            "controlled": {k: v for k, v in ctrl_verdict.items() if k != "strata_details"},
        },
        "censoring_by_bin": censor_rows,
        "support_test": {
            "shortest_expiry_no_zero": bool(no_zero.iloc[0]),
            "shortest_expiry_supported_pts": int(supp_df.iloc[0].sum()),
            "longest_expiry_no_zero": bool(no_zero.iloc[1]),
            "longest_expiry_supported_pts": int(supp_df.iloc[1].sum()),
        },
        "mlflow_run_id": mlflow_run_id,
    }
    out_json = REPORTS_M3_DIR / "m3_t05_diagnostics_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved diagnostics JSON: {out_json}")
    print(f"Logged MLflow run: {mlflow_run_id}")


if __name__ == "__main__":
    main()
