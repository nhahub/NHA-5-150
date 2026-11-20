"""
scripts/M3_T05_addendum_r1.py - M3-T05 Addendum: Rule Verification, Region Support & R1 Depth Effect.

Executes strictly on training rows (split == "train", N = 80,042):
1. Verifies exact discount rule (expiry_remaining_ratio <= 0.40 OR quality_grade == 'C').
2. Characterizes regions R0 (never discounted) and R1 (always discounted).
3. Tests region-based discount_support() on R0 and R1 example rows.
4. Analyzes depth assignment inside R1 (plot, distinct values, OOF R^2, residual SD, Rule E).
5. Runs NAIVE & CONTROLLED inside R1 only + cross-fitted partialling-out slope (week bootstrap CI).
6. Logs addendum run & figures to MLflow and saves reports/M3/m3_t05_addendum_summary.json.
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

from src.config import CLEANED_DATASET_PATH, FIGURES_DIR, REPORTS_DIR
from src.m2_feature_pipeline import TRACK_3B_FEATURES
from src.m3_t04_scenario import assign_discount_bin
from src.m3_t05_confounding import (
    DEFAULT_DISCOUNT_GRID,
    _build_region_support_reference,
    discount_support,
    load_3b_train_only,
    run_r1_depth_diagnostics,
    run_r1_local_models_and_partialling_out,
)
from src.mlflow_utils import log_figure_artifact, setup_tracking_environment, start_tracked_run

REPORTS_M3_DIR = REPORTS_DIR / "M3"
FIGURES_M3_DIR = FIGURES_DIR / "m3"


def _region_stats(sub: pd.DataFrame) -> dict:
    p_unit = sub["profit"] / sub["raw_initial_quantity"]
    return {
        "n_rows": int(len(sub)),
        "share_pct": float(len(sub) / 80042 * 100.0),
        "s_mean": float(sub["s"].mean()),
        "s_std": float(sub["s"].std()),
        "s_median": float(sub["s"].median()),
        "s_eq_0_pct": float((sub["s"] == 0.0).mean() * 100.0),
        "s_eq_1_pct": float((sub["s"] == 1.0).mean() * 100.0),
        "profit_per_unit_mean": float(p_unit.mean()),
        "loss_share_pct": float((sub["profit"] < 0).mean() * 100.0),
    }


def _plot_r1_depth_vs_expiry(tr_r1: pd.DataFrame, out_path: Path) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, is_c, title in [
        (axes[0], False, "R1: Quality Grade A/B (expiry_remaining_ratio <= 0.40)"),
        (axes[1], True, "R1: Quality Grade C (all expiry ratios)"),
    ]:
        sub = tr_r1[(tr_r1["quality_grade"] == "C") == is_c]
        if len(sub) > 4000:
            sub = sub.sample(4000, random_state=42)
        ax.scatter(sub["raw_expiry_remaining_ratio"], sub["raw_discount_pct"] * 100.0, alpha=0.22, s=8)
        ax.axvline(0.40, color="red", linestyle="--", linewidth=1, label="ratio = 0.40")
        ax.set_xlabel("Raw expiry_remaining_ratio")
        ax.set_title(f"{title}\n(n={( (tr_r1['quality_grade']=='C')==is_c ).sum()})", fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Discount Depth (%)")
    fig.suptitle("M3-T05 Addendum: Discount Depth vs. Expiry Remaining Ratio inside R1", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    return fig


def _plot_r1_binned_2x3(
    tr_r1: pd.DataFrame,
    r1_local: dict,
    metric: str,
    out_path: Path,
) -> plt.Figure:
    ref = _build_region_support_reference()
    r1_edges = np.asarray(ref["r1_exp_tercile_edges"], dtype=float)
    r1_terc = np.digitize(tr_r1["raw_expiry_remaining_ratio"].to_numpy(dtype=float), r1_edges[1:-1], right=True) + 1
    is_c = (tr_r1["quality_grade"] == "C").astype(int).to_numpy()

    r1_grid = np.asarray(r1_local["r1_grid"], dtype=float)
    supp_df, _ = discount_support(tr_r1, grid=r1_grid)
    supp_mat = supp_df.to_numpy(dtype=bool)

    grid_s_n = r1_local["r1_naive_grid_s"]
    grid_s_c = r1_local["r1_controlled_grid_s"]

    if metric == "profit":
        base_p = tr_r1["raw_base_price"].to_numpy(dtype=float)
        cost_p = tr_r1["raw_cost_price"].to_numpy(dtype=float)
        sp = np.round(base_p[:, None] * (1.0 - r1_grid[None, :]), 2)
        y_n_mat = grid_s_n * sp - cost_p[:, None]
        y_c_mat = grid_s_c * sp - cost_p[:, None]
        obs_vals = (tr_r1["profit"] / tr_r1["raw_initial_quantity"]).to_numpy(dtype=float)
        ylabel = "Profit / Initial Unit ($)"
    else:
        y_n_mat = grid_s_n
        y_c_mat = grid_s_c
        obs_vals = tr_r1["s"].to_numpy(dtype=float)
        ylabel = "Sell-Through s"

    tr_tmp = pd.DataFrame({
        "is_c": is_c,
        "terc": r1_terc,
        "bin": assign_discount_bin(tr_r1["raw_discount_pct"]).astype(str),
        "obs": obs_vals,
    })
    grid_bins = [str(assign_discount_bin(pd.Series([d]))[0]) for d in r1_grid]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for row_i, c_flag in enumerate([0, 1]):
        for col_i, t_idx in enumerate([1, 2, 3]):
            ax = axes[row_i, col_i]
            m = (is_c == c_flag) & (r1_terc == t_idx)
            if m.sum() == 0:
                ax.set_title(f"{'Grade C' if c_flag == 1 else 'Grade A/B'} | R1 Exp Tercile {t_idx} (n=0)", fontsize=9)
                ax.text(0.5, 0.5, "Empty stratum\n(Rule-excluded)", ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray")
                continue
            supp = supp_mat[m][0]
            yn = y_n_mat[m].mean(axis=0)
            yc = y_c_mat[m].mean(axis=0)

            ax.plot(r1_grid, yn, color="#ff7f0e", linestyle="--", alpha=0.45)
            ax.plot(r1_grid, yc, color="#1f77b4", linestyle="--", alpha=0.45)
            if supp.any():
                ax.plot(r1_grid[supp], yn[supp], color="#ff7f0e", linewidth=2.0, label="NAIVE (R1)")
                ax.plot(r1_grid[supp], yc[supp], color="#1f77b4", linewidth=2.2, label="CONTROLLED (R1)")

            sub_obs = tr_tmp[(tr_tmp["is_c"] == c_flag) & (tr_tmp["terc"] == t_idx)]
            obs_mean = sub_obs.groupby("bin", observed=True)["obs"].mean().to_dict()
            obs_cnt = sub_obs.groupby("bin", observed=True).size().to_dict()
            ox = [r1_grid[j] for j, b in enumerate(grid_bins) if obs_cnt.get(b, 0) >= 30 and b in obs_mean]
            oy = [obs_mean[grid_bins[j]] for j, b in enumerate(grid_bins) if obs_cnt.get(b, 0) >= 30 and b in obs_mean]
            if ox:
                ax.scatter(ox, oy, color="black", s=22, zorder=4, label="Observed (>=30)")

            c_lbl = "Grade C" if c_flag == 1 else "Grade A/B"
            ax.set_title(f"{c_lbl} | R1 Exp Tercile {t_idx} (n={int(m.sum())})", fontsize=9)
            ax.grid(alpha=0.25)
            if col_i == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row_i == 1:
                ax.set_xlabel("Discount Depth", fontsize=9)
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=8)

    fig.suptitle(f"M3-T05 Addendum: R1-Only Diagnostic ({ylabel}, Solid = Region Supported >=30)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    return fig


def main() -> None:
    setup_tracking_environment()
    REPORTS_M3_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_M3_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("M3-T05 Addendum: Rule Verification, Region Support & R1 Depth")
    print("==================================================================")

    tr = load_3b_train_only()
    cl_extra = pd.read_csv(
        CLEANED_DATASET_PATH,
        usecols=[
            "record_id", "expiry_remaining_ratio", "quality_grade",
            "store_id", "supplier_id", "product_name",
        ],
    ).rename(columns={"expiry_remaining_ratio": "raw_expiry_remaining_ratio"})
    tr = tr.merge(cl_extra, on="record_id", how="inner")
    tr["month"] = tr["transaction_date"].dt.month
    tr["dow"] = tr["transaction_date"].dt.dayofweek

    # 1. Verify exact rule
    y_disc = tr["raw_discount_pct"] > 0
    is_c = tr["quality_grade"] == "C"
    le_40 = tr["raw_expiry_remaining_ratio"] <= 0.40
    lt_40 = tr["raw_expiry_remaining_ratio"] < 0.40
    rule_le = le_40 | is_c
    rule_lt = lt_40 | is_c
    viol_le = int((y_disc != rule_le).sum())
    viol_lt = int((y_disc != rule_lt).sum())

    print(f"\n[1] Rule Verification (N={len(tr)}):")
    print(f"    (ratio <= 0.40 OR grade=='C') violations: {viol_le}")
    print(f"    (ratio <  0.40 OR grade=='C') violations: {viol_lt}")
    print(f"    Grade C share: {is_c.mean()*100:.2f}% | Ratio <= 0.40 share: {le_40.mean()*100:.2f}% | Both: {(is_c & le_40).mean()*100:.2f}%")

    # 2. Region stats
    tr_r0 = tr.loc[~rule_le].copy().reset_index(drop=True)
    tr_r1 = tr.loc[rule_le].copy().reset_index(drop=True)
    r0_stats = _region_stats(tr_r0)
    r1_stats = _region_stats(tr_r1)
    print("\n[2] Region Stats:")
    print(f"    R0 (no discount): n={r0_stats['n_rows']} ({r0_stats['share_pct']:.2f}%), s_mean={r0_stats['s_mean']:.4f}, profit/unit=${r0_stats['profit_per_unit_mean']:.2f}, loss={r0_stats['loss_share_pct']:.1f}%")
    print(f"    R1 (discounted):  n={r1_stats['n_rows']} ({r1_stats['share_pct']:.2f}%), s_mean={r1_stats['s_mean']:.4f}, profit/unit=${r1_stats['profit_per_unit_mean']:.2f}, loss={r1_stats['loss_share_pct']:.1f}%")

    # 3. Test region-based discount_support() on R0 and R1 example rows
    ex_rows = pd.concat([tr_r0.iloc[[0]], tr_r1.iloc[[0]]], ignore_index=True)
    supp_ex, reason_ex = discount_support(ex_rows)
    print("\n[3] New Region-Based discount_support() Examples:")
    print(f"    R0 row: supported={int(supp_ex.iloc[0].sum())}/{len(DEFAULT_DISCOUNT_GRID)} | d=0.00: {supp_ex.iloc[0]['d_0.00']} ({reason_ex.iloc[0]['d_0.00']}) | d=0.20: {supp_ex.iloc[0]['d_0.20']} ({reason_ex.iloc[0]['d_0.20']})")
    print(f"    R1 row: supported={int(supp_ex.iloc[1].sum())}/{len(DEFAULT_DISCOUNT_GRID)} | d=0.00: {supp_ex.iloc[1]['d_0.00']} ({reason_ex.iloc[1]['d_0.00']}) | d=0.20: {supp_ex.iloc[1]['d_0.20']} ({reason_ex.iloc[1]['d_0.20']}) | d=0.75: {supp_ex.iloc[1]['d_0.75']} ({reason_ex.iloc[1]['d_0.75']})")

    # 4. Depth assignment inside R1
    print("\n[4] Depth Assignment Inside R1:")
    distinct_c = int(tr_r1.loc[tr_r1["quality_grade"] == "C", "raw_discount_pct"].nunique())
    distinct_ab = int(tr_r1.loc[tr_r1["quality_grade"] != "C", "raw_discount_pct"].nunique())
    print(f"    Distinct depth values: Grade C={distinct_c}, Grade A/B={distinct_ab}")
    fig_d = _plot_r1_depth_vs_expiry(tr_r1, FIGURES_M3_DIR / "m3_t05_r1_depth_vs_expiry.png")

    r1_depth = run_r1_depth_diagnostics(tr_r1, n_blocks=5)
    print(f"    depth ~ controls OOF R^2={r1_depth['depth_oof_r2']:.4f} | residual SD={r1_depth['residual_sd_pts']:.2f} pts")
    print(f"    residual ~ other features OOF R^2={r1_depth['residual_oof_r2']:.4f}")
    print(f"    Rule E verdict: SD>=3.0={r1_depth['rule_e_sd_ge_3pts']}, resid_R2<0.05={r1_depth['rule_e_resid_r2_lt_005']} -> PASSED={r1_depth['rule_e_passed']}")

    # 5. Local depth effect inside R1 only
    print("\n[5] R1-Only Local Models & Cross-Fitted Partialling-Out Slope:")
    r1_local = run_r1_local_models_and_partialling_out(tr_r1, r1_depth["resid_depth"], n_blocks=5, n_boot=500)
    print(f"    R1 NAIVE      OOF R^2={r1_local['r1_naive_metrics']['oof_r2']:.4f} (RMSE={r1_local['r1_naive_metrics']['oof_rmse']:.4f})")
    print(f"    R1 CONTROLLED OOF R^2={r1_local['r1_controlled_metrics']['oof_r2']:.4f} (RMSE={r1_local['r1_controlled_metrics']['oof_rmse']:.4f})")
    print(
        f"    Partialling-out slope (ds / d_discount): {r1_local['partialling_out_slope']:+.4f} "
        f"(per +10% discount: {r1_local['slope_per_10pct_pts']*100:+.2f} sell-through pts) | "
        f"95% week-boot CI: [{r1_local['boot_ci95_low']:+.4f}, {r1_local['boot_ci95_high']:+.4f}]"
    )
    for a in r1_local["ablation_slopes"]:
        print(f"      {a['ablation']:14s}: slope={a['slope']:+.4f} (shift={a['shift_vs_full']:+.4f})")
    for b in r1_local["band_slopes"]:
        print(f"      band {b['band']:7s} (n={b['n']:5d}): slope={b['slope']:+.4f}")

    fig_s = _plot_r1_binned_2x3(tr_r1, r1_local, "s", FIGURES_M3_DIR / "m3_t05_r1_binned_s.png")
    fig_p = _plot_r1_binned_2x3(tr_r1, r1_local, "profit", FIGURES_M3_DIR / "m3_t05_r1_binned_profit.png")

    # 6. Log to MLflow
    with start_tracked_run(
        track="3B",
        model_type="lgbm_r1_depth_diagnostic",
        feature_set_name="track_3b_43_plus_oof",
        shelf_life_version="expiry_remaining_ratio",
        features=list(TRACK_3B_FEATURES),
        run_name="m3_t05_addendum_r1_depth",
        purpose="confounding_diagnosis_addendum",
    ) as run:
        mlflow.log_metrics({
            "rule_le_40_violations": viol_le,
            "r1_depth_oof_r2": r1_depth["depth_oof_r2"],
            "r1_resid_sd_pts": r1_depth["residual_sd_pts"],
            "r1_resid_oof_r2": r1_depth["residual_oof_r2"],
            "r1_naive_oof_r2": r1_local["r1_naive_metrics"]["oof_r2"],
            "r1_controlled_oof_r2": r1_local["r1_controlled_metrics"]["oof_r2"],
            "r1_partialling_out_slope": r1_local["partialling_out_slope"],
            "r1_boot_ci95_low": r1_local["boot_ci95_low"],
            "r1_boot_ci95_high": r1_local["boot_ci95_high"],
        })
        log_figure_artifact(fig_d, "m3_t05_r1_depth_vs_expiry.png")
        log_figure_artifact(fig_s, "m3_t05_r1_binned_s.png")
        log_figure_artifact(fig_p, "m3_t05_r1_binned_profit.png")
        mlflow_run_id = run.info.run_id

    plt.close("all")

    out_summary = {
        "rule_verification": {
            "violations_le_040": viol_le,
            "violations_lt_040": viol_lt,
            "grade_c_share_pct": float(is_c.mean() * 100.0),
            "ratio_le_040_share_pct": float(le_40.mean() * 100.0),
            "overlap_both_pct": float((is_c & le_40).mean() * 100.0),
        },
        "regions": {"R0": r0_stats, "R1": r1_stats},
        "support_examples": {
            "r0_d0": [bool(supp_ex.iloc[0]["d_0.00"]), str(reason_ex.iloc[0]["d_0.00"])],
            "r0_d20": [bool(supp_ex.iloc[0]["d_0.20"]), str(reason_ex.iloc[0]["d_0.20"])],
            "r1_d0": [bool(supp_ex.iloc[1]["d_0.00"]), str(reason_ex.iloc[1]["d_0.00"])],
            "r1_d20": [bool(supp_ex.iloc[1]["d_0.20"]), str(reason_ex.iloc[1]["d_0.20"])],
            "r1_d75": [bool(supp_ex.iloc[1]["d_0.75"]), str(reason_ex.iloc[1]["d_0.75"])],
        },
        "r1_depth_diagnostics": {
            k: v for k, v in r1_depth.items() if k not in ("resid_depth", "date_blocks")
        } | {"distinct_depths_grade_c": distinct_c, "distinct_depths_grade_ab": distinct_ab},
        "r1_local_effect": {
            k: v for k, v in r1_local.items() if k not in ("r1_naive_grid_s", "r1_controlled_grid_s", "r1_grid")
        },
        "mlflow_run_id": mlflow_run_id,
    }
    out_json = REPORTS_M3_DIR / "m3_t05_addendum_summary.json"
    out_json.write_text(json.dumps(out_summary, indent=2))
    print(f"\nSaved addendum JSON: {out_json}")
    print(f"Logged MLflow run: {mlflow_run_id}")


if __name__ == "__main__":
    main()
