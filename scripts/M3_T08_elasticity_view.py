"""
scripts/M3_T08_elasticity_view.py - Generate Track 3B Category Elasticity View (M3-T08).

Uses frozen models/track_3b_regressor.joblib on TRAINING R1 rows only (no refit):
1. Saves dashboard-ready table to reports/tables/M3_T08_category_elasticity.csv.
2. Computes per-band model slopes & cross-fitted model-free slopes with 95% week-boot CIs.
3. Computes descriptive spoilage_sensitivity Spearman cross-check & censoring table.
4. Generates per-category units_sold curves and slope comparison summary plots.
5. Logs run & figures to MLflow smart-shelf-3b-revenue-response.
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

from src.config import FIGURES_DIR, REPORTS_DIR, TABLES_DIR, TRACK_3B_MODEL_PATH
from src.m3_t08_elasticity import (
    category_elasticity_table,
    compute_category_band_slopes,
    compute_category_censoring,
    compute_model_free_category_slopes,
    compute_spoilage_sensitivity_check,
)
from src.mlflow_utils import (
    load_model_with_metadata,
    log_figure_artifact,
    setup_tracking_environment,
    start_tracked_run,
)

REPORTS_M3_DIR = REPORTS_DIR / "M3"
FIGURES_M3_DIR = FIGURES_DIR / "m3"


def _plot_category_units_curves(elast_df: pd.DataFrame, out_path: Path) -> plt.Figure:
    cats = sorted(elast_df["category"].unique().tolist())
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.5), sharex=True)
    axes_flat = axes.flatten()
    for ax, cat in zip(axes_flat, cats):
        sub = elast_df[elast_df["category"] == cat].sort_values("discount_pct")
        d = sub["discount_pct"].to_numpy(dtype=float)
        q = sub["mean_units_sold"].to_numpy(dtype=float)
        supp = sub["supported"].to_numpy(dtype=bool)
        n_r1 = int(sub["n_cat_r1"].iloc[0])
        n_b = int(sub["n_supported_bins_cat"].iloc[0])

        ax.plot(d, q, color="gray", linestyle="--", alpha=0.55, linewidth=1.3, label="Unsupported" if cat == cats[0] else None)
        if supp.any():
            ax.plot(d[supp], q[supp], "o-", color="#1f77b4", linewidth=2.0, markersize=4, label="Supported (>=30)" if cat == cats[0] else None)
        ax.set_title(f"{cat}\n(n={n_r1}, {n_b}/13 bins)", fontsize=9)
        ax.grid(alpha=0.25)
        if cat in (cats[0], cats[5]):
            ax.set_ylabel("Mean Pred Units Sold", fontsize=9)
        if cat in cats[5:]:
            ax.set_xlabel("Discount Depth", fontsize=9)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("M3-T08 Per-Category Demand Response: Predicted Units Sold vs. Depth (Train R1)", y=0.95, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    return fig


def _plot_slopes_summary(merged_slopes: pd.DataFrame, out_path: Path) -> plt.Figure:
    df = merged_slopes.sort_values("slope_per10_10-30%", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    mf = df["mf_slope_per10"].to_numpy(dtype=float) * 100.0
    lo = (df["mf_slope_per10"] - df["mf_ci95_low_per10"]).to_numpy(dtype=float) * 100.0
    hi = (df["mf_ci95_high_per10"] - df["mf_slope_per10"]).to_numpy(dtype=float) * 100.0
    mod_10_30 = df["slope_per10_10-30%"].to_numpy(dtype=float) * 100.0

    ax.errorbar(mf, y - 0.12, xerr=[lo, hi], fmt="o", color="#2ca02c", capsize=4, label="Model-free linear residual slope (all depths, 95% week-boot CI)")
    ax.scatter(mod_10_30, y + 0.12, marker="D", s=45, color="#1f77b4", label="Frozen model slope in common 10-30% band")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['category']} ({r['n_supported_bins']}/13)" for _, r in df.iterrows()])
    ax.set_xlabel("Sell-Through Lift (percentage points per +10% discount)")
    ax.set_title("M3-T08 Category Demand-Response Slope: Common 10-30% Band vs. Model-Free (Train R1)")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig


def main() -> None:
    setup_tracking_environment()
    REPORTS_M3_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_M3_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("M3-T08: Track 3B Per-Category Elasticity View (Frozen Model, Train R1)")
    print("==================================================================")

    _, meta = load_model_with_metadata(TRACK_3B_MODEL_PATH)
    features_3b = list(meta["feature_list"])

    elast_df = category_elasticity_table(save_csv=True)
    csv_path = TABLES_DIR / "M3_T08_category_elasticity.csv"
    print(f"\n[1] Saved {csv_path} ({len(elast_df)} rows, {elast_df['category'].nunique()} categories)")

    band_slopes = compute_category_band_slopes(elast_df)
    mf_slopes = compute_model_free_category_slopes(n_blocks=5, n_boot=500)
    merged_slopes = band_slopes.merge(mf_slopes, on="category", how="inner")

    print("\n[2-3] Model-Implied vs. Model-Free Slopes (pts per +10% discount):")
    for _, r in merged_slopes.sort_values("slope_per10_overall", ascending=False).iterrows():
        in_ci = (r["mf_ci95_low_per10"] <= r["slope_per10_overall"] <= r["mf_ci95_high_per10"])
        print(
            f"    {r['category']:16s} ({r['n_supported_bins']:2d}/13) | "
            f"model={r['slope_per10_overall']*100:+5.2f} (10-30:{r['slope_per10_10-30%']*100:+5.2f}) | "
            f"model-free={r['mf_slope_per10']*100:+5.2f} [{r['mf_ci95_low_per10']*100:+5.2f}, {r['mf_ci95_high_per10']*100:+5.2f}] | "
            f"in_CI={in_ci}"
        )

    sens_check = compute_spoilage_sensitivity_check(elast_df, band_slopes)
    print(
        f"\n[4] Descriptive spoilage_sensitivity Spearman (n={sens_check['n_categories']}): "
        f"vs baseline_s rho={sens_check['spearman_sens_vs_baseline_s']:+.3f} | "
        f"vs model_slope rho={sens_check['spearman_sens_vs_model_slope']:+.3f}"
    )

    censor_df = compute_category_censoring()
    print("\n[5] Censoring Share (s == 1.0 %) by Category & Depth Band:")
    print(censor_df.to_string(index=False))

    fig_curves = _plot_category_units_curves(elast_df, FIGURES_M3_DIR / "m3_t08_category_units_sold_curves.png")
    fig_slopes = _plot_slopes_summary(merged_slopes, FIGURES_M3_DIR / "m3_t08_category_slopes_summary.png")

    with start_tracked_run(
        track="3B",
        model_type="lgbm_ce_monotone_elasticity_view",
        feature_set_name="track_3b_r1_44",
        shelf_life_version="expiry_remaining_ratio",
        features=features_3b,
        run_name="m3_t08_elasticity_view",
        purpose="elasticity_view",
        train_rows=24475,
    ) as run:
        mlflow.log_metrics({
            "n_categories": float(elast_df["category"].nunique()),
            "n_low_conf_categories": float((merged_slopes["n_cat_r1"] < 300).sum()),
            "spearman_sens_vs_baseline_s": sens_check["spearman_sens_vs_baseline_s"],
            "spearman_sens_vs_model_slope": sens_check["spearman_sens_vs_model_slope"],
        })
        log_figure_artifact(fig_curves, "m3_t08_category_units_sold_curves.png")
        log_figure_artifact(fig_slopes, "m3_t08_category_slopes_summary.png")
        mlflow.log_artifact(str(csv_path))
        mlflow_run_id = run.info.run_id

    plt.close("all")

    summary = {
        "slopes_comparison": merged_slopes.to_dict(orient="records"),
        "spoilage_sensitivity_check": sens_check,
        "censoring_by_category_band": censor_df.to_dict(orient="records"),
        "mlflow_run_id": mlflow_run_id,
    }
    out_json = REPORTS_M3_DIR / "m3_t08_results_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved M3-T08 summary JSON: {out_json}")
    print(f"Logged MLflow run: {mlflow_run_id}")


if __name__ == "__main__":
    main()
