"""
scripts/M3_T04_build_and_diagnose.py - Build Track 3B Scenario Dataset & Overlap Diagnostics.

Executes M3-T04:
1. Builds data/processed/model_ready_3b_dataset.csv (no models trained).
2. Runs integrity checks & leakage guard.
3. Computes training-only selection & overlap diagnostics.
4. Generates 3 diagnostic figures in reports/figures/m3/.
5. Logs diagnostics & figures to MLflow experiment smart-shelf-3b-revenue-response.
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

from src.config import (
    FIGURES_DIR,
    MODEL_READY_3B_DATASET_PATH,
    REPORTS_DIR,
)
from src.m2_feature_pipeline import TRACK_3B_FEATURES
from src.m3_t04_scenario import (
    DISCOUNT_BIN_LABELS,
    build_3b_dataset,
    compute_selection_and_overlap_diagnostics,
    run_integrity_checks,
)
from src.mlflow_utils import (
    log_figure_artifact,
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
    print("M3-T04: Track 3B Scenario Construction & Overlap Diagnostics")
    print("==================================================================")

    # 1. Build dataset
    df = build_3b_dataset(save=True)
    print(f"\n[1] Saved {MODEL_READY_3B_DATASET_PATH} | shape={df.shape}")

    # 2. Integrity checks
    integrity = run_integrity_checks(df)
    print("\n[2] Integrity Checks:")
    print(f"    Rows: {integrity['n_total']} (train={integrity['n_train']}, test={integrity['n_test']})")
    print(f"    record_id unique={integrity['record_id_unique']}, train_oof_missing={integrity['train_oof_missing']}")
    print(f"    inventory_max_diff={integrity['inventory_max_diff']}, profit_identity_max_err={integrity['profit_identity_max_err']:.2e}")
    print(f"    leakage_guard_passed={integrity['leakage_guard_passed']}")
    print(f"    feature_nan_counts={integrity['feature_nan_counts']}")
    sd = integrity["s_distribution"]
    print(f"    s: mean={sd['overall_mean']:.4f}, std={sd['overall_std']:.4f}, share_0={sd['share_exactly_0']*100:.2f}%, share_1={sd['share_exactly_1']*100:.2f}%")
    print(f"    s quantiles: {sd['quantiles']}")

    # 3. Selection & overlap diagnostics (training partition only)
    diag = compute_selection_and_overlap_diagnostics(df)

    print("\n[3a] Discount-Bin Support Table (Train, N=80,042):")
    for r in diag["discount_bins"]:
        thin = " [THIN <300]" if r["thin_support"] else ""
        print(
            f"    {r['discount_bin']:8s} | n={r['n_rows']:5d} ({r['share_pct']:5.2f}%) "
            f"| mean_s={r['mean_s']:.4f} | mean_profit/unit={r['mean_profit_per_unit']:6.2f} "
            f"| loss={r['loss_share_pct']:5.1f}%{thin}"
        )

    print("\n[3b] Spearman Correlations with discount_pct (Train):")
    for r in diag["spearman_with_discount"]:
        print(f"    {r['feature']:24s}: rho={r['spearman_rho']:+.4f} (n={r['n_valid']}, p={r['p_value']:.2e})")

    ov = diag["overlap"]
    print(
        f"\n[3c] Overlap across {ov['n_strata']} strata (category x expiry_tercile x risk_tercile): "
        f"both(0 & >0)={ov['share_strata_both']*100:.1f}% strata ({ov['share_rows_both']*100:.1f}% rows), "
        f"3+ bins={ov['share_strata_3plus_bins']*100:.1f}% strata ({ov['share_rows_3plus_bins']*100:.1f}% rows)"
    )

    print("\n[3d] Promoted vs Non-Promoted (Train):")
    for r in diag["promoted_vs_not"]:
        print(
            f"    is_promoted={r['is_promoted']} | n={r['n_rows']:5d} | "
            f"discount_rate={r['discount_rate']*100:.2f}% | mean_disc_all={r['mean_discount_all']*100:.2f}% | "
            f"mean_disc_md={r['mean_discount_when_md']*100:.2f}%"
        )

    # 4. Generate 3 figures
    # Fig 1: Discount-bin support
    bins_df = pd.DataFrame(diag["discount_bins"])
    fig1, ax1 = plt.subplots(figsize=(9, 4.5))
    colors = ["#d62728" if t else "#1f77b4" for t in bins_df["thin_support"]]
    ax1.bar(bins_df["discount_bin"], bins_df["n_rows"], color=colors, alpha=0.88)
    ax1.axhline(300, color="red", linestyle="--", linewidth=1, label="Thin support threshold (300)")
    ax1.set_yscale("log")
    ax1.set_ylabel("Training Rows (log scale)")
    ax1.set_xlabel("Discount Bin")
    ax1.set_title("Track 3B Training Support by Discount Bin (Red = Thin < 300 rows)")
    ax1.tick_params(axis="x", rotation=40)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.25)
    fig1.tight_layout()
    f1_path = FIGURES_M3_DIR / "m3_t04_discount_bin_support.png"
    fig1.savefig(f1_path, dpi=150)

    # Fig 2: Discount rate by oof_proba decile
    dec_df = pd.DataFrame(diag["discount_by_oof_decile"])
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.2))
    ax2.plot(dec_df["oof_decile"], dec_df["discount_rate"] * 100.0, marker="o", linewidth=2, color="#2ca02c")
    ax2.set_xticks(dec_df["oof_decile"])
    ax2.set_xlabel("Track 3A Spoilage Risk Decile (1=Lowest, 10=Highest)")
    ax2.set_ylabel("Historical Markdown Rate (%)")
    ax2.set_title("Historical Discount Rate by Predicted Spoilage Risk Decile (Train)")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    f2_path = FIGURES_M3_DIR / "m3_t04_discount_rate_by_risk_decile.png"
    fig2.savefig(f2_path, dpi=150)

    # Fig 3: Raw mean s by discount bin within oof_proba terciles
    tb_df = pd.DataFrame(diag["raw_by_tercile_and_bin"])
    fig3, ax3 = plt.subplots(figsize=(9.5, 4.8))
    for terc in ["LowRisk", "MidRisk", "HighRisk"]:
        sub = tb_df[tb_df["oof_tercile"] == terc].set_index("discount_bin").reindex(DISCOUNT_BIN_LABELS).reset_index()
        ax3.plot(sub["discount_bin"], sub["mean_s"], marker="o", label=terc)
    ax3.set_xlabel("Discount Bin")
    ax3.set_ylabel("Observed Mean Sell-Through s")
    ax3.set_title("Descriptive Raw Sell-Through by Discount Bin & Risk Tercile (Not Causal)")
    ax3.tick_params(axis="x", rotation=40)
    ax3.legend()
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    f3_path = FIGURES_M3_DIR / "m3_t04_raw_sellthrough_by_risk_tercile.png"
    fig3.savefig(f3_path, dpi=150)

    # 5. Log to MLflow smart-shelf-3b-revenue-response
    with start_tracked_run(
        track="3B",
        model_type="scenario_construction",
        feature_set_name="track_3b_43_plus_oof",
        shelf_life_version="expiry_remaining_ratio",
        features=list(TRACK_3B_FEATURES),
        run_name="m3_t04_scenario_and_overlap",
        purpose="scenario_construction",
    ) as run:
        mlflow.log_metrics({
            "n_train": integrity["n_train"],
            "n_test": integrity["n_test"],
            "profit_identity_max_err": integrity["profit_identity_max_err"],
            "overlap_share_strata_both": ov["share_strata_both"],
            "overlap_share_rows_both": ov["share_rows_both"],
            "overlap_share_strata_3plus_bins": ov["share_strata_3plus_bins"],
            "s_train_mean": sd["train_mean"],
        })
        log_figure_artifact(fig1, "m3_t04_discount_bin_support.png")
        log_figure_artifact(fig2, "m3_t04_discount_rate_by_risk_decile.png")
        log_figure_artifact(fig3, "m3_t04_raw_sellthrough_by_risk_tercile.png")
        mlflow_run_id = run.info.run_id

    plt.close("all")

    payload = {
        "integrity": integrity,
        "diagnostics": diag,
        "mlflow_run_id": mlflow_run_id,
    }
    out_json = REPORTS_M3_DIR / "m3_t04_diagnostics_summary.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved diagnostics JSON: {out_json}")
    print(f"Logged MLflow run to smart-shelf-3b-revenue-response: {mlflow_run_id}")


if __name__ == "__main__":
    main()
