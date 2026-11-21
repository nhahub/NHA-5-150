"""
scripts/M3_T09_run_optimizer.py - Run & Validate Track 3C What-If Scorer and Optimizer.

Executes M3-T09:
1. Validates score_whatif() and optimize_batches() (bounds, R0 guard, profit identity,
   single-row vs batch parity, determinism, candidate counts).
2. Runs optimizer on training R1 (in-sample, for tier derivation in M3-T10) and test R1,
   saving optimizer_recs_train.csv and optimizer_recs_test.csv.
3. Computes descriptive depth distributions, flat_curve shares, historical agreement,
   and heterogeneity tables.
4. Logs metrics and distribution figure to MLflow smart-shelf-3c-discount-optimizer.
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
    OPTIMIZER_RECS_TEST_PATH,
    OPTIMIZER_RECS_TRAIN_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TRACK_3B_MODEL_PATH,
)
from src.m3_t05_confounding import _build_region_support_reference, discount_support, scale_discount_values
from src.m3_t06_regressor import R1_DEPTH_GRID
from src.m3_t07_eval import load_train_test_3b_with_raw
from src.m3_t09_optimizer import get_flat_curve_threshold, optimize_batches, score_whatif
from src.mlflow_utils import (
    load_model_with_metadata,
    log_figure_artifact,
    log_track_3c_metrics,
    setup_tracking_environment,
    start_tracked_run,
)

REPORTS_M3_DIR = REPORTS_DIR / "M3"
FIGURES_M3_DIR = FIGURES_DIR / "m3"


def _describe_recs(r1_df: pd.DataFrame, recs: pd.DataFrame, base_s_10: np.ndarray, ref: dict) -> dict:
    supp_df, _ = discount_support(r1_df, grid=R1_DEPTH_GRID)
    grid_arr = np.asarray(R1_DEPTH_GRID, dtype=float)
    supp_mat = supp_df.to_numpy(dtype=bool)

    shallowest_d = np.array([grid_arr[supp_mat[i]][0] for i in range(len(r1_df))])
    deepest_d = np.array([grid_arr[supp_mat[i]][-1] for i in range(len(r1_df))])

    rec_d = recs["recommended_depth"].to_numpy(dtype=float)
    act_d = r1_df["raw_discount_pct"].to_numpy(dtype=float)
    diff = rec_d - act_d
    within_025 = np.abs(diff) <= 0.025 + 1e-9

    at_shallowest = np.isclose(rec_d, shallowest_d)
    at_deepest = np.isclose(rec_d, deepest_d)

    dist_counts = recs["recommended_depth"].value_counts().sort_index()
    dist_shares = {f"{k:.2f}": float(v / len(recs) * 100.0) for k, v in dist_counts.items()}
    max_single_share = float(max(dist_shares.values())) if dist_shares else 0.0

    margins = recs["margin_per_unit"].dropna().to_numpy(dtype=float)
    margin_q = {
        "min": float(np.min(margins)),
        "p05": float(np.percentile(margins, 5)),
        "p25": float(np.percentile(margins, 25)),
        "p50": float(np.percentile(margins, 50)),
        "p75": float(np.percentile(margins, 75)),
        "p95": float(np.percentile(margins, 95)),
        "max": float(np.max(margins)),
    }

    r1_edges = np.asarray(ref["r1_exp_tercile_edges"], dtype=float)
    exp_terc = np.digitize(r1_df["raw_expiry_remaining_ratio"].to_numpy(dtype=float), r1_edges[1:-1], right=True) + 1
    grade_grp = np.where(r1_df["quality_grade"] == "C", "Grade_C", "Grade_AB")
    base_terc = pd.qcut(base_s_10, q=3, labels=["Low_Base_s", "Mid_Base_s", "High_Base_s"]).astype(str)

    tmp = pd.DataFrame({
        "exp_terc": exp_terc,
        "grade_grp": grade_grp,
        "base_terc": base_terc,
        "rec_d": rec_d,
        "act_d": act_d,
        "diff": diff,
        "within_025": within_025,
        "at_shallowest": at_shallowest,
        "flat_curve": recs["flat_curve"].to_numpy(dtype=bool),
    })

    by_exp_grade = (
        tmp.groupby(["exp_terc", "grade_grp"], observed=True)
        .agg(
            n=("rec_d", "count"),
            mean_rec_d=("rec_d", "mean"),
            mean_act_d=("act_d", "mean"),
            mean_signed_diff=("diff", "mean"),
            within_025_pct=("within_025", lambda s: float(s.mean() * 100.0)),
            at_shallowest_pct=("at_shallowest", lambda s: float(s.mean() * 100.0)),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    het_table = (
        tmp.groupby(["exp_terc", "base_terc", "grade_grp"], observed=True)
        .agg(
            n=("rec_d", "count"),
            mean_rec_d=("rec_d", "mean"),
            median_rec_d=("rec_d", "median"),
            at_shallowest_pct=("at_shallowest", lambda s: float(s.mean() * 100.0)),
            flat_curve_pct=("flat_curve", lambda s: float(s.mean() * 100.0)),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    return {
        "n_rows": int(len(recs)),
        "zero_supported_rows": int((recs["n_supported_depths"] == 0).sum()),
        "depth_shares_pct": dist_shares,
        "max_single_depth_share_pct": max_single_share,
        "gt_80pct_single_depth": bool(max_single_share > 80.0),
        "at_shallowest_supported_pct": float(at_shallowest.mean() * 100.0),
        "at_deepest_supported_pct": float(at_deepest.mean() * 100.0),
        "flat_curve_pct": float(recs["flat_curve"].mean() * 100.0),
        "margin_per_unit_quantiles": margin_q,
        "within_025_of_actual_pct": float(within_025.mean() * 100.0),
        "mean_signed_diff_rec_minus_act": float(diff.mean()),
        "by_exp_terc_and_grade": by_exp_grade,
        "heterogeneity_table": het_table,
    }


def main() -> None:
    setup_tracking_environment()
    REPORTS_M3_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_M3_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("M3-T09: Track 3C What-If Scoring & Discrete Discount Optimizer")
    print("==================================================================")

    tr, te = load_train_test_3b_with_raw()
    tr_r1 = tr[tr["raw_discount_pct"] > 0].reset_index(drop=True)
    te_r1 = te[te["raw_discount_pct"] > 0].reset_index(drop=True)
    te_r0 = te[te["raw_discount_pct"] == 0].reset_index(drop=True)

    flat_thr = get_flat_curve_threshold()
    print(f"\n[D3] Frozen flat_curve threshold (CV profit_unit_mae): ${flat_thr:.4f}/unit")

    # ------------------------------------------------------------------
    # Run optimizer on train R1 and test R1
    # ------------------------------------------------------------------
    recs_tr = optimize_batches(tr_r1)
    recs_te = optimize_batches(te_r1)

    # Save to data/processed/
    OPTIMIZER_RECS_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    recs_tr.to_csv(OPTIMIZER_RECS_TRAIN_PATH, index=False)
    recs_te.to_csv(OPTIMIZER_RECS_TEST_PATH, index=False)
    print(f"[3] Saved {OPTIMIZER_RECS_TRAIN_PATH} ({len(recs_tr)} rows) and {OPTIMIZER_RECS_TEST_PATH} ({len(recs_te)} rows)")

    # ------------------------------------------------------------------
    # Step 2: Validation checks (a-d)
    # ------------------------------------------------------------------
    # 2a: Bounds, no R1 == 0, no R0 recommendation
    r0_sample_recs = optimize_batches(te_r0.head(25))
    assert (r0_sample_recs["status"] == "no_markdown_historical_practice").all()
    assert (r0_sample_recs["reason"] == "no_discount_region").all()
    assert r0_sample_recs["recommended_depth"].isna().all()

    for name, r_df in [("train_R1", recs_tr), ("test_R1", recs_te)]:
        valid_d = r_df["recommended_depth"].dropna()
        assert ((valid_d >= 0.10 - 1e-9) & (valid_d <= 0.75 + 1e-9)).all(), f"{name} depth out of bounds!"
        assert not (valid_d == 0.0).any(), f"{name} recommended 0%!"

    # 2b: Profit identity on random 1,000 batch x depth cells
    rng = np.random.default_rng(RANDOM_STATE)
    sample_batches = te_r1.sample(n=100, random_state=RANDOM_STATE)
    whatif_sample = score_whatif(sample_batches).merge(
        sample_batches[["record_id", "raw_base_price", "raw_cost_price", "raw_initial_quantity"]],
        on="record_id",
    )
    cells_1000 = whatif_sample.sample(n=1000, random_state=RANDOM_STATE)
    sp = pd.Series(
        [round(float(b) * (1.0 - float(d)), 2) for b, d in zip(cells_1000["raw_base_price"], cells_1000["depth"])],
        index=cells_1000.index,
    )
    expected_batch_p = cells_1000["raw_initial_quantity"] * (cells_1000["s_hat"] * sp - cells_1000["raw_cost_price"])
    profit_id_max_err = float((cells_1000["batch_profit"] - expected_batch_p).abs().max())
    assert profit_id_max_err < 1e-9, f"Profit identity error: {profit_id_max_err}"

    # 2c: Single-row vs batch parity & repeated-call determinism
    b5 = te_r1.iloc[:5]
    out_a = optimize_batches(b5)
    out_b = optimize_batches(b5)
    out_single = optimize_batches(b5.iloc[[0]])
    pd.testing.assert_frame_equal(out_a, out_b)
    pd.testing.assert_frame_equal(out_a.iloc[[0]].reset_index(drop=True), out_single.reset_index(drop=True))

    # 2d: Candidate count >= 1 for R1
    zero_cand_tr = int((recs_tr["n_supported_depths"] == 0).sum())
    zero_cand_te = int((recs_te["n_supported_depths"] == 0).sum())
    assert zero_cand_tr == 0 and zero_cand_te == 0, f"Found R1 rows with 0 supported depths: {zero_cand_tr}, {zero_cand_te}"

    print("\n[2] Validation Checks (a-d): ALL PASSED")
    print(f"    2a: R0 guard=PASS (25/25 NaN + no_discount_region); R1 depths in [0.10, 0.75] & != 0")
    print(f"    2b: Profit identity max error (1,000 cells) = {profit_id_max_err:.2e}")
    print(f"    2c: Single-row vs batch & repeated-call determinism = PASS (identical)")
    print(f"    2d: R1 rows with 0 supported depths: train={zero_cand_tr}, test={zero_cand_te}")

    # ------------------------------------------------------------------
    # Step 4: Descriptive outputs
    # ------------------------------------------------------------------
    model_3b, meta_3b = load_model_with_metadata(TRACK_3B_MODEL_PATH)
    feats_3b = list(meta_3b["feature_list"])
    d_10_scaled = scale_discount_values(np.array([0.10]), float(meta_3b["disc_scale_slope"]), float(meta_3b["disc_scale_intercept"]))[0]
    ref = _build_region_support_reference()

    X_tr10 = tr_r1[feats_3b].copy()
    X_tr10["discount_pct"] = d_10_scaled
    base_s_tr = np.clip(model_3b.predict(X_tr10), 0.0, 1.0)

    X_te10 = te_r1[feats_3b].copy()
    X_te10["discount_pct"] = d_10_scaled
    base_s_te = np.clip(model_3b.predict(X_te10), 0.0, 1.0)

    desc_tr = _describe_recs(tr_r1, recs_tr, base_s_tr, ref)
    desc_te = _describe_recs(te_r1, recs_te, base_s_te, ref)

    for lbl, dsc in [("TRAIN R1 (in-sample)", desc_tr), ("TEST R1", desc_te)]:
        print(f"\n[4] Descriptive Output — {lbl} (N={dsc['n_rows']}):")
        print(f"    Depth shares (%): {dsc['depth_shares_pct']}")
        print(f"    At shallowest supported={dsc['at_shallowest_supported_pct']:.1f}% | deepest={dsc['at_deepest_supported_pct']:.1f}% | >80% single depth={dsc['gt_80pct_single_depth']}")
        print(f"    flat_curve={dsc['flat_curve_pct']:.1f}% | margin_per_unit p50=${dsc['margin_per_unit_quantiles']['p50']:.2f} (p25=${dsc['margin_per_unit_quantiles']['p25']:.2f}, p75=${dsc['margin_per_unit_quantiles']['p75']:.2f})")
        print(f"    Within +/-0.025 of actual={dsc['within_025_of_actual_pct']:.1f}% | mean(rec - actual)={dsc['mean_signed_diff_rec_minus_act']:+.4f}")

    # Distribution figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, title, dsc in [(axes[0], "Train R1 (In-Sample)", desc_tr), (axes[1], "Test R1", desc_te)]:
        ks = sorted(dsc["depth_shares_pct"].keys())
        vs = [dsc["depth_shares_pct"][k] for k in ks]
        ax.bar(ks, vs, color="#1f77b4", alpha=0.88)
        ax.set_title(f"{title} (n={dsc['n_rows']})\nFlat curve: {dsc['flat_curve_pct']:.1f}%", fontsize=10)
        ax.set_xlabel("Recommended Depth")
        ax.tick_params(axis="x", rotation=40)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Share of R1 Batches (%)")
    fig.suptitle("M3-T09 Recommended Discount Depth Distribution (R1 Supported Depths)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURES_M3_DIR / "m3_t09_rec_depth_distribution.png", dpi=150)

    # Log to MLflow smart-shelf-3c-discount-optimizer
    with start_tracked_run(
        track="3C",
        model_type="whatif_profit_optimizer",
        feature_set_name="track_3b_r1_44",
        shelf_life_version="expiry_remaining_ratio",
        features=feats_3b,
        run_name="m3_t09_optimizer_run",
        purpose="optimizer_validation",
        train_rows=len(tr_r1),
        test_rows=len(te_r1),
        params={"flat_curve_threshold": flat_thr, "tie_rule": "shallower_depth", "grid": "0.10..0.75_step_0.05"},
    ) as run:
        log_track_3c_metrics(
            lane1_metrics={},
            lane2_metrics={"counterfactual_count": float(len(te_r1))},
            extra_metrics={
                "test_r1_flat_curve_pct": desc_te["flat_curve_pct"],
                "test_r1_at_shallowest_pct": desc_te["at_shallowest_supported_pct"],
                "test_r1_within_025_actual_pct": desc_te["within_025_of_actual_pct"],
                "profit_identity_max_err": profit_id_max_err,
            },
        )
        log_figure_artifact(fig, "m3_t09_rec_depth_distribution.png")
        mlflow_run_id = run.info.run_id

    plt.close("all")

    summary = {
        "flat_curve_threshold": flat_thr,
        "validation": {
            "r0_guard_passed": True,
            "profit_identity_max_err_1000": profit_id_max_err,
            "single_row_and_determinism_passed": True,
            "zero_supported_r1_train": zero_cand_tr,
            "zero_supported_r1_test": zero_cand_te,
        },
        "train_r1_descriptive": desc_tr,
        "test_r1_descriptive": desc_te,
        "mlflow_run_id": mlflow_run_id,
    }
    out_json = REPORTS_M3_DIR / "m3_t09_results_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary JSON: {out_json}")
    print(f"Logged MLflow run to smart-shelf-3c-discount-optimizer: {mlflow_run_id}")


if __name__ == "__main__":
    main()
