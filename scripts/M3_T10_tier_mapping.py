"""
scripts/M3_T10_tier_mapping.py - Derive Track 3C Tier Edges, Examples & Validation (M3-T10).

Executes M3-T10:
1. Derives frozen tier edges from training R1 recommendations -> models/track_3c_tier_edges.json.
2. Computes per-tier training documentation & composition.
3. Picks 1 representative training batch per tier by fixed distance-to-median rule
   and scores its supported profit curve via score_whatif().
4. Applies frozen edges to test R1 recommendations (flags tiers with < 20 test batches).
5. Validates monotonicity, single-row parity, JSON reload, and R0/R1 assignment rules.
6. Generates figures and logs run to MLflow smart-shelf-3c-discount-optimizer.
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
    REPORTS_DIR,
    TIER_EDGES_PATH,
)
from src.m2_feature_pipeline import TRACK_3B_FEATURES
from src.m3_t05_confounding import _build_region_support_reference
from src.m3_t07_eval import load_train_test_3b_with_raw
from src.m3_t09_optimizer import score_whatif
from src.m3_t10_tiers import (
    R0_STATUS_LABEL,
    TIER_WORDING,
    assign_tier,
    derive_tier_edges,
    load_tier_edges,
)
from src.mlflow_utils import log_figure_artifact, setup_tracking_environment, start_tracked_run

REPORTS_M3_DIR = REPORTS_DIR / "M3"
FIGURES_M3_DIR = FIGURES_DIR / "m3"
TIER_ORDER = ["Low", "Medium", "High", "Critical"]


def main() -> None:
    setup_tracking_environment()
    REPORTS_M3_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_M3_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("M3-T10: Track 3C Tier Mapping for the Dashboard")
    print("==================================================================")

    recs_tr = pd.read_csv(OPTIMIZER_RECS_TRAIN_PATH)
    recs_te = pd.read_csv(OPTIMIZER_RECS_TEST_PATH)

    tr_full, te_full = load_train_test_3b_with_raw()
    tr_r1 = tr_full[tr_full["raw_discount_pct"] > 0].reset_index(drop=True)
    te_r1 = te_full[te_full["raw_discount_pct"] > 0].reset_index(drop=True)
    te_r0 = te_full[te_full["raw_discount_pct"] == 0].reset_index(drop=True)

    # 1. Derive edges from training R1 only
    edges_payload = derive_tier_edges(recs_tr, save=True)
    e = edges_payload["edges"]
    print(f"\n[1] Derived Edges (Train R1, N={edges_payload['training_row_count']}): Edge1={e['edge1']:.2f}, Edge2={e['edge2']:.2f}, Edge3={e['edge3']:.2f}")
    print(f"    Saved {TIER_EDGES_PATH} (source sha256={edges_payload['source_sha256'][:12]}...)")

    # Reload check
    reloaded = load_tier_edges()
    assert reloaded == edges_payload, "Reloaded tier edges JSON differs from derived payload!"

    # Assign tiers
    tr_tiers = assign_tier(recs_tr, reloaded)
    te_tiers = assign_tier(recs_te, reloaded)
    recs_tr = pd.concat([recs_tr, tr_tiers], axis=1)
    recs_te = pd.concat([recs_te, te_tiers], axis=1)

    # 6. Validation checks
    grid_test_df = pd.DataFrame({
        "region": ["R1"] * 14 + ["R0"],
        "recommended_depth": [round(0.10 + 0.05 * i, 2) for i in range(14)] + [np.nan],
    })
    grid_out = assign_tier(grid_test_df, reloaded)
    tier_rank = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    r1_ranks = [tier_rank[t] for t in grid_out["tier"].iloc[:14]]
    assert r1_ranks == sorted(r1_ranks), "assign_tier is not monotone in depth!"
    assert pd.isna(grid_out["tier"].iloc[14]) and grid_out["tier_label"].iloc[14] == R0_STATUS_LABEL

    # Single-row vs batch parity
    single_t = assign_tier(recs_te.iloc[[0]][["region", "recommended_depth"]], reloaded)
    pd.testing.assert_frame_equal(single_t.reset_index(drop=True), te_tiers.iloc[[0]].reset_index(drop=True))
    assert recs_tr["tier"].isin(TIER_ORDER).all()
    assert recs_te["tier"].isin(TIER_ORDER).all()
    print("[6] Validation Checks: ALL PASSED (monotone, single-row parity, JSON reload, 100% R1 tiered, R0 untiered)")

    # 2. Per-tier training documentation
    ref = _build_region_support_reference()
    r1_edges = np.asarray(ref["r1_exp_tercile_edges"], dtype=float)
    tr_merged = tr_r1.merge(recs_tr, on="record_id", how="inner")
    tr_merged["exp_terc"] = np.digitize(tr_merged["raw_expiry_remaining_ratio"].to_numpy(dtype=float), r1_edges[1:-1], right=True) + 1
    tr_merged["grade_grp"] = np.where(tr_merged["quality_grade"] == "C", "Grade_C", "Grade_AB")

    tier_docs = []
    print("\n[2] Per-Tier Training Documentation (N=24,475):")
    for t in TIER_ORDER:
        sub = tr_merged[tr_merged["tier"] == t]
        comp = sub.groupby(["exp_terc", "grade_grp"]).size().to_dict()
        comp_str = {f"T{k[0]}_{k[1]}": int(v) for k, v in sorted(comp.items())}
        doc = {
            "tier": t,
            "depth_min": float(sub["recommended_depth"].min()),
            "depth_max": float(sub["recommended_depth"].max()),
            "n_rows": int(len(sub)),
            "share_pct": float(len(sub) / len(tr_merged) * 100.0),
            "median_pred_profit_per_unit": float(sub["predicted_profit_per_unit_at_rec"].median()),
            "flat_curve_pct": float(sub["flat_curve"].mean() * 100.0),
            "median_historical_actual_depth": float(sub["raw_discount_pct"].median()),
            "composition_exp_terc_x_grade": comp_str,
        }
        tier_docs.append(doc)
        print(
            f"    {t:8s} | [{doc['depth_min']:.2f}-{doc['depth_max']:.2f}] | n={doc['n_rows']:5d} ({doc['share_pct']:5.2f}%) "
            f"| med_profit/u=${doc['median_pred_profit_per_unit']:6.2f} | flat={doc['flat_curve_pct']:5.1f}% "
            f"| med_actual={doc['median_historical_actual_depth']:.2f} | comp={comp_str}"
        )

    # 3. Representative example batch per tier (fixed rule: closest to tier median on standardized expiry_remaining_ratio, oof_proba, recommended_depth; ties -> min record_id)
    std_cols = ["expiry_remaining_ratio", "oof_proba", "recommended_depth"]
    mu = tr_merged[std_cols].mean()
    sd = tr_merged[std_cols].std().replace(0.0, 1.0)

    examples_out = []
    fig_ex, axes_ex = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes_ex.flatten()
    print("\n[3] Representative Example Batches (closest to tier median):")
    for idx_t, t in enumerate(TIER_ORDER):
        sub = tr_merged[tr_merged["tier"] == t].copy()
        med = sub[std_cols].median()
        z_sub = (sub[std_cols] - mu) / sd
        z_med = (med - mu) / sd
        sub["_dist"] = np.sqrt(((z_sub - z_med) ** 2).sum(axis=1))
        chosen_row = sub.sort_values(["_dist", "record_id"]).iloc[[0]]
        rid = int(chosen_row["record_id"].iloc[0])

        curve = score_whatif(chosen_row)
        supp_curve = curve[curve["supported"]].copy()
        ex_rec = {
            "tier": t,
            "record_id": rid,
            "category": str(chosen_row["category"].iloc[0]),
            "quality_grade": str(chosen_row["quality_grade"].iloc[0]),
            "raw_expiry_remaining_ratio": float(chosen_row["raw_expiry_remaining_ratio"].iloc[0]),
            "oof_proba": float(chosen_row["oof_proba"].iloc[0]),
            "recommended_depth": float(chosen_row["recommended_depth"].iloc[0]),
            "flat_curve": bool(chosen_row["flat_curve"].iloc[0]),
            "margin_per_unit": float(chosen_row["margin_per_unit"].iloc[0]),
            "supported_profit_curve": [
                {
                    "depth": float(r["depth"]),
                    "s_hat": float(r["s_hat"]),
                    "profit_per_unit": float(r["profit_per_unit"]),
                    "batch_profit": float(r["batch_profit"]),
                }
                for _, r in supp_curve.iterrows()
            ],
        }
        examples_out.append(ex_rec)
        print(
            f"    {t:8s}: record_id={rid} ({ex_rec['category']}, Grade {ex_rec['quality_grade']}, "
            f"exp={ex_rec['raw_expiry_remaining_ratio']:.3f}, risk={ex_rec['oof_proba']:.3f}, "
            f"rec_d={ex_rec['recommended_depth']:.2f}, flat={ex_rec['flat_curve']})"
        )

        ax = axes_flat[idx_t]
        ax.plot(supp_curve["depth"], supp_curve["profit_per_unit"], "o-", color="#1f77b4", linewidth=2)
        ax.axvline(ex_rec["recommended_depth"], color="red", linestyle="--", linewidth=1, label=f"Rec={ex_rec['recommended_depth']:.2f}")
        ax.set_title(f"{t} Tier Example: record_id={rid}\n({ex_rec['category']}, Grade {ex_rec['quality_grade']}, exp={ex_rec['raw_expiry_remaining_ratio']:.2f})", fontsize=9)
        ax.set_xlabel("Supported Discount Depth")
        ax.set_ylabel("Predicted Profit / Unit ($)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig_ex.suptitle("M3-T10 Representative Batch Profit Curves per Tier (Supported Depths Only)", fontsize=12)
    fig_ex.tight_layout(rect=[0, 0, 1, 0.95])
    fig_ex.savefig(FIGURES_M3_DIR / "m3_t10_tier_example_curves.png", dpi=150)

    # 4. Apply frozen edges to test R1 recommendations
    test_docs = []
    print("\n[4] Test R1 Tier Application (N=6,176):")
    for t in TIER_ORDER:
        te_cnt = int((recs_te["tier"] == t).sum())
        te_share = float(te_cnt / len(recs_te) * 100.0)
        tr_share = next(d["share_pct"] for d in tier_docs if d["tier"] == t)
        lt20 = bool(te_cnt < 20)
        test_docs.append({
            "tier": t,
            "n_test_rows": te_cnt,
            "test_share_pct": te_share,
            "train_share_pct": tr_share,
            "diff_test_minus_train_pct": te_share - tr_share,
            "lt_20_test_batches": lt20,
        })
        flag_str = " [<20 BATCHES]" if lt20 else ""
        print(f"    {t:8s} | n={te_cnt:4d} ({te_share:5.2f}%) | diff vs train={te_share - tr_share:+5.2f}%{flag_str}")

    # Tier shares comparison chart
    fig_sh, ax_sh = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(len(TIER_ORDER))
    w = 0.35
    ax_sh.bar(x - w / 2, [d["share_pct"] for d in tier_docs], width=w, label="Train R1 (N=24,475)")
    ax_sh.bar(x + w / 2, [d["test_share_pct"] for d in test_docs], width=w, label="Test R1 (N=6,176)")
    ax_sh.set_xticks(x)
    ax_sh.set_xticklabels(TIER_ORDER)
    ax_sh.set_ylabel("Share of R1 Batches (%)")
    ax_sh.set_title("M3-T10 Track 3C Tier Shares (Train vs. Test R1)")
    ax_sh.legend()
    ax_sh.grid(axis="y", alpha=0.25)
    fig_sh.tight_layout()
    fig_sh.savefig(FIGURES_M3_DIR / "m3_t10_tier_shares.png", dpi=150)

    # Log to MLflow smart-shelf-3c-discount-optimizer
    with start_tracked_run(
        track="3C",
        model_type="tier_mapping",
        feature_set_name="track_3b_r1_44",
        shelf_life_version="expiry_remaining_ratio",
        features=list(TRACK_3B_FEATURES),
        run_name="m3_t10_tier_mapping",
        purpose="tier_mapping",
        train_rows=len(recs_tr),
        test_rows=len(recs_te),
        params={"edge1": e["edge1"], "edge2": e["edge2"], "edge3": e["edge3"]},
    ) as run:
        mlflow.log_metrics({
            "train_share_low": tier_docs[0]["share_pct"],
            "train_share_medium": tier_docs[1]["share_pct"],
            "train_share_high": tier_docs[2]["share_pct"],
            "train_share_critical": tier_docs[3]["share_pct"],
            "test_share_low": test_docs[0]["test_share_pct"],
            "test_share_medium": test_docs[1]["test_share_pct"],
            "test_share_high": test_docs[2]["test_share_pct"],
            "test_share_critical": test_docs[3]["test_share_pct"],
        })
        log_figure_artifact(fig_sh, "m3_t10_tier_shares.png")
        log_figure_artifact(fig_ex, "m3_t10_tier_example_curves.png")
        mlflow.log_artifact(str(TIER_EDGES_PATH))
        mlflow_run_id = run.info.run_id

    plt.close("all")

    summary = {
        "edges_payload": edges_payload,
        "train_tier_docs": tier_docs,
        "test_tier_docs": test_docs,
        "examples": examples_out,
        "tier_wording": TIER_WORDING,
        "mlflow_run_id": mlflow_run_id,
    }
    out_json = REPORTS_M3_DIR / "m3_t10_results_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary JSON: {out_json}")
    print(f"Logged MLflow run: {mlflow_run_id}")


if __name__ == "__main__":
    main()
