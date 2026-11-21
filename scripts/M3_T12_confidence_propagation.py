"""
scripts/M3_T12_confidence_propagation.py - Track 3C Confidence Propagation & Error Check (M3-T12).

Executes:
1. Attaches confidence_level, confidence_reasons, component flags, and near_tie to
   optimizer_recs_train.csv and optimizer_recs_test.csv -> saves *_with_confidence_*.csv.
2. Runs validation checks (weakest-link, no High, R0->No recommendation, single-row parity).
3. Computes train & test R1 distributions, flag overlaps, tier/category/expiry*grade breakdowns.
4. Runs descriptive actual-depth regressor error check by confidence level on test R1
   with 1,000-resample week-grouped bootstrap 95% CIs.
5. Saves figures and logs to MLflow smart-shelf-3c-discount-optimizer.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FIGURES_DIR,
    OPTIMIZER_RECS_CONF_TEST_PATH,
    OPTIMIZER_RECS_CONF_TRAIN_PATH,
    OPTIMIZER_RECS_TEST_PATH,
    OPTIMIZER_RECS_TRAIN_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TRACK_3B_MODEL_PATH,
)
from src.m2_feature_pipeline import TRACK_3B_FEATURES
from src.m3_t05_confounding import _build_region_support_reference
from src.m3_t07_eval import load_train_test_3b_with_raw
from src.m3_t09_optimizer import optimize_batches
from src.m3_t12_confidence import CONFIDENCE_WORDING, attach_confidence
from src.mlflow_utils import load_model_with_metadata, log_figure_artifact, setup_tracking_environment, start_tracked_run

REPORTS_M3_DIR = REPORTS_DIR / "M3"
FIGURES_M3_DIR = FIGURES_DIR / "m3"
FLAGS = ["low_history_3a", "limited_depth_category", "support_edge", "critical_tier"]


def _summarize_r1(df_conf: pd.DataFrame, r1_raw: pd.DataFrame, ref: dict) -> dict:
    m = df_conf.merge(
        r1_raw[["record_id", "category", "quality_grade", "raw_expiry_remaining_ratio"]],
        on="record_id",
        how="inner",
    )
    n = len(m)
    lvl_shares = {k: float((m["confidence_level"] == k).mean() * 100.0) for k in ["Low", "Moderate"]}
    flag_shares = {f: float(m[f].mean() * 100.0) for f in FLAGS + ["near_tie"]}
    n_flags_dist = m[FLAGS].sum(axis=1).value_counts().sort_index().to_dict()

    # Pairwise flag overlap (% of R1 rows where both are True)
    overlap = {}
    for i, f1 in enumerate(FLAGS):
        for f2 in FLAGS[i + 1:]:
            overlap[f"{f1}_AND_{f2}"] = float((m[f1] & m[f2]).mean() * 100.0)

    by_tier = (
        m.groupby("tier", observed=True)["confidence_level"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .mul(100.0)
        .to_dict(orient="index")
    )
    by_cat = (
        m.groupby("category", observed=True)["confidence_level"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .mul(100.0)
        .to_dict(orient="index")
    )

    r1_edges = np.asarray(ref["r1_exp_tercile_edges"], dtype=float)
    m["exp_terc"] = np.digitize(m["raw_expiry_remaining_ratio"].to_numpy(dtype=float), r1_edges[1:-1], right=True) + 1
    m["grade_grp"] = np.where(m["quality_grade"] == "C", "Grade_C", "Grade_AB")
    m["is_mod"] = (m["confidence_level"] == "Moderate").astype(float) * 100.0
    mod_by_eg = (
        m.groupby(["exp_terc", "grade_grp"], observed=True)
        .agg(n=("record_id", "count"), moderate_pct=("is_mod", "mean"))
        .reset_index()
        .to_dict(orient="records")
    )

    return {
        "n_rows": int(n),
        "level_shares_pct": lvl_shares,
        "flag_shares_pct": flag_shares,
        "active_flag_count_dist": {int(k): int(v) for k, v in n_flags_dist.items()},
        "pairwise_overlap_pct": overlap,
        "by_tier_pct": by_tier,
        "by_category_pct": by_cat,
        "moderate_by_exp_terc_and_grade": mod_by_eg,
    }


def _error_by_level_test(te_conf: pd.DataFrame, te_r1: pd.DataFrame, n_boot: int = 1000) -> dict:
    model, meta = load_model_with_metadata(TRACK_3B_MODEL_PATH)
    feats = list(meta["feature_list"])
    pred_s = np.clip(model.predict(te_r1[feats]), 0.0, 1.0)
    obs_s = te_r1["s"].to_numpy(dtype=float)
    obs_pu = (te_r1["profit"] / te_r1["raw_initial_quantity"]).to_numpy(dtype=float)
    pred_pu = pred_s * te_r1["raw_selling_price"].to_numpy(dtype=float) - te_r1["raw_cost_price"].to_numpy(dtype=float)

    df = pd.DataFrame({
        "level": te_conf["confidence_level"].values,
        "week": pd.to_datetime(te_r1["transaction_date"]).dt.to_period("W").astype(str).values,
        "obs_s": obs_s,
        "pred_s": pred_s,
        "obs_pu": obs_pu,
        "pred_pu": pred_pu,
    })

    rng = np.random.default_rng(RANDOM_STATE)
    out = {}
    for lvl in ["Low", "Moderate"]:
        sub = df[df["level"] == lvl].reset_index(drop=True)
        weeks = sub["week"].to_numpy()
        w_idx = [np.where(weeks == w)[0] for w in np.unique(weeks)]

        mae_pu = float(mean_absolute_error(sub["obs_pu"], sub["pred_pu"]))
        rmse_pu = float(np.sqrt(mean_squared_error(sub["obs_pu"], sub["pred_pu"])))
        mae_s = float(mean_absolute_error(sub["obs_s"], sub["pred_s"]))
        rmse_s = float(np.sqrt(mean_squared_error(sub["obs_s"], sub["pred_s"])))

        b_mae_pu, b_rmse_pu, b_mae_s, b_rmse_s = [], [], [], []
        o_pu = sub["obs_pu"].to_numpy()
        p_pu = sub["pred_pu"].to_numpy()
        o_s = sub["obs_s"].to_numpy()
        p_s = sub["pred_s"].to_numpy()
        for _ in range(n_boot):
            chosen = rng.integers(0, len(w_idx), size=len(w_idx))
            idx = np.concatenate([w_idx[w] for w in chosen])
            b_mae_pu.append(mean_absolute_error(o_pu[idx], p_pu[idx]))
            b_rmse_pu.append(np.sqrt(mean_squared_error(o_pu[idx], p_pu[idx])))
            b_mae_s.append(mean_absolute_error(o_s[idx], p_s[idx]))
            b_rmse_s.append(np.sqrt(mean_squared_error(o_s[idx], p_s[idx])))

        out[lvl] = {
            "n_rows": int(len(sub)),
            "mae_profit_unit": mae_pu,
            "mae_profit_unit_ci95": [float(np.percentile(b_mae_pu, 2.5)), float(np.percentile(b_mae_pu, 97.5))],
            "rmse_profit_unit": rmse_pu,
            "rmse_profit_unit_ci95": [float(np.percentile(b_rmse_pu, 2.5)), float(np.percentile(b_rmse_pu, 97.5))],
            "mae_s": mae_s,
            "mae_s_ci95": [float(np.percentile(b_mae_s, 2.5)), float(np.percentile(b_mae_s, 97.5))],
            "rmse_s": rmse_s,
            "rmse_s_ci95": [float(np.percentile(b_rmse_s, 2.5)), float(np.percentile(b_rmse_s, 97.5))],
            "obs_profit_unit_mean": float(o_pu.mean()),
            "obs_profit_unit_median": float(np.median(o_pu)),
        }
    return out


def main() -> None:
    setup_tracking_environment()
    REPORTS_M3_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_M3_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("M3-T12: Track 3C Confidence Propagation & Error-by-Level Check")
    print("==================================================================")

    recs_tr = pd.read_csv(OPTIMIZER_RECS_TRAIN_PATH)
    recs_te = pd.read_csv(OPTIMIZER_RECS_TEST_PATH)
    tr, te = load_train_test_3b_with_raw()
    tr_r1 = tr[tr["raw_discount_pct"] > 0].reset_index(drop=True)
    te_r1 = te[te["raw_discount_pct"] > 0].reset_index(drop=True)
    te_r0 = te[te["raw_discount_pct"] == 0].reset_index(drop=True)

    # 1 & 4. Attach confidence and save
    conf_tr = attach_confidence(recs_tr, tr_r1)
    conf_te = attach_confidence(recs_te, te_r1)
    conf_tr.to_csv(OPTIMIZER_RECS_CONF_TRAIN_PATH, index=False)
    conf_te.to_csv(OPTIMIZER_RECS_CONF_TEST_PATH, index=False)
    print(f"\n[1,4] Saved {OPTIMIZER_RECS_CONF_TRAIN_PATH} ({len(conf_tr)}) & {OPTIMIZER_RECS_CONF_TEST_PATH} ({len(conf_te)})")

    # 2. Validation checks
    r0_recs = optimize_batches(te_r0.head(10))
    r0_conf = attach_confidence(r0_recs, te_r0.head(10))
    assert (r0_conf["confidence_level"] == "No recommendation").all()
    assert not conf_tr["confidence_level"].isin(["High"]).any()
    assert not conf_te["confidence_level"].isin(["High"]).any()
    assert conf_tr["confidence_level"].isin(["Low", "Moderate"]).all()
    assert conf_te["confidence_level"].isin(["Low", "Moderate"]).all()

    # Single-row vs batch parity
    single_c = attach_confidence(recs_te.iloc[[0]], te_r1.iloc[[0]])
    pd.testing.assert_frame_equal(single_c.reset_index(drop=True), conf_te.iloc[[0]].reset_index(drop=True))

    # Weakest-link property check on Moderate rows
    mod_idx = conf_te.index[conf_te["confidence_level"] == "Moderate"][0]
    mod_rec = recs_te.iloc[[mod_idx]].copy()
    mod_batch = te_r1.iloc[[mod_idx]].copy()
    mod_batch["category"] = "Frozen_Meals"  # force F3B1
    forced_low = attach_confidence(mod_rec, mod_batch)
    assert forced_low["confidence_level"].iloc[0] == "Low"
    print("[2] Validation Checks: ALL PASSED (weakest-link, no High, R0->No recommendation, single-row parity)")

    # 3. Distributions
    ref = _build_region_support_reference()
    dist_tr = _summarize_r1(conf_tr, tr_r1, ref)
    dist_te = _summarize_r1(conf_te, te_r1, ref)
    for lbl, d in [("TRAIN R1", dist_tr), ("TEST R1", dist_te)]:
        print(f"\n[3] {lbl} (N={d['n_rows']}): Low={d['level_shares_pct']['Low']:.2f}%, Moderate={d['level_shares_pct']['Moderate']:.2f}%")
        print(f"    Flags (%): { {k: round(v, 2) for k, v in d['flag_shares_pct'].items()} }")
        print(f"    Active flag counts: {d['active_flag_count_dist']}")

    # 5. Error by level (Test R1 actual depth)
    err_lvl = _error_by_level_test(conf_te, te_r1, n_boot=1000)
    print("\n[5] Test R1 Actual-Depth Error by Confidence Level:")
    for lvl in ["Low", "Moderate"]:
        e = err_lvl[lvl]
        print(
            f"    {lvl:8s} (n={e['n_rows']:4d}) | p/u MAE=${e['mae_profit_unit']:.2f} {e['mae_profit_unit_ci95']} "
            f"| p/u RMSE=${e['rmse_profit_unit']:.2f} | s MAE={e['mae_s']:.4f} | obs p/u=${e['obs_profit_unit_mean']:.2f}"
        )

    # Figures
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4.3))
    lvls = ["Low", "Moderate"]
    x = np.arange(len(lvls))
    axes1[0].bar(x - 0.18, [dist_tr["level_shares_pct"][k] for k in lvls], width=0.35, label="Train R1")
    axes1[0].bar(x + 0.18, [dist_te["level_shares_pct"][k] for k in lvls], width=0.35, label="Test R1")
    axes1[0].set_xticks(x)
    axes1[0].set_xticklabels(lvls)
    axes1[0].set_ylabel("Share of R1 Batches (%)")
    axes1[0].set_title("Confidence Level Shares (No 'High' Level)")
    axes1[0].legend()
    axes1[0].grid(axis="y", alpha=0.25)

    fl_names = FLAGS + ["near_tie"]
    xf = np.arange(len(fl_names))
    axes1[1].barh(xf - 0.18, [dist_tr["flag_shares_pct"][k] for k in fl_names], height=0.35, label="Train R1")
    axes1[1].barh(xf + 0.18, [dist_te["flag_shares_pct"][k] for k in fl_names], height=0.35, label="Test R1")
    axes1[1].set_yticks(xf)
    axes1[1].set_yticklabels(fl_names)
    axes1[1].set_xlabel("Share of R1 Batches (%)")
    axes1[1].set_title("Component Flags & Informational near_tie")
    axes1[1].legend()
    axes1[1].grid(axis="x", alpha=0.25)
    fig1.tight_layout()
    fig1.savefig(FIGURES_M3_DIR / "m3_t12_confidence_distribution.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(7, 4.3))
    maes = [err_lvl[k]["mae_profit_unit"] for k in lvls]
    los = [err_lvl[k]["mae_profit_unit"] - err_lvl[k]["mae_profit_unit_ci95"][0] for k in lvls]
    his = [err_lvl[k]["mae_profit_unit_ci95"][1] - err_lvl[k]["mae_profit_unit"] for k in lvls]
    ax2.errorbar(lvls, maes, yerr=[los, his], fmt="o", capsize=6, markersize=8, linewidth=2, color="#1f77b4")
    ax2.set_ylabel("Actual-Depth Profit/Unit MAE ($)")
    ax2.set_title("Test R1 Regressor Error by Confidence Level (95% Week-Boot CI)")
    ax2.grid(axis="y", alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(FIGURES_M3_DIR / "m3_t12_error_by_confidence_level.png", dpi=150)

    with start_tracked_run(
        track="3C",
        model_type="confidence_propagation",
        feature_set_name="track_3b_r1_44",
        shelf_life_version="expiry_remaining_ratio",
        features=list(TRACK_3B_FEATURES),
        run_name="m3_t12_confidence_propagation",
        purpose="confidence_propagation",
        train_rows=len(conf_tr),
        test_rows=len(conf_te),
    ) as run:
        mlflow.log_metrics({
            "train_r1_low_pct": dist_tr["level_shares_pct"]["Low"],
            "train_r1_moderate_pct": dist_tr["level_shares_pct"]["Moderate"],
            "test_r1_low_pct": dist_te["level_shares_pct"]["Low"],
            "test_r1_moderate_pct": dist_te["level_shares_pct"]["Moderate"],
            "test_low_mae_unit": err_lvl["Low"]["mae_profit_unit"],
            "test_moderate_mae_unit": err_lvl["Moderate"]["mae_profit_unit"],
        })
        log_figure_artifact(fig1, "m3_t12_confidence_distribution.png")
        log_figure_artifact(fig2, "m3_t12_error_by_confidence_level.png")
        mlflow_run_id = run.info.run_id

    plt.close("all")

    summary = {
        "train_r1": dist_tr,
        "test_r1": dist_te,
        "error_by_level_test_r1": err_lvl,
        "wording": CONFIDENCE_WORDING,
        "mlflow_run_id": mlflow_run_id,
    }
    out_json = REPORTS_M3_DIR / "m3_t12_results_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary JSON: {out_json}")
    print(f"Logged MLflow run: {mlflow_run_id}")


if __name__ == "__main__":
    main()
