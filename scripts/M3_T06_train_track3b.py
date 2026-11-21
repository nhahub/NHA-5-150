"""
scripts/M3_T06_train_track3b.py - Track 3B Regressor Training, Tuning & Freezing (R1 Domain).

Executes M3-T06 strictly on training rows (split == "train"):
1. Extreme-outcome analysis on R1 (N = 24,475).
2. 5-fold time-blocked CV candidates (GroupMeanBaseline, LightGBM cross_entropy,
   LightGBM cross_entropy monotone-constrained, XGBoost reg:logistic, RandomForestRegressor)
   with 40 Optuna trials per ML family.
3. 1-SE model selection (preferring monotone-constrained LightGBM if within 1 SE).
4. Model-implied OOF depth slope diagnostic vs M3-T05 (+0.2614).
5. Heterogeneity check by baseline sell-through tercile.
6. Freeze final model on all R1 training rows, save via save_model_with_metadata,
   and verify predict_profit_curve() on R1, R0, and single-row inputs.
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

from src.config import FIGURES_DIR, MODEL_READY_3B_DATASET_PATH, RANDOM_STATE, REPORTS_DIR
from src.m3_t04_scenario import DISCOUNT_BIN_LABELS, assign_discount_bin
from src.m3_t05_confounding import (
    _build_region_support_reference,
    discount_support,
    scale_discount_values,
)
from src.m3_t06_regressor import (
    GroupMeanBaseline,
    R1_DEPTH_GRID,
    compute_extreme_outcome_stats,
    evaluate_3b_oof,
    load_r1_train_data,
    make_3b_model,
    predict_profit_curve,
    tune_3b_candidate,
)
from src.mlflow_utils import (
    log_figure_artifact,
    log_track_3b_metrics,
    save_model_with_metadata,
    setup_tracking_environment,
    start_tracked_run,
    to_repo_relative_path,
)

REPORTS_M3_DIR = REPORTS_DIR / "M3"
FIGURES_M3_DIR = FIGURES_DIR / "m3"


def _calibration_tables(r1: pd.DataFrame, oof_s: np.ndarray) -> dict:
    df = pd.DataFrame({
        "obs_s": r1["s"].to_numpy(dtype=float),
        "pred_s": oof_s,
        "bin": assign_discount_bin(r1["raw_discount_pct"]).astype(str),
    })
    df["decile"] = pd.qcut(df["pred_s"].rank(method="first"), q=10, labels=False) + 1
    by_dec = (
        df.groupby("decile", observed=True)
        .agg(n=("obs_s", "count"), mean_pred=("pred_s", "mean"), mean_obs=("obs_s", "mean"))
        .reset_index()
        .to_dict(orient="records")
    )
    by_bin = (
        df.groupby("bin", observed=True)
        .agg(n=("obs_s", "count"), mean_pred=("pred_s", "mean"), mean_obs=("obs_s", "mean"))
        .reindex(DISCOUNT_BIN_LABELS[1:])
        .reset_index()
        .to_dict(orient="records")
    )
    return {"by_pred_decile": by_dec, "by_depth_bin": by_bin}


def main() -> None:
    setup_tracking_environment()
    REPORTS_M3_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_M3_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("M3-T06: Track 3B Regressor Training & Freezing (R1 Domain Only)")
    print("==================================================================")

    r1, features_3b, const_cols, disc_slope, disc_intercept = load_r1_train_data()
    ref = _build_region_support_reference()
    grade_c_cutoff = float(ref["grade_c_ord_cutoff"])
    print(f"\n[D1-D3] R1 training rows={len(r1)} | features={len(features_3b)} | dropped constant={const_cols}")

    # 1. Extreme-outcome analysis
    ext_stats = compute_extreme_outcome_stats(r1)
    q = ext_stats["profit_per_unit_quantiles"]
    print(
        f"\n[1] Extreme-Outcome Stats (R1): profit/unit p01=${q['p01']:.2f}, p50=${q['p50']:.2f}, "
        f"p99=${q['p99']:.2f} | loss_share={ext_stats['loss_share_pct']:.2f}% | "
        f"below_-1x_cost={ext_stats['below_minus_1x_cost_share_pct']:.2f}%"
    )

    # Date-blocked 5-fold CV splits
    dates = pd.to_datetime(r1["transaction_date"]).values
    unique_dates = np.sort(np.unique(dates))
    date_blocks = np.array_split(unique_dates, 5)

    # 2. Candidates
    print("\n[2] Evaluating Candidates (5-fold time-blocked CV, 40 Optuna trials each)...")
    candidates = []

    # 2a. Baseline
    base_res = evaluate_3b_oof(
        r1=r1,
        features_3b=features_3b,
        model_builder=GroupMeanBaseline,
        date_blocks=date_blocks,
        is_baseline=True,
        disc_slope=disc_slope,
        disc_intercept=disc_intercept,
        grade_c_cutoff=grade_c_cutoff,
    )
    candidates.append({"family": "group_mean_baseline", "params": {"grouping": "exp_terc x grade_C x 10pt_depth"}, **base_res})

    # 2b-2e. ML Candidates (40 Optuna trials each, or --skip-optuna using tuned params)
    tuned_defaults = {
        "lgbm_ce": {"colsample_bytree": 0.9889333398041387, "learning_rate": 0.06477779459794802, "min_child_samples": 86, "num_leaves": 22, "n_estimators": 400, "reg_lambda": 6.815049987925894, "subsample": 0.9920525952002122},
        "lgbm_ce_monotone": {"colsample_bytree": 0.9950553116457923, "learning_rate": 0.08149619212376244, "min_child_samples": 93, "num_leaves": 15, "n_estimators": 500, "reg_lambda": 7.480534388702453, "subsample": 0.985296571922335},
        "xgboost_logistic": {"colsample_bytree": 0.9009280029539052, "learning_rate": 0.06225622467074855, "max_depth": 5, "min_child_weight": 13, "n_estimators": 500, "reg_lambda": 0.2978597194178599, "subsample": 0.9950167345650736},
        "rf": {"max_depth": 14, "max_features": 0.7, "min_samples_leaf": 12, "n_estimators": 200},
    }
    skip_optuna = "--skip-optuna" in sys.argv
    for fam in ["lgbm_ce", "lgbm_ce_monotone", "xgboost_logistic", "rf"]:
        if skip_optuna:
            p = tuned_defaults[fam]
            res = {"family": fam, "params": p, **evaluate_3b_oof(r1, features_3b, lambda f=fam, pr=p: make_3b_model(f, pr, features_3b), date_blocks)}
        else:
            res = tune_3b_candidate(fam, r1, features_3b, date_blocks, n_trials=40)
        candidates.append(res)

    for c in candidates:
        print(
            f"  {c['family']:20s} | s_RMSE={c['s_rmse']:.5f} (SE={c['s_rmse_se']:.5f}) | "
            f"s_MAE={c['s_mae']:.5f} | s_R2={c['s_r2']:.4f} | "
            f"p_u_RMSE=${c['profit_unit_rmse']:.2f} | p_RMSE(all/no1%)=${c['profit_rmse_all']:.0f}/${c['profit_rmse_no_ext1pct']:.0f}"
        )
        with start_tracked_run(
            track="3B",
            model_type=c["family"],
            feature_set_name="track_3b_r1_44",
            shelf_life_version="expiry_remaining_ratio",
            features=features_3b,
            run_name=f"m3_t06_cv_{c['family']}",
            purpose="cv_candidate",
            train_rows=len(r1),
            params=c["params"],
        ):
            log_track_3b_metrics(
                profit_mae=c["profit_mae_all"],
                profit_rmse=c["profit_rmse_all"],
                profit_r2=c["s_r2"],
                sell_through_mae=c["s_mae"],
                sell_through_rmse=c["s_rmse"],
                sell_through_r2=c["s_r2"],
                extra_metrics={
                    "s_rmse_se": c["s_rmse_se"],
                    "profit_unit_mae": c["profit_unit_mae"],
                    "profit_unit_rmse": c["profit_unit_rmse"],
                    "profit_rmse_no_ext1pct": c["profit_rmse_no_ext1pct"],
                },
            )

    # Selection via 1-SE rule (monotone LightGBM preferred if within 1 SE of best)
    best_cand = min(candidates, key=lambda c: c["s_rmse"])
    se_cutoff = best_cand["s_rmse"] + best_cand["s_rmse_se"]
    lgbm_mono = next(c for c in candidates if c["family"] == "lgbm_ce_monotone")
    if lgbm_mono["s_rmse"] <= se_cutoff:
        chosen = lgbm_mono
        sel_reason = f"lgbm_ce_monotone ({lgbm_mono['s_rmse']:.5f}) is within 1 SE ({se_cutoff:.5f}) of best ({best_cand['family']} {best_cand['s_rmse']:.5f}); chosen per pre-registered monotonicity & parsimony rule."
    else:
        chosen = best_cand
        sel_reason = f"{best_cand['family']} ({best_cand['s_rmse']:.5f}) beats monotone constraint beyond 1 SE ({se_cutoff:.5f})."
    print(f"\n[Selected] {chosen['family']} | {sel_reason}")

    # Calibration tables & figure
    cal_chosen = _calibration_tables(r1, chosen["oof_s"])
    fig_cal, axes_c = plt.subplots(1, 2, figsize=(12, 4.5))
    dec_df = pd.DataFrame(cal_chosen["by_pred_decile"])
    axes_c[0].plot(dec_df["mean_pred"], dec_df["mean_obs"], marker="o", label="OOF Observed")
    axes_c[0].plot([0, 1], [0, 1], "k--", label="Perfect")
    axes_c[0].set_xlabel("Mean Predicted s")
    axes_c[0].set_ylabel("Mean Observed s")
    axes_c[0].set_title(f"Calibration by Predicted-s Decile ({chosen['family']})")
    axes_c[0].legend()
    axes_c[0].grid(alpha=0.25)

    bin_df = pd.DataFrame(cal_chosen["by_depth_bin"])
    axes_c[1].plot(bin_df["bin"], bin_df["mean_pred"], marker="o", label="Mean Predicted s")
    axes_c[1].plot(bin_df["bin"], bin_df["mean_obs"], marker="s", label="Mean Observed s")
    axes_c[1].set_xlabel("Discount Depth Bin")
    axes_c[1].set_ylabel("Mean Sell-Through s")
    axes_c[1].set_title("Calibration by 5-Point Depth Bin (R1 OOF)")
    axes_c[1].tick_params(axis="x", rotation=40)
    axes_c[1].legend()
    axes_c[1].grid(alpha=0.25)
    fig_cal.tight_layout()
    fig_cal.savefig(FIGURES_M3_DIR / "m3_t06_cv_calibration.png", dpi=150)

    # 3 & 4. OOF Counterfactual Grid for Model-Implied Slope & Heterogeneity Check
    print("\n[3-4] Computing OOF counterfactual grid for slope & heterogeneity diagnostics...")
    grid_arr = np.asarray(R1_DEPTH_GRID, dtype=float)
    scaled_grid = scale_discount_values(grid_arr, disc_slope, disc_intercept)
    supp_df, _ = discount_support(r1, grid=R1_DEPTH_GRID)
    supp_mat = supp_df.to_numpy(dtype=bool)

    X_r1 = r1[features_3b]
    y_s = r1["s"].to_numpy(dtype=float)
    oof_grid_s = np.zeros((len(r1), len(grid_arr)), dtype=float)
    for val_dates in date_blocks:
        val_m = np.isin(dates, val_dates)
        m_fold = make_3b_model(chosen["family"], chosen["params"], features_3b)
        m_fold.fit(X_r1.loc[~val_m], y_s[~val_m])
        X_val = X_r1.loc[val_m].copy()
        for j, d_sc in enumerate(scaled_grid):
            X_val["discount_pct"] = d_sc
            oof_grid_s[val_m, j] = np.clip(m_fold.predict(X_val), 0.0, 1.0)

    # Model-implied finite-difference slopes over consecutive supported grid points
    ds_grid = np.diff(oof_grid_s, axis=1) / np.diff(grid_arr)[None, :]
    pair_supp = supp_mat[:, :-1] & supp_mat[:, 1:]
    overall_pd_slope = float(ds_grid[pair_supp].mean())

    mid_d = 0.5 * (grid_arr[:-1] + grid_arr[1:])
    slope_by_band = {}
    for lbl, lo, hi in [("10-30%", 0.10, 0.30), ("30-50%", 0.30, 0.50), ("50-75%", 0.50, 0.76)]:
        bm = (mid_d >= lo) & (mid_d < hi)
        m_sub = pair_supp[:, bm]
        slope_by_band[lbl] = float(ds_grid[:, bm][m_sub].mean())

    r1_edges = np.asarray(ref["r1_exp_tercile_edges"], dtype=float)
    r1_terc = np.digitize(r1["raw_expiry_remaining_ratio"].to_numpy(dtype=float), r1_edges[1:-1], right=True) + 1
    slope_by_terc = {}
    for t_i in [1, 2, 3]:
        tm = r1_terc == t_i
        slope_by_terc[f"R1_exp_tercile_{t_i}"] = float(ds_grid[tm][pair_supp[tm]].mean())

    print(f"  Model-implied OOF slope (supported): {overall_pd_slope:+.4f} (M3-T05 estimate: +0.2614, 95% CI [+0.2342, +0.2880])")
    print(f"  Slope by depth band: {slope_by_band}")
    print(f"  Slope by R1 expiry tercile: {slope_by_terc}")

    # Heterogeneity check by baseline sell-through tercile (at d=0.10)
    base_p = r1["raw_base_price"].to_numpy(dtype=float)
    cost_p = r1["raw_cost_price"].to_numpy(dtype=float)
    sp_grid = np.round(base_p[:, None] * (1.0 - grid_arr[None, :]), 2)
    oof_grid_profit_u = oof_grid_s * sp_grid - cost_p[:, None]

    base_s_terc = pd.qcut(oof_grid_s[:, 0], q=3, labels=["Low_Base_s", "Mid_Base_s", "High_Base_s"])
    het_summary = []
    fig_het, ax_h = plt.subplots(figsize=(8.5, 4.8))
    for grp in ["Low_Base_s", "Mid_Base_s", "High_Base_s"]:
        gm = np.asarray(base_s_terc == grp)
        # Mean over supported rows at each depth d
        mean_prof_u = []
        valid_d = []
        for j, d in enumerate(grid_arr):
            sm = gm & supp_mat[:, j]
            if sm.sum() >= 30:
                valid_d.append(d)
                mean_prof_u.append(float(oof_grid_profit_u[sm, j].mean()))
        best_idx = int(np.argmax(mean_prof_u))
        het_summary.append({
            "baseline_s_tercile": grp,
            "n_rows": int(gm.sum()),
            "mean_s_at_10pct": float(oof_grid_s[gm, 0].mean()),
            "profit_max_supported_depth": float(valid_d[best_idx]),
            "max_mean_profit_unit": float(mean_prof_u[best_idx]),
            "profit_unit_at_10pct": float(mean_prof_u[0]),
            "curve_depths": valid_d,
            "curve_profit_unit": mean_prof_u,
        })
        ax_h.plot(valid_d, mean_prof_u, marker="o", linewidth=2, label=f"{grp} (best d={valid_d[best_idx]:.2f})")
        print(f"  Heterogeneity {grp:11s}: base_s@10%={oof_grid_s[gm, 0].mean():.3f} | best supported depth={valid_d[best_idx]:.2f} (profit/u=${mean_prof_u[best_idx]:.2f} vs @10%=${mean_prof_u[0]:.2f})")

    ax_h.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax_h.set_xlabel("Candidate Discount Depth (Supported Grid)")
    ax_h.set_ylabel("Mean Predicted Profit / Initial Unit ($)")
    ax_h.set_title("M3-T06 Heterogeneity: Profit Curve by Baseline Sell-Through Tercile (R1 OOF)")
    ax_h.legend()
    ax_h.grid(alpha=0.25)
    fig_het.tight_layout()
    fig_het.savefig(FIGURES_M3_DIR / "m3_t06_heterogeneity_profit_curves.png", dpi=150)

    # 5. Fit final model on all 24,475 R1 rows & save with metadata
    print("\n[5] Fitting final Track 3B regressor on all R1 training rows and freezing...")
    final_model = make_3b_model(chosen["family"], chosen["params"], features_3b)
    final_model.fit(X_r1, y_s)

    with start_tracked_run(
        track="3B",
        model_type=chosen["family"],
        feature_set_name="track_3b_r1_44",
        shelf_life_version="expiry_remaining_ratio",
        features=features_3b,
        run_name="m3_t06_final_track_3b",
        purpose="production",
        train_rows=len(r1),
        params={**chosen["params"], "region_domain": "R1_only", "oof_version": "v2"},
    ) as run:
        log_track_3b_metrics(
            profit_mae=chosen["profit_mae_all"],
            profit_rmse=chosen["profit_rmse_all"],
            profit_r2=chosen["s_r2"],
            sell_through_mae=chosen["s_mae"],
            sell_through_rmse=chosen["s_rmse"],
            sell_through_r2=chosen["s_r2"],
            extra_metrics={
                "profit_unit_mae": chosen["profit_unit_mae"],
                "profit_unit_rmse": chosen["profit_unit_rmse"],
                "model_implied_slope": overall_pd_slope,
            },
        )
        log_figure_artifact(fig_cal, "m3_t06_cv_calibration.png")
        log_figure_artifact(fig_het, "m3_t06_heterogeneity_profit_curves.png")

        model_path, meta_path = save_model_with_metadata(
            model=final_model,
            track="3B",
            feature_list=features_3b,
            shelf_life_version="expiry_remaining_ratio",
            train_period="2023-01-01 to 2024-08-07 (R1 only)",
            mlflow_run_id=run.info.run_id,
            overwrite=True,
            log_as_artifact=True,
        )
        final_run_id = run.info.run_id

    # Enrich JSON metadata with required M3-T06 fields
    meta_obj = json.loads(meta_path.read_text())
    meta_obj.update({
        "oof_version": "v2",
        "region_domain": "R1_only (discount_pct > 0)",
        "discount_support_ref": "models/track_3b_region_support_ref.json",
        "target_definition": "s = units_sold / initial_quantity in [0, 1]; profit = initial_quantity * (s * round(base_price*(1-d), 2) - cost_price)",
        "disc_scale_slope": disc_slope,
        "disc_scale_intercept": disc_intercept,
        "cv_scores": {
            "s_mae": chosen["s_mae"],
            "s_rmse": chosen["s_rmse"],
            "s_rmse_se": chosen["s_rmse_se"],
            "s_r2": chosen["s_r2"],
            "profit_unit_mae": chosen["profit_unit_mae"],
            "profit_unit_rmse": chosen["profit_unit_rmse"],
        },
    })
    meta_path.write_text(json.dumps(meta_obj, indent=2))

    # 6. Test predict_profit_curve() on R1, R0, and single-row inputs
    full_tr = pd.read_csv(MODEL_READY_3B_DATASET_PATH)
    r0_sample = full_tr[(full_tr["split"] == "train") & (full_tr["raw_discount_pct"] == 0)].iloc[[0]]
    r1_sample = r1.iloc[[0]]
    single_curve = predict_profit_curve(r1_sample)
    both_curves = predict_profit_curve(pd.concat([r1_sample, r0_sample], ignore_index=True))
    assert len(single_curve) == len(R1_DEPTH_GRID)
    assert both_curves[both_curves["record_id"] == r0_sample["record_id"].iloc[0]]["s_hat"].isna().all()
    assert (both_curves[both_curves["record_id"] == r0_sample["record_id"].iloc[0]]["reason"] == "region_R0_no_discount_evidence").all()
    print(f"\n[6] predict_profit_curve() verified on single-row R1 ({int(single_curve['supported'].sum())}/14 supported) and R0 (0/14 supported, NaN + region_R0_no_discount_evidence).")

    plt.close("all")

    summary = {
        "dropped_constant_cols": const_cols,
        "extreme_outcomes": ext_stats,
        "candidates": [
            {k: v for k, v in c.items() if k != "oof_s"} for c in candidates
        ],
        "selected": {
            "family": chosen["family"],
            "reason": sel_reason,
            "calibration": cal_chosen,
        },
        "slope_comparison": {
            "model_implied_overall": overall_pd_slope,
            "by_depth_band": slope_by_band,
            "by_r1_expiry_tercile": slope_by_terc,
            "m3_t05_cross_fitted": {"slope": 0.2614, "ci95": [0.2342, 0.2880]},
        },
        "heterogeneity": het_summary,
        "final_mlflow_run_id": final_run_id,
        "model_path": to_repo_relative_path(model_path),
        "meta_path": to_repo_relative_path(meta_path),
    }
    out_json = REPORTS_M3_DIR / "m3_t06_results_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary JSON: {out_json}")


if __name__ == "__main__":
    main()
