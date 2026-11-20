"""
src/m3_t07_eval.py - Track 3B Final Test Evaluation & Shape-Check Helpers (M3-T07).

Provides reusable evaluation functions for:
  - Test-set region integrity and distribution shift checks (test R1 vs train R1).
  - R1 primary test metrics, decile/depth calibration, and subgroup errors.
  - Rule T1 (generalization) and Rule T2 (unconstrained partial-slope shape) checks.
  - Train-fit residualization slope on test R1 with week-grouped bootstrap 95% CI.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import CLEANED_DATASET_PATH, MODEL_READY_3B_DATASET_PATH, RANDOM_STATE
from src.m3_t04_scenario import DISCOUNT_BIN_LABELS, assign_discount_bin
from src.m3_t05_confounding import (
    _build_region_support_reference,
    discount_support,
    scale_discount_values,
)
from src.m3_t06_regressor import R1_DEPTH_GRID


def load_train_test_3b_with_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads model_ready_3b_dataset.csv merged with raw expiry_remaining_ratio & quality_grade."""
    df = pd.read_csv(MODEL_READY_3B_DATASET_PATH)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    cl = pd.read_csv(
        CLEANED_DATASET_PATH,
        usecols=["record_id", "expiry_remaining_ratio", "quality_grade"],
    ).rename(columns={"expiry_remaining_ratio": "raw_expiry_remaining_ratio"})
    df = df.merge(cl, on="record_id", how="inner")

    tr = df[df["split"] == "train"].sort_values(["transaction_date", "record_id"]).reset_index(drop=True)
    te = df[df["split"] == "test"].sort_values(["transaction_date", "record_id"]).reset_index(drop=True)
    assert len(tr) == 80042 and len(te) == 19958
    return tr, te


def dist_summary(arr: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def compute_r1_metrics(df_r1: pd.DataFrame, pred_s: np.ndarray) -> Dict[str, float]:
    """Computes s-scale and derived profit-scale metrics on R1 rows."""
    y_s = df_r1["s"].to_numpy(dtype=float)
    pred_clip = np.clip(pred_s, 0.0, 1.0)
    sell_p = df_r1["raw_selling_price"].to_numpy(dtype=float)
    cost_p = df_r1["raw_cost_price"].to_numpy(dtype=float)
    init_q = df_r1["raw_initial_quantity"].to_numpy(dtype=float)

    obs_prof_u = (df_r1["profit"] / init_q).to_numpy(dtype=float)
    pred_prof_u = pred_clip * sell_p - cost_p
    obs_prof = df_r1["profit"].to_numpy(dtype=float)
    pred_prof = pred_prof_u * init_q

    p01, p99 = np.percentile(obs_prof, [1, 99])
    non_ext = (obs_prof >= p01) & (obs_prof <= p99)

    return {
        "n_rows": int(len(df_r1)),
        "s_mae": float(mean_absolute_error(y_s, pred_clip)),
        "s_rmse": float(np.sqrt(mean_squared_error(y_s, pred_clip))),
        "s_r2": float(r2_score(y_s, pred_clip)),
        "profit_unit_mae": float(mean_absolute_error(obs_prof_u, pred_prof_u)),
        "profit_unit_rmse": float(np.sqrt(mean_squared_error(obs_prof_u, pred_prof_u))),
        "profit_mae_all": float(mean_absolute_error(obs_prof, pred_prof)),
        "profit_rmse_all": float(np.sqrt(mean_squared_error(obs_prof, pred_prof))),
        "profit_rmse_no_ext1pct": float(np.sqrt(mean_squared_error(obs_prof[non_ext], pred_prof[non_ext]))),
    }


def compute_test_residual_slope(
    tr_r1: pd.DataFrame,
    te_r1: pd.DataFrame,
    controls_only: List[str],
    n_boot: int = 500,
) -> Dict[str, Any]:
    """
    Fits s ~ controls (cross_entropy) and raw_discount_pct ~ controls (regression)
    on TRAIN R1 only (no refitting on test), residualizes TEST R1, and estimates
    the residual-on-residual slope with a week-grouped bootstrap 95% CI.
    """
    m_s = lgb.LGBMRegressor(
        objective="cross_entropy",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=400,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    m_d = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=400,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    m_s.fit(tr_r1[controls_only], tr_r1["s"].to_numpy(dtype=float))
    m_d.fit(tr_r1[controls_only], tr_r1["raw_discount_pct"].to_numpy(dtype=float))

    r_s = te_r1["s"].to_numpy(dtype=float) - np.clip(m_s.predict(te_r1[controls_only]), 0.0, 1.0)
    r_d = te_r1["raw_discount_pct"].to_numpy(dtype=float) - m_d.predict(te_r1[controls_only])
    slope = float(np.dot(r_d, r_s) / np.dot(r_d, r_d))

    weeks = pd.to_datetime(te_r1["transaction_date"]).dt.to_period("W").astype(str).to_numpy()
    unique_weeks = np.unique(weeks)
    week_idx = [np.where(weeks == w)[0] for w in unique_weeks]
    rng = np.random.default_rng(RANDOM_STATE)
    boot = np.zeros(n_boot, dtype=float)
    for k in range(n_boot):
        chosen = rng.integers(0, len(week_idx), size=len(week_idx))
        idx = np.concatenate([week_idx[w] for w in chosen])
        boot[k] = np.dot(r_d[idx], r_s[idx]) / np.dot(r_d[idx], r_d[idx])

    return {
        "n_test_r1": int(len(te_r1)),
        "n_test_weeks": int(len(unique_weeks)),
        "slope": slope,
        "slope_per_10pct": slope * 0.10,
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
    }


def evaluate_rule_t2_shape(
    tr_r1: pd.DataFrame,
    unconstrained_model: Any,
    frozen_model: Any,
    features_3b: List[str],
    disc_slope: float,
    disc_intercept: float,
    n_boot: int = 200,
) -> Dict[str, Any]:
    """
    Compares partial dependence of constrained (frozen) vs unconstrained LightGBM on
    training R1 rows across supported depths, and applies Pre-Registered Rule T2.
    """
    grid = np.asarray(R1_DEPTH_GRID, dtype=float)
    scaled_grid = scale_discount_values(grid, disc_slope, disc_intercept)
    supp_df, _ = discount_support(tr_r1, grid=R1_DEPTH_GRID)
    supp_mat = supp_df.to_numpy(dtype=bool)

    X_base = tr_r1[features_3b].copy()
    pd_unc = np.zeros((len(tr_r1), len(grid)), dtype=float)
    pd_con = np.zeros((len(tr_r1), len(grid)), dtype=float)
    for j, d_sc in enumerate(scaled_grid):
        X_base["discount_pct"] = d_sc
        pd_unc[:, j] = np.clip(unconstrained_model.predict(X_base), 0.0, 1.0)
        pd_con[:, j] = np.clip(frozen_model.predict(X_base), 0.0, 1.0)

    # Slopes between consecutive supported grid points
    ds_unc = np.diff(pd_unc, axis=1) / np.diff(grid)[None, :]
    pair_supp = supp_mat[:, :-1] & supp_mat[:, 1:]
    mid_d = 0.5 * (grid[:-1] + grid[1:])

    ref = _build_region_support_reference()
    r1_edges = np.asarray(ref["r1_exp_tercile_edges"], dtype=float)
    exp_terc = np.digitize(tr_r1["raw_expiry_remaining_ratio"].to_numpy(dtype=float), r1_edges[1:-1], right=True) + 1
    base_terc = pd.qcut(pd_con[:, 0], q=3, labels=[1, 2, 3]).to_numpy(dtype=int)

    # Rule T2 cells: (exp_terc x base_terc x depth_band) with >= 200 training rows
    bands = [("10-30%", 0.10, 0.30), ("30-50%", 0.30, 0.50), ("50-75%", 0.50, 0.76)]
    obs_d = tr_r1["raw_discount_pct"].to_numpy(dtype=float)
    rng = np.random.default_rng(RANDOM_STATE)

    t2_cells: List[Dict[str, Any]] = []
    for et in [1, 2, 3]:
        for bt in [1, 2, 3]:
            row_m = (exp_terc == et) & (base_terc == bt)
            for b_lbl, lo, hi in bands:
                bm = (mid_d >= lo) & (mid_d < hi)
                n_obs_cell = int((row_m & (obs_d >= lo) & (obs_d < hi)).sum())
                if n_obs_cell < 200:
                    continue
                # Average slope per row across supported pairs in this band
                sub_ds = ds_unc[row_m][:, bm]
                sub_sp = pair_supp[row_m][:, bm]
                valid_rows = sub_sp.any(axis=1)
                if valid_rows.sum() == 0:
                    continue
                row_slopes = np.array([
                    sub_ds[r, sub_sp[r]].mean() for r in range(len(sub_ds)) if sub_sp[r].any()
                ])
                mean_sl = float(row_slopes.mean())
                boots = np.array([
                    rng.choice(row_slopes, size=len(row_slopes), replace=True).mean()
                    for _ in range(n_boot)
                ])
                ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
                t2_cells.append({
                    "exp_tercile": et,
                    "base_s_tercile": bt,
                    "depth_band": b_lbl,
                    "n_train_rows": n_obs_cell,
                    "mean_unc_slope": mean_sl,
                    "ci95_low": ci_lo,
                    "ci95_high": ci_hi,
                    "positive_slope": bool(mean_sl > 0.0),
                    "sig_negative": bool(ci_hi < 0.0),
                })

    pos_share = float(np.mean([c["positive_slope"] for c in t2_cells])) if t2_cells else 0.0
    any_sig_neg = any(c["sig_negative"] for c in t2_cells)
    rule_t2_passed = bool((pos_share >= 0.80) and (not any_sig_neg))

    # Group profit curves (for Step 4d) across expiry terciles & baseline terciles
    base_p = tr_r1["raw_base_price"].to_numpy(dtype=float)
    cost_p = tr_r1["raw_cost_price"].to_numpy(dtype=float)
    sp_grid = np.round(base_p[:, None] * (1.0 - grid[None, :]), 2)
    prof_con = pd_con * sp_grid - cost_p[:, None]
    prof_unc = pd_unc * sp_grid - cost_p[:, None]

    group_shapes = []
    for g_name, g_arr in [("Exp_Tercile", exp_terc), ("Base_s_Tercile", base_terc)]:
        for val in [1, 2, 3]:
            gm = g_arr == val
            d_list, pc_list, pu_list, sc_list, su_list = [], [], [], [], []
            for j, d in enumerate(grid):
                sm = gm & supp_mat[:, j]
                if sm.sum() >= 30:
                    d_list.append(float(d))
                    pc_list.append(float(prof_con[sm, j].mean()))
                    pu_list.append(float(prof_unc[sm, j].mean()))
                    sc_list.append(float(pd_con[sm, j].mean()))
                    su_list.append(float(pd_unc[sm, j].mean()))
            best_i = int(np.argmax(pc_list))
            if best_i == 0:
                shape_lbl = "highest_at_shallowest (0.10)"
            elif best_i == len(d_list) - 1:
                shape_lbl = f"highest_at_deepest ({d_list[-1]:.2f})"
            else:
                shape_lbl = f"interior_max (peaks at {d_list[best_i]:.2f})"
            group_shapes.append({
                "group": f"{g_name}_{val}",
                "n_rows": int(gm.sum()),
                "best_depth_constrained": d_list[best_i],
                "shape_constrained": shape_lbl,
                "depths": d_list,
                "profit_constrained": pc_list,
                "profit_unconstrained": pu_list,
                "s_constrained": sc_list,
                "s_unconstrained": su_list,
            })

    return {
        "n_t2_cells_ge_200": len(t2_cells),
        "positive_slope_share": pos_share,
        "any_sig_negative": any_sig_neg,
        "rule_t2_passed": rule_t2_passed,
        "t2_cells": t2_cells,
        "group_shapes": group_shapes,
    }
