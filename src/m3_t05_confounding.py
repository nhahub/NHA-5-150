"""
src/m3_t05_confounding.py - Track 3B Confounding Diagnostics & Discount Support.

Provides:
  - discount_support(df, grid=...) using pre-registered Rule S (25 strata = oof_proba
    quintile x expiry_remaining_ratio quintile, >= 30 training rows per stratum x 5-pt bin),
    plus `no_undiscounted_history` flag for Track 3C.
  - Propensity diagnostics (time-blocked 5-fold CV classifier, discounted-only R^2,
    depth-3 decision tree rules).
  - Fixed diagnostic LightGBM cross_entropy models (NAIVE, CONTROLLED, and 6 ablations)
    with out-of-fold counterfactual grid scoring.
  - Pre-registered Rule U, Rule M, and Rule D evaluators.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

from src.config import CLEANED_DATASET_PATH, MODEL_READY_3B_DATASET_PATH, MODELS_DIR, RANDOM_STATE
from src.m2_feature_pipeline import TRACK_3B_FEATURES
from src.m3_oof import generate_oof_predictions
from src.m3_t04_scenario import DISCOUNT_BIN_LABELS, assign_discount_bin

DEFAULT_DISCOUNT_GRID: Tuple[float, ...] = (
    0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75
)
SUPPORT_MIN_ROWS: int = 30

ABLATION_GROUPS: Dict[str, List[str]] = {
    "drop_oof_proba": ["oof_proba"],
    "drop_expiry": ["days_until_expiry", "expiry_remaining_ratio"],
    "drop_quality": ["quality_grade_ord"],
    "drop_velocity": ["sales_velocity", "sell_through_risk", "sell_through_ratio", "at_risk"],
    "drop_promoted": ["is_promoted"],
    "drop_store_md": ["store_md_freq_hist"],
}


def load_3b_train_only() -> pd.DataFrame:
    """Loads training partition (split == 'train', N = 80,042) sorted chronologically."""
    df = pd.read_csv(MODEL_READY_3B_DATASET_PATH)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    tr = df[df["split"] == "train"].sort_values(["transaction_date", "record_id"]).reset_index(drop=True)
    assert len(tr) == 80042, f"Expected 80,042 train rows, got {len(tr)}"
    return tr


def fit_discount_scaler_params(tr: pd.DataFrame) -> Tuple[float, float]:
    """Recovers exact log1p -> standardized discount_pct linear transform from training rows."""
    pairs = tr[["raw_discount_pct", "discount_pct"]].drop_duplicates()
    slope, intercept = np.polyfit(np.log1p(pairs["raw_discount_pct"].values), pairs["discount_pct"].values, 1)
    return float(slope), float(intercept)


def scale_discount_values(d_raw: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Maps raw discount values in [0, 0.75] to the standardized discount_pct scale."""
    return slope * np.log1p(d_raw) + intercept


def _assign_quintile_from_edges(vals: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assigns 1..5 quintile indices using training edges (clamping out-of-range values)."""
    # edges has length 6: [min, q20, q40, q60, q80, max]
    inner = edges[1:-1]
    return np.digitize(vals, inner, right=True) + 1


@lru_cache(maxsize=1)
def _build_support_reference() -> Dict[str, Any]:
    """Computes training-only 5x5 quintile edges and (stratum, bin) support counts."""
    tr = load_3b_train_only()
    oof_edges = np.quantile(tr["oof_proba"].to_numpy(dtype=float), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    exp_edges = np.quantile(tr["expiry_remaining_ratio"].to_numpy(dtype=float), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    oof_q = _assign_quintile_from_edges(tr["oof_proba"].to_numpy(dtype=float), oof_edges)
    exp_q = _assign_quintile_from_edges(tr["expiry_remaining_ratio"].to_numpy(dtype=float), exp_edges)
    bins = assign_discount_bin(tr["raw_discount_pct"]).astype(str)

    strata_df = pd.DataFrame({"oof_q": oof_q, "exp_q": exp_q, "bin": bins})
    counts = strata_df.groupby(["oof_q", "exp_q", "bin"], observed=True).size().to_dict()

    return {
        "oof_edges": oof_edges,
        "exp_edges": exp_edges,
        "counts": counts,
    }


def discount_support_coarse(
    df: pd.DataFrame,
    grid: Sequence[float] = DEFAULT_DISCOUNT_GRID,
    min_rows: int = SUPPORT_MIN_ROWS,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    SUPERSEDED by region-based `discount_support()`.
    Kept for historical reference: coarse 5x5 quintile strata crossed the
    deterministic R0/R1 assignment boundary (expiry_remaining_ratio <= 0.40 or grade == 'C').
    """
    for req in ("oof_proba", "expiry_remaining_ratio"):
        if req not in df.columns:
            raise KeyError(f"Missing required column '{req}' for discount_support().")

    ref = _build_support_reference()
    oof_q = _assign_quintile_from_edges(df["oof_proba"].to_numpy(dtype=float), ref["oof_edges"])
    exp_q = _assign_quintile_from_edges(df["expiry_remaining_ratio"].to_numpy(dtype=float), ref["exp_edges"])
    counts = ref["counts"]

    grid_bins = [str(assign_discount_bin(pd.Series([d]))[0]) for d in grid]
    mat = np.zeros((len(df), len(grid)), dtype=bool)
    no_zero = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        oq, eq = int(oof_q[i]), int(exp_q[i])
        no_zero[i] = counts.get((oq, eq, "0%"), 0) < min_rows
        for j, b in enumerate(grid_bins):
            mat[i, j] = counts.get((oq, eq, b), 0) >= min_rows

    col_names = [f"d_{d:.2f}" for d in grid]
    supported_df = pd.DataFrame(mat, index=df.index, columns=col_names)
    no_zero_s = pd.Series(no_zero, index=df.index, name="no_undiscounted_history")
    return supported_df, no_zero_s


def run_propensity_diagnostics(tr: pd.DataFrame, n_blocks: int = 5) -> Dict[str, Any]:
    """Runs propensity classifier CV, discounted-only discount_pct R^2, and depth-3 tree."""
    controls = [f for f in TRACK_3B_FEATURES if f != "discount_pct"] + ["oof_proba"]
    X = tr[controls]
    y_treat = (tr["raw_discount_pct"] > 0).astype(int).to_numpy()
    dates = tr["transaction_date"].to_numpy()
    rec_ids = tr["record_id"].to_numpy()

    def clf_factory() -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
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

    prop_oof = generate_oof_predictions(
        model_factory=clf_factory,
        X=X,
        y=y_treat,
        dates=dates,
        record_ids=rec_ids,
        n_blocks=n_blocks,
        scheme="blocked",
        feature_set_name="track_3b_controls",
        model_type="lightgbm_propensity",
    )
    p_hat = prop_oof["oof_proba"].to_numpy(dtype=float)
    prop_auc = float(roc_auc_score(y_treat, p_hat))
    prop_brier = float(brier_score_loss(y_treat, p_hat))
    extreme_low = float((p_hat < 0.02).mean())
    extreme_high = float((p_hat > 0.98).mean())
    extreme_total = float(((p_hat < 0.02) | (p_hat > 0.98)).mean())

    # 2b: Among discounted rows, OOF R^2 for raw_discount_pct
    disc_mask = y_treat == 1
    tr_disc = tr.loc[disc_mask].reset_index(drop=True)
    unique_dates = np.sort(np.unique(pd.to_datetime(tr_disc["transaction_date"]).values))
    date_blocks = np.array_split(unique_dates, n_blocks)
    disc_dates = pd.to_datetime(tr_disc["transaction_date"]).values
    y_disc = tr_disc["raw_discount_pct"].to_numpy(dtype=float)
    X_disc = tr_disc[controls]
    disc_oof_pred = np.zeros(len(tr_disc), dtype=float)

    for b in range(n_blocks):
        val_m = np.isin(disc_dates, date_blocks[b])
        reg = lgb.LGBMRegressor(
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
        reg.fit(X_disc.loc[~val_m], y_disc[~val_m])
        disc_oof_pred[val_m] = reg.predict(X_disc.loc[val_m])

    disc_r2 = float(r2_score(y_disc, disc_oof_pred))
    disc_rmse = float(np.sqrt(mean_squared_error(y_disc, disc_oof_pred)))

    # 2c: Depth-3 decision tree for discount > 0
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, random_state=RANDOM_STATE)
    tree.fit(X, y_treat)
    tree_acc = float(tree.score(X, y_treat))
    tree_rules = export_text(tree, feature_names=controls, decimals=4)

    return {
        "propensity_auc": prop_auc,
        "propensity_brier": prop_brier,
        "extreme_lt_002_share": extreme_low,
        "extreme_gt_098_share": extreme_high,
        "extreme_total_share": extreme_total,
        "discounted_only_n": int(len(tr_disc)),
        "discounted_only_oof_r2": disc_r2,
        "discounted_only_oof_rmse": disc_rmse,
        "depth3_tree_accuracy": tree_acc,
        "depth3_tree_rules": tree_rules,
        "propensity_oof": p_hat,
        "y_treat": y_treat,
    }


def fit_cross_entropy_oof_and_grid(
    tr: pd.DataFrame,
    feature_cols: List[str],
    grid: Sequence[float] = DEFAULT_DISCOUNT_GRID,
    n_blocks: int = 5,
    compute_grid: bool = True,
) -> Dict[str, Any]:
    """
    Fits fixed LightGBM (objective='cross_entropy') on s in [0, 1] across 5 date-blocked
    folds, returning OOF metrics and (optionally) OOF counterfactual grid predictions.
    """
    dates = pd.to_datetime(tr["transaction_date"]).values
    unique_dates = np.sort(np.unique(dates))
    date_blocks = np.array_split(unique_dates, n_blocks)

    X = tr[feature_cols].copy()
    y_s = tr["s"].to_numpy(dtype=float)
    oof_s = np.zeros(len(tr), dtype=float)

    slope, intercept = fit_discount_scaler_params(tr)
    scaled_grid = scale_discount_values(np.asarray(grid, dtype=float), slope, intercept)
    grid_preds = np.zeros((len(tr), len(grid)), dtype=float) if compute_grid else None

    for b in range(n_blocks):
        val_m = np.isin(dates, date_blocks[b])
        model = lgb.LGBMRegressor(
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
        model.fit(X.loc[~val_m], y_s[~val_m])
        oof_s[val_m] = np.clip(model.predict(X.loc[val_m]), 0.0, 1.0)

        if compute_grid and grid_preds is not None:
            X_val = X.loc[val_m].copy()
            for j, d_scaled in enumerate(scaled_grid):
                X_val["discount_pct"] = d_scaled
                grid_preds[val_m, j] = np.clip(model.predict(X_val), 0.0, 1.0)

    metrics = {
        "oof_mae": float(mean_absolute_error(y_s, oof_s)),
        "oof_rmse": float(np.sqrt(mean_squared_error(y_s, oof_s))),
        "oof_r2": float(r2_score(y_s, oof_s)),
    }
    return {
        "metrics": metrics,
        "oof_s": oof_s,
        "grid_s": grid_preds,
    }


def evaluate_pre_registered_rules(
    tr: pd.DataFrame,
    grid_s: np.ndarray,
    grid: Sequence[float] = DEFAULT_DISCOUNT_GRID,
    extreme_propensity_share: float = 0.0,
) -> Dict[str, Any]:
    """
    Applies pre-registered Rules S, U, M, and D mechanically across the 25 strata.
    """
    ref = _build_support_reference()
    oof_q = _assign_quintile_from_edges(tr["oof_proba"].to_numpy(dtype=float), ref["oof_edges"])
    exp_q = _assign_quintile_from_edges(tr["expiry_remaining_ratio"].to_numpy(dtype=float), ref["exp_edges"])
    supported_df, _ = discount_support_coarse(tr, grid=grid, min_rows=SUPPORT_MIN_ROWS)
    supp_mat = supported_df.to_numpy(dtype=bool)

    base_p = tr["raw_base_price"].to_numpy(dtype=float)
    cost_p = tr["raw_cost_price"].to_numpy(dtype=float)
    grid_arr = np.asarray(grid, dtype=float)

    # Profit per initial unit at each grid discount d
    sell_prices = np.round(base_p[:, None] * (1.0 - grid_arr[None, :]), 2)
    grid_profit_unit = grid_s * sell_prices - cost_p[:, None]

    n_strata = 25
    harmful_count = 0
    monotone_demand_count = 0
    evaluable_strata = 0
    strata_rows: List[Dict[str, Any]] = []

    for oq in range(1, 6):
        for eq in range(1, 6):
            s_mask = (oof_q == oq) & (exp_q == eq)
            # A grid point j is supported in this stratum if all rows in this stratum have supp_mat[row, j] == True
            supp_j = supp_mat[s_mask][0]  # identical within stratum
            supp_idx = np.where(supp_j)[0]

            mean_s_vec = grid_s[s_mask].mean(axis=0)
            mean_prof_vec = grid_profit_unit[s_mask].mean(axis=0)

            is_harmful = False
            is_monotone_s = False
            avg_slope_s = None
            if len(supp_idx) >= 2:
                evaluable_strata += 1
                prof_supp = mean_prof_vec[supp_idx]
                s_supp = mean_s_vec[supp_idx]
                d_supp = grid_arr[supp_idx]

                # Rule U: profit per unit falls monotonically from lowest to highest supported discount
                is_harmful = bool(np.all(np.diff(prof_supp) <= 1e-9))
                if is_harmful:
                    harmful_count += 1

                # Rule M: average slope of s vs discount across supported cells >= 0
                avg_slope_s = float(np.polyfit(d_supp, s_supp, 1)[0])
                is_monotone_s = bool(avg_slope_s >= 0.0)
                if is_monotone_s:
                    monotone_demand_count += 1

            strata_rows.append({
                "oof_q": oq,
                "exp_q": eq,
                "n_rows": int(s_mask.sum()),
                "n_supported_grid_pts": int(len(supp_idx)),
                "supported_mask": supp_j.tolist(),
                "mean_pred_s": mean_s_vec.tolist(),
                "mean_pred_profit_unit": mean_prof_vec.tolist(),
                "rule_u_harmful": is_harmful,
                "rule_m_nonneg_slope": is_monotone_s,
                "avg_slope_s": avg_slope_s,
            })

    harmful_share = harmful_count / evaluable_strata if evaluable_strata > 0 else 0.0
    monotone_share = monotone_demand_count / evaluable_strata if evaluable_strata > 0 else 0.0
    rule_u_triggered = bool(harmful_share >= 0.80)
    rule_m_passed = bool(monotone_share >= 0.80)
    overlap_adequate = bool(extreme_propensity_share < 0.05)
    rule_d_recommend_dml = bool((rule_u_triggered or (not rule_m_passed)) and overlap_adequate)

    return {
        "evaluable_strata": evaluable_strata,
        "harmful_strata": harmful_count,
        "harmful_share": float(harmful_share),
        "rule_u_triggered": rule_u_triggered,
        "monotone_demand_strata": monotone_demand_count,
        "monotone_demand_share": float(monotone_share),
        "rule_m_passed": rule_m_passed,
        "overlap_adequate_lt_5pct": overlap_adequate,
        "rule_d_recommend_dml": rule_d_recommend_dml,
        "strata_details": strata_rows,
    }


REGION_SUPPORT_REF_PATH: Path = MODELS_DIR / "track_3b_region_support_ref.json"


@lru_cache(maxsize=1)
def _build_region_support_reference() -> Dict[str, Any]:
    """
    Computes and caches region-based support reference from training rows only:
      - R0: raw expiry_remaining_ratio > 0.40 AND quality_grade != 'C' (never discounted)
      - R1: raw expiry_remaining_ratio <= 0.40 OR quality_grade == 'C' (always discounted)
      - R1 strata: quality_grade == 'C' (bool) x expiry_remaining_ratio tercile within R1
    Saves edges and counts to models/track_3b_region_support_ref.json.
    """
    tr = load_3b_train_only()
    cl = pd.read_csv(
        CLEANED_DATASET_PATH,
        usecols=["record_id", "expiry_remaining_ratio", "quality_grade"],
    ).rename(columns={"expiry_remaining_ratio": "raw_expiry_remaining_ratio"})
    tr = tr.merge(cl, on="record_id", how="inner")

    # Fit exact linear mapping between raw_expiry_remaining_ratio and scaled expiry_remaining_ratio
    exp_slope, exp_intercept = np.polyfit(
        tr["raw_expiry_remaining_ratio"].to_numpy(dtype=float),
        tr["expiry_remaining_ratio"].to_numpy(dtype=float),
        1,
    )
    grade_c_ord_cutoff = float(tr.loc[tr["quality_grade"] == "C", "quality_grade_ord"].max()) + 1e-6

    r1_mask = (tr["raw_expiry_remaining_ratio"] <= 0.40) | (tr["quality_grade"] == "C")
    tr_r1 = tr.loc[r1_mask].copy()
    r1_exp_edges = np.quantile(
        tr_r1["raw_expiry_remaining_ratio"].to_numpy(dtype=float), [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
    )
    tr_r1["r1_exp_tercile"] = np.digitize(
        tr_r1["raw_expiry_remaining_ratio"].to_numpy(dtype=float), r1_exp_edges[1:-1], right=True
    ) + 1
    tr_r1["is_grade_c"] = (tr_r1["quality_grade"] == "C").astype(int)
    tr_r1["depth_bin"] = assign_discount_bin(tr_r1["raw_discount_pct"]).astype(str)

    raw_counts = (
        tr_r1.groupby(["is_grade_c", "r1_exp_tercile", "depth_bin"], observed=True)
        .size()
        .to_dict()
    )
    counts_str = {f"{k[0]}|{k[1]}|{k[2]}": int(v) for k, v in raw_counts.items()}

    ref = {
        "exp_scale_slope": float(exp_slope),
        "exp_scale_intercept": float(exp_intercept),
        "grade_c_ord_cutoff": float(grade_c_ord_cutoff),
        "r1_exp_tercile_edges": [float(x) for x in r1_exp_edges],
        "counts": counts_str,
    }
    REGION_SUPPORT_REF_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGION_SUPPORT_REF_PATH.write_text(json.dumps(ref, indent=2))
    return ref


def discount_support(
    df: pd.DataFrame,
    grid: Sequence[float] = DEFAULT_DISCOUNT_GRID,
    min_rows: int = SUPPORT_MIN_ROWS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Region-based discount support (M3-T05 addendum):
      - R0 (raw expiry_remaining_ratio > 0.40 and grade != 'C'):
          0% -> supported=True, reason='ok'
          d > 0 -> supported=False, reason='no_discount_region'
      - R1 (raw expiry_remaining_ratio <= 0.40 or grade == 'C'):
          0% -> supported=False, reason='no_zero_history'
          d > 0 -> supported=True ('ok') iff training R1 has >= min_rows in that
                   5-pt depth bin within (is_grade_c, r1_exp_tercile), else 'thin_depth_bin'.

    Accepts either raw ('raw_expiry_remaining_ratio', 'quality_grade') or scaled
    ('expiry_remaining_ratio', 'quality_grade_ord') columns.
    """
    ref = _build_region_support_reference()
    if "raw_expiry_remaining_ratio" in df.columns:
        raw_exp = df["raw_expiry_remaining_ratio"].to_numpy(dtype=float)
    elif "expiry_remaining_ratio" in df.columns:
        raw_exp = (df["expiry_remaining_ratio"].to_numpy(dtype=float) - ref["exp_scale_intercept"]) / ref["exp_scale_slope"]
    else:
        raise KeyError("Need 'raw_expiry_remaining_ratio' or 'expiry_remaining_ratio'.")

    if "quality_grade" in df.columns:
        is_c = (df["quality_grade"].astype(str) == "C").to_numpy(dtype=int)
    elif "quality_grade_ord" in df.columns:
        is_c = (df["quality_grade_ord"].to_numpy(dtype=float) <= ref["grade_c_ord_cutoff"]).astype(int)
    else:
        raise KeyError("Need 'quality_grade' or 'quality_grade_ord'.")

    is_r1 = (raw_exp <= 0.40 + 1e-9) | (is_c == 1)
    r1_edges = np.asarray(ref["r1_exp_tercile_edges"], dtype=float)
    r1_terc = np.digitize(raw_exp, r1_edges[1:-1], right=True) + 1
    counts = ref["counts"]

    grid_bins = [str(assign_discount_bin(pd.Series([d]))[0]) for d in grid]
    supp_mat = np.zeros((len(df), len(grid)), dtype=bool)
    reason_mat = np.empty((len(df), len(grid)), dtype=object)

    for i in range(len(df)):
        if not is_r1[i]:
            for j, d in enumerate(grid):
                if d == 0.0:
                    supp_mat[i, j] = True
                    reason_mat[i, j] = "ok"
                else:
                    supp_mat[i, j] = False
                    reason_mat[i, j] = "no_discount_region"
        else:
            c_flag = int(is_c[i])
            t_idx = int(r1_terc[i])
            for j, (d, b) in enumerate(zip(grid, grid_bins)):
                if d == 0.0:
                    supp_mat[i, j] = False
                    reason_mat[i, j] = "no_zero_history"
                else:
                    cnt = counts.get(f"{c_flag}|{t_idx}|{b}", 0)
                    if cnt >= min_rows:
                        supp_mat[i, j] = True
                        reason_mat[i, j] = "ok"
                    else:
                        supp_mat[i, j] = False
                        reason_mat[i, j] = "thin_depth_bin"

    cols = [f"d_{d:.2f}" for d in grid]
    return (
        pd.DataFrame(supp_mat, index=df.index, columns=cols),
        pd.DataFrame(reason_mat, index=df.index, columns=cols),
    )


def _lgbm_fixed_reg() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
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


def _lgbm_fixed_ce() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
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


def run_r1_depth_diagnostics(tr_r1: pd.DataFrame, n_blocks: int = 5) -> Dict[str, Any]:
    """
    Fits depth ~ controls within R1 (5-fold time-blocked CV) and tests whether
    OOF residual depth is predictable from store/supplier/product/calendar/handling/risk.
    Evaluates Pre-Registered Rule E (SD >= 3.0 pts AND residual OOF R^2 < 0.05).
    """
    controls = [f for f in TRACK_3B_FEATURES if f != "discount_pct"] + ["oof_proba"]
    dates = pd.to_datetime(tr_r1["transaction_date"]).values
    unique_dates = np.sort(np.unique(dates))
    date_blocks = np.array_split(unique_dates, n_blocks)
    y_depth = tr_r1["raw_discount_pct"].to_numpy(dtype=float)

    oof_depth_pred = np.zeros(len(tr_r1), dtype=float)
    X_ctrl = tr_r1[controls]
    for b in range(n_blocks):
        val_m = np.isin(dates, date_blocks[b])
        reg = _lgbm_fixed_reg()
        reg.fit(X_ctrl.loc[~val_m], y_depth[~val_m])
        oof_depth_pred[val_m] = reg.predict(X_ctrl.loc[val_m])

    resid = y_depth - oof_depth_pred
    depth_oof_r2 = float(r2_score(y_depth, oof_depth_pred))
    resid_sd_pts = float(np.std(resid) * 100.0)

    # Test predictability of residual depth from store, supplier, product, month, dow, handling, packaging, temp_abuse, daily_demand, oof_proba
    resid_feats_cat = ["store_id", "supplier_id", "product_name"]
    resid_feats_num = ["month", "dow", "handling_score", "packaging_score", "temp_abuse_events", "daily_demand", "oof_proba"]
    X_res = tr_r1[resid_feats_cat + resid_feats_num].copy()
    for c in resid_feats_cat:
        X_res[c] = X_res[c].astype("category")

    oof_resid_pred = np.zeros(len(tr_r1), dtype=float)
    for b in range(n_blocks):
        val_m = np.isin(dates, date_blocks[b])
        reg_r = _lgbm_fixed_reg()
        reg_r.fit(X_res.loc[~val_m], resid[~val_m])
        oof_resid_pred[val_m] = reg_r.predict(X_res.loc[val_m])

    resid_oof_r2 = float(r2_score(resid, oof_resid_pred))
    corrs = {c: float(tr_r1[c].corr(pd.Series(resid, index=tr_r1.index))) for c in resid_feats_num}

    rule_e_sd_ok = bool(resid_sd_pts >= 3.0)
    rule_e_r2_ok = bool(resid_oof_r2 < 0.05)
    rule_e_passed = bool(rule_e_sd_ok and rule_e_r2_ok)

    return {
        "n_r1": int(len(tr_r1)),
        "depth_oof_r2": depth_oof_r2,
        "residual_sd_pts": resid_sd_pts,
        "residual_oof_r2": resid_oof_r2,
        "residual_numeric_corrs": corrs,
        "rule_e_sd_ge_3pts": rule_e_sd_ok,
        "rule_e_resid_r2_lt_005": rule_e_r2_ok,
        "rule_e_passed": rule_e_passed,
        "resid_depth": resid,
        "date_blocks": date_blocks,
    }


def run_r1_local_models_and_partialling_out(
    tr_r1: pd.DataFrame,
    resid_depth: np.ndarray,
    n_blocks: int = 5,
    n_boot: int = 500,
) -> Dict[str, Any]:
    """
    Runs NAIVE and CONTROLLED within R1 only, plus cross-fitted partialling-out
    slope (residual s ~ residual depth) with week-grouped bootstrap 95% CI,
    control-group ablation sensitivity, and depth-band slopes.
    """
    dates = pd.to_datetime(tr_r1["transaction_date"]).values
    unique_dates = np.sort(np.unique(dates))
    date_blocks = np.array_split(unique_dates, n_blocks)

    cat_cols = [c for c in TRACK_3B_FEATURES if c.startswith("category_")]
    naive_cols = ["discount_pct"] + cat_cols
    ctrl_cols = list(TRACK_3B_FEATURES) + ["oof_proba"]
    controls_only = [f for f in TRACK_3B_FEATURES if f != "discount_pct"] + ["oof_proba"]

    r1_grid = [d for d in DEFAULT_DISCOUNT_GRID if d > 0.0]
    naive_res = fit_cross_entropy_oof_and_grid(tr_r1, naive_cols, grid=r1_grid, compute_grid=True)
    ctrl_res = fit_cross_entropy_oof_and_grid(tr_r1, ctrl_cols, grid=r1_grid, compute_grid=True)

    # Cross-fitted partialling-out for s ~ controls (excluding discount_pct)
    y_s = tr_r1["s"].to_numpy(dtype=float)
    X_ctrl = tr_r1[controls_only]
    oof_s_ctrl = np.zeros(len(tr_r1), dtype=float)
    for b in range(n_blocks):
        val_m = np.isin(dates, date_blocks[b])
        m = _lgbm_fixed_ce()
        m.fit(X_ctrl.loc[~val_m], y_s[~val_m])
        oof_s_ctrl[val_m] = np.clip(m.predict(X_ctrl.loc[val_m]), 0.0, 1.0)

    r_s = y_s - oof_s_ctrl
    r_d = resid_depth
    slope = float(np.dot(r_d, r_s) / np.dot(r_d, r_d))

    # Week-grouped bootstrap 95% CI (500 resamples)
    weeks = pd.to_datetime(tr_r1["transaction_date"]).dt.to_period("W").astype(str).to_numpy()
    unique_weeks = np.unique(weeks)
    week_indices = [np.where(weeks == w)[0] for w in unique_weeks]
    rng = np.random.default_rng(RANDOM_STATE)
    boot_slopes = np.zeros(n_boot, dtype=float)
    for k in range(n_boot):
        chosen = rng.integers(0, len(week_indices), size=len(week_indices))
        idx = np.concatenate([week_indices[w] for w in chosen])
        rd_b = r_d[idx]
        rs_b = r_s[idx]
        boot_slopes[k] = np.dot(rd_b, rs_b) / np.dot(rd_b, rd_b)

    ci_low, ci_high = float(np.percentile(boot_slopes, 2.5)), float(np.percentile(boot_slopes, 97.5))

    # Ablation sensitivity of partialling-out slope
    abl_slopes = []
    y_depth = tr_r1["raw_discount_pct"].to_numpy(dtype=float)
    for abl_name, drop_cols in ABLATION_GROUPS.items():
        sub_ctrl = [c for c in controls_only if c not in drop_cols]
        X_sub = tr_r1[sub_ctrl]
        s_hat = np.zeros(len(tr_r1), dtype=float)
        d_hat = np.zeros(len(tr_r1), dtype=float)
        for b in range(n_blocks):
            val_m = np.isin(dates, date_blocks[b])
            ms = _lgbm_fixed_ce()
            ms.fit(X_sub.loc[~val_m], y_s[~val_m])
            s_hat[val_m] = np.clip(ms.predict(X_sub.loc[val_m]), 0.0, 1.0)
            md = _lgbm_fixed_reg()
            md.fit(X_sub.loc[~val_m], y_depth[~val_m])
            d_hat[val_m] = md.predict(X_sub.loc[val_m])
        rs_a = y_s - s_hat
        rd_a = y_depth - d_hat
        sl_a = float(np.dot(rd_a, rs_a) / np.dot(rd_a, rd_a))
        abl_slopes.append({"ablation": abl_name, "slope": sl_a, "shift_vs_full": sl_a - slope})

    # Slope by depth band
    band_slopes = []
    for label, lo, hi in [("10-30%", 0.10, 0.30), ("30-50%", 0.30, 0.50), ("50-75%", 0.50, 0.751)]:
        m_band = (y_depth >= lo) & (y_depth < hi)
        if m_band.sum() >= 100:
            sl_b = float(np.dot(r_d[m_band], r_s[m_band]) / np.dot(r_d[m_band], r_d[m_band]))
            band_slopes.append({"band": label, "n": int(m_band.sum()), "slope": sl_b})

    return {
        "r1_naive_metrics": naive_res["metrics"],
        "r1_controlled_metrics": ctrl_res["metrics"],
        "r1_naive_grid_s": naive_res["grid_s"],
        "r1_controlled_grid_s": ctrl_res["grid_s"],
        "r1_grid": r1_grid,
        "partialling_out_slope": slope,
        "slope_per_10pct_pts": slope * 0.10,
        "boot_ci95_low": ci_low,
        "boot_ci95_high": ci_high,
        "ablation_slopes": abl_slopes,
        "band_slopes": band_slopes,
    }

