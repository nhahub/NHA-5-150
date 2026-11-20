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

from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

from src.config import MODEL_READY_3B_DATASET_PATH, RANDOM_STATE
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


def discount_support(
    df: pd.DataFrame,
    grid: Sequence[float] = DEFAULT_DISCOUNT_GRID,
    min_rows: int = SUPPORT_MIN_ROWS,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies pre-registered Rule S using training-only quintile edges (oof_proba x
    expiry_remaining_ratio = 25 strata) and 5-point discount bins.

    Returns
    -------
    supported_df : pd.DataFrame (len(df) x len(grid))
        Boolean per row and per candidate discount d in `grid`.
    no_undiscounted_history : pd.Series (bool, len(df))
        True when the row's stratum has fewer than `min_rows` undiscounted (0%) rows.
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
    supported_df, _ = discount_support(tr, grid=grid, min_rows=SUPPORT_MIN_ROWS)
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
