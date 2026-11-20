"""
src/m3_t06_regressor.py - Track 3B Sell-Through Regressor, CV Tuning & Profit Curve Helper.

Implements M3-T06:
  - Design decisions D1-D3 (R1 domain, sell-through target s in [0,1], TRACK_3B_FEATURES + oof_proba).
  - Extreme-outcome diagnostics on R1 training rows.
  - 5-fold time-blocked CV candidates (GroupMeanBaseline, LightGBM cross_entropy,
    LightGBM cross_entropy monotone-constrained on discount_pct, XGBoost reg:logistic,
    RandomForestRegressor) with Optuna 40-trial tuning.
  - predict_profit_curve(batch_df, grid) using frozen model + M3-T05 discount_support().
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.config import (
    CLEANED_DATASET_PATH,
    MODEL_READY_3B_DATASET_PATH,
    RANDOM_STATE,
    TRACK_3B_MODEL_PATH,
)
from src.m2_feature_pipeline import TRACK_3B_FEATURES
from src.m3_t04_scenario import DISCOUNT_BIN_LABELS, assign_discount_bin
from src.m3_t05_confounding import (
    REGION_SUPPORT_REF_PATH,
    discount_support,
    fit_discount_scaler_params,
    scale_discount_values,
)
from src.mlflow_utils import load_model_with_metadata

optuna.logging.set_verbosity(optuna.logging.WARNING)

R1_DEPTH_GRID: Tuple[float, ...] = (
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75
)


def load_r1_train_data() -> Tuple[pd.DataFrame, List[str], List[str], float, float]:
    """
    Loads R1 training rows (split == 'train' and raw_discount_pct > 0, N = 24,475),
    drops any constant columns inside R1, and returns:
      (r1_df, features_3b, dropped_constant_cols, disc_slope, disc_intercept)
    """
    df = pd.read_csv(MODEL_READY_3B_DATASET_PATH)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    tr = df[df["split"] == "train"].copy()
    disc_slope, disc_intercept = fit_discount_scaler_params(tr)

    cl_extra = pd.read_csv(
        CLEANED_DATASET_PATH,
        usecols=["record_id", "expiry_remaining_ratio", "quality_grade"],
    ).rename(columns={"expiry_remaining_ratio": "raw_expiry_remaining_ratio"})

    r1 = (
        tr[tr["raw_discount_pct"] > 0]
        .merge(cl_extra, on="record_id", how="inner")
        .sort_values(["transaction_date", "record_id"])
        .reset_index(drop=True)
    )
    assert len(r1) == 24475, f"Expected 24,475 R1 train rows, got {len(r1)}"

    all_feats = list(TRACK_3B_FEATURES) + ["oof_proba"]
    const_cols = [c for c in all_feats if r1[c].nunique(dropna=False) <= 1]
    features_3b = [c for c in all_feats if c not in const_cols]
    return r1, features_3b, const_cols, disc_slope, disc_intercept


def compute_extreme_outcome_stats(r1: pd.DataFrame) -> Dict[str, Any]:
    """Computes R1 profit-per-unit tail statistics and s=0 / s=1 shares by depth band."""
    p_unit = (r1["profit"] / r1["raw_initial_quantity"]).to_numpy(dtype=float)
    cost_u = r1["raw_cost_price"].to_numpy(dtype=float)
    profit_abs = r1["profit"].to_numpy(dtype=float)

    r1_tmp = r1.copy()
    r1_tmp["depth_bin"] = assign_discount_bin(r1_tmp["raw_discount_pct"])
    censor_rows = []
    for b in DISCOUNT_BIN_LABELS[1:]:  # skip 0%
        sub = r1_tmp[r1_tmp["depth_bin"] == b]
        censor_rows.append({
            "depth_bin": b,
            "n_rows": int(len(sub)),
            "s_eq_0_pct": float((sub["s"] == 0.0).mean() * 100.0) if len(sub) else 0.0,
            "s_eq_1_pct": float((sub["s"] == 1.0).mean() * 100.0) if len(sub) else 0.0,
        })

    return {
        "n_r1": int(len(r1)),
        "profit_per_unit_quantiles": {
            "min": float(np.min(p_unit)),
            "p01": float(np.percentile(p_unit, 1)),
            "p05": float(np.percentile(p_unit, 5)),
            "p25": float(np.percentile(p_unit, 25)),
            "p50": float(np.percentile(p_unit, 50)),
            "p75": float(np.percentile(p_unit, 75)),
            "p95": float(np.percentile(p_unit, 95)),
            "p99": float(np.percentile(p_unit, 99)),
            "max": float(np.max(p_unit)),
        },
        "loss_share_pct": float((profit_abs < 0).mean() * 100.0),
        "below_minus_1x_cost_share_pct": float((p_unit < -cost_u).mean() * 100.0),
        "below_minus_05x_cost_share_pct": float((p_unit < -0.5 * cost_u).mean() * 100.0),
        "profit_p01": float(np.percentile(profit_abs, 1)),
        "profit_p99": float(np.percentile(profit_abs, 99)),
        "censoring_by_depth_bin": censor_rows,
    }


class GroupMeanBaseline:
    """Mean s by expiry_remaining_ratio tercile x grade_C x 10-point depth band."""

    def __init__(self) -> None:
        self.exp_edges_: Optional[np.ndarray] = None
        self.table_: Dict[Tuple[int, int, int], float] = {}
        self.fallback_: Dict[Tuple[int, int], float] = {}
        self.global_mean_: float = 0.8

    @staticmethod
    def _depth_band_10(d_scaled: np.ndarray, slope: float, intercept: float) -> np.ndarray:
        d_raw = np.expm1((d_scaled - intercept) / slope)
        return np.clip(np.floor(d_raw * 10.0).astype(int), 1, 7)

    def fit(self, X: pd.DataFrame, y: np.ndarray, disc_slope: float, disc_intercept: float, grade_c_cutoff: float) -> "GroupMeanBaseline":
        self.disc_slope_ = disc_slope
        self.disc_intercept_ = disc_intercept
        self.grade_c_cutoff_ = grade_c_cutoff
        exp_vals = X["expiry_remaining_ratio"].to_numpy(dtype=float)
        self.exp_edges_ = np.quantile(exp_vals, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        terc = np.digitize(exp_vals, self.exp_edges_[1:-1], right=True) + 1
        is_c = (X["quality_grade_ord"].to_numpy(dtype=float) <= grade_c_cutoff).astype(int)
        dband = self._depth_band_10(X["discount_pct"].to_numpy(dtype=float), disc_slope, disc_intercept)

        df_fit = pd.DataFrame({"terc": terc, "is_c": is_c, "dband": dband, "y": y})
        self.table_ = df_fit.groupby(["terc", "is_c", "dband"])["y"].mean().to_dict()
        self.fallback_ = df_fit.groupby(["is_c", "dband"])["y"].mean().to_dict()
        self.global_mean_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.exp_edges_ is not None
        exp_vals = X["expiry_remaining_ratio"].to_numpy(dtype=float)
        terc = np.digitize(exp_vals, self.exp_edges_[1:-1], right=True) + 1
        is_c = (X["quality_grade_ord"].to_numpy(dtype=float) <= self.grade_c_cutoff_).astype(int)
        dband = self._depth_band_10(X["discount_pct"].to_numpy(dtype=float), self.disc_slope_, self.disc_intercept_)
        out = np.empty(len(X), dtype=float)
        for i in range(len(X)):
            key = (int(terc[i]), int(is_c[i]), int(dband[i]))
            if key in self.table_:
                out[i] = self.table_[key]
            else:
                out[i] = self.fallback_.get((int(is_c[i]), int(dband[i])), self.global_mean_)
        return np.clip(out, 0.0, 1.0)


def make_3b_model(family: str, params: Dict[str, Any], features_3b: List[str]) -> Any:
    """Instantiates an un-fitted Track 3B model for the given family and hyperparameters."""
    fam = family.lower()
    if fam == "lgbm_ce":
        return lgb.LGBMRegressor(
            objective="cross_entropy",
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            min_child_samples=params["min_child_samples"],
            subsample=params["subsample"],
            subsample_freq=1,
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    if fam == "lgbm_ce_monotone":
        mono = [1 if c == "discount_pct" else 0 for c in features_3b]
        return lgb.LGBMRegressor(
            objective="cross_entropy",
            monotone_constraints=mono,
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            min_child_samples=params["min_child_samples"],
            subsample=params["subsample"],
            subsample_freq=1,
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    if fam == "xgboost_logistic":
        return xgb.XGBRegressor(
            objective="reg:logistic",
            eval_metric="rmse",
            tree_method="hist",
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            min_child_weight=params["min_child_weight"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
    if fam == "rf":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("reg", RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])
    raise ValueError(f"Unknown model family: {family}")


def _sample_3b_params(trial: optuna.Trial, family: str) -> Dict[str, Any]:
    fam = family.lower()
    if fam in ("lgbm_ce", "lgbm_ce_monotone"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
    if fam == "xgboost_logistic":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
    if fam == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 200, step=50),
            "max_depth": trial.suggest_int("max_depth", 8, 16, step=2),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 80),
            "max_features": trial.suggest_categorical("max_features", [0.3, 0.5, 0.7]),
        }
    raise ValueError(f"Unknown model family: {family}")


def evaluate_3b_oof(
    r1: pd.DataFrame,
    features_3b: List[str],
    model_builder: Callable[[], Any],
    date_blocks: List[np.ndarray],
    is_baseline: bool = False,
    disc_slope: float = 1.0,
    disc_intercept: float = 0.0,
    grade_c_cutoff: float = -1.0,
) -> Dict[str, Any]:
    """Evaluates a Track 3B candidate across 5 date-blocked folds inside R1."""
    dates = pd.to_datetime(r1["transaction_date"]).values
    X = r1[features_3b]
    y_s = r1["s"].to_numpy(dtype=float)
    oof_s = np.zeros(len(r1), dtype=float)
    fold_rmses: List[float] = []

    for b, val_dates in enumerate(date_blocks):
        val_m = np.isin(dates, val_dates)
        model = model_builder()
        if is_baseline:
            model.fit(X.loc[~val_m], y_s[~val_m], disc_slope, disc_intercept, grade_c_cutoff)
        else:
            model.fit(X.loc[~val_m], y_s[~val_m])
        preds = np.clip(model.predict(X.loc[val_m]), 0.0, 1.0)
        oof_s[val_m] = preds
        fold_rmses.append(float(np.sqrt(mean_squared_error(y_s[val_m], preds))))

    # Derived profit per initial unit
    sell_p = r1["raw_selling_price"].to_numpy(dtype=float)
    cost_p = r1["raw_cost_price"].to_numpy(dtype=float)
    init_q = r1["raw_initial_quantity"].to_numpy(dtype=float)
    obs_profit_u = (r1["profit"] / init_q).to_numpy(dtype=float)
    pred_profit_u = oof_s * sell_p - cost_p
    obs_profit = r1["profit"].to_numpy(dtype=float)
    pred_profit = pred_profit_u * init_q

    # With vs without most extreme 1% profit rows
    p01, p99 = np.percentile(obs_profit, [1, 99])
    non_extreme = (obs_profit >= p01) & (obs_profit <= p99)

    return {
        "s_mae": float(mean_absolute_error(y_s, oof_s)),
        "s_rmse": float(np.sqrt(mean_squared_error(y_s, oof_s))),
        "s_rmse_se": float(np.std(fold_rmses, ddof=1) / np.sqrt(len(fold_rmses))),
        "s_r2": float(r2_score(y_s, oof_s)),
        "profit_unit_mae": float(mean_absolute_error(obs_profit_u, pred_profit_u)),
        "profit_unit_rmse": float(np.sqrt(mean_squared_error(obs_profit_u, pred_profit_u))),
        "profit_mae_all": float(mean_absolute_error(obs_profit, pred_profit)),
        "profit_rmse_all": float(np.sqrt(mean_squared_error(obs_profit, pred_profit))),
        "profit_mae_no_ext1pct": float(mean_absolute_error(obs_profit[non_extreme], pred_profit[non_extreme])),
        "profit_rmse_no_ext1pct": float(np.sqrt(mean_squared_error(obs_profit[non_extreme], pred_profit[non_extreme]))),
        "oof_s": oof_s,
    }


def tune_3b_candidate(
    family: str,
    r1: pd.DataFrame,
    features_3b: List[str],
    date_blocks: List[np.ndarray],
    n_trials: int = 40,
) -> Dict[str, Any]:
    """Runs Optuna tuning (40 trials) minimizing 5-fold time-blocked CV RMSE on s."""
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    cache: Dict[int, Dict[str, Any]] = {}

    def objective(trial: optuna.Trial) -> float:
        params = _sample_3b_params(trial, family)
        res = evaluate_3b_oof(
            r1=r1,
            features_3b=features_3b,
            model_builder=lambda: make_3b_model(family, params, features_3b),
            date_blocks=date_blocks,
        )
        cache[trial.number] = {"params": params, "res": res}
        return res["s_rmse"]

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = cache[study.best_trial.number]
    return {"family": family, "params": best["params"], **best["res"]}


def predict_profit_curve(
    batch_df: pd.DataFrame,
    grid: Sequence[float] = R1_DEPTH_GRID,
    model_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Scores each batch across candidate discount depths in [0.10, 0.75] using the
    frozen Track 3B regressor and M3-T05 region-based discount_support().

    For R0 batches (expiry_remaining_ratio > 0.40 and grade != 'C'), returns
    supported=False, NaN predictions, and reason='region_R0_no_discount_evidence'.
    Never extrapolates below 0.10 or above 0.75.
    """
    if batch_df.empty:
        raise ValueError("Input batch_df is empty.")
    grid_arr = np.asarray(grid, dtype=float)
    if np.any(grid_arr < 0.10 - 1e-9) or np.any(grid_arr > 0.75 + 1e-9):
        raise ValueError("Grid values must stay strictly within [0.10, 0.75] (no extrapolation).")

    target_path = Path(model_path) if model_path is not None else TRACK_3B_MODEL_PATH
    model, metadata = load_model_with_metadata(target_path)
    feature_list: List[str] = list(metadata["feature_list"])

    missing = [c for c in feature_list if c not in batch_df.columns]
    if missing:
        raise KeyError(f"Missing required Track 3B features: {missing[:10]}")
    for req in ("raw_base_price", "raw_cost_price", "raw_initial_quantity"):
        if req not in batch_df.columns:
            raise KeyError(f"Missing required auxiliary column '{req}' for profit derivation.")

    ref = json.loads(REGION_SUPPORT_REF_PATH.read_text())
    disc_slope = float(metadata.get("disc_scale_slope", ref.get("disc_scale_slope")))
    disc_intercept = float(metadata.get("disc_scale_intercept", ref.get("disc_scale_intercept")))

    supp_df, reason_df = discount_support(batch_df, grid=grid)

    # Identify R0 rows directly from reason_df (any R0 row has 'no_discount_region' on d > 0)
    is_r0 = (reason_df.iloc[:, 0] == "no_discount_region").to_numpy()
    rec_ids = batch_df["record_id"].tolist() if "record_id" in batch_df.columns else list(range(len(batch_df)))

    rows_out: List[Dict[str, Any]] = []
    X_base = batch_df[feature_list].copy()

    # Pre-score R1 rows across grid
    r1_mask = ~is_r0
    s_hat_grid = np.full((len(batch_df), len(grid_arr)), np.nan, dtype=float)
    if r1_mask.any():
        X_r1 = X_base.loc[r1_mask].copy()
        scaled_grid = scale_discount_values(grid_arr, disc_slope, disc_intercept)
        for j, d_sc in enumerate(scaled_grid):
            X_r1["discount_pct"] = d_sc
            s_hat_grid[r1_mask, j] = np.clip(model.predict(X_r1), 0.0, 1.0)

    base_prices = batch_df["raw_base_price"].to_numpy(dtype=float)
    cost_prices = batch_df["raw_cost_price"].to_numpy(dtype=float)
    init_qtys = batch_df["raw_initial_quantity"].to_numpy(dtype=float)

    for i in range(len(batch_df)):
        rid = rec_ids[i]
        bp, cp, iq = float(base_prices[i]), float(cost_prices[i]), float(init_qtys[i])
        for j, d in enumerate(grid_arr):
            if is_r0[i]:
                rows_out.append({
                    "record_id": rid,
                    "discount_pct": float(d),
                    "s_hat": np.nan,
                    "expected_units_sold": np.nan,
                    "profit_per_unit": np.nan,
                    "profit": np.nan,
                    "supported": False,
                    "reason": "region_R0_no_discount_evidence",
                })
            else:
                s_hat = float(s_hat_grid[i, j])
                sp = round(bp * (1.0 - float(d)), 2)
                p_unit = s_hat * sp - cp
                rows_out.append({
                    "record_id": rid,
                    "discount_pct": float(d),
                    "s_hat": s_hat,
                    "expected_units_sold": s_hat * iq,
                    "profit_per_unit": p_unit,
                    "profit": p_unit * iq,
                    "supported": bool(supp_df.iloc[i, j]),
                    "reason": str(reason_df.iloc[i, j]),
                })

    return pd.DataFrame(rows_out)
