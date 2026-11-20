"""
src/m3_t08_elasticity.py - Track 3B Per-Category Demand-Response (Elasticity) View.

Extracts category-level counterfactual sell-through, derived units_sold, arc price
elasticity, and per-band slopes strictly from the frozen Track 3B regressor
(models/track_3b_regressor.joblib) on TRAINING R1 rows (N = 24,475).
"""

from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

from src.config import CLEANED_DATASET_PATH, RANDOM_STATE, TABLES_DIR, TRACK_3B_MODEL_PATH
from src.m3_t04_scenario import DISCOUNT_BIN_LABELS, assign_discount_bin
from src.m3_t05_confounding import scale_discount_values
from src.m3_t06_regressor import R1_DEPTH_GRID, load_r1_train_data
from src.mlflow_utils import load_model_with_metadata

RULE_C1_MIN_CAT_ROWS: int = 300
RULE_C1_MIN_BANDS: int = 3
RULE_C2_MIN_CELL_ROWS: int = 30


def _load_r1_with_sens() -> pd.DataFrame:
    r1, _, _, _, _ = load_r1_train_data()
    cl = pd.read_csv(
        CLEANED_DATASET_PATH,
        usecols=["record_id", "spoilage_sensitivity"],
    ).rename(columns={"spoilage_sensitivity": "raw_spoilage_sensitivity"})
    return r1.merge(cl, on="record_id", how="inner")


def category_elasticity_table(
    categories: Optional[Sequence[str]] = None,
    save_csv: bool = False,
) -> pd.DataFrame:
    """
    Scores every training R1 row in each category at each depth d in {0.10..0.75}
    using the frozen Track 3B regressor (no refit). Applies Pre-Registered Rules C1 & C2.
    """
    model, meta = load_model_with_metadata(TRACK_3B_MODEL_PATH)
    features_3b: List[str] = list(meta["feature_list"])
    disc_slope = float(meta["disc_scale_slope"])
    disc_intercept = float(meta["disc_scale_intercept"])

    r1 = _load_r1_with_sens()
    r1["depth_bin"] = assign_discount_bin(r1["raw_discount_pct"]).astype(str)

    all_cats = sorted(r1["category"].unique().tolist())
    target_cats = list(categories) if categories is not None else all_cats

    grid = np.asarray(R1_DEPTH_GRID, dtype=float)
    scaled_grid = scale_discount_values(grid, disc_slope, disc_intercept)
    grid_bins = [str(assign_discount_bin(pd.Series([d]))[0]) for d in grid]

    rows_out: List[Dict[str, Any]] = []
    for cat in target_cats:
        sub = r1[r1["category"] == cat].reset_index(drop=True)
        n_cat = int(len(sub))
        bin_counts = sub["depth_bin"].value_counts().to_dict()
        n_supported_bins = int(sum(1 for b in DISCOUNT_BIN_LABELS[1:] if bin_counts.get(b, 0) >= RULE_C2_MIN_CELL_ROWS))
        cat_ok = (n_cat >= RULE_C1_MIN_CAT_ROWS) and (n_supported_bins >= RULE_C1_MIN_BANDS)
        limited_depth_range = bool(n_supported_bins < 8)

        X_cat = sub[features_3b].copy()
        init_q = sub["raw_initial_quantity"].to_numpy(dtype=float)
        base_p = sub["raw_base_price"].to_numpy(dtype=float)
        cost_p = sub["raw_cost_price"].to_numpy(dtype=float)

        cat_rows: List[Dict[str, Any]] = []
        for j, (d, d_sc, b_lbl) in enumerate(zip(grid, scaled_grid, grid_bins)):
            X_cat["discount_pct"] = d_sc
            s_hat = np.clip(model.predict(X_cat), 0.0, 1.0)
            units_hat = s_hat * init_q
            sp = np.round(base_p * (1.0 - float(d)), 2)
            prof_u = s_hat * sp - cost_p

            n_bin = int(bin_counts.get(b_lbl, 0))
            if not cat_ok:
                conf = "low_confidence_category"
                supp = False
            elif n_bin < RULE_C2_MIN_CELL_ROWS:
                conf = "thin_depth_band"
                supp = False
            else:
                conf = "ok"
                supp = True

            cat_rows.append({
                "category": cat,
                "discount_pct": float(d),
                "depth_bin": b_lbl,
                "rel_price": float(1.0 - d),
                "n_cat_r1": n_cat,
                "n_supported_bins_cat": n_supported_bins,
                "n_obs_in_bin": n_bin,
                "mean_s_hat": float(s_hat.mean()),
                "mean_units_sold": float(units_hat.mean()),
                "mean_profit_per_unit": float(prof_u.mean()),
                "arc_price_elasticity": np.nan,
                "supported": bool(supp),
                "confidence": conf,
                "limited_depth_range": limited_depth_range,
            })

        # Compute arc elasticity between adjacent supported depth points
        for j in range(1, len(cat_rows)):
            prev_r, cur_r = cat_rows[j - 1], cat_rows[j]
            if prev_r["supported"] and cur_r["supported"]:
                q1, q2 = prev_r["mean_units_sold"], cur_r["mean_units_sold"]
                p1, p2 = prev_r["rel_price"], cur_r["rel_price"]
                q_mid = 0.5 * (q1 + q2)
                p_mid = 0.5 * (p1 + p2)
                if q_mid > 0 and (p2 - p1) != 0:
                    arc_e = ((q2 - q1) / q_mid) / ((p2 - p1) / p_mid)
                    cur_r["arc_price_elasticity"] = float(arc_e)

        rows_out.extend(cat_rows)

    out_df = pd.DataFrame(rows_out)
    if save_csv:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(TABLES_DIR / "M3_T08_category_elasticity.csv", index=False)
    return out_df


def compute_category_band_slopes(elast_df: pd.DataFrame) -> pd.DataFrame:
    """Computes average slope of s_hat per +10 discount points across supported pairs."""
    bands = [("overall", 0.10, 0.76), ("10-30%", 0.10, 0.30), ("30-50%", 0.30, 0.50), ("50-75%", 0.50, 0.76)]
    rows = []
    for cat, sub in elast_df.groupby("category", sort=True):
        sub = sub.sort_values("discount_pct").reset_index(drop=True)
        d = sub["discount_pct"].to_numpy(dtype=float)
        s = sub["mean_s_hat"].to_numpy(dtype=float)
        supp = sub["supported"].to_numpy(dtype=bool)
        pair_ok = supp[:-1] & supp[1:]
        slopes_10 = ((s[1:] - s[:-1]) / (d[1:] - d[:-1])) * 0.10
        mid_d = 0.5 * (d[:-1] + d[1:])

        row: Dict[str, Any] = {
            "category": cat,
            "n_cat_r1": int(sub["n_cat_r1"].iloc[0]),
            "n_supported_bins": int(sub["n_supported_bins_cat"].iloc[0]),
        }
        for b_lbl, lo, hi in bands:
            m = pair_ok & (mid_d >= lo) & (mid_d < hi)
            row[f"slope_per10_{b_lbl}"] = float(slopes_10[m].mean()) if m.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compute_model_free_category_slopes(n_blocks: int = 5, n_boot: int = 500) -> pd.DataFrame:
    """
    Cross-fitted 5-fold time-blocked residual-on-residual slope of s on raw_discount_pct
    per category (models fit on train R1 only, week-grouped bootstrap 95% CI).
    """
    r1 = _load_r1_with_sens()
    _, meta = load_model_with_metadata(TRACK_3B_MODEL_PATH)
    controls = [c for c in meta["feature_list"] if c != "discount_pct" and not c.startswith("category_")]

    rows = []
    for cat, sub in r1.groupby("category", sort=True):
        sub = sub.sort_values(["transaction_date", "record_id"]).reset_index(drop=True)
        dates = pd.to_datetime(sub["transaction_date"]).values
        unique_dates = np.sort(np.unique(dates))
        date_blocks = np.array_split(unique_dates, n_blocks)

        y_s = sub["s"].to_numpy(dtype=float)
        y_d = sub["raw_discount_pct"].to_numpy(dtype=float)
        X = sub[controls]
        s_hat = np.zeros(len(sub), dtype=float)
        d_hat = np.zeros(len(sub), dtype=float)

        for val_dates in date_blocks:
            vm = np.isin(dates, val_dates)
            ms = lgb.LGBMRegressor(
                objective="cross_entropy", num_leaves=15, learning_rate=0.05,
                n_estimators=200, min_child_samples=30, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, random_state=RANDOM_STATE,
                n_jobs=-1, verbose=-1,
            )
            md = lgb.LGBMRegressor(
                num_leaves=15, learning_rate=0.05, n_estimators=200,
                min_child_samples=30, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, random_state=RANDOM_STATE,
                n_jobs=-1, verbose=-1,
            )
            ms.fit(X.loc[~vm], y_s[~vm])
            md.fit(X.loc[~vm], y_d[~vm])
            s_hat[vm] = np.clip(ms.predict(X.loc[vm]), 0.0, 1.0)
            d_hat[vm] = md.predict(X.loc[vm])

        r_s = y_s - s_hat
        r_d = y_d - d_hat
        slope = float(np.dot(r_d, r_s) / np.dot(r_d, r_d))

        weeks = pd.to_datetime(sub["transaction_date"]).dt.to_period("W").astype(str).to_numpy()
        week_idx = [np.where(weeks == w)[0] for w in np.unique(weeks)]
        rng = np.random.default_rng(RANDOM_STATE)
        boots = np.zeros(n_boot, dtype=float)
        for k in range(n_boot):
            chosen = rng.integers(0, len(week_idx), size=len(week_idx))
            idx = np.concatenate([week_idx[w] for w in chosen])
            boots[k] = np.dot(r_d[idx], r_s[idx]) / np.dot(r_d[idx], r_d[idx])

        rows.append({
            "category": cat,
            "n_r1": int(len(sub)),
            "mf_slope_per10": slope * 0.10,
            "mf_ci95_low_per10": float(np.percentile(boots, 2.5)) * 0.10,
            "mf_ci95_high_per10": float(np.percentile(boots, 97.5)) * 0.10,
        })
    return pd.DataFrame(rows)


def compute_spoilage_sensitivity_check(elast_df: pd.DataFrame, slopes_df: pd.DataFrame) -> Dict[str, Any]:
    """Descriptive Spearman rank correlation across categories passing Rule C1."""
    r1 = _load_r1_with_sens()
    sens_map = r1.groupby("category")["raw_spoilage_sensitivity"].first().to_dict()

    # Baseline sell-through = mean_s_hat at shallowest supported depth per category
    supp_only = elast_df[elast_df["supported"]].sort_values(["category", "discount_pct"])
    base_s = supp_only.groupby("category")["mean_s_hat"].first().to_dict()

    rows = []
    for _, r in slopes_df.iterrows():
        cat = r["category"]
        if cat in base_s and not np.isnan(r["slope_per10_overall"]):
            rows.append({
                "category": cat,
                "spoilage_sensitivity": float(sens_map[cat]),
                "baseline_s_hat": float(base_s[cat]),
                "model_slope_per10": float(r["slope_per10_overall"]),
            })
    df_c = pd.DataFrame(rows)
    rho_base, _ = spearmanr(df_c["spoilage_sensitivity"], df_c["baseline_s_hat"])
    rho_slope, _ = spearmanr(df_c["spoilage_sensitivity"], df_c["model_slope_per10"])
    return {
        "n_categories": int(len(df_c)),
        "spearman_sens_vs_baseline_s": float(rho_base),
        "spearman_sens_vs_model_slope": float(rho_slope),
        "table": rows,
    }


def compute_category_censoring() -> pd.DataFrame:
    """Share of s == 1 by category and depth band (10-30%, 30-50%, 50-75%)."""
    r1 = _load_r1_with_sens()
    d = r1["raw_discount_pct"].to_numpy(dtype=float)
    r1["band"] = np.where(d < 0.30, "10-30%", np.where(d < 0.50, "30-50%", "50-75%"))
    r1["is_s1"] = (r1["s"] == 1.0).astype(float) * 100.0
    piv = r1.pivot_table(index="category", columns="band", values="is_s1", aggfunc="mean").reset_index()
    return piv
