"""
src/m3_t04_scenario.py - Track 3B Scenario Dataset Construction & Overlap Diagnostics.

Builds data/processed/model_ready_3b_dataset.csv (100,000 rows) combining:
  - record_id, transaction_date, split
  - TRACK_3B_FEATURES (43 scaled/encoded features from model_ready_m2_dataset.csv)
  - oof_proba (v2 OOF for train rows; frozen-model test proba for test rows)
  - Primary target s = units_sold / raw_initial_quantity in [0, 1], plus secondary
    observed outcomes (waste_pct, revenue, profit)
  - Unscaled auxiliary columns (raw_base_price, raw_cost_price, raw_initial_quantity,
    raw_selling_price, raw_discount_pct, units_sold, units_wasted, category) for
    the exact profit identity and Track 3C backtesting (NOT model features).

Provides integrity checks and training-only selection/overlap diagnostics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import (
    CLEANED_DATASET_PATH,
    MODEL_READY_3B_DATASET_PATH,
    MODEL_READY_M2_DATASET_PATH,
    TRACK_3A_OOF_TRAIN_V2_PATH,
    TRACK_3A_PROBA_TEST_PATH,
)
from src.m2_feature_pipeline import TRACK_3B_FEATURES, validate_feature_columns_exist
from src.mlflow_utils import validate_feature_leakage

# 5-point discount bins: {0}, [0.10, 0.15), ..., [0.65, 0.70), [0.70, 0.75]
DISCOUNT_BIN_LABELS: List[str] = [
    "0%",
    "10-15%",
    "15-20%",
    "20-25%",
    "25-30%",
    "30-35%",
    "35-40%",
    "40-45%",
    "45-50%",
    "50-55%",
    "55-60%",
    "60-65%",
    "65-70%",
    "70-75%",
]


def assign_discount_bin(discount_pct: pd.Series) -> pd.Categorical:
    """Assigns raw discount_pct in {0} U [0.10, 0.75] into 5-point bins."""
    vals = discount_pct.to_numpy(dtype=float)
    out = np.full(len(vals), "", dtype=object)
    out[vals == 0.0] = "0%"
    edges = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        label = DISCOUNT_BIN_LABELS[i + 1]
        if i < len(edges) - 2:
            mask = (vals >= lo - 1e-12) & (vals < hi - 1e-12)
        else:
            mask = (vals >= lo - 1e-12) & (vals <= hi + 1e-12)
        out[mask] = label
    return pd.Categorical(out, categories=DISCOUNT_BIN_LABELS, ordered=True)


def build_3b_dataset(
    save: bool = True,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Constructs the Track 3B scenario dataset (100,000 rows) without training any model.
    Explicitly leaves NaN-velocity rows as NaN (for LightGBM native missing handling
    or explicit indicator ablation in M3-T05).
    """
    m2_df = pd.read_csv(MODEL_READY_M2_DATASET_PATH)
    validate_feature_columns_exist(TRACK_3B_FEATURES, m2_df.columns)

    cl_cols = [
        "record_id",
        "transaction_date",
        "category",
        "base_price",
        "cost_price",
        "initial_quantity",
        "selling_price",
        "discount_pct",
        "units_sold",
        "units_wasted",
        "waste_pct",
        "revenue",
        "profit",
    ]
    cl_df = pd.read_csv(CLEANED_DATASET_PATH, usecols=cl_cols)
    cl_df["transaction_date"] = pd.to_datetime(cl_df["transaction_date"])
    cl_df = cl_df.rename(columns={
        "base_price": "raw_base_price",
        "cost_price": "raw_cost_price",
        "initial_quantity": "raw_initial_quantity",
        "selling_price": "raw_selling_price",
        "discount_pct": "raw_discount_pct",
    })

    oof_v2 = pd.read_csv(TRACK_3A_OOF_TRAIN_V2_PATH, usecols=["record_id", "oof_proba"])
    proba_test = pd.read_csv(TRACK_3A_PROBA_TEST_PATH, usecols=["record_id", "proba"])
    proba_test = proba_test.rename(columns={"proba": "oof_proba"})
    risk_all = pd.concat([oof_v2, proba_test], ignore_index=True)

    # Assemble base DataFrame
    base_cols = ["record_id", "split"] + list(TRACK_3B_FEATURES)
    df = m2_df[base_cols].merge(cl_df, on="record_id", how="inner")
    df = df.merge(risk_all, on="record_id", how="inner")

    # Primary bounded sell-through target s in [0, 1]
    df["s"] = df["units_sold"] / df["raw_initial_quantity"]

    # Sort chronologically by transaction_date, record_id
    df = df.sort_values(["transaction_date", "record_id"]).reset_index(drop=True)

    # Reorder columns cleanly
    ordered_cols = (
        ["record_id", "transaction_date", "split"]
        + list(TRACK_3B_FEATURES)
        + ["oof_proba", "s", "waste_pct", "revenue", "profit"]
        + [
            "raw_base_price",
            "raw_cost_price",
            "raw_initial_quantity",
            "raw_selling_price",
            "raw_discount_pct",
            "units_sold",
            "units_wasted",
            "category",
        ]
    )
    df = df[ordered_cols]

    if save:
        out_file = Path(output_path) if output_path is not None else MODEL_READY_3B_DATASET_PATH
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)

    return df


def run_integrity_checks(df: pd.DataFrame) -> Dict[str, Any]:
    """Runs and asserts all M3-T04 integrity checks on the 3B dataset."""
    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"
    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())

    assert n_train == 80042, f"Expected 80,042 train rows, got {n_train}"
    assert n_test == 19958, f"Expected 19,958 test rows, got {n_test}"
    assert df["record_id"].is_unique, "record_id is not unique!"
    assert not df.loc[train_mask, "oof_proba"].isna().any(), "Train rows missing oof_proba!"
    assert not df.loc[test_mask, "oof_proba"].isna().any(), "Test rows missing oof_proba!"

    # Target & accounting identities
    assert (df["s"] >= 0.0).all() and (df["s"] <= 1.0).all(), "s outside [0, 1]!"
    inv_diff = (df["units_sold"] + df["units_wasted"] - df["raw_initial_quantity"]).abs().max()
    assert inv_diff == 0, f"Inventory identity violated: max diff={inv_diff}"

    recon_profit = df["raw_initial_quantity"] * (df["s"] * df["raw_selling_price"] - df["raw_cost_price"])
    profit_max_err = float((df["profit"] - recon_profit).abs().max())
    assert profit_max_err < 1e-4, f"Profit identity error too large: {profit_max_err}"

    # Leakage guard check (TRACK_3B_FEATURES + oof_proba)
    validate_feature_leakage("3B", list(TRACK_3B_FEATURES) + ["oof_proba"], "expiry_remaining_ratio")

    # NaN counts in features (explicitly report the 576 NaN-velocity rows)
    nan_counts = df[list(TRACK_3B_FEATURES) + ["oof_proba"]].isna().sum()
    nan_dict = {k: int(v) for k, v in nan_counts[nan_counts > 0].items()}

    # Distribution of s
    s_vals = df["s"].to_numpy(dtype=float)
    s_train = df.loc[train_mask, "s"].to_numpy(dtype=float)
    s_dist = {
        "overall_mean": float(np.mean(s_vals)),
        "overall_std": float(np.std(s_vals)),
        "share_exactly_0": float((s_vals == 0.0).mean()),
        "share_exactly_1": float((s_vals == 1.0).mean()),
        "quantiles": {
            "min": float(np.min(s_vals)),
            "p05": float(np.percentile(s_vals, 5)),
            "p25": float(np.percentile(s_vals, 25)),
            "p50": float(np.percentile(s_vals, 50)),
            "p75": float(np.percentile(s_vals, 75)),
            "p95": float(np.percentile(s_vals, 95)),
            "max": float(np.max(s_vals)),
        },
        "train_mean": float(np.mean(s_train)),
    }

    return {
        "n_total": int(len(df)),
        "n_train": n_train,
        "n_test": n_test,
        "record_id_unique": True,
        "train_oof_missing": 0,
        "test_oof_missing": 0,
        "inventory_max_diff": int(inv_diff),
        "profit_identity_max_err": profit_max_err,
        "leakage_guard_passed": True,
        "feature_nan_counts": nan_dict,
        "s_distribution": s_dist,
    }


def compute_selection_and_overlap_diagnostics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes selection and overlap diagnostics strictly on the training partition
    (transaction_date <= 2024-08-07, N = 80,042).
    """
    tr = df[df["split"] == "train"].copy()
    tr["discount_bin"] = assign_discount_bin(tr["raw_discount_pct"])
    tr["is_discounted"] = (tr["raw_discount_pct"] > 0).astype(int)
    tr["profit_per_unit"] = tr["profit"] / tr["raw_initial_quantity"]
    tr["is_loss"] = (tr["profit"] < 0).astype(int)

    # 1. Discount-bin support & descriptive outcome table
    bin_rows: List[Dict[str, Any]] = []
    n_tr = len(tr)
    for b in DISCOUNT_BIN_LABELS:
        sub = tr[tr["discount_bin"] == b]
        cnt = int(len(sub))
        bin_rows.append({
            "discount_bin": b,
            "n_rows": cnt,
            "share_pct": float(cnt / n_tr * 100.0),
            "thin_support": bool(cnt < 300),
            "mean_s": float(sub["s"].mean()) if cnt > 0 else None,
            "mean_profit": float(sub["profit"].mean()) if cnt > 0 else None,
            "mean_profit_per_unit": float(sub["profit_per_unit"].mean()) if cnt > 0 else None,
            "loss_share_pct": float(sub["is_loss"].mean() * 100.0) if cnt > 0 else None,
        })

    # 2. Spearman correlations with raw_discount_pct
    corr_cols = [
        "oof_proba",
        "expiry_remaining_ratio",
        "store_md_freq_hist",
        "sales_velocity",
        "quality_grade_ord",
        "is_promoted",
    ]
    spearman_rows: List[Dict[str, Any]] = []
    for col in corr_cols:
        valid = tr[["raw_discount_pct", col]].dropna()
        rho, pval = spearmanr(valid["raw_discount_pct"], valid[col])
        spearman_rows.append({
            "feature": col,
            "n_valid": int(len(valid)),
            "spearman_rho": float(rho),
            "p_value": float(pval),
        })

    # Discount rate by oof_proba decile & expiry_remaining_ratio decile
    tr["oof_decile"] = pd.qcut(tr["oof_proba"], q=10, labels=False) + 1
    tr["exp_decile"] = pd.qcut(tr["expiry_remaining_ratio"].rank(method="first"), q=10, labels=False) + 1

    oof_decile_stats = (
        tr.groupby("oof_decile", observed=True)
        .agg(
            n_rows=("record_id", "count"),
            mean_oof=("oof_proba", "mean"),
            discount_rate=("is_discounted", "mean"),
            mean_discount_all=("raw_discount_pct", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    exp_decile_stats = (
        tr.groupby("exp_decile", observed=True)
        .agg(
            n_rows=("record_id", "count"),
            mean_exp_ratio=("expiry_remaining_ratio", "mean"),
            discount_rate=("is_discounted", "mean"),
            mean_discount_all=("raw_discount_pct", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    # 3. Overlap across strata = category x expiry_remaining_ratio tercile x oof_proba tercile
    tr["oof_tercile"] = pd.qcut(tr["oof_proba"], q=3, labels=["LowRisk", "MidRisk", "HighRisk"])
    tr["exp_tercile"] = pd.qcut(
        tr["expiry_remaining_ratio"].rank(method="first"), q=3, labels=["ShortExp", "MidExp", "LongExp"]
    )
    strata_stats = (
        tr.groupby(["category", "exp_tercile", "oof_tercile"], observed=True)
        .agg(
            n_rows=("record_id", "count"),
            n_undiscounted=("is_discounted", lambda s: int((s == 0).sum())),
            n_discounted=("is_discounted", lambda s: int((s == 1).sum())),
            n_distinct_bins=("discount_bin", "nunique"),
        )
        .reset_index()
    )
    strata_stats["has_both"] = (strata_stats["n_undiscounted"] > 0) & (strata_stats["n_discounted"] > 0)
    strata_stats["has_3plus_bins"] = strata_stats["n_distinct_bins"] >= 3

    n_strata = int(len(strata_stats))
    share_strata_both = float(strata_stats["has_both"].mean())
    share_rows_both = float(strata_stats.loc[strata_stats["has_both"], "n_rows"].sum() / n_tr)
    share_strata_3plus = float(strata_stats["has_3plus_bins"].mean())
    share_rows_3plus = float(strata_stats.loc[strata_stats["has_3plus_bins"], "n_rows"].sum() / n_tr)

    # 4. Promoted vs not
    promo_stats = (
        tr.groupby("is_promoted", observed=True)
        .agg(
            n_rows=("record_id", "count"),
            discount_rate=("is_discounted", "mean"),
            mean_discount_all=("raw_discount_pct", "mean"),
            mean_discount_when_md=("raw_discount_pct", lambda s: float(s[s > 0].mean()) if (s > 0).any() else 0.0),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    # 5. Raw association: mean s and mean profit by discount bin within oof_proba terciles
    tercile_bin_df = (
        tr.groupby(["oof_tercile", "discount_bin"], observed=True)
        .agg(
            n_rows=("record_id", "count"),
            mean_s=("s", "mean"),
            mean_profit=("profit", "mean"),
        )
        .reset_index()
    )

    return {
        "discount_bins": bin_rows,
        "spearman_with_discount": spearman_rows,
        "discount_by_oof_decile": oof_decile_stats,
        "discount_by_exp_decile": exp_decile_stats,
        "overlap": {
            "n_strata": n_strata,
            "share_strata_both": share_strata_both,
            "share_rows_both": share_rows_both,
            "share_strata_3plus_bins": share_strata_3plus,
            "share_rows_3plus_bins": share_rows_3plus,
            "min_rows_per_stratum": int(strata_stats["n_rows"].min()),
            "median_rows_per_stratum": float(strata_stats["n_rows"].median()),
        },
        "promoted_vs_not": promo_stats,
        "raw_by_tercile_and_bin": tercile_bin_df.to_dict(orient="records"),
    }
