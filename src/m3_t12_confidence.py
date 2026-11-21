"""
src/m3_t12_confidence.py - Track 3C Confidence Propagation (Weakest-Link Rule).

Implements M3-T12 Pre-Registered Rules:
  - Levels: "No recommendation" (R0), "Low", "Moderate" (deliberately no "High").
  - Component flags per R1 batch:
      F3A  low_history_3a: product_name x store_id < 20 training rows (src/support.py).
      F3B1 limited_depth_category: category has < 8 of 13 depth bands with >= 30 R1 train rows.
      F3B2 support_edge: recommended_depth equals shallowest or deepest supported depth.
      F3B3 critical_tier: tier == "Critical".
      INFO near_tie: flat_curve (informational only, NOT used for confidence_level).
  - Weakest-link combination: Low if ANY of {F3A, F3B1, F3B2, F3B3} is True; else Moderate.
"""

from functools import lru_cache
import json
from typing import Any, Dict, List, Set
import numpy as np
import pandas as pd

from src.config import CLEANED_DATASET_PATH, MODEL_READY_3B_DATASET_PATH
from src.m3_t04_scenario import DISCOUNT_BIN_LABELS, assign_discount_bin
from src.m3_t05_confounding import discount_support
from src.m3_t06_regressor import R1_DEPTH_GRID
from src.m3_t10_tiers import assign_tier, load_tier_edges
from src.support import (
    DEFAULT_SUPPORT_KEYS,
    DEFAULT_SUPPORT_THRESHOLD,
    compute_support,
    flag_low_confidence,
)

GLOBAL_CAVEAT = "Advisory estimate. The observed-data backtest was inconclusive; see the validation report."

CONFIDENCE_WORDING: Dict[str, Dict[str, str]] = {
    "levels": {
        "No recommendation": "No markdown recommended: this batch is outside the range where the data supports a discount decision.",
        "Low": "Low confidence: one or more data-support limits apply to this recommendation; treat as a tentative estimate.",
        "Moderate": "Moderate confidence: this batch lies inside supported data ranges, but remains an advisory model estimate rather than a guarantee.",
    },
    "reasons": {
        "no_discount_region": "Batch is in region R0 where historical practice is no markdown.",
        "low_history_3a": "Limited history for this product at this store (fewer than 20 training rows).",
        "limited_depth_category": "Category has limited historical discount-depth coverage (fewer than 8 of 13 depth bands supported).",
        "support_edge": "Recommended depth sits on the shallowest or deepest supported depth boundary for this batch.",
        "critical_tier": "Critical tier has very thin historical support (fewer than 20 test batches).",
        "near_tie": "Predicted profit is nearly the same across depths.",
    },
    "global_caveat": {"text": GLOBAL_CAVEAT},
}


@lru_cache(maxsize=1)
def _get_3a_support_table() -> pd.DataFrame:
    """Computes product_name x store_id support table strictly on training rows."""
    cl = pd.read_csv(
        CLEANED_DATASET_PATH,
        usecols=["transaction_date", "product_name", "store_id"],
    )
    cl["transaction_date"] = pd.to_datetime(cl["transaction_date"])
    tr = cl[cl["transaction_date"] <= pd.Timestamp("2024-08-07")]
    return compute_support(tr, keys=DEFAULT_SUPPORT_KEYS)


@lru_cache(maxsize=1)
def _get_limited_depth_categories() -> Set[str]:
    """
    Computes in code (without reading reports/tables/*.csv) which categories have
    fewer than 8 of the 13 5-point depth bands with >= 30 training R1 rows.
    """
    df = pd.read_csv(
        MODEL_READY_3B_DATASET_PATH,
        usecols=["split", "raw_discount_pct", "category"],
    )
    r1 = df[(df["split"] == "train") & (df["raw_discount_pct"] > 0)].copy()
    r1["bin"] = assign_discount_bin(r1["raw_discount_pct"]).astype(str)
    limited: Set[str] = set()
    for cat, sub in r1.groupby("category"):
        bcnt = sub["bin"].value_counts().to_dict()
        n_ok = sum(1 for b in DISCOUNT_BIN_LABELS[1:] if bcnt.get(b, 0) >= 30)
        if n_ok < 8:
            limited.add(str(cat))
    return limited


@lru_cache(maxsize=1)
def _get_record_meta_lookup() -> pd.DataFrame:
    """Lookup for product_name and store_id by record_id when omitted from batch_df."""
    return pd.read_csv(CLEANED_DATASET_PATH, usecols=["record_id", "product_name", "store_id"])


def attach_confidence(recs_df: pd.DataFrame, batch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attaches confidence_level, confidence_reasons, component flags (low_history_3a,
    limited_depth_category, support_edge, critical_tier), and informational near_tie
    to optimizer recommendations. Works on single-row or multi-row DataFrames.
    """
    if len(recs_df) != len(batch_df):
        raise ValueError(f"Length mismatch: recs_df ({len(recs_df)}) vs batch_df ({len(batch_df)})")
    if recs_df.empty:
        raise ValueError("Input DataFrames are empty.")

    out = recs_df.reset_index(drop=True).copy()
    b_df = batch_df.reset_index(drop=True).copy()

    # Attach tier if missing
    if "tier" not in out.columns:
        t_df = assign_tier(out, load_tier_edges())
        out["tier"] = t_df["tier"].values
        out["tier_label"] = t_df["tier_label"].values

    # Ensure product_name and store_id are available for 3A support lookup
    if "product_name" not in b_df.columns or "store_id" not in b_df.columns:
        if "record_id" not in b_df.columns:
            raise KeyError("batch_df must contain ('product_name', 'store_id') or 'record_id'.")
        lkp = _get_record_meta_lookup()
        b_df = b_df.merge(lkp, on="record_id", how="left")

    # Component Flag F3A: product_name x store_id < 20 training rows
    supp_3a = _get_3a_support_table()
    f3a_series = flag_low_confidence(
        b_df, supp_3a, threshold=DEFAULT_SUPPORT_THRESHOLD, keys=DEFAULT_SUPPORT_KEYS
    ).to_numpy(dtype=bool)

    # Component Flag F3B1: category < 8/13 supported depth bands
    limited_cats = _get_limited_depth_categories()
    if "category" not in b_df.columns:
        raise KeyError("batch_df must contain 'category' for limited_depth_category flag.")
    f3b1_arr = b_df["category"].astype(str).isin(limited_cats).to_numpy(dtype=bool)

    # Component Flag F3B2: recommended_depth == shallowest or deepest supported depth
    grid_arr = np.asarray(R1_DEPTH_GRID, dtype=float)
    supp_df, _ = discount_support(b_df, grid=R1_DEPTH_GRID)
    supp_mat = supp_df.to_numpy(dtype=bool)

    # Informational near_tie: flat_curve
    near_tie_arr = out["flat_curve"].to_numpy(dtype=bool)

    levels: List[str] = []
    reasons_json: List[str] = []
    f3a_out: List[bool] = []
    f3b1_out: List[bool] = []
    f3b2_out: List[bool] = []
    f3b3_out: List[bool] = []

    for i in range(len(out)):
        region = str(out["region"].iloc[i])
        if region == "R0":
            levels.append("No recommendation")
            reasons_json.append(json.dumps(["no_discount_region"]))
            f3a_out.append(bool(f3a_series[i]))
            f3b1_out.append(False)
            f3b2_out.append(False)
            f3b3_out.append(False)
            continue

        rec_d = float(out["recommended_depth"].iloc[i])
        supp_d = grid_arr[supp_mat[i]]
        is_edge = bool(len(supp_d) > 0 and (np.isclose(rec_d, supp_d[0]) or np.isclose(rec_d, supp_d[-1])))
        is_crit = bool(out["tier"].iloc[i] == "Critical")
        is_f3a = bool(f3a_series[i])
        is_f3b1 = bool(f3b1_arr[i])

        r_codes: List[str] = []
        if is_f3a:
            r_codes.append("low_history_3a")
        if is_f3b1:
            r_codes.append("limited_depth_category")
        if is_edge:
            r_codes.append("support_edge")
        if is_crit:
            r_codes.append("critical_tier")

        lvl = "Low" if len(r_codes) > 0 else "Moderate"
        levels.append(lvl)
        reasons_json.append(json.dumps(r_codes))
        f3a_out.append(is_f3a)
        f3b1_out.append(is_f3b1)
        f3b2_out.append(is_edge)
        f3b3_out.append(is_crit)

    out["confidence_level"] = levels
    out["confidence_reasons"] = reasons_json
    out["low_history_3a"] = f3a_out
    out["limited_depth_category"] = f3b1_out
    out["support_edge"] = f3b2_out
    out["critical_tier"] = f3b3_out
    out["near_tie"] = near_tie_arr
    out["global_caveat"] = GLOBAL_CAVEAT
    return out
