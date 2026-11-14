"""
src/m3_t03_explain.py - Track 3A Per-Batch Exact Linear Explanations & Limited-History Flag.

Provides explain_batch(df, top_k=5) and explain_text(row_explanation) for the frozen
Track 3A regularized logistic regression model (TRACK_3A_FEATURES_COMPACT, 90 features),
using training-only reference statistics and src/support.py (product_name x store_id < 20).
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import (
    CLEANED_DATASET_PATH,
    MODEL_READY_M2_DATASET_PATH,
    TRACK_3A_MODEL_PATH,
)
from src.m3_t01_track3a import score_batches
from src.mlflow_utils import load_model_with_metadata
from src.support import (
    DEFAULT_SUPPORT_KEYS,
    DEFAULT_SUPPORT_THRESHOLD,
    compute_support,
    flag_low_confidence,
)

LIMITED_HISTORY_UI_TEXT = "Limited history for this product at this store"
SUFFICIENT_HISTORY_UI_TEXT = "Sufficient history for this product at this store"

PLAIN_NAME_MAP: Dict[str, str] = {
    "packaging_score": "packaging quality",
    "handling_score": "handling quality",
    "days_until_expiry": "days until expiry",
    "days_since_receipt": "days since receipt",
    "temp_deviation": "storage temperature deviation",
    "distribution_hours": "distribution transit time",
    "daily_demand": "expected daily demand",
    "base_price": "base price",
    "initial_quantity": "initial batch quantity",
    "temp_abuse_events": "temperature abuse events",
    "expiry_remaining_ratio": "remaining shelf-life ratio",
    "quality_grade_ord": "quality grade",
    "is_weekend": "weekend arrival",
    "category": "product category",
    "product_name": "product type",
    "seasonality_month": "month of year",
    "day_of_week": "day of week",
}


def _map_feature_to_group(feat_name: str) -> str:
    """Maps a model feature name (including one-hot/cyclical) to a grouped driver key."""
    if feat_name.startswith("category_"):
        return "category"
    if feat_name.startswith("product_name_"):
        return "product_name"
    if feat_name in ("month_sin", "month_cos"):
        return "seasonality_month"
    if feat_name in ("dow_sin", "dow_cos"):
        return "day_of_week"
    return feat_name


@lru_cache(maxsize=1)
def _load_reference_context() -> Dict[str, Any]:
    """
    Loads frozen Track 3A model and computes training-only reference statistics
    (feature means, base logit, base predicted probability, and product x store support).
    """
    model, metadata = load_model_with_metadata(TRACK_3A_MODEL_PATH)
    feature_list: List[str] = list(metadata["feature_list"])

    m2_df = pd.read_csv(MODEL_READY_M2_DATASET_PATH, usecols=["record_id", "split"] + feature_list)
    cl_df = pd.read_csv(CLEANED_DATASET_PATH, usecols=["record_id", "product_name", "store_id"])

    train_m2 = m2_df[m2_df["split"] == "train"].merge(cl_df, on="record_id", how="inner")
    assert len(train_m2) == 80042, f"Expected 80,042 training rows, got {len(train_m2)}"

    X_train = train_m2[feature_list]
    train_mean = X_train.mean(axis=0).values.astype(float)

    coef = np.asarray(model.coef_[0], dtype=float)
    intercept = float(model.intercept_[0])
    base_logit = float(intercept + np.dot(coef, train_mean))
    train_probas = np.asarray(model.predict_proba(X_train)[:, 1], dtype=float)
    base_rate = float(np.mean(train_probas))

    support_table = compute_support(train_m2, keys=DEFAULT_SUPPORT_KEYS)
    group_keys = [_map_feature_to_group(f) for f in feature_list]

    return {
        "feature_list": feature_list,
        "group_keys": group_keys,
        "coef": coef,
        "intercept": intercept,
        "train_mean": train_mean,
        "base_logit": base_logit,
        "base_rate": base_rate,
        "support_table": support_table,
    }


def explain_batch(df: pd.DataFrame, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Computes exact linear log-odds decompositions, grouped plain-language drivers,
    proportional probability-point contributions, and product x store support flags
    for each row in `df` (supports single-row simulator inputs and batch DataFrames).

    Probability-point allocation method:
      delta_p_total = (predicted_proba - base_rate) * 100
      delta_p_group = delta_p_total * (group_logit_contrib / total_logit_shift)
    so grouped probability-point contributions sum to (predicted_proba - base_rate) * 100.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty; expected at least 1 row.")

    ctx = _load_reference_context()
    feature_list: List[str] = ctx["feature_list"]
    group_keys: List[str] = ctx["group_keys"]
    coef: np.ndarray = ctx["coef"]
    intercept: float = ctx["intercept"]
    train_mean: np.ndarray = ctx["train_mean"]
    base_logit: float = ctx["base_logit"]
    base_rate: float = ctx["base_rate"]
    support_table: pd.DataFrame = ctx["support_table"]

    # Calibrated probabilities via frozen scorer (validates feature columns & no NaNs)
    probas = score_batches(df)

    # Support lookup on product_name x store_id (threshold 20)
    low_flags = flag_low_confidence(
        df, support_table, threshold=DEFAULT_SUPPORT_THRESHOLD, keys=DEFAULT_SUPPORT_KEYS
    )
    merged_counts = df[list(DEFAULT_SUPPORT_KEYS)].merge(
        support_table[list(DEFAULT_SUPPORT_KEYS) + ["cell_count"]],
        on=list(DEFAULT_SUPPORT_KEYS),
        how="left",
    )["cell_count"].fillna(0).astype(int).values

    X_mat = df[feature_list].to_numpy(dtype=float)
    # Exact linear decomposition: phi_j = coef_j * (x_j - E_train[x_j])
    contribs_mat = (X_mat - train_mean[None, :]) * coef[None, :]
    model_logits = intercept + np.dot(X_mat, coef)

    unique_groups = list(dict.fromkeys(group_keys))
    group_indices: Dict[str, List[int]] = {
        g: [i for i, gk in enumerate(group_keys) if gk == g] for g in unique_groups
    }

    has_record_id = "record_id" in df.columns
    record_ids = df["record_id"].tolist() if has_record_id else [None] * len(df)

    explanations: List[Dict[str, Any]] = []
    for idx in range(len(df)):
        p = float(probas[idx])
        m_logit = float(model_logits[idx])
        logit_diff = m_logit - base_logit
        row_contribs = contribs_mat[idx]
        sum_contribs = float(np.sum(row_contribs))

        # Group contributions
        grouped_logits: Dict[str, float] = {
            g: float(np.sum(row_contribs[idxs])) for g, idxs in group_indices.items()
        }

        # Proportional probability-point allocation (sums to (p - base_rate)*100)
        delta_p_pts = (p - base_rate) * 100.0
        if abs(logit_diff) > 1e-9:
            scale = delta_p_pts / logit_diff
        else:
            scale = p * (1.0 - p) * 100.0

        drivers: List[Dict[str, Any]] = []
        for g, g_logit in grouped_logits.items():
            pts = float(g_logit * scale)
            drivers.append({
                "driver": g,
                "label": PLAIN_NAME_MAP.get(g, g.replace("_", " ")),
                "direction": "raises risk" if g_logit >= 0 else "lowers risk",
                "log_odds_contrib": g_logit,
                "prob_pts_contrib": pts,
            })

        push_up = sorted(
            [d for d in drivers if d["log_odds_contrib"] > 0],
            key=lambda d: d["log_odds_contrib"],
            reverse=True,
        )[:top_k]
        push_down = sorted(
            [d for d in drivers if d["log_odds_contrib"] < 0],
            key=lambda d: d["log_odds_contrib"],
        )[:top_k]
        top_overall = sorted(
            drivers, key=lambda d: abs(d["log_odds_contrib"]), reverse=True
        )[:top_k]

        is_low = bool(low_flags.iloc[idx])
        hist_rows = int(merged_counts[idx])

        explanations.append({
            "record_id": record_ids[idx],
            "predicted_proba": p,
            "base_rate": base_rate,
            "model_logit": m_logit,
            "base_logit": base_logit,
            "logit_diff": float(logit_diff),
            "sum_contributions_logit": sum_contribs,
            "grouped_contributions": drivers,
            "top_push_up": push_up,
            "top_push_down": push_down,
            "top_drivers": top_overall,
            "low_history_flag": is_low,
            "history_rows": hist_rows,
            "confidence_note": LIMITED_HISTORY_UI_TEXT if is_low else SUFFICIENT_HISTORY_UI_TEXT,
        })

    return explanations


def explain_text(row_explanation: Dict[str, Any], top_n: int = 3) -> str:
    """
    Formats a concise, calibrated plain-English explanation for a single batch.
    Avoids overclaiming certainty (probabilities mostly span ~12%-28%, test AUC ~0.59).
    """
    p_pct = row_explanation["predicted_proba"] * 100.0
    base_pct = row_explanation["base_rate"] * 100.0
    drivers = row_explanation["top_drivers"][:top_n]

    driver_strs = []
    for d in drivers:
        pts = d["prob_pts_contrib"]
        sign = "+" if pts >= 0 else ""
        driver_strs.append(f"{d['label']} {d['direction']} ({sign}{pts:.1f} pts)")

    text = (
        f"Predicted spoilage risk {p_pct:.1f}% (typical batch: {base_pct:.1f}%). "
        f"Main drivers: {', '.join(driver_strs)}."
    )
    if row_explanation.get("low_history_flag"):
        hist_n = row_explanation.get("history_rows", 0)
        text += f" Note: {LIMITED_HISTORY_UI_TEXT} ({hist_n} training rows)."
    return text
