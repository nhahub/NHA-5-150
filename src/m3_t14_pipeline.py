"""
src/m3_t14_pipeline.py - Track 3 End-to-End Inference Pipeline & Inventory Snapshot Helper.

Implements M3-T14 Pre-Registered Design Decisions P1-P5:
  - P1: Input contract (5 required fields, 19 optional fields with training-only defaults,
        derived fields, and causal M2 history lookup in 'real' mode vs training lookup
        in 'hypothetical' mode).
  - P2: Training-only defaults artifact saved to models/track_3_inference_defaults.json
        (strictly from split == 'train', transaction_date <= 2024-08-07).
  - P3: Input validation + defaulted-input Low confidence rule:
        unknown category -> ValueError; negative/impossible values -> ValueError;
        unknown product/store -> category/store defaults + Low confidence ('unknown_entity');
        >50% of optional fields defaulted -> Low confidence ('defaults_dominate').
  - P4: Inventory snapshot on latest test date with >= 100 batches (2024-12-31, 145 rows).
  - P5: Global caveat attached to every output (GLOBAL_CAVEAT from src.m3_t12_confidence).

Reuses frozen preprocessors, Track 3A scorer/explainer, Track 3B regressor/support,
Track 3C optimizer, tier mapper, and confidence propagator without modification.
"""

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from src.config import (
    CLEANED_DATASET_PATH,
    INFERENCE_DEFAULTS_PATH,
    INVENTORY_SNAPSHOT_PATH,
    MODEL_READY_M2_DATASET_PATH,
    PREPROCESSOR_PATH,
)

TRAIN_END_DATE = "2024-08-07"
from src.m2_velocity import sell_through
from src.m3_t01_track3a import score_batches
from src.m3_t03_explain import explain_batch, explain_text
from src.m3_t06_regressor import R1_DEPTH_GRID
from src.m3_t09_optimizer import optimize_batches, score_whatif
from src.m3_t10_tiers import R0_STATUS_LABEL, assign_tier
from src.m3_t12_confidence import CONFIDENCE_WORDING, GLOBAL_CAVEAT, attach_confidence

REQUIRED_FIELDS: Tuple[str, ...] = (
    "category",
    "days_until_expiry",
    "initial_quantity",
    "base_price",
    "cost_price",
)

OPTIONAL_NUMERIC_FIELDS: Tuple[str, ...] = (
    "shelf_life_days",
    "days_remaining_at_purchase",
    "days_since_receipt",
    "storage_temp",
    "temp_deviation",
    "temp_abuse_events",
    "distribution_hours",
    "handling_score",
    "packaging_score",
    "supplier_score",
    "spoilage_sensitivity",
    "daily_demand",
    "demand_variability",
)

OPTIONAL_CATEGORICAL_FIELDS: Tuple[str, ...] = (
    "product_name",
    "store_id",
    "region",
    "supplier_id",
    "quality_grade",
    "transaction_date",
)

ALL_OPTIONAL_TRACKED: Tuple[str, ...] = OPTIONAL_NUMERIC_FIELDS + OPTIONAL_CATEGORICAL_FIELDS

KEY_DEFAULT_SENSITIVE_FIELDS: Tuple[str, ...] = (
    "packaging_score",
    "handling_score",
    "days_until_expiry",
    "days_since_receipt",
    "temp_deviation",
)

SNAPSHOT_FORBIDDEN_COLS: Tuple[str, ...] = (
    "was_spoiled",
    "spoilage_risk",
    "units_sold",
    "units_wasted",
    "waste_pct",
    "revenue",
    "waste_cost",
    "profit",
    "profit_margin_pct",
    "markdown_applied",
    "discount_pct",
    "selling_price",
)

QUALITY_GRADE_MAP: Dict[str, int] = {"A": 3, "B": 2, "C": 1}

EXTRA_CONFIDENCE_REASONS: Dict[str, str] = {
    "unknown_entity": "Product or store was not seen in training history; category defaults applied.",
    "defaults_dominate": "More than half of optional batch inputs were defaulted from training category averages.",
}


def build_and_save_defaults(output_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Computes training-only category medians/modes and product x store / category x store
    historical velocity and markdown frequency fallbacks from split == 'train'.
    Saves JSON to models/track_3_inference_defaults.json.
    """
    target = Path(output_path) if output_path is not None else INFERENCE_DEFAULTS_PATH
    cl = pd.read_csv(CLEANED_DATASET_PATH)
    cl["transaction_date_dt"] = pd.to_datetime(cl["transaction_date"])
    tr = cl[cl["split"] == "train"].copy()
    assert (tr["transaction_date_dt"] <= pd.Timestamp(TRAIN_END_DATE)).all(), (
        "Training split contains dates after TRAIN_END_DATE."
    )

    m2 = pd.read_csv(
        MODEL_READY_M2_DATASET_PATH,
        usecols=["record_id", "split", "sales_velocity", "sales_velocity_per_day", "store_md_freq_hist"],
    )
    m2_tr = m2[m2["split"] == "train"].drop(columns=["split"])
    tr = tr.merge(m2_tr, on="record_id", how="inner")
    assert len(tr) == 80042, f"Expected 80,042 training rows, got {len(tr)}"

    def _cat_mode(series: pd.Series) -> str:
        m = series.dropna().mode()
        return str(m.iloc[0]) if len(m) > 0 else ""

    global_num = {c: float(tr[c].median()) for c in OPTIONAL_NUMERIC_FIELDS}
    global_cat = {
        c: (_cat_mode(tr[c]) if c != "transaction_date" else str(TRAIN_END_DATE))
        for c in OPTIONAL_CATEGORICAL_FIELDS
    }
    global_hist = {
        "sales_velocity": float(tr["sales_velocity"].median()),
        "sales_velocity_per_day": float(tr["sales_velocity_per_day"].median()),
        "store_md_freq_hist": float(tr["store_md_freq_hist"].median()),
    }

    by_category: Dict[str, Dict[str, Any]] = {}
    for cat, sub in tr.groupby("category"):
        num_meds = {c: float(sub[c].median()) for c in OPTIONAL_NUMERIC_FIELDS}
        cat_modes = {
            c: (_cat_mode(sub[c]) if c != "transaction_date" else str(TRAIN_END_DATE))
            for c in OPTIONAL_CATEGORICAL_FIELDS
        }
        hist_meds = {
            "sales_velocity": float(sub["sales_velocity"].median()),
            "sales_velocity_per_day": float(sub["sales_velocity_per_day"].median()),
            "store_md_freq_hist": float(sub["store_md_freq_hist"].median()),
        }
        by_category[str(cat)] = {
            "numeric_medians": num_meds,
            "categorical_modes": cat_modes,
            "history_medians": hist_meds,
        }

    # Product x Store and Category x Store history medians for hypothetical/new batches
    ps_hist: Dict[str, Dict[str, float]] = {}
    for (prod, store), sub in tr.groupby(["product_name", "store_id"]):
        ps_hist[f"{prod}||{store}"] = {
            "sales_velocity": float(sub["sales_velocity"].median()),
            "sales_velocity_per_day": float(sub["sales_velocity_per_day"].median()),
            "store_md_freq_hist": float(sub["store_md_freq_hist"].median()),
        }

    cs_hist: Dict[str, Dict[str, float]] = {}
    for (cat, store), sub in tr.groupby(["category", "store_id"]):
        cs_hist[f"{cat}||{store}"] = {
            "sales_velocity": float(sub["sales_velocity"].median()),
            "sales_velocity_per_day": float(sub["sales_velocity_per_day"].median()),
            "store_md_freq_hist": float(sub["store_md_freq_hist"].median()),
        }

    store_region: Dict[str, str] = {}
    for store, sub in tr.groupby("store_id"):
        store_region[str(store)] = _cat_mode(sub["region"])

    payload: Dict[str, Any] = {
        "train_end_date": str(TRAIN_END_DATE),
        "training_row_count": int(len(tr)),
        "valid_categories": sorted(tr["category"].dropna().astype(str).unique().tolist()),
        "valid_products": sorted(tr["product_name"].dropna().astype(str).unique().tolist()),
        "valid_stores": sorted(tr["store_id"].dropna().astype(str).unique().tolist()),
        "valid_regions": sorted(tr["region"].dropna().astype(str).unique().tolist()),
        "valid_suppliers": sorted(tr["supplier_id"].dropna().astype(str).unique().tolist()),
        "valid_quality_grades": ["A", "B", "C"],
        "store_region_map": store_region,
        "global_defaults": {
            "numeric_medians": global_num,
            "categorical_modes": global_cat,
            "history_medians": global_hist,
        },
        "by_category": by_category,
        "history_by_product_store": ps_hist,
        "history_by_category_store": cs_hist,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    load_inference_defaults.cache_clear()
    return payload


@lru_cache(maxsize=1)
def load_inference_defaults(path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads training-only defaults JSON (building it if missing)."""
    target = Path(path) if path is not None else INFERENCE_DEFAULTS_PATH
    if not target.exists():
        return build_and_save_defaults(target)
    return json.loads(target.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_preprocessor():
    """Loads the frozen M1 ColumnTransformer preprocessor."""
    return joblib.load(PREPROCESSOR_PATH)


@lru_cache(maxsize=1)
def _load_m2_history_lookup() -> pd.DataFrame:
    """Loads M2 causal history columns keyed by record_id for 'real' mode parity."""
    cols = [
        "record_id",
        "sales_velocity",
        "sales_velocity_per_day",
        "store_md_freq_hist",
        "sell_through_risk",
        "sell_through_ratio",
        "at_risk",
    ]
    return pd.read_csv(MODEL_READY_M2_DATASET_PATH, usecols=cols)


def _is_missing(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return bool(pd.isna(val))


def validate_input(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validates raw batch input against the Track 3 input contract (P1, P3).

    Raises ValueError on:
      - Empty DataFrame
      - Missing or NaN required fields (category, days_until_expiry, initial_quantity,
        base_price, cost_price)
      - Negative/impossible values
      - Unknown category

    Returns:
      (filled_df, lineage_df) where lineage_df tracks per-row:
        record_id, n_optional_defaulted, defaulted_share, defaults_dominate,
        unknown_entity, unknown_fields
    """
    if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
        raise ValueError("Input raw_df must be a non-empty pandas DataFrame.")

    missing_req = [c for c in REQUIRED_FIELDS if c not in raw_df.columns]
    if missing_req:
        raise ValueError(f"Missing required input columns: {missing_req}")

    defaults = load_inference_defaults()
    valid_cats = set(defaults["valid_categories"])
    valid_prods = set(defaults["valid_products"])
    valid_stores = set(defaults["valid_stores"])
    valid_regions = set(defaults["valid_regions"])
    valid_suppliers = set(defaults["valid_suppliers"])
    store_region_map = defaults["store_region_map"]

    df = raw_df.copy().reset_index(drop=True)
    if "record_id" not in df.columns:
        df["record_id"] = [f"INF_{i:05d}" for i in range(len(df))]

    # Validate required fields row-by-row
    for idx, row in df.iterrows():
        for req in REQUIRED_FIELDS:
            if _is_missing(row[req]):
                raise ValueError(f"Row {idx}: required field '{req}' is missing/NaN.")
        cat = str(row["category"])
        if cat not in valid_cats:
            raise ValueError(
                f"Row {idx}: unknown category '{cat}'. Valid categories: {sorted(valid_cats)}"
            )
        if float(row["days_until_expiry"]) < 0:
            raise ValueError(f"Row {idx}: days_until_expiry must be >= 0.")
        if float(row["initial_quantity"]) <= 0:
            raise ValueError(f"Row {idx}: initial_quantity must be > 0.")
        if float(row["base_price"]) <= 0:
            raise ValueError(f"Row {idx}: base_price must be > 0.")
        if float(row["cost_price"]) < 0:
            raise ValueError(f"Row {idx}: cost_price must be >= 0.")
        if "discount_pct" in df.columns and not _is_missing(row["discount_pct"]):
            d_val = float(row["discount_pct"])
            if d_val < 0.0 or d_val >= 1.0:
                raise ValueError(f"Row {idx}: discount_pct must be in [0, 1).")

    # Fill optional fields with training-only category defaults & track lineage
    for col in OPTIONAL_NUMERIC_FIELDS:
        if col not in df.columns:
            df[col] = np.nan
    for col in OPTIONAL_CATEGORICAL_FIELDS:
        if col not in df.columns:
            df[col] = pd.Series([None] * len(df), dtype=object)
        else:
            df[col] = df[col].astype(object)

    for binary_col in ("is_promoted", "markdown_applied"):
        if binary_col not in df.columns:
            df[binary_col] = 0
        else:
            df[binary_col] = df[binary_col].fillna(0).astype(int)

    if "discount_pct" not in df.columns:
        df["discount_pct"] = 0.0
    else:
        df["discount_pct"] = df["discount_pct"].fillna(0.0).astype(float)

    lineage_rows: List[Dict[str, Any]] = []
    n_opt_total = len(ALL_OPTIONAL_TRACKED)

    for idx in range(len(df)):
        cat = str(df.at[idx, "category"])
        cat_def = defaults["by_category"][cat]
        num_meds = cat_def["numeric_medians"]
        cat_modes = cat_def["categorical_modes"]

        n_defaulted = 0
        defaulted_cols: List[str] = []
        unknown_fields: List[str] = []

        for col in OPTIONAL_NUMERIC_FIELDS:
            if _is_missing(df.at[idx, col]):
                df.at[idx, col] = num_meds[col]
                n_defaulted += 1
                defaulted_cols.append(col)
            else:
                df.at[idx, col] = float(df.at[idx, col])

        # Ensure numeric cols have clean float dtype
        for col in OPTIONAL_CATEGORICAL_FIELDS:
            if _is_missing(df.at[idx, col]):
                if col == "region" and not _is_missing(df.at[idx, "store_id"]):
                    st = str(df.at[idx, "store_id"])
                    df.at[idx, col] = store_region_map.get(st, cat_modes[col])
                else:
                    df.at[idx, col] = cat_modes[col]
                n_defaulted += 1
                defaulted_cols.append(col)
            else:
                val = str(df.at[idx, col])
                df.at[idx, col] = val
                if col == "product_name" and val not in valid_prods:
                    unknown_fields.append("product_name")
                elif col == "store_id" and val not in valid_stores:
                    unknown_fields.append("store_id")
                elif col == "region" and val not in valid_regions:
                    unknown_fields.append("region")
                    df.at[idx, col] = cat_modes["region"]
                elif col == "supplier_id" and val not in valid_suppliers:
                    unknown_fields.append("supplier_id")
                    df.at[idx, col] = cat_modes["supplier_id"]
                elif col == "quality_grade" and val not in QUALITY_GRADE_MAP:
                    raise ValueError(
                        f"Row {idx}: invalid quality_grade '{val}'. Expected A, B, or C."
                    )

        share = float(n_defaulted / n_opt_total)
        key_defaulted = any(c in defaulted_cols for c in KEY_DEFAULT_SENSITIVE_FIELDS)
        lineage_rows.append({
            "record_id": df.at[idx, "record_id"],
            "n_optional_defaulted": int(n_defaulted),
            "defaulted_share": share,
            "defaults_dominate": bool(share > 0.50 or key_defaulted),
            "unknown_entity": bool(len(unknown_fields) > 0),
            "unknown_fields": ",".join(unknown_fields),
            "defaulted_fields": ",".join(defaulted_cols),
        })

    # Enforce clean numeric dtypes on numeric columns
    for col in OPTIONAL_NUMERIC_FIELDS + ("days_until_expiry", "initial_quantity", "base_price", "cost_price", "discount_pct"):
        df[col] = df[col].astype(float)

    # Derive model-ready raw columns if omitted
    if "quality_grade_ord" not in df.columns or df["quality_grade_ord"].isna().any():
        df["quality_grade_ord"] = df["quality_grade"].map(QUALITY_GRADE_MAP).astype(float)

    if "selling_price" not in df.columns or df["selling_price"].isna().any():
        df["selling_price"] = np.round(df["base_price"] * (1.0 - df["discount_pct"]), 2)

    if "expiry_remaining_ratio" not in df.columns or df["expiry_remaining_ratio"].isna().any():
        denom = np.maximum(df["shelf_life_days"].to_numpy(dtype=float), 1.0)
        df["expiry_remaining_ratio"] = np.clip(
            df["days_until_expiry"].to_numpy(dtype=float) / denom, 0.0, 1.0
        )

    if "shelf_life_used_ratio" not in df.columns or df["shelf_life_used_ratio"].isna().any():
        df["shelf_life_used_ratio"] = np.clip(1.0 - df["expiry_remaining_ratio"].to_numpy(dtype=float), 0.0, 1.0)

    if "expiry_lag_days" not in df.columns or df["expiry_lag_days"].isna().any():
        df["expiry_lag_days"] = (
            df["days_until_expiry"].to_numpy(dtype=float)
            - df["days_remaining_at_purchase"].to_numpy(dtype=float)
        )

    dt = pd.to_datetime(df["transaction_date"])
    if "is_weekend" not in df.columns or df["is_weekend"].isna().any():
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    if "month_sin" not in df.columns or df["month_sin"].isna().any():
        df["month_sin"] = np.sin(2.0 * np.pi * dt.dt.month / 12.0)
    if "month_cos" not in df.columns or df["month_cos"].isna().any():
        df["month_cos"] = np.cos(2.0 * np.pi * dt.dt.month / 12.0)
    if "dow_sin" not in df.columns or df["dow_sin"].isna().any():
        df["dow_sin"] = np.sin(2.0 * np.pi * dt.dt.dayofweek / 7.0)
    if "dow_cos" not in df.columns or df["dow_cos"].isna().any():
        df["dow_cos"] = np.cos(2.0 * np.pi * dt.dt.dayofweek / 7.0)

    lineage_df = pd.DataFrame(lineage_rows)
    return df, lineage_df


def _prepare_model_features(
    validated_df: pd.DataFrame,
    mode: str = "real",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies the frozen M1 preprocessor and attaches M2 history features + raw auxiliary
    columns required by Track 3A (score_batches, explain_batch) and Track 3B/3C
    (predict_profit_curve, optimize_batches).
    """
    if mode not in ("real", "hypothetical"):
        raise ValueError(f"Unsupported mode '{mode}'. Expected 'real' or 'hypothetical'.")

    prep = _load_preprocessor()
    feat_names = list(prep.get_feature_names_out())
    X_m1 = pd.DataFrame(prep.transform(validated_df), columns=feat_names, index=validated_df.index)

    # Preserve identity/raw columns needed by explain_batch, predict_profit_curve, attach_confidence
    work = X_m1.copy()
    work["record_id"] = validated_df["record_id"].values
    work["product_name"] = validated_df["product_name"].values
    work["store_id"] = validated_df["store_id"].values
    work["category"] = validated_df["category"].values
    work["quality_grade"] = validated_df["quality_grade"].values
    work["raw_base_price"] = validated_df["base_price"].to_numpy(dtype=float)
    work["raw_cost_price"] = validated_df["cost_price"].to_numpy(dtype=float)
    work["raw_initial_quantity"] = validated_df["initial_quantity"].to_numpy(dtype=float)
    work["raw_expiry_remaining_ratio"] = validated_df["expiry_remaining_ratio"].to_numpy(dtype=float)
    work["raw_discount_pct"] = validated_df["discount_pct"].to_numpy(dtype=float)
    work["days_until_expiry_raw"] = validated_df["days_until_expiry"].to_numpy(dtype=float)

    # M2 history features (sales_velocity, store_md_freq_hist, sell_through_risk, sell_through_ratio, at_risk)
    defaults = load_inference_defaults()
    ps_hist = defaults["history_by_product_store"]
    cs_hist = defaults["history_by_category_store"]

    if mode == "real":
        m2_lookup = _load_m2_history_lookup()
        merged_hist = validated_df[["record_id"]].merge(m2_lookup, on="record_id", how="left")
    else:
        merged_hist = pd.DataFrame({"record_id": validated_df["record_id"]})
        for col in ("sales_velocity", "sales_velocity_per_day", "store_md_freq_hist", "sell_through_risk", "sell_through_ratio", "at_risk"):
            merged_hist[col] = np.nan

    # Fill any rows without precomputed M2 history (hypothetical mode or new real batches)
    need_fill = merged_hist["sales_velocity"].isna()
    if need_fill.any():
        sv_list, svd_list, smf_list = [], [], []
        for idx in validated_df.index[need_fill]:
            prod = str(validated_df.at[idx, "product_name"])
            store = str(validated_df.at[idx, "store_id"])
            cat = str(validated_df.at[idx, "category"])
            rec = ps_hist.get(f"{prod}||{store}")
            if rec is None:
                rec = cs_hist.get(f"{cat}||{store}")
            if rec is None:
                rec = defaults["by_category"][cat]["history_medians"]
            sv_list.append(rec["sales_velocity"])
            svd_list.append(rec["sales_velocity_per_day"])
            smf_list.append(rec["store_md_freq_hist"])

        merged_hist.loc[need_fill, "sales_velocity"] = sv_list
        merged_hist.loc[need_fill, "sales_velocity_per_day"] = svd_list
        merged_hist.loc[need_fill, "store_md_freq_hist"] = smf_list

        # Compute sell_through_risk, sell_through_ratio, at_risk dynamically via src.m2_velocity
        st_in = pd.DataFrame({
            "current_stock": validated_df.loc[need_fill, "initial_quantity"].to_numpy(dtype=float),
            "days_until_expiry": validated_df.loc[need_fill, "days_until_expiry"].to_numpy(dtype=float),
            "sales_velocity_per_day": np.asarray(svd_list, dtype=float),
        })
        days_to_sell, ratio, at_risk, _ = sell_through(st_in, "sales_velocity_per_day")
        merged_hist.loc[need_fill, "sell_through_risk"] = days_to_sell
        merged_hist.loc[need_fill, "sell_through_ratio"] = ratio
        merged_hist.loc[need_fill, "at_risk"] = at_risk

    for col in ("sales_velocity", "store_md_freq_hist", "sell_through_risk", "sell_through_ratio", "at_risk"):
        work[col] = merged_hist[col].to_numpy(dtype=float)

    return work, X_m1


def run_pipeline(
    raw_df: pd.DataFrame,
    mode: str = "real",
    top_k_drivers: int = 5,
) -> pd.DataFrame:
    """
    Runs the full frozen Track 3 inference pipeline on raw batch rows:
      validate_input -> preprocessor -> Track 3A score_batches & explain_batch ->
      Track 3B/3C optimize_batches -> assign_tier -> attach_confidence ->
      apply P3 defaulted/unknown-entity Low confidence overrides + P5 global caveat.
    """
    validated_df, lineage_df = validate_input(raw_df)
    work_df, _ = _prepare_model_features(validated_df, mode=mode)

    # Track 3A calibrated spoilage risk + exact linear explanations
    spoil_proba = score_batches(work_df)
    work_df["oof_proba"] = spoil_proba
    explanations = explain_batch(work_df, top_k=top_k_drivers)

    # Track 3C discrete discount optimizer -> tiers -> confidence
    recs_df = optimize_batches(work_df, grid=R1_DEPTH_GRID)
    tier_cols = assign_tier(recs_df)
    tiered_df = pd.concat([recs_df, tier_cols], axis=1)
    conf_df = attach_confidence(tiered_df, work_df)

    # Merge explanation summary & lineage & P3 overrides
    out = conf_df.copy()
    out["spoil_proba"] = spoil_proba
    out["spoil_chance_pct"] = np.round(spoil_proba * 100.0, 2)
    out["limited_history_3a"] = [bool(e["low_history_flag"]) for e in explanations]
    out["support_cell_count"] = [int(e["history_rows"]) for e in explanations]
    out["top_drivers_text"] = [explain_text(e, top_n=3) for e in explanations]
    out["top_drivers_json"] = [json.dumps(e["top_drivers"]) for e in explanations]

    out["n_optional_defaulted"] = lineage_df["n_optional_defaulted"].to_numpy()
    out["defaulted_share"] = lineage_df["defaulted_share"].to_numpy()
    out["defaults_dominate"] = lineage_df["defaults_dominate"].to_numpy()
    out["unknown_entity"] = lineage_df["unknown_entity"].to_numpy()

    # P3 override: defaulted inputs (>50%) or unknown product/store force Low confidence on R1
    for i in range(len(out)):
        if out.at[i, "region"] == "R0":
            continue
        reasons = [r for r in str(out.at[i, "confidence_reasons"]).split("|") if r]
        changed = False
        if bool(out.at[i, "unknown_entity"]) and "unknown_entity" not in reasons:
            reasons.append("unknown_entity")
            changed = True
        if bool(out.at[i, "defaults_dominate"]) and "defaults_dominate" not in reasons:
            reasons.append("defaults_dominate")
            changed = True
        if changed:
            out.at[i, "confidence_level"] = "Low"
            out.at[i, "confidence_reasons"] = "|".join(reasons)

    # Attach input summary fields for downstream UI / parity checks
    out["category"] = validated_df["category"].values
    out["product_name"] = validated_df["product_name"].values
    out["store_id"] = validated_df["store_id"].values
    out["days_until_expiry"] = validated_df["days_until_expiry"].to_numpy(dtype=float)
    out["initial_quantity"] = validated_df["initial_quantity"].to_numpy(dtype=float)
    out["base_price"] = validated_df["base_price"].to_numpy(dtype=float)
    out["cost_price"] = validated_df["cost_price"].to_numpy(dtype=float)
    out["global_caveat"] = GLOBAL_CAVEAT
    return out


def _coerce_single_row(batch: Union[pd.DataFrame, pd.Series, Dict[str, Any]]) -> pd.DataFrame:
    if isinstance(batch, dict):
        return pd.DataFrame([batch])
    if isinstance(batch, pd.Series):
        return pd.DataFrame([batch.to_dict()])
    if isinstance(batch, pd.DataFrame):
        if len(batch) != 1:
            raise ValueError(f"Expected a single-row DataFrame, got {len(batch)} rows.")
        return batch.copy().reset_index(drop=True)
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def whatif_table(
    batch: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
    mode: str = "hypothetical",
) -> pd.DataFrame:
    """
    Returns the 14-depth what-if profit curve (0.10..0.75) for a single batch,
    reusing score_whatif() without extrapolation.
    """
    one_row = _coerce_single_row(batch)
    validated_df, _ = validate_input(one_row)
    work_df, _ = _prepare_model_features(validated_df, mode=mode)
    work_df["oof_proba"] = score_batches(work_df)
    curve = score_whatif(work_df, grid=R1_DEPTH_GRID)
    curve["depth_pct"] = np.round(curve["depth"] * 100.0, 1)
    return curve


def simulate(
    base_batch: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
    overrides: Optional[Dict[str, Any]] = None,
    chosen_depth: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Runs a live dashboard Simulator scenario in 'hypothetical' mode:
      1. Applies `overrides` to `base_batch`.
      2. Runs `run_pipeline(..., mode='hypothetical')` and `whatif_table(...)`.
      3. If `chosen_depth` is provided (e.g., 0.25), evaluates that depth on the
         what-if curve (reporting whether it is supported and its simulated metrics).
    """
    row_df = _coerce_single_row(base_batch)
    if overrides:
        for k, v in overrides.items():
            row_df[k] = v

    res_df = run_pipeline(row_df, mode="hypothetical")
    curve_df = whatif_table(row_df, mode="hypothetical")
    rec = res_df.iloc[0].to_dict()

    chosen_eval: Optional[Dict[str, Any]] = None
    if chosen_depth is not None:
        d_target = float(chosen_depth)
        if rec["region"] == "R0":
            chosen_eval = {
                "chosen_depth": d_target,
                "supported": False,
                "reason": "region_R0_no_discount_evidence",
                "s_hat": None,
                "expected_units_sold": None,
                "profit_per_unit": None,
                "batch_profit": None,
            }
        else:
            match = curve_df[np.isclose(curve_df["depth"].to_numpy(dtype=float), d_target, atol=1e-6)]
            if match.empty:
                chosen_eval = {
                    "chosen_depth": d_target,
                    "supported": False,
                    "reason": "outside_supported_grid",
                    "s_hat": None,
                    "expected_units_sold": None,
                    "profit_per_unit": None,
                    "batch_profit": None,
                }
            else:
                m = match.iloc[0]
                chosen_eval = {
                    "chosen_depth": float(m["depth"]),
                    "supported": bool(m["supported"]),
                    "reason": str(m["reason"]),
                    "s_hat": float(m["s_hat"]) if pd.notna(m["s_hat"]) else None,
                    "expected_units_sold": float(m["expected_units_sold"]) if pd.notna(m["expected_units_sold"]) else None,
                    "profit_per_unit": float(m["profit_per_unit"]) if pd.notna(m["profit_per_unit"]) else None,
                    "batch_profit": float(m["batch_profit"]) if pd.notna(m["batch_profit"]) else None,
                }

    return {
        "recommendation": rec,
        "whatif_curve": curve_df,
        "chosen_depth_eval": chosen_eval,
        "global_caveat": GLOBAL_CAVEAT,
    }


def _risk_band(spoil_proba: float) -> str:
    if spoil_proba >= 0.70:
        return "Critical"
    if spoil_proba >= 0.50:
        return "High"
    if spoil_proba >= 0.25:
        return "Medium"
    return "Low"


def manager_view_table(snapshot_df: pd.DataFrame, mode: str = "real") -> Dict[str, Any]:
    """
    Formats the Manager View (Smart_Shelf_Project_Documentation.md Sections 5.4 & 5.8):
      - summary_strip: batches_today, high_or_critical_risk_count,
        estimated_value_at_risk_sim, global_caveat
      - table: Product, Days left, Stock, Spoil chance, Recommended discount tier + %,
        Confidence, simulated profit estimate, top factors, and caveat.
    """
    res = run_pipeline(snapshot_df, mode=mode)
    risk_bands = [_risk_band(float(p)) for p in res["spoil_proba"]]
    is_high_crit = np.array([rb in ("High", "Critical") for rb in risk_bands], dtype=bool)
    value_at_risk_sim = float(
        np.sum(
            res["spoil_proba"].to_numpy(dtype=float)
            * res["initial_quantity"].to_numpy(dtype=float)
            * res["cost_price"].to_numpy(dtype=float)
        )
    )

    rec_pct = np.where(
        res["recommended_depth"].notna(),
        np.round(res["recommended_depth"].to_numpy(dtype=float) * 100.0, 0),
        np.nan,
    )

    table = pd.DataFrame({
        "record_id": res["record_id"],
        "product_name": res["product_name"],
        "category": res["category"],
        "store_id": res["store_id"],
        "days_left": res["days_until_expiry"].astype(int),
        "stock_units": res["initial_quantity"].astype(int),
        "spoil_chance_pct": res["spoil_chance_pct"],
        "risk_band": risk_bands,
        "region": res["region"],
        "recommended_tier": res["tier"],
        "recommended_discount_pct": rec_pct,
        "simulated_profit_at_rec": np.round(res["predicted_profit_at_rec"].to_numpy(dtype=float), 2),
        "confidence_level": res["confidence_level"],
        "confidence_reasons": res["confidence_reasons"],
        "limited_history_flag": res["limited_history_3a"],
        "top_factors": res["top_drivers_text"],
    }).sort_values(["spoil_chance_pct", "days_left"], ascending=[False, True]).reset_index(drop=True)

    summary_strip = {
        "batches_today": int(len(table)),
        "high_or_critical_risk_count": int(is_high_crit.sum()),
        "estimated_value_at_risk_sim": round(value_at_risk_sim, 2),
        "r1_batches_count": int((res["region"] == "R1").sum()),
        "r0_batches_count": int((res["region"] == "R0").sum()),
        "global_caveat": GLOBAL_CAVEAT,
        "wording_note": (
            "All profit and value-at-risk figures are simulated estimates; "
            "no markdown is guaranteed to improve observed profit."
        ),
    }

    return {"summary_strip": summary_strip, "table": table, "raw_results": res}


def export_inventory_snapshot(output_path: Optional[Path] = None) -> Tuple[pd.DataFrame, str]:
    """
    Selects the latest test date with >= 100 batches (2024-12-31, N = 145) from
    cleaned_dataset.csv and exports it to data/processed/inventory_snapshot.csv.
    """
    target = Path(output_path) if output_path is not None else INVENTORY_SNAPSHOT_PATH
    cl = pd.read_csv(CLEANED_DATASET_PATH)
    test_df = cl[cl["split"] == "test"].copy()
    date_counts = test_df.groupby("transaction_date").size().sort_index()
    eligible = date_counts[date_counts >= 100]
    if eligible.empty:
        raise RuntimeError("No test date found with >= 100 batches.")
    snapshot_date = str(eligible.index[-1])
    snap = (
        test_df[test_df["transaction_date"] == snapshot_date]
        .sort_values("record_id")
        .reset_index(drop=True)
    )
    drop_cols = [c for c in SNAPSHOT_FORBIDDEN_COLS if c in snap.columns]
    if drop_cols:
        snap = snap.drop(columns=drop_cols)
    target.parent.mkdir(parents=True, exist_ok=True)
    snap.to_csv(target, index=False)
    return snap, snapshot_date
