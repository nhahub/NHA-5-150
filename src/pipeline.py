"""
src/pipeline.py - Feature engineering and preprocessing pipelines for Smart Shelf.

Provides an end-to-end, scikit-learn compatible transformer (PerishableFeatureEngineer)
and factory functions to construct the complete preprocessing pipeline.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Canonical column role groups established in Milestone 1 (M1-T04)
LOG_COLS: List[str] = [
    "shelf_life_days",
    "days_remaining_at_purchase",
    "days_until_expiry",
    "days_since_receipt",
    "shelf_life_used_ratio",
    "temp_abuse_events",
    "base_price",
    "cost_price",
    "selling_price",
    "discount_pct",
    "daily_demand",
]

SCALE_COLS: List[str] = [
    "expiry_remaining_ratio",
    "expiry_lag_days",
    "storage_temp",
    "temp_deviation",
    "distribution_hours",
    "handling_score",
    "packaging_score",
    "supplier_score",
    "spoilage_sensitivity",
    "initial_quantity",
    "demand_variability",
    "quality_grade_ord",
]

NOMINAL_COLS: List[str] = [
    "category",
    "region",
    "product_name",
    "store_id",
    "supplier_id",
]

BINARY_COLS: List[str] = [
    "is_weekend",
    "is_promoted",
    "markdown_applied",
]

CYCLICAL_COLS: List[str] = [
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
]


class PerishableFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms raw retail batch inputs into engineered temporal, ratio,
    cyclical, and ordinal features without relying on external notebook state.
    Designed for seamless use in both batch training and real-time Streamlit simulation.
    """

    def __init__(self, date_format: str = "%Y-%m-%d"):
        self.date_format = date_format

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "PerishableFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Parse Datetimes
        t_date = pd.to_datetime(df["transaction_date"], errors="coerce")
        e_date = pd.to_datetime(df["expiration_date"], errors="coerce")
        shelf_life = df["shelf_life_days"].astype(float)

        # 2. Derive Receipt Date & Days Since Receipt
        receipt_date = e_date - pd.to_timedelta(shelf_life, unit="D")
        df["days_since_receipt"] = (t_date - receipt_date).dt.days.fillna(0).clip(lower=0)

        # 3. Ensure Days Remaining at Purchase & Days Until Expiry
        if "days_remaining_at_purchase" not in df.columns:
            df["days_remaining_at_purchase"] = (e_date - t_date).dt.days

        if "days_until_expiry" not in df.columns:
            # Fallback for simulator where only transaction and expiration dates are supplied
            df["days_until_expiry"] = df["days_remaining_at_purchase"]

        # 4. Ratios and Temporal Lags
        df["shelf_life_used_ratio"] = (df["days_since_receipt"] / shelf_life).clip(0.0, 1.0)
        df["expiry_remaining_ratio"] = (df["days_until_expiry"] / shelf_life).clip(0.0, 1.0)
        df["expiry_lag_days"] = df["days_remaining_at_purchase"] - df["days_until_expiry"]

        # 5. Cyclical Calendar Features
        month = t_date.dt.month if "month" not in df.columns else df["month"].astype(float)
        dow = t_date.dt.dayofweek if "day_of_week" not in df.columns else df["day_of_week"].astype(float)

        df["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
        df["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
        df["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
        df["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

        # 6. Ordinal Quality Mapping
        quality_map: Dict[str, int] = {"C": 1, "B": 2, "A": 3}
        if "quality_grade_ord" not in df.columns:
            df["quality_grade_ord"] = df["quality_grade"].map(quality_map).fillna(2).astype(int)

        # 7. Ensure Binary Treatments Default Intact
        if "markdown_applied" not in df.columns and "discount_pct" in df.columns:
            df["markdown_applied"] = (df["discount_pct"] > 0).astype(int)

        if "selling_price" not in df.columns and "base_price" in df.columns and "discount_pct" in df.columns:
            df["selling_price"] = (df["base_price"] * (1.0 - df["discount_pct"])).round(2)

        return df


def build_column_transformer() -> ColumnTransformer:
    """
    Constructs the ColumnTransformer matching Milestone 1 specifications:
    - log1p + StandardScaler on skewed numerical columns
    - StandardScaler on remaining numerical columns
    - OneHotEncoder on nominal categorical columns (ignoring unknown categories)
    - Passthrough for binary and cyclical trigonometric features
    """
    log_branch = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num_log_scaled", log_branch, LOG_COLS),
            ("num_scaled", StandardScaler(), SCALE_COLS),
            ("cat_onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), NOMINAL_COLS),
            ("binary_cyclical_passthrough", "passthrough", BINARY_COLS + CYCLICAL_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def build_full_pipeline() -> Pipeline:
    """
    Returns an end-to-end scikit-learn Pipeline that accepts raw batch DataFrames
    directly and outputs the normalized 178-feature matrix.
    """
    return Pipeline([
        ("feature_engineer", PerishableFeatureEngineer()),
        ("encoder_scaler", build_column_transformer()),
    ])
