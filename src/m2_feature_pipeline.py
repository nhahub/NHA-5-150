"""
src/m2_feature_pipeline.py - Feature engineering and consolidation pipeline for Milestone 2.
Smart Shelf: Perishable Food Waste Reduction and Dynamic Markdown Optimizer

Consolidates all Milestone 2 engineered features into reproducible, code-built datasets:
- Sales velocity and sell-through risk (M2-T02, M2-T03, re-using src.m2_velocity)
- Store-level historical markdown frequency (M2-T08)
- Quality and handling signal transformations (M2-T06)
- Standardized Track 3A and Track 3B feature list constants (M2-T11, M3 Design Decisions)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import (
    CLEANED_DATASET_PATH,
    MODEL_READY_DATASET_PATH,
    MODEL_READY_M2_DATASET_PATH,
    PREPROCESSOR_M2_PATH,
    RANDOM_STATE,
)
from src.m2_velocity import add_sales_velocity, sell_through

# Training period boundary established in Milestone 1 (M1-T04)
TRAIN_END_DATE: str = "2024-08-07"
TRAIN_ROWS: int = 80042
TEST_ROWS: int = 19958

# Forbidden outcome and leakage column sets
FORBIDDEN_OUTCOME_COLS: set = {
    "was_spoiled",
    "spoilage_risk",
    "units_sold",
    "units_wasted",
    "waste_pct",
    "revenue",
    "waste_cost",
    "profit",
    "profit_margin_pct",
}

FORBIDDEN_TRACK_3A_DISCOUNT_COLS: set = {
    "markdown_applied",
    "discount_pct",
    "selling_price",
}

FORBIDDEN_DISCOUNT_TIER_FLAGS: set = {
    "discount_tier",
    "is_light",
    "is_medium",
    "is_deep",
}

# -----------------------------------------------------------------------------
# Reusable Feature Generation Functions
# -----------------------------------------------------------------------------


def compute_velocity_and_sell_through(data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes M2 sales velocity and sell-through risk features using src.m2_velocity.

    Parameters
    ----------
    data : pd.DataFrame
        Cleaned dataset containing product_name, store_id, category,
        transaction_date, units_sold, days_remaining_at_purchase,
        days_until_expiry, initial_quantity.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed like data containing:
        - sales_velocity: trailing 7-day product-store sum / 7 (with category-store fallback)
        - sales_velocity_per_day: trailing 7-day daily rate average
        - velocity_source: attribution flag (product_store_7d, category_store_avg, no_history)
        - batches_in_window_7d: count of earlier batches in product-store window
        - current_stock: pre-sale batch arrival stock (= initial_quantity)
        - sell_through_risk: estimated days to sell out (= current_stock / sales_velocity_per_day)
        - sell_through_ratio: days to sell / days until expiry (capped at 10.0)
        - at_risk: binary flag (1.0 if ratio > 1.0 or zero velocity with stock, 0.0 otherwise)
        - risk_status: detailed status (at_risk, at_risk_zero_velocity, ok, no_history)
    """
    df = data.copy()
    if not np.issubdtype(df["transaction_date"].dtype, np.datetime64):
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    if not np.issubdtype(df["expiration_date"].dtype, np.datetime64):
        df["expiration_date"] = pd.to_datetime(df["expiration_date"])

    # Re-use src.m2_velocity.add_sales_velocity
    df = add_sales_velocity(df)

    # Set current stock strictly from pre-sale arrival quantity (no outcome leakage)
    df["current_stock"] = df["initial_quantity"]

    # Re-use src.m2_velocity.sell_through with sales_velocity_per_day
    days_to_sell, ratio, at_risk, status = sell_through(df, "sales_velocity_per_day")
    df["sell_through_risk"] = days_to_sell
    df["sell_through_ratio"] = ratio
    df["at_risk"] = at_risk
    df["risk_status"] = status

    cols = [
        "record_id",
        "sales_velocity",
        "sales_velocity_per_day",
        "velocity_source",
        "batches_in_window_7d",
        "current_stock",
        "sell_through_risk",
        "sell_through_ratio",
        "at_risk",
        "risk_status",
    ]
    # Explicitly restore exact row alignment matching the input record_id order
    aligned = df.set_index("record_id").loc[data["record_id"].values].reset_index()
    return aligned[cols]


def compute_store_markdown_frequency(data: pd.DataFrame, k_smooth: int = 20, window_d: int = 90) -> pd.DataFrame:
    """
    Computes store-level historical markdown frequency strictly on earlier dates (M2-T08).

    Parameters
    ----------
    data : pd.DataFrame
        Cleaned dataset containing record_id, store_id, transaction_date,
        expiration_date, markdown_applied.
    k_smooth : int, default 20
        Empirical Bayes smoothing prior weight.
    window_d : int, default 90
        Trailing window days for 90-day robustness check.

    Returns
    -------
    pd.DataFrame
        DataFrame containing record_id and the store markdown frequency features.
    """
    df = data.copy()
    if not np.issubdtype(df["transaction_date"].dtype, np.datetime64):
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    if not np.issubdtype(df["expiration_date"].dtype, np.datetime64):
        df["expiration_date"] = pd.to_datetime(df["expiration_date"])

    # Daily aggregation per store and transaction_date
    daily = (
        df.groupby(["store_id", "transaction_date"])
        .agg(n=("record_id", "size"), md=("markdown_applied", "sum"))
        .reset_index()
        .sort_values(["store_id", "transaction_date"])
        .reset_index(drop=True)
    )
    g = daily.groupby("store_id")

    # Strictly before current day = cumulative sum minus current day's total
    daily["n_before"] = g["n"].cumsum() - daily["n"]
    daily["md_before"] = g["md"].cumsum() - daily["md"]

    # Global all-store rate strictly before current day
    all_days = df.groupby("transaction_date").agg(n=("record_id", "size"), md=("markdown_applied", "sum")).sort_index()
    all_days["g_n_before"] = all_days["n"].cumsum() - all_days["n"]
    all_days["g_md_before"] = all_days["md"].cumsum() - all_days["md"]
    all_days["global_rate_before"] = all_days["g_md_before"] / all_days["g_n_before"].replace(0, np.nan)
    daily = daily.merge(all_days[["global_rate_before"]], left_on="transaction_date", right_index=True, how="left")

    daily["store_md_prior_n"] = daily["n_before"]
    daily["store_md_freq_hist"] = daily["md_before"] / daily["n_before"].replace(0, np.nan)
    daily["store_md_freq_smoothed"] = (daily["md_before"] + k_smooth * daily["global_rate_before"]) / (
        daily["n_before"] + k_smooth
    )

    # 90-day window on earlier dates (closed='left' = [t-90d, t))
    roll = daily.groupby("store_id").rolling(f"{window_d}D", on="transaction_date", closed="left")
    w_n, w_md = roll["n"].sum(), roll["md"].sum()
    daily["store_md_prior_n_90d"] = w_n.fillna(0).astype(int).values
    daily["store_md_freq_90d"] = (w_md / w_n.replace(0, np.nan)).values

    feat_cols = [
        "store_md_prior_n",
        "store_md_freq_hist",
        "store_md_freq_smoothed",
        "store_md_prior_n_90d",
        "store_md_freq_90d",
    ]
    merged = df[["record_id", "store_id", "transaction_date", "expiration_date"]].merge(
        daily[["store_id", "transaction_date"] + feat_cols],
        on=["store_id", "transaction_date"],
        how="left",
    )

    # Strict finished-batches version (only batches already expired before day t)
    fin_n = np.zeros(len(merged))
    fin_md = np.zeros(len(merged))
    for store, idx in merged.groupby("store_id").indices.items():
        sub = df.iloc[idx]
        order = np.argsort(sub["expiration_date"].values)
        exp_sorted = sub["expiration_date"].values[order]
        cum_md = np.concatenate([[0], np.cumsum(sub["markdown_applied"].values[order])])
        pos = np.searchsorted(exp_sorted, sub["transaction_date"].values, side="left")
        fin_n[idx] = pos
        fin_md[idx] = cum_md[pos]

    merged["store_md_prior_n_finished"] = fin_n.astype(int)
    merged["store_md_freq_finished"] = np.where(fin_n > 0, fin_md / np.where(fin_n > 0, fin_n, 1), np.nan)

    res_cols = [
        "record_id",
        "store_md_prior_n",
        "store_md_freq_hist",
        "store_md_freq_smoothed",
        "store_md_prior_n_90d",
        "store_md_freq_90d",
        "store_md_prior_n_finished",
        "store_md_freq_finished",
    ]
    aligned = merged.set_index("record_id").loc[data["record_id"].values].reset_index()
    return aligned[res_cols]


def fit_quality_handling_preprocessor(
    data: pd.DataFrame, train_mask: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """
    Fits transforms for quality and handling signals strictly on training period rows (M2-T06).

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing temp_abuse_events, handling_score, packaging_score,
        supplier_score, spoilage_sensitivity.
    train_mask : pd.Series
        Boolean series indicating rows in the training split.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Pipeline]]
        DataFrame containing z-scaled columns and dictionary of fitted pipelines.
    """
    scalers: Dict[str, Pipeline] = {}
    z_df = pd.DataFrame(index=data.index)
    z_df["record_id"] = data["record_id"].values

    cols = [
        "temp_abuse_events",
        "handling_score",
        "packaging_score",
        "supplier_score",
        "spoilage_sensitivity",
    ]

    for col in cols:
        steps = []
        if col == "temp_abuse_events":
            steps.append(("log1p", FunctionTransformer(np.log1p)))
        steps.append(("scale", StandardScaler()))
        pipe = Pipeline(steps)
        pipe.fit(data.loc[train_mask, [col]])
        scalers[col] = pipe
        z_df[f"{col}_z"] = pipe.transform(data[[col]]).ravel()

    return z_df, scalers


# -----------------------------------------------------------------------------
# Canonical Feature Lists for Milestone 3 (Decision 2 & 3)
# -----------------------------------------------------------------------------

# 1. Full Track 3A feature list: M1 pre-sale features + M2 allowed features
# Excludes discount treatments (discount_pct, markdown_applied, selling_price)
# Excludes shelf_life_used_ratio (Decision 3: expiry_remaining_ratio is standard)
# Excludes outcome/spoilage_risk targets and discount tier flags
TRACK_3A_FEATURES_FULL: List[str] = [
    # M1 Pre-Sale Numeric Features (scaled/log-transformed)
    "shelf_life_days",
    "days_remaining_at_purchase",
    "days_until_expiry",
    "days_since_receipt",
    "temp_abuse_events",
    "base_price",
    "cost_price",
    "daily_demand",
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
    # M1 Cyclical & Calendar
    "is_weekend",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    # M1 Nominal One-Hot Features
    "category_Bakery",
    "category_Beverages",
    "category_Dairy",
    "category_Deli",
    "category_Frozen_Meals",
    "category_Meat",
    "category_Pharmaceuticals",
    "category_Produce",
    "category_Ready_to_Eat",
    "category_Seafood",
    "region_Midwest",
    "region_Northeast",
    "region_South",
    "region_Southeast",
    "region_West",
    # 63 Product One-Hot Features
    "product_name_Antibiotics",
    "product_name_Apples",
    "product_name_Avocados",
    "product_name_Bagels",
    "product_name_Bananas",
    "product_name_Beef",
    "product_name_Berries",
    "product_name_Biologics",
    "product_name_Bread",
    "product_name_Butter",
    "product_name_Cakes",
    "product_name_Cheese",
    "product_name_Chicken",
    "product_name_Cod",
    "product_name_Cold Brew Coffee",
    "product_name_Cottage Cheese",
    "product_name_Crab",
    "product_name_Cream",
    "product_name_Croissants",
    "product_name_Dips",
    "product_name_Donuts",
    "product_name_Eye Drops",
    "product_name_Fresh Juice",
    "product_name_Fresh Pasta",
    "product_name_Frozen Desserts",
    "product_name_Frozen Dinners",
    "product_name_Frozen Vegetables",
    "product_name_Grapes",
    "product_name_Ground Beef",
    "product_name_Ham",
    "product_name_Hummus",
    "product_name_Ice Cream",
    "product_name_Insulin",
    "product_name_Kombucha",
    "product_name_Lamb",
    "product_name_Lettuce",
    "product_name_Lobster",
    "product_name_Milk",
    "product_name_Milk Alternatives",
    "product_name_Muffins",
    "product_name_Pastries",
    "product_name_Pizza",
    "product_name_Pork",
    "product_name_Prepared Salads",
    "product_name_Probiotics",
    "product_name_Salads",
    "product_name_Salami",
    "product_name_Salmon",
    "product_name_Sandwiches",
    "product_name_Sausages",
    "product_name_Shrimp",
    "product_name_Smoothies",
    "product_name_Soups",
    "product_name_Sour Cream",
    "product_name_Sushi",
    "product_name_Tilapia",
    "product_name_Tomatoes",
    "product_name_Tuna",
    "product_name_Turkey",
    "product_name_Turkey Slices",
    "product_name_Vaccines",
    "product_name_Wraps",
    "product_name_Yogurt",
    # 50 Store ID One-Hot Features
    "store_id_S001",
    "store_id_S002",
    "store_id_S003",
    "store_id_S004",
    "store_id_S005",
    "store_id_S006",
    "store_id_S007",
    "store_id_S008",
    "store_id_S009",
    "store_id_S010",
    "store_id_S011",
    "store_id_S012",
    "store_id_S013",
    "store_id_S014",
    "store_id_S015",
    "store_id_S016",
    "store_id_S017",
    "store_id_S018",
    "store_id_S019",
    "store_id_S020",
    "store_id_S021",
    "store_id_S022",
    "store_id_S023",
    "store_id_S024",
    "store_id_S025",
    "store_id_S026",
    "store_id_S027",
    "store_id_S028",
    "store_id_S029",
    "store_id_S030",
    "store_id_S031",
    "store_id_S032",
    "store_id_S033",
    "store_id_S034",
    "store_id_S035",
    "store_id_S036",
    "store_id_S037",
    "store_id_S038",
    "store_id_S039",
    "store_id_S040",
    "store_id_S041",
    "store_id_S042",
    "store_id_S043",
    "store_id_S044",
    "store_id_S045",
    "store_id_S046",
    "store_id_S047",
    "store_id_S048",
    "store_id_S049",
    "store_id_S050",
    # 20 Supplier ID One-Hot Features
    "supplier_id_SUP001",
    "supplier_id_SUP002",
    "supplier_id_SUP003",
    "supplier_id_SUP004",
    "supplier_id_SUP005",
    "supplier_id_SUP006",
    "supplier_id_SUP007",
    "supplier_id_SUP008",
    "supplier_id_SUP009",
    "supplier_id_SUP010",
    "supplier_id_SUP011",
    "supplier_id_SUP012",
    "supplier_id_SUP013",
    "supplier_id_SUP014",
    "supplier_id_SUP015",
    "supplier_id_SUP016",
    "supplier_id_SUP017",
    "supplier_id_SUP018",
    "supplier_id_SUP019",
    "supplier_id_SUP020",
    # M2 Engineered Allowed Features
    "sales_velocity",
    "sales_velocity_per_day",
    "sell_through_risk",
    "sell_through_ratio",
    "at_risk",
    "store_md_freq_hist",
    "store_md_freq_smoothed",
]

# 2. Compact Track 3A feature list:
# Drops features identified in M2-T01 / M2-T05 as carrying zero signal or severe redundancy:
# - region_* (5 cols): Cramér's V = 0.0066, p = 0.356 (no signal)
# - store_id_* (50 cols): Cramér's V = 0.0226, p = 0.399 (no signal)
# - supplier_id_* (20 cols): Cramér's V = 0.0135, p = 0.510 (no signal)
# - is_promoted: Cramér's V = 0.0018, p = 0.559 (no signal)
# - demand_variability: adjusted p >= 0.05 (no signal)
# - storage_temp: adjusted p >= 0.05 (no signal)
# - supplier_score: r = +0.0029, p = 0.36 (no signal)
# - spoilage_sensitivity: redundant with category (R2 = 1.0)
# - shelf_life_days: collinear with days_until_expiry (r = 0.9917)
# - days_remaining_at_purchase: collinear with days_until_expiry (r = 0.986)
# - cost_price: collinear with base_price (r = 0.986)
# - expiry_lag_days: lag between expiry dates has no spoilage signal
# - M2 velocity / sell-through / store markdown: M2-T05 confirms AUC 0.47 - 0.52 (sell-out logic only, no spoilage signal)
TRACK_3A_FEATURES_COMPACT: List[str] = [
    "days_until_expiry",
    "days_since_receipt",
    "temp_abuse_events",
    "base_price",
    "daily_demand",
    "expiry_remaining_ratio",
    "temp_deviation",
    "distribution_hours",
    "handling_score",
    "packaging_score",
    "initial_quantity",
    "quality_grade_ord",
    # Cyclical & Calendar
    "is_weekend",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    # Category One-Hot (meaningful signal, V = 0.039)
    "category_Bakery",
    "category_Beverages",
    "category_Dairy",
    "category_Deli",
    "category_Frozen_Meals",
    "category_Meat",
    "category_Pharmaceuticals",
    "category_Produce",
    "category_Ready_to_Eat",
    "category_Seafood",
    # Product Name One-Hot
    "product_name_Antibiotics",
    "product_name_Apples",
    "product_name_Avocados",
    "product_name_Bagels",
    "product_name_Bananas",
    "product_name_Beef",
    "product_name_Berries",
    "product_name_Biologics",
    "product_name_Bread",
    "product_name_Butter",
    "product_name_Cakes",
    "product_name_Cheese",
    "product_name_Chicken",
    "product_name_Cod",
    "product_name_Cold Brew Coffee",
    "product_name_Cottage Cheese",
    "product_name_Crab",
    "product_name_Cream",
    "product_name_Croissants",
    "product_name_Dips",
    "product_name_Donuts",
    "product_name_Eye Drops",
    "product_name_Fresh Juice",
    "product_name_Fresh Pasta",
    "product_name_Frozen Desserts",
    "product_name_Frozen Dinners",
    "product_name_Frozen Vegetables",
    "product_name_Grapes",
    "product_name_Ground Beef",
    "product_name_Ham",
    "product_name_Hummus",
    "product_name_Ice Cream",
    "product_name_Insulin",
    "product_name_Kombucha",
    "product_name_Lamb",
    "product_name_Lettuce",
    "product_name_Lobster",
    "product_name_Milk",
    "product_name_Milk Alternatives",
    "product_name_Muffins",
    "product_name_Pastries",
    "product_name_Pizza",
    "product_name_Pork",
    "product_name_Prepared Salads",
    "product_name_Probiotics",
    "product_name_Salads",
    "product_name_Salami",
    "product_name_Salmon",
    "product_name_Sandwiches",
    "product_name_Sausages",
    "product_name_Shrimp",
    "product_name_Smoothies",
    "product_name_Soups",
    "product_name_Sour Cream",
    "product_name_Sushi",
    "product_name_Tilapia",
    "product_name_Tomatoes",
    "product_name_Tuna",
    "product_name_Turkey",
    "product_name_Turkey Slices",
    "product_name_Vaccines",
    "product_name_Wraps",
    "product_name_Yogurt",
]

# 3. Track 3B feature list:
# Pre-sale batch features + discount_pct (treatment) + confounding controls
# Note: Track 3A OOF predicted spoilage probability will be appended in M2-T12.
TRACK_3B_FEATURES: List[str] = [
    # Treatment Variable
    "discount_pct",
    # Confounding Controls (M3-T05)
    "days_until_expiry",
    "quality_grade_ord",
    "sales_velocity",
    "store_md_freq_hist",
    "is_promoted",
    # Batch Physical & Demand Attributes
    "base_price",
    "cost_price",
    "daily_demand",
    "demand_variability",
    "initial_quantity",
    "expiry_remaining_ratio",
    "storage_temp",
    "temp_deviation",
    "distribution_hours",
    "temp_abuse_events",
    "handling_score",
    "packaging_score",
    "supplier_score",
    "spoilage_sensitivity",
    "sell_through_risk",
    "sell_through_ratio",
    "at_risk",
    # Cyclical & Calendar
    "is_weekend",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    # Category One-Hot
    "category_Bakery",
    "category_Beverages",
    "category_Dairy",
    "category_Deli",
    "category_Frozen_Meals",
    "category_Meat",
    "category_Pharmaceuticals",
    "category_Produce",
    "category_Ready_to_Eat",
    "category_Seafood",
    # Region One-Hot
    "region_Midwest",
    "region_Northeast",
    "region_South",
    "region_Southeast",
    "region_West",
]


# -----------------------------------------------------------------------------
# End-to-End Consolidation Pipeline Function
# -----------------------------------------------------------------------------


def build_consolidated_m2_dataset(
    cleaned_path: Optional[str] = None,
    model_ready_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """
    Constructs the consolidated Milestone 2 dataset by executing all feature
    engineering modules deterministically on cleaned_dataset.csv and merging
    them with the Milestone 1 model-ready dataset.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Pipeline]]
        The consolidated DataFrame and the dictionary of fitted M2 signal scalers.
    """
    cl_path = cleaned_path or CLEANED_DATASET_PATH
    mr_path = model_ready_path or MODEL_READY_DATASET_PATH

    cleaned_df = pd.read_csv(cl_path, parse_dates=["transaction_date", "expiration_date"])
    model_ready_df = pd.read_csv(mr_path)

    # 1. Verify 1:1 row alignment on record_id
    assert (
        cleaned_df["record_id"] == model_ready_df["record_id"]
    ).all(), "Fatal mismatch: record_id order between cleaned and model_ready differs!"
    assert len(cleaned_df) == 100000, f"Expected 100,000 rows, found {len(cleaned_df)}"

    # 2. Compute velocity and sell-through risk
    vel_df = compute_velocity_and_sell_through(cleaned_df)

    # 3. Compute store markdown frequencies
    md_df = compute_store_markdown_frequency(cleaned_df)

    # 4. Fit quality and handling preprocessors on training period only
    is_train = cleaned_df["transaction_date"] <= pd.Timestamp(TRAIN_END_DATE)
    assert (
        is_train.sum() == TRAIN_ROWS
    ), f"Expected {TRAIN_ROWS} training rows, found {is_train.sum()}"
    z_df, preprocessor_m2 = fit_quality_handling_preprocessor(cleaned_df, is_train)

    # 5. Assemble consolidated DataFrame via explicit key-based merge on record_id
    m2_full = (
        model_ready_df
        .merge(vel_df, on="record_id", how="left")
        .merge(md_df, on="record_id", how="left")
        .merge(z_df, on="record_id", how="left")
    )

    # Strict check: ensure discount tier flags are NOT present
    for bad_col in FORBIDDEN_DISCOUNT_TIER_FLAGS:
        assert (
            bad_col not in m2_full.columns
        ), f"Leakage Violation: forbidden discount tier column '{bad_col}' found in dataset!"

    return m2_full, preprocessor_m2
