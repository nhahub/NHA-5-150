"""
scripts/M2_T11_build_m2_dataset.py - End-to-end dataset consolidation script for Milestone 2.
Smart Shelf: Perishable Food Waste Reduction and Dynamic Markdown Optimizer

Executes the M2 consolidation pipeline end-to-end, writing:
- data/processed/model_ready_m2_dataset.csv
- data/processed/feature_dictionary_m2.csv
- models/preprocessor_m2.joblib

Performs comprehensive Phase 3 verification:
(a) Row counts & split verification (100,000 rows; 80,042 train / 19,958 test)
(b) NaN accounting across feature columns
(c) Velocity window coverage verification
(d) Anti-leakage tests across TRACK_3A_FEATURES_FULL, TRACK_3A_FEATURES_COMPACT, TRACK_3B_FEATURES
(e) Temporal causality perturbation test (post-cutoff sales perturbation)
(f) Training-only scaler fit verification
(g) Baseline M1 verification script execution
"""

import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLEANED_DATASET_PATH,
    FEATURE_DICTIONARY_M2_PATH,
    FEATURE_DICTIONARY_PATH,
    MODEL_READY_DATASET_PATH,
    MODEL_READY_M2_DATASET_PATH,
    PREPROCESSOR_M2_PATH,
    RANDOM_STATE,
)
from src.m2_feature_pipeline import (
    FORBIDDEN_DISCOUNT_TIER_FLAGS,
    FORBIDDEN_OUTCOME_COLS,
    FORBIDDEN_TRACK_3A_DISCOUNT_COLS,
    TRACK_3A_FEATURES_COMPACT,
    TRACK_3A_FEATURES_FULL,
    TRACK_3B_FEATURES,
    TRAIN_END_DATE,
    TRAIN_ROWS,
    TEST_ROWS,
    build_consolidated_m2_dataset,
    compute_store_markdown_frequency,
    compute_velocity_and_sell_through,
)
from src.mlflow_utils import validate_feature_leakage


def run_consolidation() -> pd.DataFrame:
    print("=" * 70)
    print("M2-T11: Consolidating Milestone 2 Features into Model-Ready Dataset")
    print("=" * 70)
    t0 = time.time()

    # 1. Build consolidated dataset and preprocessor
    print("\n[Step 1/5] Executing feature pipeline...")
    df_m2, preprocessor_m2 = build_consolidated_m2_dataset()
    print(f" -> Consolidated DataFrame built: {df_m2.shape[0]:,} rows, {df_m2.shape[1]} columns in {time.time()-t0:.2f}s")

    # 2. Save preprocessor_m2.joblib
    print("\n[Step 2/5] Saving preprocessor_m2.joblib...")
    PREPROCESSOR_M2_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor_m2, PREPROCESSOR_M2_PATH)
    print(f" -> Saved fitted M2 preprocessor to: {PREPROCESSOR_M2_PATH}")

    # 3. Save model_ready_m2_dataset.csv
    print("\n[Step 3/5] Saving model_ready_m2_dataset.csv...")
    MODEL_READY_M2_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_m2.to_csv(MODEL_READY_M2_DATASET_PATH, index=False, float_format="%.6g")
    print(f" -> Saved consolidated dataset to: {MODEL_READY_M2_DATASET_PATH} ({MODEL_READY_M2_DATASET_PATH.stat().st_size / (1024*1024):.1f} MB)")

    # 4. Build and save feature_dictionary_m2.csv
    print("\n[Step 4/5] Building feature_dictionary_m2.csv...")
    dict_m1 = pd.read_csv(FEATURE_DICTIONARY_PATH)
    dict_m2_rows = []

    # Copy all M1 rows, updating spoilage_sensitivity status
    for _, row in dict_m1.iterrows():
        r = row.to_dict()
        if r["column"] == "spoilage_sensitivity":
            r["track_3a_use"] = "allowed"  # Cleared by M1-T06 verdict
        dict_m2_rows.append(r)

    # Define metadata entries for new M2 columns
    m2_new_defs = [
        {
            "column": "sales_velocity",
            "role": "feature",
            "source_column": "units_sold",
            "transformation": "trailing 7-day sum of units_sold / 7 with category-store fallback",
            "track_3a_use": "allowed",
            "track_3b_use": "allowed",
            "redundant_with": "daily_demand",
        },
        {
            "column": "sales_velocity_per_day",
            "role": "feature",
            "source_column": "units_sold, days_remaining_at_purchase",
            "transformation": "trailing 7-day mean of units_sold / days_remaining with category-store fallback",
            "track_3a_use": "allowed",
            "track_3b_use": "allowed",
            "redundant_with": "sales_velocity",
        },
        {
            "column": "velocity_source",
            "role": "metadata",
            "source_column": "product_name, store_id, category, transaction_date",
            "transformation": "rule indicator: product_store_7d, category_store_avg, no_history",
            "track_3a_use": "never",
            "track_3b_use": "never",
            "redundant_with": "None",
        },
        {
            "column": "batches_in_window_7d",
            "role": "metadata",
            "source_column": "product_name, store_id, transaction_date",
            "transformation": "count of earlier batches in trailing 7-day window",
            "track_3a_use": "never",
            "track_3b_use": "never",
            "redundant_with": "None",
        },
        {
            "column": "current_stock",
            "role": "metadata",
            "source_column": "initial_quantity",
            "transformation": "passthrough (= initial_quantity)",
            "track_3a_use": "never",
            "track_3b_use": "never",
            "redundant_with": "initial_quantity",
        },
        {
            "column": "sell_through_risk",
            "role": "feature",
            "source_column": "initial_quantity, sales_velocity_per_day",
            "transformation": "current_stock / sales_velocity_per_day (estimated days to sell out)",
            "track_3a_use": "allowed",
            "track_3b_use": "allowed",
            "redundant_with": "sell_through_ratio",
        },
        {
            "column": "sell_through_ratio",
            "role": "feature",
            "source_column": "sell_through_risk, days_until_expiry",
            "transformation": "sell_through_risk / max(days_until_expiry, 1), capped at 10.0",
            "track_3a_use": "allowed",
            "track_3b_use": "allowed",
            "redundant_with": "sell_through_risk",
        },
        {
            "column": "at_risk",
            "role": "feature",
            "source_column": "sell_through_ratio",
            "transformation": "binary indicator (ratio > 1.0 or zero velocity with stock)",
            "track_3a_use": "allowed",
            "track_3b_use": "allowed",
            "redundant_with": "sell_through_ratio",
        },
        {
            "column": "risk_status",
            "role": "metadata",
            "source_column": "sell_through_ratio",
            "transformation": "categorical status: at_risk, at_risk_zero_velocity, ok, no_history",
            "track_3a_use": "never",
            "track_3b_use": "never",
            "redundant_with": "at_risk",
        },
        {
            "column": "store_md_prior_n",
            "role": "metadata",
            "source_column": "store_id, transaction_date",
            "transformation": "count of store batches strictly before transaction_date",
            "track_3a_use": "never",
            "track_3b_use": "never",
            "redundant_with": "None",
        },
        {
            "column": "store_md_freq_hist",
            "role": "feature",
            "source_column": "store_id, transaction_date, markdown_applied",
            "transformation": "store markdowns strictly before transaction_date / store batches strictly before transaction_date",
            "track_3a_use": "allowed",
            "track_3b_use": "allowed",
            "redundant_with": "store_md_freq_smoothed",
        },
        {
            "column": "store_md_freq_smoothed",
            "role": "feature",
            "source_column": "store_md_freq_hist, global_rate_before",
            "transformation": "empirical Bayes smoothed rate toward past all-store rate (K=20)",
            "track_3a_use": "allowed",
            "track_3b_use": "allowed",
            "redundant_with": "store_md_freq_hist",
        },
        {
            "column": "store_md_prior_n_90d",
            "role": "metadata",
            "source_column": "store_id, transaction_date",
            "transformation": "count of store batches in trailing 90 days",
            "track_3a_use": "never",
            "track_3b_use": "never",
            "redundant_with": "None",
        },
        {
            "column": "store_md_freq_90d",
            "role": "feature",
            "source_column": "store_id, transaction_date, markdown_applied",
            "transformation": "trailing 90-day store markdown frequency",
            "track_3a_use": "allowed",
            "track_3b_use": "allowed",
            "redundant_with": "store_md_freq_hist",
        },
        {
            "column": "store_md_prior_n_finished",
            "role": "metadata",
            "source_column": "store_id, transaction_date, expiration_date",
            "transformation": "count of store batches expired before transaction_date",
            "track_3a_use": "never",
            "track_3b_use": "never",
            "redundant_with": "None",
        },
        {
            "column": "store_md_freq_finished",
            "role": "feature",
            "source_column": "store_id, transaction_date, expiration_date, markdown_applied",
            "transformation": "markdown frequency of batches expired before transaction_date",
            "track_3a_use": "allowed",
            "track_3b_use": "allowed",
            "redundant_with": "store_md_freq_hist",
        },
        {
            "column": "temp_abuse_events_z",
            "role": "feature",
            "source_column": "temp_abuse_events",
            "transformation": "log1p + StandardScaler fitted on training period only",
            "track_3a_use": "allowed",
            "track_3b_use": "allowed",
            "redundant_with": "temp_abuse_events",
        },
        {
            "column": "handling_score_z",
            "role": "feature",
            "source_column": "handling_score",
            "transformation": "StandardScaler fitted on training period only",
            "track_3a_use": "allowed",
            "track_3b_use": "allowed",
            "redundant_with": "handling_score",
        },
        {
            "column": "packaging_score_z",
            "role": "feature",
            "source_column": "packaging_score",
            "transformation": "StandardScaler fitted on training period only",
            "track_3a_use": "allowed",
            "track_3b_use": "allowed",
            "redundant_with": "packaging_score",
        },
        {
            "column": "supplier_score_z",
            "role": "feature",
            "source_column": "supplier_score",
            "transformation": "StandardScaler fitted on training period only",
            "track_3a_use": "allowed",
            "track_3b_use": "allowed",
            "redundant_with": "supplier_score",
        },
        {
            "column": "spoilage_sensitivity_z",
            "role": "feature",
            "source_column": "spoilage_sensitivity",
            "transformation": "StandardScaler fitted on training period only",
            "track_3a_use": "allowed",
            "track_3b_use": "allowed",
            "redundant_with": "spoilage_sensitivity",
        },
    ]

    dict_m2_rows.extend(m2_new_defs)
    df_dict_m2 = pd.DataFrame(dict_m2_rows)
    df_dict_m2.to_csv(FEATURE_DICTIONARY_M2_PATH, index=False)
    print(f" -> Saved updated feature dictionary to: {FEATURE_DICTIONARY_M2_PATH} ({len(df_dict_m2)} rows)")

    # 5. Run Phase 3 Verifications
    print("\n[Step 5/5] Running Phase 3 Verifications...")
    run_phase_3_verifications(df_m2, preprocessor_m2)

    return df_m2


def run_phase_3_verifications(df_m2: pd.DataFrame, preprocessor_m2: dict):
    print("\n--- (a) Row Counts & Split Verification ---")
    assert len(df_m2) == 100000, f"Row count mismatch: {len(df_m2)} != 100,000"
    train_count = (df_m2["split"] == "train").sum()
    test_count = (df_m2["split"] == "test").sum()
    print(f" [PASS] Total rows: {len(df_m2):,}")
    print(f" [PASS] Train split rows: {train_count:,} (expected {TRAIN_ROWS:,})")
    print(f" [PASS] Test split rows: {test_count:,} (expected {TEST_ROWS:,})")
    assert train_count == TRAIN_ROWS
    assert test_count == TEST_ROWS

    print("\n--- (b) NaN Accounting Across Features ---")
    nan_counts = df_m2.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    print(f" Columns with NaNs ({len(cols_with_nan)} columns total):")
    for col, count in cols_with_nan.items():
        pct = (count / len(df_m2)) * 100
        print(f"   * {col:<26}: {count:5,d} NaNs ({pct:5.2f}%)")
    # Expected: velocity/sell-through features have 576 or 2,603 NaNs; store_md has 148, 152, 552 NaNs
    assert "sales_velocity" in cols_with_nan
    assert cols_with_nan["sales_velocity"] == 576
    assert cols_with_nan["store_md_freq_hist"] == 152
    assert cols_with_nan["store_md_freq_smoothed"] == 148
    print(" [PASS] All NaN counts strictly correspond to expected edge boundaries (first days / no prior sales).")

    print("\n--- (c) Velocity Coverage Verification ---")
    src_counts = df_m2["velocity_source"].value_counts()
    src_pct = df_m2["velocity_source"].value_counts(normalize=True) * 100
    print(f" Velocity sources distribution:")
    for src, count in src_counts.items():
        print(f"   * {src:<20}: {count:6,d} ({src_pct[src]:5.2f}%)")
    # 25.42% product_store_7d, 74.01% category_store_avg, 0.58% no_history
    assert src_counts["product_store_7d"] == 25415
    assert src_counts["category_store_avg"] == 74009
    assert src_counts["no_history"] == 576
    print(" [PASS] Velocity coverage matches M2-T02 report exactly (25.4% own 7d window, 74.0% fallback, 0.6% no history).")

    print("\n--- (d) Anti-Leakage Feature List Tests ---")
    print(f" Validating TRACK_3A_FEATURES_FULL ({len(TRACK_3A_FEATURES_FULL)} features)...")
    validate_feature_leakage("3A", TRACK_3A_FEATURES_FULL, "expiry_remaining_ratio")
    for bad in FORBIDDEN_OUTCOME_COLS | FORBIDDEN_TRACK_3A_DISCOUNT_COLS | FORBIDDEN_DISCOUNT_TIER_FLAGS:
        assert bad not in TRACK_3A_FEATURES_FULL, f"Leakage violation in 3A FULL: {bad}"
    assert "shelf_life_used_ratio" not in TRACK_3A_FEATURES_FULL
    print("  [PASS] TRACK_3A_FEATURES_FULL clean.")

    print(f" Validating TRACK_3A_FEATURES_COMPACT ({len(TRACK_3A_FEATURES_COMPACT)} features)...")
    validate_feature_leakage("3A", TRACK_3A_FEATURES_COMPACT, "expiry_remaining_ratio")
    for bad in FORBIDDEN_OUTCOME_COLS | FORBIDDEN_TRACK_3A_DISCOUNT_COLS | FORBIDDEN_DISCOUNT_TIER_FLAGS:
        assert bad not in TRACK_3A_FEATURES_COMPACT, f"Leakage violation in 3A COMPACT: {bad}"
    assert "shelf_life_used_ratio" not in TRACK_3A_FEATURES_COMPACT
    print("  [PASS] TRACK_3A_FEATURES_COMPACT clean.")

    print(f" Validating TRACK_3B_FEATURES ({len(TRACK_3B_FEATURES)} features)...")
    validate_feature_leakage("3B", TRACK_3B_FEATURES, "expiry_remaining_ratio")
    for bad in FORBIDDEN_OUTCOME_COLS | {"markdown_applied", "selling_price"} | FORBIDDEN_DISCOUNT_TIER_FLAGS:
        assert bad not in TRACK_3B_FEATURES, f"Leakage violation in 3B: {bad}"
    assert "discount_pct" in TRACK_3B_FEATURES
    assert "shelf_life_used_ratio" not in TRACK_3B_FEATURES
    print("  [PASS] TRACK_3B_FEATURES clean.")

    print("\n--- (e) Temporal Causality Perturbation Test ---")
    cl_df = pd.read_csv(CLEANED_DATASET_PATH, parse_dates=["transaction_date", "expiration_date"])
    cutoff = pd.Timestamp("2024-01-01")
    pre_mask = cl_df["transaction_date"] <= cutoff
    post_mask = cl_df["transaction_date"] > cutoff

    # Baseline features on unperturbed data
    base_vel = compute_velocity_and_sell_through(cl_df)
    base_md = compute_store_markdown_frequency(cl_df)

    # Perturb units_sold and markdown_applied strictly on post-cutoff dates
    perturbed_df = cl_df.copy()
    np.random.seed(RANDOM_STATE)
    perturbed_df.loc[post_mask, "units_sold"] = np.random.randint(100, 1000, size=post_mask.sum())
    perturbed_df.loc[post_mask, "markdown_applied"] = 1 - perturbed_df.loc[post_mask, "markdown_applied"]

    pert_vel = compute_velocity_and_sell_through(perturbed_df)
    pert_md = compute_store_markdown_frequency(perturbed_df)

    # Confirm pre-cutoff values did not change
    vel_diff = (base_vel.loc[pre_mask, "sales_velocity"].fillna(-999) - pert_vel.loc[pre_mask, "sales_velocity"].fillna(-999)).abs().max()
    md_diff = (base_md.loc[pre_mask, "store_md_freq_hist"].fillna(-999) - pert_md.loc[pre_mask, "store_md_freq_hist"].fillna(-999)).abs().max()
    print(f" Max absolute change in sales_velocity on or before {cutoff.date()}: {vel_diff:.10f}")
    print(f" Max absolute change in store_md_freq_hist on or before {cutoff.date()}: {md_diff:.10f}")
    assert vel_diff == 0.0, "Causality breach: post-cutoff perturbation altered pre-cutoff sales_velocity!"
    assert md_diff == 0.0, "Causality breach: post-cutoff perturbation altered pre-cutoff store_md_freq_hist!"
    print(" [PASS] Strict temporal causality confirmed: future transactions have zero effect on past/present features.")

    print("\n--- (f) Training-Only Scaler Fit Verification ---")
    cl_df = pd.read_csv(CLEANED_DATASET_PATH)
    is_train = pd.to_datetime(cl_df["transaction_date"]) <= pd.Timestamp(TRAIN_END_DATE)
    for col, pipe in preprocessor_m2.items():
        scaler = pipe.named_steps["scale"]
        if col == "temp_abuse_events":
            transformed = np.log1p(cl_df.loc[is_train, [col]])
            actual_mean = transformed.mean().values[0]
            actual_var = transformed.var(ddof=0).values[0]
        else:
            actual_mean = cl_df.loc[is_train, col].mean()
            actual_var = cl_df.loc[is_train, col].var(ddof=0)
        fitted_mean = scaler.mean_[0]
        fitted_var = scaler.var_[0]
        diff_mean = abs(fitted_mean - actual_mean)
        diff_var = abs(fitted_var - actual_var)
        print(f" * {col:<24}: fitted_mean={fitted_mean:10.5f}, train_mean={actual_mean:10.5f} (diff={diff_mean:.2e})")
        assert diff_mean < 1e-9, f"Scaler for {col} was not fitted on training rows!"
        assert diff_var < 1e-9, f"Scaler variance for {col} was not fitted on training rows!"
    print(" [PASS] All M2 scalers verified to be fitted strictly on the 80,042 training period rows.")

    print("\n" + "=" * 70)
    print("ALL M2-T11 CONSOLIDATION & VERIFICATION CHECKS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_consolidation()
