"""
src/m3_oof.py - Reusable Out-of-Fold (OOF) Generator for Track 3A.

Provides time-blocked and expanding-window OOF probability generation for the
Track 3A spoilage classifier, ensuring zero target leakage into downstream
Track 3B models and Track 3C optimizers.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

from src.config import (
    CLEANED_DATASET_PATH,
    MODEL_READY_M2_DATASET_PATH,
    RANDOM_STATE,
    TRACK_3A_OOF_TRAIN_PATH,
)
from src.m2_feature_pipeline import (
    TRACK_3A_FEATURES_FULL,
    validate_feature_columns_exist,
)
from src.mlflow_utils import log_track_3a_metrics, start_tracked_run


def generate_oof_predictions(
    model_factory: Callable[[], Any],
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    dates: Union[pd.Series, Sequence[Any], np.ndarray],
    record_ids: Union[pd.Series, Sequence[Any], np.ndarray],
    n_blocks: int = 5,
    scheme: str = "blocked",
    feature_set_name: str = "track_3a_full",
    model_type: str = "lightgbm",
    shelf_life_version: str = "expiry_remaining_ratio",
) -> pd.DataFrame:
    """
    Generates out-of-fold probability predictions on the training partition.

    Parameters
    ----------
    model_factory : Callable[[], Any]
        Callable returning a fresh, un-fitted scikit-learn compatible classifier.
    X : pd.DataFrame or np.ndarray
        Feature matrix (training rows only, N=80,042).
    y : pd.Series or np.ndarray
        Binary target ('was_spoiled', training rows only).
    dates : pd.Series or Sequence
        Transaction dates corresponding to X rows.
    record_ids : pd.Series or Sequence
        Unique record IDs corresponding to X rows.
    n_blocks : int, default 5
        Number of time blocks or folds.
    scheme : str, default 'blocked'
        Cross-validation scheme:
        - 'blocked': Contiguous time-blocked k-fold (train on all other blocks).
        - 'expanding': Expanding window (train only on earlier blocks; block 0 missing).
        - 'stratified': Stratified k-fold (for reference comparison).
    feature_set_name : str, default 'track_3a_full'
        Name of feature set metadata.
    model_type : str, default 'lightgbm'
        Model type metadata.
    shelf_life_version : str, default 'expiry_remaining_ratio'
        Standard shelf-life feature name.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        [record_id, oof_proba, block, scheme, feature_set_name, model_type, shelf_life_version]
    """
    n_samples = len(X)
    assert len(y) == n_samples, f"Length mismatch: X ({n_samples}) vs y ({len(y)})"
    assert len(dates) == n_samples, f"Length mismatch: X ({n_samples}) vs dates ({len(dates)})"
    assert len(record_ids) == n_samples, f"Length mismatch: X ({n_samples}) vs record_ids ({len(record_ids)})"

    y_arr = np.asarray(y)
    rec_arr = np.asarray(record_ids)
    dates_s = pd.to_datetime(pd.Series(dates))

    # Convert X to DataFrame for uniform indexing
    if not isinstance(X, pd.DataFrame):
        X_df = pd.DataFrame(X)
    else:
        X_df = X.copy()

    scheme_norm = scheme.strip().lower()
    if scheme_norm not in ("blocked", "expanding", "stratified"):
        raise ValueError(f"Unknown scheme '{scheme}'. Must be 'blocked', 'expanding', or 'stratified'.")

    oof_proba = np.full(n_samples, np.nan, dtype=float)
    assigned_block = np.full(n_samples, -1, dtype=int)

    if scheme_norm in ("blocked", "expanding"):
        # Step 3: Cut blocks strictly at date boundaries so rows with the same
        # transaction_date are never split across two blocks.
        unique_dates = np.unique(dates_s.values)
        unique_dates.sort()
        date_blocks = np.array_split(unique_dates, n_blocks)

        for b in range(n_blocks):
            val_dates = date_blocks[b]
            val_mask = np.isin(dates_s.values, val_dates)
            val_idx = np.where(val_mask)[0]
            assigned_block[val_idx] = b

            if scheme_norm == "blocked":
                train_mask = ~val_mask
                train_idx = np.where(train_mask)[0]

                # Assertions
                assert len(np.intersect1d(train_idx, val_idx)) == 0, f"Block {b} leaked validation rows!"
                assert len(train_idx) + len(val_idx) == n_samples, f"Block {b} row partition is incomplete!"
                assert len(np.intersect1d(dates_s.values[train_idx], val_dates)) == 0, f"Block {b} split dates across blocks!"

                clf = model_factory()
                clf.fit(X_df.iloc[train_idx], y_arr[train_idx])
                oof_proba[val_idx] = clf.predict_proba(X_df.iloc[val_idx])[:, 1]

            elif scheme_norm == "expanding":
                if b == 0:
                    # Block 0 has no strictly earlier dates; left as NaN
                    continue
                prior_dates = np.concatenate(date_blocks[:b])
                train_mask = np.isin(dates_s.values, prior_dates)
                train_idx = np.where(train_mask)[0]

                # Assertions: strictly prior dates, zero leakage
                assert len(np.intersect1d(train_idx, val_idx)) == 0, f"Block {b} leaked validation rows!"
                assert dates_s.values[train_idx].max() < val_dates.min(), f"Block {b} expanding window violated date ordering!"

                clf = model_factory()
                clf.fit(X_df.iloc[train_idx], y_arr[train_idx])
                oof_proba[val_idx] = clf.predict_proba(X_df.iloc[val_idx])[:, 1]

    elif scheme_norm == "stratified":
        skf = StratifiedKFold(n_splits=n_blocks, shuffle=True, random_state=RANDOM_STATE)
        for b, (train_idx, val_idx) in enumerate(skf.split(X_df, y_arr)):
            assigned_block[val_idx] = b
            assert len(np.intersect1d(train_idx, val_idx)) == 0, f"Fold {b} leaked validation rows!"
            clf = model_factory()
            clf.fit(X_df.iloc[train_idx], y_arr[train_idx])
            oof_proba[val_idx] = clf.predict_proba(X_df.iloc[val_idx])[:, 1]

    # Structural Assertions
    if scheme_norm == "blocked":
        # Every training row appears exactly once
        assert not np.isnan(oof_proba).any(), "Found unexpected NaN in blocked OOF predictions!"
        assert (assigned_block >= 0).all(), "Unassigned block detected in blocked OOF!"
        assert (oof_proba >= 0.0).all() and (oof_proba <= 1.0).all(), "OOF probabilities out of bounds [0, 1]!"

    elif scheme_norm == "expanding":
        # Block 0 rows are NaN, remaining blocks have valid probabilities
        block_0_mask = assigned_block == 0
        assert np.isnan(oof_proba[block_0_mask]).all(), "Block 0 in expanding window must be NaN!"
        assert not np.isnan(oof_proba[~block_0_mask]).any(), "Unexpected NaN in expanding window predictions!"

    result_df = pd.DataFrame({
        "record_id": rec_arr,
        "oof_proba": oof_proba,
        "block": assigned_block,
        "scheme": scheme_norm,
        "feature_set_name": feature_set_name,
        "model_type": model_type,
        "shelf_life_version": shelf_life_version,
    })

    return result_df


def fit_final_and_predict_test(
    model_factory: Callable[[], Any],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
) -> np.ndarray:
    """
    Fits the final model on all training partition rows and generates test predictions.

    Parameters
    ----------
    model_factory : Callable[[], Any]
        Model constructor.
    X_train : pd.DataFrame or np.ndarray
        Training features (N=80,042).
    y_train : pd.Series or np.ndarray
        Training targets.
    X_test : pd.DataFrame or np.ndarray
        Test features (N=19,958).

    Returns
    -------
    np.ndarray
        Array of predicted spoilage probabilities on X_test.
    """
    clf = model_factory()
    clf.fit(X_train, y_train)
    test_proba = clf.predict_proba(X_test)[:, 1]
    assert len(test_proba) == len(X_test), "Mismatch in test prediction length!"
    assert not np.isnan(test_proba).any(), "NaN found in test probabilities!"
    return test_proba


def build_baseline_oof_dataset(
    m2_path: Optional[str] = None,
    cleaned_path: Optional[str] = None,
    output_path: Optional[str] = None,
    log_mlflow: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Constructs the baseline Track 3A OOF dataset using LightGBM on TRACK_3A_FEATURES_FULL
    under the time-blocked scheme, writes track_3a_oof_train.csv, and logs an MLflow run.
    """
    m2_file = m2_path or MODEL_READY_M2_DATASET_PATH
    cl_file = cleaned_path or CLEANED_DATASET_PATH
    out_file = output_path or TRACK_3A_OOF_TRAIN_PATH

    m2_df = pd.read_csv(m2_file)
    cl_df = pd.read_csv(cl_file, usecols=["record_id", "transaction_date"])
    cl_df["transaction_date"] = pd.to_datetime(cl_df["transaction_date"])

    train_df = m2_df[m2_df["split"] == "train"].merge(cl_df, on="record_id")
    assert len(train_df) == 80042, f"Expected 80,042 train rows, found {len(train_df)}"
    train_df = train_df.sort_values(["transaction_date", "record_id"]).reset_index(drop=True)

    validate_feature_columns_exist(TRACK_3A_FEATURES_FULL, train_df.columns)
    feature_cols = list(TRACK_3A_FEATURES_FULL)

    X_train = train_df[feature_cols]
    y_train = train_df["was_spoiled"].values
    dates_train = train_df["transaction_date"].values
    records_train = train_df["record_id"].values

    def lgb_factory():
        return lgb.LGBMClassifier(
            random_state=RANDOM_STATE,
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=100,
            n_jobs=-1,
            verbose=-1,
        )

    oof_df = generate_oof_predictions(
        model_factory=lgb_factory,
        X=X_train,
        y=y_train,
        dates=dates_train,
        record_ids=records_train,
        n_blocks=5,
        scheme="blocked",
        feature_set_name="track_3a_full",
        model_type="lightgbm",
        shelf_life_version="expiry_remaining_ratio",
    )

    # Save to data/processed/track_3a_oof_train.csv
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(out_path, index=False)

    # Compute metrics strictly aligned by record_id
    eval_df = train_df[["record_id", "was_spoiled", "spoilage_risk"]].merge(
        oof_df[["record_id", "oof_proba"]], on="record_id"
    )

    brier = float(brier_score_loss(eval_df["was_spoiled"], eval_df["oof_proba"]))
    ll = float(log_loss(eval_df["was_spoiled"], eval_df["oof_proba"]))
    auc = float(roc_auc_score(eval_df["was_spoiled"], eval_df["oof_proba"]))
    bench_auc = float(roc_auc_score(eval_df["was_spoiled"], eval_df["spoilage_risk"]))

    metrics = {
        "brier_score": brier,
        "log_loss": ll,
        "roc_auc": auc,
        "benchmark_auc": bench_auc,
    }

    if log_mlflow:
        with start_tracked_run(
            track="3A",
            model_type="lightgbm",
            feature_set_name="track_3a_full",
            shelf_life_version="expiry_remaining_ratio",
            features=feature_cols,
            run_name="track_3a_oof_baseline",
            purpose="oof_baseline",
        ):
            log_track_3a_metrics(
                brier_score=brier,
                log_loss_val=ll,
                roc_auc=auc,
                benchmark_auc=bench_auc,
            )

    return oof_df, metrics
