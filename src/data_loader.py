"""
src/data_loader.py - Safe data loading helper for Milestone 2 & 3 modeling.

Enforces strict feature separation between Track 3A (spoilage classifier)
and Track 3B (revenue-response regressor) to eliminate treatment-assignment
confounding and data leakage.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

try:
    from src.config import DATA_PROCESSED
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import DATA_PROCESSED


def _resolve_data_dir(data_dir: Optional[Union[str, Path]] = None) -> Path:
    """Finds the folder containing model_ready_dataset.csv and feature_dictionary.csv."""
    if data_dir is not None:
        p = Path(data_dir)
        if (p / "model_ready_dataset.csv").exists():
            return p
        raise FileNotFoundError(f"model_ready_dataset.csv not found in provided path: {p}")

    candidates = [
        DATA_PROCESSED,
        Path.cwd() / "data" / "processed",
        Path.cwd() / "processed",
        Path.cwd(),
        Path(__file__).resolve().parent.parent / "data" / "processed",
        Path(__file__).resolve().parent.parent / "processed",
        Path(__file__).resolve().parent.parent,
    ]
    for cand in candidates:
        if (cand / "model_ready_dataset.csv").exists():
            return cand

    raise FileNotFoundError("Could not locate model_ready_dataset.csv in standard workspace paths.")


def get_feature_names(track: str = "3A", data_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """
    Returns the exact list of allowed model feature names for the specified track.

    - Track 3A: 175 legitimate pre-sale/handling features (excludes markdown_applied,
      discount_pct, selling_price to prevent treatment confounding).
    - Track 3B: Pre-sale features + discount_pct (the treatment input; excludes
      selling_price and markdown_applied which are derived from it).
    """
    p = _resolve_data_dir(data_dir)
    fd = pd.read_csv(p / "feature_dictionary.csv")

    track_norm = track.strip().upper()
    if track_norm == "3A":
        # Strictly allowed pre-sale features
        features = fd.loc[fd["track_3a_use"] == "allowed", "column"].tolist()
    elif track_norm == "3B":
        # Pre-sale features + the single discount treatment
        features = fd.loc[fd["track_3b_use"].isin(["allowed", "treatment (varied by 3C)"]), "column"].tolist()
    elif track_norm == "ALL":
        # All 178 model-ready features (for diagnostic/benchmark comparisons only)
        features = fd.loc[fd["role"] == "feature", "column"].tolist()
    else:
        raise ValueError(f"Unknown track: {track}. Must be '3A', '3B', or 'ALL'.")

    return features


def load_dataset(
    track: str = "3A",
    split: Optional[str] = None,
    data_dir: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Loads features and targets for the chosen modeling track.

    Parameters
    ----------
    track : str, default '3A'
        '3A' for Spoilage Risk Classifier (target: was_spoiled)
        '3B' for Revenue-Response Regressor (target: profit)
    split : Optional[str], default None
        If 'train' or 'test', returns (X, y, None, None) for that slice.
        If None, returns (X_train, y_train, X_test, y_test).
    data_dir : Optional[Union[str, Path]], default None
        Directory containing model_ready_dataset.csv.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]
        Train and test features and target arrays.
    """
    p = _resolve_data_dir(data_dir)
    df = pd.read_csv(p / "model_ready_dataset.csv")
    features = get_feature_names(track=track, data_dir=p)

    track_norm = track.strip().upper()
    target_col = "was_spoiled" if track_norm == "3A" else "profit"

    # Leakage guard check
    forbidden = {"was_spoiled", "spoilage_risk", "units_sold", "units_wasted",
                 "waste_pct", "revenue", "waste_cost", "profit", "profit_margin_pct"}
    assert not (set(features) & forbidden), f"Leakage detected! Target/outcome found in features: {set(features) & forbidden}"

    if track_norm == "3A":
        treatment_cols = {"markdown_applied", "discount_pct", "selling_price"}
        assert not (set(features) & treatment_cols), f"Confounding detected! Treatment variable found in Track 3A features: {set(features) & treatment_cols}"

    if split in ("train", "test"):
        sub = df[df["split"] == split]
        return sub[features], sub[target_col], None, None

    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"

    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, target_col]
    X_test = df.loc[test_mask, features]
    y_test = df.loc[test_mask, target_col]

    return X_train, y_train, X_test, y_test


def load_track_3a_benchmark(data_dir: Optional[Union[str, Path]] = None) -> Tuple[pd.Series, pd.Series]:
    """
    Loads the reserved benchmark spoilage_risk and true was_spoiled labels
    for evaluating model calibration and ROC-AUC.
    """
    p = _resolve_data_dir(data_dir)
    df = pd.read_csv(p / "model_ready_dataset.csv")
    test_mask = df["split"] == "test"
    return df.loc[test_mask, "spoilage_risk"], df.loc[test_mask, "was_spoiled"]
