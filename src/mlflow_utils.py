"""
src/mlflow_utils.py - Reusable MLflow experiment tracking utilities for Smart Shelf.

Provides standardized experiment management, run tracking, leakage enforcement,
metric logging contracts (Tracks 3A, 3B, 3C), artifact logging, and per-author
run summary export and merge utilities for the Smart Shelf project.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, confusion_matrix

from src.config import (
    EXPERIMENT_NAMES,
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_TRACKING_DIR,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    SMART_SHELF_AUTHOR_ENV_VAR,
    TABLES_DIR,
    TRACK_3A_MODEL_PATH,
    TRACK_3B_MODEL_PATH,
)

# Canonical forbidden column sets to enforce strict leakage prevention
FORBIDDEN_OUTCOME_COLS = {
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

FORBIDDEN_TRACK_3A_DISCOUNT_COLS = {
    "markdown_applied",
    "discount_pct",
    "selling_price",
}

VALID_SHELF_LIFE_VERSIONS = {
    "expiry_remaining_ratio",
    "shelf_life_used_ratio",
    "both",
}


def get_run_author() -> str:
    """
    Resolves the author name for run attribution.
    Prioritizes SMART_SHELF_AUTHOR environment variable, falling back to git config user.name.
    Guarantees no emails, machine usernames, or local filepaths are logged.
    """
    author = os.environ.get(SMART_SHELF_AUTHOR_ENV_VAR, "").strip()
    if author:
        return author

    try:
        res = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(PROJECT_ROOT),
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass

    return "teammate"


def get_git_commit_hash() -> str:
    """Returns the current short git commit hash, or 'untracked' if git is unavailable."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(PROJECT_ROOT),
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "untracked"


def sanitize_author_filename(author: str) -> str:
    """Converts an author name into a safe filesystem string for CSV export."""
    sanitized = re.sub(r"[^\w\-_]", "_", author.strip())
    return sanitized.strip("_") or "default"


def to_repo_relative_path(path: Union[str, Path]) -> str:
    """Converts an absolute filesystem path to a clean repo-relative POSIX path."""
    p = Path(path)
    try:
        rel = p.resolve().relative_to(PROJECT_ROOT.resolve())
        return rel.as_posix()
    except ValueError:
        # If not relative to PROJECT_ROOT, strip Windows drive letters and return safe name
        return p.name


def setup_tracking_environment() -> str:
    """
    Configures MLflow tracking URI from configuration or environment.
    Creates local tracking and artifact directories if needed.
    """
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MLFLOW_TRACKING_URI


def get_or_create_experiment(track: str) -> str:
    """
    Idempotently retrieves or creates the MLflow experiment for the specified track.

    Parameters
    ----------
    track : str
        '3A' for Spoilage Risk Classifier
        '3B' for Revenue-Response Regressor
        '3C' for Discount Optimizer

    Returns
    -------
    str
        The experiment ID.
    """
    track_norm = track.strip().upper()
    if track_norm not in EXPERIMENT_NAMES:
        raise ValueError(f"Unknown track: '{track}'. Must be one of {list(EXPERIMENT_NAMES.keys())}")

    setup_tracking_environment()
    exp_name = EXPERIMENT_NAMES[track_norm]
    client = MlflowClient()

    exp = client.get_experiment_by_name(exp_name)
    if exp is not None:
        if exp.lifecycle_stage == "deleted":
            client.restore_experiment(exp.experiment_id)
        return exp.experiment_id

    # Create new experiment with repo-relative local artifact root under models/mlartifacts
    artifact_loc = (MLFLOW_ARTIFACTS_DIR / exp_name).as_uri()
    return client.create_experiment(name=exp_name, artifact_location=artifact_loc)


def validate_feature_leakage(track: str, features: Sequence[str], shelf_life_version: str) -> None:
    """
    Enforces strict anti-leakage rules. Raises ValueError if any forbidden column is present.

    Rules enforced:
    1. shelf_life_version must be explicitly specified ('expiry_remaining_ratio' or 'shelf_life_used_ratio').
    2. spoilage_risk and all post-sale outcomes are forbidden across ALL tracks.
    3. Track 3A strictly forbids all discount fields (discount_pct, markdown_applied, selling_price).
    """
    if shelf_life_version not in VALID_SHELF_LIFE_VERSIONS:
        raise ValueError(
            f"Invalid shelf_life_version: '{shelf_life_version}'. "
            f"Must be one of {sorted(VALID_SHELF_LIFE_VERSIONS)}."
        )

    feature_set = set(features)

    # 1. Spoilage risk and post-sale outcome leakage check
    leaked_outcomes = feature_set & FORBIDDEN_OUTCOME_COLS
    if leaked_outcomes:
        raise ValueError(
            f"Leakage Violation! Target or outcome columns found in model feature set: {sorted(leaked_outcomes)}"
        )

    # 2. Track 3A discount treatment confounding check
    track_norm = track.strip().upper()
    if track_norm == "3A":
        leaked_discounts = feature_set & FORBIDDEN_TRACK_3A_DISCOUNT_COLS
        if leaked_discounts:
            raise ValueError(
                f"Leakage Violation! Discount treatments found in Track 3A classifier features: {sorted(leaked_discounts)}"
            )


@contextmanager
def start_tracked_run(
    track: str,
    model_type: str,
    feature_set_name: str,
    shelf_life_version: str,
    features: Optional[Sequence[str]] = None,
    run_name: Optional[str] = None,
    purpose: str = "production",
    train_end_date: str = "2024-08-07",
    train_rows: int = 80042,
    test_rows: int = 19958,
    dataset_version: str = "de94302b867c9debedfd45c431306623fdfc038f5ed8ca17736339b4460a6674",
    tags: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Generator[Run, None, None]:
    """
    Context manager that initializes and standardizes an MLflow tracking run.

    Enforces mandatory tags, leakage checks, split metadata, and clean attribution.
    """
    track_norm = track.strip().upper()

    # Enforce anti-leakage governance before initializing or starting any run
    if features is not None:
        validate_feature_leakage(track_norm, features, shelf_life_version)
    elif shelf_life_version not in VALID_SHELF_LIFE_VERSIONS:
        raise ValueError(
            f"Invalid shelf_life_version: '{shelf_life_version}'. "
            f"Must be one of {sorted(VALID_SHELF_LIFE_VERSIONS)}."
        )

    exp_id = get_or_create_experiment(track_norm)


    author = get_run_author()
    git_commit = get_git_commit_hash()

    standard_tags = {
        "track": track_norm,
        "model_type": model_type,
        "feature_set_name": feature_set_name,
        "shelf_life_version": shelf_life_version,
        "split_type": "time-ordered",
        "train_end_date": train_end_date,
        "train_rows": str(train_rows),
        "test_rows": str(test_rows),
        "git_commit": git_commit,
        "dataset_version": dataset_version,
        "no_spoilage_risk_leakage": "true",
        "purpose": purpose,
        "run_author": author,
    }

    if tags:
        for k, v in tags.items():
            # Guarantee no absolute paths in tags
            if isinstance(v, (str, Path)) and (Path(str(v)).is_absolute() or ":\\" in str(v)):
                standard_tags[k] = to_repo_relative_path(v)
            else:
                standard_tags[k] = str(v)

    standard_params = {
        "model_type": model_type,
        "feature_set_name": feature_set_name,
        "shelf_life_version": shelf_life_version,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "random_state": RANDOM_STATE,
    }

    if params:
        for k, v in params.items():
            if isinstance(v, (str, Path)) and (Path(str(v)).is_absolute() or ":\\" in str(v)):
                standard_params[k] = to_repo_relative_path(v)
            else:
                standard_params[k] = v

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as active_run:
        mlflow.set_tags(standard_tags)
        mlflow.log_params(standard_params)
        yield active_run


def log_track_3a_metrics(
    brier_score: float,
    log_loss_val: float,
    roc_auc: float,
    benchmark_auc: float,
    threshold: float = 0.5,
    precision: Optional[float] = None,
    recall: Optional[float] = None,
    f1: Optional[float] = None,
    extra_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Logs standard metrics for Track 3A (Spoilage Classifier).
    Headline metric is probability calibration (Brier score & log loss).
    """
    metrics = {
        "test_brier_score": float(brier_score),
        "test_log_loss": float(log_loss_val),
        "test_roc_auc": float(roc_auc),
        "benchmark_spoilage_risk_auc": float(benchmark_auc),
        "auc_gap_to_benchmark": float(roc_auc - benchmark_auc),
        "decision_threshold": float(threshold),
    }
    if precision is not None:
        metrics["test_precision"] = float(precision)
    if recall is not None:
        metrics["test_recall"] = float(recall)
    if f1 is not None:
        metrics["test_f1"] = float(f1)

    if extra_metrics:
        metrics.update({k: float(v) for k, v in extra_metrics.items()})

    mlflow.log_metrics(metrics)


def log_track_3b_metrics(
    profit_mae: float,
    profit_rmse: float,
    profit_r2: float,
    sell_through_mae: Optional[float] = None,
    sell_through_rmse: Optional[float] = None,
    sell_through_r2: Optional[float] = None,
    waste_mae: Optional[float] = None,
    waste_rmse: Optional[float] = None,
    waste_r2: Optional[float] = None,
    extra_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Logs standard metrics for Track 3B (Revenue-Response Regressor).
    Primary target is profit; sell-through and waste are secondary targets.
    """
    metrics = {
        "profit_mae": float(profit_mae),
        "profit_rmse": float(profit_rmse),
        "profit_r2": float(profit_r2),
    }
    if sell_through_mae is not None:
        metrics["sell_through_mae"] = float(sell_through_mae)
    if sell_through_rmse is not None:
        metrics["sell_through_rmse"] = float(sell_through_rmse)
    if sell_through_r2 is not None:
        metrics["sell_through_r2"] = float(sell_through_r2)

    if waste_mae is not None:
        metrics["waste_mae"] = float(waste_mae)
    if waste_rmse is not None:
        metrics["waste_rmse"] = float(waste_rmse)
    if waste_r2 is not None:
        metrics["waste_r2"] = float(waste_r2)

    if extra_metrics:
        metrics.update({k: float(v) for k, v in extra_metrics.items()})

    mlflow.log_metrics(metrics)


def log_track_3c_metrics(
    lane1_metrics: Dict[str, float],
    lane2_metrics: Dict[str, float],
    extra_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Logs standard metrics for Track 3C (Discount Optimizer).
    Lane 1 (matched historical) and Lane 2 (counterfactual simulated) are strictly separated.
    """
    metrics = {}
    for k, v in lane1_metrics.items():
        name = k if k.startswith("lane1_") else f"lane1_{k}"
        metrics[name] = float(v)

    for k, v in lane2_metrics.items():
        name = k if k.startswith("lane2_") else f"lane2_{k}"
        metrics[name] = float(v)

    if extra_metrics:
        metrics.update({k: float(v) for k, v in extra_metrics.items()})

    mlflow.log_metrics(metrics)


def log_feature_list_artifact(features: Sequence[str], artifact_name: str = "feature_list.json") -> None:
    """Logs the list of feature column names as an artifact."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / artifact_name
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(list(features), f, indent=2)
        mlflow.log_artifact(str(tmp_file))


def log_figure_artifact(fig: Any, artifact_name: str) -> None:
    """Saves a matplotlib/seaborn figure and logs it as an artifact."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / artifact_name
        fig.savefig(str(tmp_file), bbox_inches="tight", dpi=150)
        mlflow.log_artifact(str(tmp_file))


def export_runs_summary(
    author: Optional[str] = None,
    include_setup_tests: bool = False,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Exports all logged runs for the author across the three experiments to a clean CSV table.

    Parameters
    ----------
    author : Optional[str]
        Author name to export. Defaults to get_run_author().
    include_setup_tests : bool, default False
        Whether to include setup verification runs (purpose='setup_test').
    output_dir : Optional[Path]
        Directory to save the CSV. Defaults to src.config.TABLES_DIR.

    Returns
    -------
    Path
        The path of the exported CSV file.
    """
    setup_tracking_environment()
    target_author = (author or get_run_author()).strip()
    author_clean = sanitize_author_filename(target_author)

    out_dir = output_dir or TABLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"M2_T09_mlflow_runs_{author_clean}.csv"

    client = MlflowClient()
    rows = []

    for track, exp_name in EXPERIMENT_NAMES.items():
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            continue

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=5000,
        )

        for run in runs:
            tags = run.data.tags
            run_author = tags.get("run_author", "").strip()

            # Filter by author
            if run_author and run_author.lower() != target_author.lower():
                continue

            # Filter setup tests unless explicitly requested
            purpose = tags.get("purpose", "")
            if not include_setup_tests and purpose == "setup_test":
                continue

            # List artifact names
            try:
                artifacts = client.list_artifacts(run.info.run_id)
                artifact_names = ";".join(a.path for a in artifacts)
            except Exception:
                artifact_names = ""

            row = {
                "run_id": run.info.run_id,
                "experiment_name": exp_name,
                "track": tags.get("track", track),
                "run_name": run.info.run_name or "",
                "run_author": run_author or target_author,
                "start_time": pd.to_datetime(run.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M:%S")
                if run.info.start_time
                else "",
                "status": run.info.status,
                "model_type": tags.get("model_type", ""),
                "feature_set_name": tags.get("feature_set_name", ""),
                "shelf_life_version": tags.get("shelf_life_version", ""),
                "purpose": purpose,
                "git_commit": tags.get("git_commit", ""),
                "artifact_names": artifact_names,
            }

            # Add metrics (flattened)
            for m_key, m_val in run.data.metrics.items():
                row[f"metric_{m_key}"] = round(m_val, 6)

            # Add parameters (flattened)
            for p_key, p_val in run.data.params.items():
                row[f"param_{p_key}"] = p_val

            rows.append(row)

    if not rows:
        # Create an empty template with expected headers if no matching runs exist
        df = pd.DataFrame(
            columns=[
                "run_id",
                "experiment_name",
                "track",
                "run_name",
                "run_author",
                "start_time",
                "status",
                "model_type",
                "feature_set_name",
                "shelf_life_version",
                "purpose",
                "git_commit",
                "artifact_names",
            ]
        )
    else:
        df = pd.DataFrame(rows)
        # Sort by start_time descending
        if "start_time" in df.columns:
            df = df.sort_values(by="start_time", ascending=False)

    df.to_csv(out_file, index=False)
    return out_file


def merge_runs_summaries(
    tables_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Merges all per-author run summaries (reports/tables/M2_T09_mlflow_runs_*.csv) into a master table.

    Parameters
    ----------
    tables_dir : Optional[Path]
        Directory containing the per-author CSV files. Defaults to src.config.TABLES_DIR.
    output_path : Optional[Path]
        Explicit destination path for the merged CSV file.
        Defaults to gitignored location: models/mlruns/M2_T09_mlflow_runs_master.csv.

    Returns
    -------
    Path
        Path to the merged master CSV file.
    """
    src_dir = tables_dir or TABLES_DIR
    target_path = output_path or (MLFLOW_TRACKING_DIR / "M2_T09_mlflow_runs_master.csv")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    csv_files = [
        f for f in src_dir.glob("M2_T09_mlflow_runs_*.csv") if not f.name.endswith("_master.csv")
    ]

    if not csv_files:
        empty_df = pd.DataFrame(columns=["run_id", "experiment_name", "track", "run_author"])
        empty_df.to_csv(target_path, index=False)
        return target_path

    dfs = []
    for f in csv_files:
        try:
            sub = pd.read_csv(f)
            if not sub.empty:
                dfs.append(sub)
        except Exception:
            pass

    if not dfs:
        empty_df = pd.DataFrame(columns=["run_id", "experiment_name", "track", "run_author"])
        empty_df.to_csv(target_path, index=False)
        return target_path

    merged_df = pd.concat(dfs, ignore_index=True)

    # Deduplicate by run_id if runs overlap
    if "run_id" in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=["run_id"], keep="last")

    if "start_time" in merged_df.columns:
        merged_df = merged_df.sort_values(by="start_time", ascending=False)

    merged_df.to_csv(target_path, index=False)
    return target_path


def save_model_with_metadata(
    model: Any,
    track: str,
    feature_list: Sequence[str],
    shelf_life_version: str,
    train_period: Union[str, Dict[str, str], Sequence[str]],
    mlflow_run_id: Optional[str] = None,
    models_dir: Optional[Union[str, Path]] = None,
    name: Optional[str] = None,
    overwrite: bool = False,
    log_as_artifact: bool = True,
) -> Tuple[Path, Path]:
    """
    Saves a trained model with joblib and a standardized JSON metadata file beside it.

    Parameters
    ----------
    model : Any
        Trained model instance (e.g. scikit-learn, LightGBM, XGBoost).
    track : str
        Track identifier ('3A', '3B', or '3C').
    feature_list : Sequence[str]
        Ordered list of feature column names used by the model.
    shelf_life_version : str
        Shelf-life formulation ('expiry_remaining_ratio' or 'shelf_life_used_ratio').
    train_period : Union[str, Dict[str, str], Sequence[str]]
        Description or boundary of training period (e.g. '2023-01-01 to 2024-08-07').
    mlflow_run_id : Optional[str]
        MLflow run ID associated with the model. Defaults to active run ID if active.
    models_dir : Optional[Union[str, Path]]
        Destination directory. Defaults to src.config.MODELS_DIR.
    name : Optional[str]
        Model base filename (without suffix or with .joblib).
        Defaults to track_3a_classifier.joblib for Track 3A and track_3b_regressor.joblib for Track 3B.
    overwrite : bool, default False
        Whether to overwrite existing files. If False, raises FileExistsError if files exist.
    log_as_artifact : bool, default True
        Whether to log both the model and metadata files as artifacts if an MLflow run is active.

    Returns
    -------
    Tuple[Path, Path]
        Paths to the saved (model_path, metadata_path).
    """
    track_norm = track.strip().upper()
    target_dir = Path(models_dir) if models_dir is not None else MODELS_DIR

    if name:
        stem = Path(name).stem
    else:
        if track_norm == "3A":
            stem = TRACK_3A_MODEL_PATH.stem
        elif track_norm == "3B":
            stem = TRACK_3B_MODEL_PATH.stem
        elif track_norm == "3C":
            stem = "track_3c_optimizer"
        else:
            stem = f"model_track_{track_norm.lower()}"

    model_path = target_dir / f"{stem}.joblib"
    meta_path = target_dir / f"{stem}.json"

    if not overwrite:
        if model_path.exists():
            raise FileExistsError(
                f"Model file already exists: {to_repo_relative_path(model_path)}. "
                "Pass overwrite=True to replace."
            )
        if meta_path.exists():
            raise FileExistsError(
                f"Metadata file already exists: {to_repo_relative_path(meta_path)}. "
                "Pass overwrite=True to replace."
            )

    import sklearn
    lib_versions: Dict[str, str] = {
        "scikit-learn": getattr(sklearn, "__version__", "unknown"),
        "pandas": getattr(pd, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown"),
        "joblib": getattr(joblib, "__version__", "unknown"),
        "mlflow": getattr(mlflow, "__version__", "unknown"),
    }
    try:
        import lightgbm
        lib_versions["lightgbm"] = lightgbm.__version__
    except ImportError:
        pass
    try:
        import xgboost
        lib_versions["xgboost"] = xgboost.__version__
    except ImportError:
        pass
    try:
        import shap
        lib_versions["shap"] = shap.__version__
    except ImportError:
        pass

    resolved_run_id = mlflow_run_id
    if not resolved_run_id:
        active = mlflow.active_run()
        if active is not None:
            resolved_run_id = active.info.run_id

    metadata = {
        "track": track_norm,
        "feature_list": list(feature_list),
        "features": list(feature_list),
        "shelf_life_version": str(shelf_life_version),
        "train_period": train_period,
        "library_versions": lib_versions,
        "mlflow_run_id": resolved_run_id,
        "git_commit": get_git_commit_hash(),
        "model_class": model.__class__.__name__,
    }

    target_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if log_as_artifact and mlflow.active_run() is not None:
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(meta_path))

    return model_path, meta_path


def load_model_with_metadata(path: Union[str, Path]) -> Tuple[Any, Dict[str, Any]]:
    """
    Loads a model and its associated metadata JSON file.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the model (.joblib) or metadata (.json) file.

    Returns
    -------
    Tuple[Any, Dict[str, Any]]
        (model, metadata)

    Raises
    ------
    FileNotFoundError
        If either the model file or the metadata JSON file is missing.
    KeyError
        If the feature list is absent from the metadata.
    """
    p = Path(path)
    if p.suffix == ".json":
        meta_path = p
        model_path = p.with_suffix(".joblib")
    elif p.suffix == ".joblib":
        model_path = p
        meta_path = p.with_suffix(".json")
    else:
        model_path = p.with_suffix(".joblib")
        meta_path = p.with_suffix(".json")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {to_repo_relative_path(model_path)}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata JSON file missing beside model: {to_repo_relative_path(meta_path)}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if "feature_list" not in metadata and "features" not in metadata:
        raise KeyError(f"Feature list is absent from model metadata: {to_repo_relative_path(meta_path)}")

    if "feature_list" not in metadata and "features" in metadata:
        metadata["feature_list"] = metadata["features"]
    elif "features" not in metadata and "feature_list" in metadata:
        metadata["features"] = metadata["feature_list"]

    model = joblib.load(model_path)
    return model, metadata


def log_calibration_curve(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    proba: Union[Sequence[float], np.ndarray, pd.Series],
    n_bins: int = 10,
    name: str = "reliability_curve",
) -> float:
    """
    Computes calibration curve, logs the reliability curve plot as an artifact,
    and returns the Brier score.

    Parameters
    ----------
    y_true : array-like
        Binary true labels.
    proba : array-like
        Predicted probabilities for the positive class.
    n_bins : int, default 10
        Number of probability bins.
    name : str, default "reliability_curve"
        Artifact base name.

    Returns
    -------
    float
        The calculated Brier score.
    """
    y_arr = np.asarray(y_true)
    p_arr = np.asarray(proba)

    brier = float(brier_score_loss(y_arr, p_arr))
    prob_true, prob_pred = calibration_curve(y_arr, p_arr, n_bins=n_bins, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, "s-", color="#1f77b4", label=f"Model (Brier = {brier:.4f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([-0.05, 1.05])
    ax.set_title("Reliability Curve (Calibration)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    filename = f"{name}.png" if not name.endswith(".png") else name
    log_figure_artifact(fig, filename)
    plt.close(fig)
    plt.close("all")

    return brier


def log_confusion_matrix(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    name: str = "confusion_matrix",
) -> np.ndarray:
    """
    Computes confusion matrix, logs a formatted matrix plot as an artifact,
    and returns the confusion matrix array.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted discrete labels.
    name : str, default "confusion_matrix"
        Artifact base name.

    Returns
    -------
    np.ndarray
        The confusion matrix array.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    cm = confusion_matrix(y_t, y_p)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    classes = ["Negative (0)", "Positive (1)"] if cm.shape == (2, 2) else [str(i) for i in range(len(cm))]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="Actual label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()

    filename = f"{name}.png" if not name.endswith(".png") else name
    log_figure_artifact(fig, filename)
    plt.close(fig)
    plt.close("all")

    return cm


def log_shap_summary(
    model: Any,
    X_sample: Union[pd.DataFrame, np.ndarray],
    max_rows: int = 2000,
    name: str = "shap_summary",
) -> Dict[str, float]:
    """
    Samples at most max_rows with seed 42, logs a SHAP summary plot for tree models,
    and returns the mean absolute SHAP value per feature as a dictionary.

    Handles both list-of-arrays and single-array SHAP outputs for binary classifiers.

    Parameters
    ----------
    model : Any
        Trained tree model (LightGBM, XGBoost, RandomForest, GradientBoosting).
    X_sample : Union[pd.DataFrame, np.ndarray]
        Input features.
    max_rows : int, default 2000
        Maximum sample rows to compute SHAP values for.
    name : str, default "shap_summary"
        Artifact base name.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping feature name to its mean absolute SHAP value.
    """
    if isinstance(X_sample, pd.DataFrame):
        feature_names = list(X_sample.columns)
        if len(X_sample) > max_rows:
            X_df = X_sample.sample(n=max_rows, random_state=42)
        else:
            X_df = X_sample.copy()
    else:
        arr = np.asarray(X_sample)
        feature_names = [f"feature_{i}" for i in range(arr.shape[1])]
        if len(arr) > max_rows:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(arr), size=max_rows, replace=False)
            X_df = pd.DataFrame(arr[idx], columns=feature_names)
        else:
            X_df = pd.DataFrame(arr, columns=feature_names)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    # Handle list-of-arrays, 3D ndarrays, and 2D ndarrays
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_matrix = np.asarray(shap_values[1])
        else:
            shap_matrix = np.asarray(shap_values[0])
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            if shap_values.shape[-1] == 2:
                shap_matrix = shap_values[:, :, 1]
            elif shap_values.shape[0] == 2:
                shap_matrix = shap_values[1, :, :]
            else:
                shap_matrix = shap_values[:, :, 0]
        else:
            shap_matrix = shap_values
    elif hasattr(shap_values, "values"):
        vals = shap_values.values
        if vals.ndim == 3 and vals.shape[-1] == 2:
            shap_matrix = vals[:, :, 1]
        else:
            shap_matrix = vals
    else:
        shap_matrix = np.asarray(shap_values)

    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    shap_dict = {str(col): float(val) for col, val in zip(feature_names, mean_abs_shap)}

    plt.figure()
    shap.summary_plot(shap_matrix, X_df, show=False)
    fig = plt.gcf()
    filename = f"{name}.png" if not name.endswith(".png") else name
    log_figure_artifact(fig, filename)
    plt.close(fig)
    plt.close("all")

    return shap_dict


def log_oof_predictions(
    oof_df: pd.DataFrame,
    name: str = "track_3a_oof",
) -> str:
    """
    Logs out-of-fold (OOF) prediction DataFrame as a CSV artifact.
    Validates presence of 'record_id' and 'oof_proba' columns.

    Parameters
    ----------
    oof_df : pd.DataFrame
        DataFrame containing OOF predictions.
    name : str, default "track_3a_oof"
        Artifact base name.

    Returns
    -------
    str
        Logged artifact filename.
    """
    if "record_id" not in oof_df.columns:
        raise ValueError("OOF DataFrame missing required column: 'record_id'")
    if "oof_proba" not in oof_df.columns:
        raise ValueError("OOF DataFrame missing required column: 'oof_proba'")

    filename = f"{name}.csv" if not name.endswith(".csv") else name
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / filename
        oof_df.to_csv(tmp_path, index=False)
        mlflow.log_artifact(str(tmp_path))

    return filename


