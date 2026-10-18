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
from typing import Any, Dict, Generator, List, Optional, Sequence, Union

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
import pandas as pd

from src.config import (
    EXPERIMENT_NAMES,
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_TRACKING_DIR,
    MLFLOW_TRACKING_URI,
    PROJECT_ROOT,
    RANDOM_STATE,
    SMART_SHELF_AUTHOR_ENV_VAR,
    TABLES_DIR,
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

