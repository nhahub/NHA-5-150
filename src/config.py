"""
Global Project Configuration and Path Management
Smart Shelf: Perishable Food Waste Reduction and Dynamic Markdown Optimizer

Defines robust, directory-independent paths to all project directories and assets.
"""

from pathlib import Path

# Resolve PROJECT_ROOT based on the location of this file (src/config.py -> parent is root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"

RAW_DATASET_PATH = DATA_RAW / "perishable_goods_management.csv"
CLEANED_DATASET_PATH = DATA_PROCESSED / "cleaned_dataset.csv"
MODEL_READY_DATASET_PATH = DATA_PROCESSED / "model_ready_dataset.csv"
FEATURE_DICTIONARY_PATH = DATA_PROCESSED / "feature_dictionary.csv"
MODEL_READY_M2_DATASET_PATH = DATA_PROCESSED / "model_ready_m2_dataset.csv"
FEATURE_DICTIONARY_M2_PATH = DATA_PROCESSED / "feature_dictionary_m2.csv"

# Models and checkpoints
MODELS_DIR = PROJECT_ROOT / "models"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
PREPROCESSOR_M2_PATH = MODELS_DIR / "preprocessor_m2.joblib"

# Notebooks and scripts
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"

# Reports, tables, and figures
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_M1_DIR = REPORTS_DIR / "M1"
REPORTS_M2_DIR = REPORTS_DIR / "M2"

TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_M1_DIR = FIGURES_DIR / "m1"
FIGURES_M2_DIR = FIGURES_DIR / "m2"
FIGURES_INTERACTIVE_DIR = FIGURES_DIR / "interactive"

# Fixed random state for reproducibility
RANDOM_STATE = 42

# MLflow tracking configuration (local store under models/mlruns, overridable by env var)
import os

# Enable filesystem tracking backend for MLflow 3.x and suppress agent hints
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

MLFLOW_TRACKING_DIR = MODELS_DIR / "mlruns"
MLFLOW_ARTIFACTS_DIR = MODELS_DIR / "mlartifacts"

# Ensure tracking directory exists
MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Default to local file URI, but allow remote server via MLFLOW_TRACKING_URI env var
DEFAULT_MLFLOW_TRACKING_URI = MLFLOW_TRACKING_DIR.as_uri()
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)

# Standard experiment names across tracks
MLFLOW_EXPERIMENT_3A = "smart-shelf-3a-spoilage-classifier"
MLFLOW_EXPERIMENT_3B = "smart-shelf-3b-revenue-response"
MLFLOW_EXPERIMENT_3C = "smart-shelf-3c-discount-optimizer"

EXPERIMENT_NAMES = {
    "3A": MLFLOW_EXPERIMENT_3A,
    "3B": MLFLOW_EXPERIMENT_3B,
    "3C": MLFLOW_EXPERIMENT_3C,
}

# Environment variable name for author attribution
SMART_SHELF_AUTHOR_ENV_VAR = "SMART_SHELF_AUTHOR"


