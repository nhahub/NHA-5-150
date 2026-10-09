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

# Models and checkpoints
MODELS_DIR = PROJECT_ROOT / "models"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

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
