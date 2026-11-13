"""
src/m3_t01_track3a.py - Track 3A Spoilage Risk Classifier Modeling & Calibration.

Implements time-blocked CV benchmarking, Optuna hyperparameter tuning across
Logistic Regression, Random Forest, XGBoost, and LightGBM on FULL (180) and
COMPACT (90) feature sets, leak-free calibration comparison (uncalibrated vs
Platt vs isotonic), final test evaluation, SHAP attribution, and support flagging.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb

from src.config import (
    CLEANED_DATASET_PATH,
    MODEL_READY_M2_DATASET_PATH,
    RANDOM_STATE,
    TRACK_3A_OOF_TRAIN_PATH,
)
from src.m2_feature_pipeline import (
    TRACK_3A_FEATURES_COMPACT,
    TRACK_3A_FEATURES_FULL,
    validate_feature_columns_exist,
)
from src.m3_oof import generate_oof_predictions

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURE_SETS: Dict[str, List[str]] = {
    "track_3a_full": list(TRACK_3A_FEATURES_FULL),
    "track_3a_compact": list(TRACK_3A_FEATURES_COMPACT),
}


def load_track_3a_splits() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads M2 dataset merged with transaction_date, split strictly into train and test."""
    m2_df = pd.read_csv(MODEL_READY_M2_DATASET_PATH)
    cl_df = pd.read_csv(
        CLEANED_DATASET_PATH,
        usecols=["record_id", "transaction_date", "product_name", "store_id"],
    )
    cl_df["transaction_date"] = pd.to_datetime(cl_df["transaction_date"])

    # Drop duplicate cols if already present in m2_df before merge
    extra_cols = [c for c in ["transaction_date", "product_name", "store_id"] if c not in m2_df.columns]
    merged = m2_df.merge(cl_df[["record_id"] + extra_cols], on="record_id", how="inner")

    train_df = (
        merged[merged["split"] == "train"]
        .sort_values(["transaction_date", "record_id"])
        .reset_index(drop=True)
    )
    test_df = (
        merged[merged["split"] == "test"]
        .sort_values(["transaction_date", "record_id"])
        .reset_index(drop=True)
    )

    assert len(train_df) == 80042, f"Expected 80,042 train rows, got {len(train_df)}"
    assert len(test_df) == 19958, f"Expected 19,958 test rows, got {len(test_df)}"
    validate_feature_columns_exist(TRACK_3A_FEATURES_FULL, train_df.columns)
    validate_feature_columns_exist(TRACK_3A_FEATURES_COMPACT, train_df.columns)
    return train_df, test_df


def compute_ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) with uniform-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (proba >= bins[i]) & (proba <= bins[i + 1])
        else:
            mask = (proba >= bins[i]) & (proba < bins[i + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / n) * abs(float(y_true[mask].mean()) - float(proba[mask].mean()))
    return float(ece)


def evaluate_probs(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """Computes headline calibration and discrimination metrics."""
    proba_clip = np.clip(proba, 1e-7, 1.0 - 1e-7)
    return {
        "brier": float(brier_score_loss(y_true, proba_clip)),
        "log_loss": float(log_loss(y_true, proba_clip)),
        "roc_auc": float(roc_auc_score(y_true, proba_clip)),
        "ece": compute_ece(y_true, proba_clip, n_bins=10),
    }


def evaluate_reference_baselines(train_df: pd.DataFrame, n_blocks: int = 5) -> Dict[str, Dict[str, float]]:
    """
    Computes reference lines on the exact 5-fold time-blocked training partition:
      (a) constant fold-mean base-rate predictor
      (b) spoilage_risk benchmark column
      (c) existing track_3a_oof_train.csv LightGBM baseline
    """
    dates_s = pd.to_datetime(train_df["transaction_date"])
    unique_dates = np.sort(np.unique(dates_s.values))
    date_blocks = np.array_split(unique_dates, n_blocks)
    y_train = train_df["was_spoiled"].values

    const_oof = np.zeros(len(train_df), dtype=float)
    for b in range(n_blocks):
        val_mask = np.isin(dates_s.values, date_blocks[b])
        const_oof[val_mask] = y_train[~val_mask].mean()

    oof_existing = pd.read_csv(TRACK_3A_OOF_TRAIN_PATH)
    aligned = train_df[["record_id", "was_spoiled", "spoilage_risk"]].merge(
        oof_existing[["record_id", "oof_proba"]], on="record_id", how="inner"
    )

    return {
        "constant_base_rate_cv": evaluate_probs(y_train, const_oof),
        "spoilage_risk_benchmark_train": evaluate_probs(
            train_df["was_spoiled"].values, train_df["spoilage_risk"].values
        ),
        "existing_lgbm_oof_baseline": evaluate_probs(
            aligned["was_spoiled"].values, aligned["oof_proba"].values
        ),
    }


def make_candidate_factory(
    model_family: str,
    params: Dict[str, Any],
    needs_imputation: bool = False,
) -> Callable[[], Any]:
    """Returns a fresh, un-fitted classifier factory for the given family and params."""
    family = model_family.lower()

    def _factory() -> Any:
        if family == "logreg":
            clf = LogisticRegression(
                C=params["C"],
                penalty="l2",
                solver="lbfgs",
                max_iter=500,
                random_state=RANDOM_STATE,
            )
            if needs_imputation:
                return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])
            return clf
        if family == "rf":
            clf = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            if needs_imputation:
                return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])
            return clf
        if family == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                min_child_weight=params["min_child_weight"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                reg_lambda=params["reg_lambda"],
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            )
        if family == "lightgbm":
            return lgb.LGBMClassifier(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                num_leaves=params["num_leaves"],
                min_child_samples=params["min_child_samples"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                reg_lambda=params["reg_lambda"],
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
        raise ValueError(f"Unknown model family: {model_family}")

    return _factory


def _sample_params(trial: optuna.Trial, model_family: str) -> Dict[str, Any]:
    """Defines modest hyperparameter search space per model family."""
    family = model_family.lower()
    if family == "logreg":
        return {"C": trial.suggest_float("C", 1e-4, 10.0, log=True)}
    if family == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 250, step=50),
            "max_depth": trial.suggest_int("max_depth", 6, 16, step=2),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 150),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.2, 0.4]),
        }
    if family == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
        }
    if family == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 250),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
        }
    raise ValueError(f"Unknown model family: {model_family}")


def tune_candidate_cv(
    model_family: str,
    feature_set_name: str,
    train_df: pd.DataFrame,
    n_trials: int = 8,
    n_blocks: int = 5,
) -> Dict[str, Any]:
    """Tunes a candidate model family on a feature set using 5-fold time-blocked CV."""
    features = FEATURE_SETS[feature_set_name]
    X_train = train_df[features]
    y_train = train_df["was_spoiled"].values
    dates_train = train_df["transaction_date"].values
    records_train = train_df["record_id"].values
    needs_imp = bool(X_train.isna().any().any())

    if model_family == "logreg":
        # 1D grid over C is cleaner and deterministic for logistic regression
        c_grid = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        best_res: Optional[Dict[str, Any]] = None
        for c_val in c_grid:
            params = {"C": c_val}
            factory = make_candidate_factory(model_family, params, needs_imputation=needs_imp)
            oof_df = generate_oof_predictions(
                model_factory=factory,
                X=X_train,
                y=y_train,
                dates=dates_train,
                record_ids=records_train,
                n_blocks=n_blocks,
                scheme="blocked",
                feature_set_name=feature_set_name,
                model_type=model_family,
            )
            metrics = evaluate_probs(y_train, oof_df["oof_proba"].values)
            if best_res is None or metrics["log_loss"] < best_res["metrics"]["log_loss"]:
                best_res = {
                    "model_family": model_family,
                    "feature_set_name": feature_set_name,
                    "params": params,
                    "metrics": metrics,
                    "oof_df": oof_df,
                    "needs_imputation": needs_imp,
                }
        assert best_res is not None
        return best_res

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    trial_cache: Dict[int, Dict[str, Any]] = {}

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, model_family)
        factory = make_candidate_factory(model_family, params, needs_imputation=needs_imp)
        oof_df = generate_oof_predictions(
            model_factory=factory,
            X=X_train,
            y=y_train,
            dates=dates_train,
            record_ids=records_train,
            n_blocks=n_blocks,
            scheme="blocked",
            feature_set_name=feature_set_name,
            model_type=model_family,
        )
        metrics = evaluate_probs(y_train, oof_df["oof_proba"].values)
        trial_cache[trial.number] = {"params": params, "metrics": metrics, "oof_df": oof_df}
        return metrics["log_loss"]

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = trial_cache[study.best_trial.number]
    return {
        "model_family": model_family,
        "feature_set_name": feature_set_name,
        "params": best["params"],
        "metrics": best["metrics"],
        "oof_df": best["oof_df"],
        "needs_imputation": needs_imp,
    }


def evaluate_calibration_cv(
    base_factory: Callable[[], Any],
    feature_set_name: str,
    model_family: str,
    train_df: pd.DataFrame,
    method: str,
    n_blocks: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """
    Evaluates uncalibrated, Platt ('sigmoid'), or 'isotonic' calibration strictly
    inside the 5-fold time-blocked CV (no cross-fold leakage).
    """
    features = FEATURE_SETS[feature_set_name]
    X_train = train_df[features]
    y_train = train_df["was_spoiled"].values
    dates_train = train_df["transaction_date"].values
    records_train = train_df["record_id"].values

    if method == "uncalibrated":
        cal_factory = base_factory
    elif method in ("sigmoid", "isotonic"):
        def cal_factory() -> CalibratedClassifierCV:
            return CalibratedClassifierCV(estimator=base_factory(), method=method, cv=3)
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    oof_df = generate_oof_predictions(
        model_factory=cal_factory,
        X=X_train,
        y=y_train,
        dates=dates_train,
        record_ids=records_train,
        n_blocks=n_blocks,
        scheme="blocked",
        feature_set_name=feature_set_name,
        model_type=f"{model_family}_{method}",
    )
    metrics = evaluate_probs(y_train, oof_df["oof_proba"].values)

    prob_true, prob_pred = calibration_curve(
        y_train, oof_df["oof_proba"].values, n_bins=10, strategy="uniform"
    )
    rel_df = pd.DataFrame({"mean_predicted_prob": prob_pred, "observed_spoilage_rate": prob_true})
    return oof_df, metrics, rel_df


def compute_threshold_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: Tuple[float, ...] = (0.20, 0.30, 0.50),
) -> List[Dict[str, Any]]:
    """Computes precision, recall, F1, positive-rate, and confusion matrix at stated thresholds."""
    rows: List[Dict[str, Any]] = []
    for thr in thresholds:
        preds = (proba >= thr).astype(int)
        cm = confusion_matrix(y_true, preds)
        rows.append({
            "threshold": thr,
            "pred_pos_rate": float(preds.mean()),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        })
    return rows

