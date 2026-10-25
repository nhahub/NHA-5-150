"""
scripts/M2_T13_test_helpers.py - Automated validation suite for Milestone 3 MLflow helpers.

Tests all helpers added in M2-T13:
1. save_model_with_metadata and load_model_with_metadata round-trip, feature-order preservation,
   overwrite protection, missing JSON error, and missing feature list error.
2. log_calibration_curve artifact logging and Brier score return.
3. log_confusion_matrix artifact logging and matrix return.
4. log_shap_summary artifact logging and feature importance dict return (one entry per feature).
5. log_oof_predictions artifact logging and schema validation.
6. MlflowClient artifact retrievability for all logged helper artifacts.

Uses tiny synthetic data only and writes test models to system temporary directories.
Every created MLflow run is tagged purpose=setup_test.
"""

from pathlib import Path
import sys
import tempfile
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from mlflow.tracking import MlflowClient

# Ensure project root is in sys.path
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from src.mlflow_utils import (
    start_tracked_run,
    save_model_with_metadata,
    load_model_with_metadata,
    log_calibration_curve,
    log_confusion_matrix,
    log_shap_summary,
    log_oof_predictions,
)


def main() -> None:
    print("======================================================================")
    print("M2-T13: Automated Testing of Milestone 3 MLflow Utilities")
    print("======================================================================")

    # 1. Create tiny synthetic dataset
    rng = np.random.RandomState(42)
    feat_names = ["feat_demand", "feat_temperature", "feat_price", "feat_stock", "feat_shelf_days"]
    X_synth = pd.DataFrame(rng.randn(60, 5), columns=feat_names)
    y_synth = np.array([0, 1] * 30)

    # Train a minimal tree model (LightGBM)
    model = lgb.LGBMClassifier(
        n_estimators=10,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_synth, y_synth)
    proba = model.predict_proba(X_synth)[:, 1]
    preds = model.predict(X_synth)

    print("\n[Step 1] Synthetic model trained successfully.")

    # 2. Run tracked test run
    print("\n[Step 2] Testing MLflow logging helpers inside a tracked run...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_models_dir = Path(tmp_dir)

        with start_tracked_run(
            track="3A",
            model_type="LGBM_Synthetic_Test",
            feature_set_name="synthetic_test_features",
            shelf_life_version="expiry_remaining_ratio",
            features=feat_names,
            purpose="setup_test",
            run_name="M2_T13_helper_verification_run",
            train_rows=60,
            test_rows=60,
        ) as run:
            run_id = run.info.run_id
            print(f" -> Active setup test run initialized: {run_id}")

            # Test A: log_calibration_curve
            brier = log_calibration_curve(y_synth, proba, n_bins=5, name="test_reliability_curve")
            assert isinstance(brier, float) and 0.0 <= brier <= 1.0, f"Invalid Brier score: {brier}"
            print(f"    [PASS] log_calibration_curve returned Brier = {brier:.4f}")

            # Test B: log_confusion_matrix
            cm = log_confusion_matrix(y_synth, preds, name="test_confusion_matrix")
            assert cm.shape == (2, 2), f"Expected (2, 2) confusion matrix, got {cm.shape}"
            assert cm.sum() == len(y_synth), "Confusion matrix sum mismatch"
            print("    [PASS] log_confusion_matrix returned 2x2 matrix")

            # Test C: log_shap_summary
            shap_dict = log_shap_summary(model, X_synth, max_rows=50, name="test_shap_summary")
            assert isinstance(shap_dict, dict), "SHAP summary must return a dictionary"
            assert len(shap_dict) == len(feat_names), (
                f"SHAP dictionary must have one entry per feature! Got {len(shap_dict)} vs {len(feat_names)}"
            )
            for f in feat_names:
                assert f in shap_dict, f"Feature '{f}' missing from SHAP dictionary"
                assert isinstance(shap_dict[f], float), f"SHAP value for '{f}' must be float"
            print(f"    [PASS] log_shap_summary returned dict with {len(shap_dict)} feature importances")

            # Test D: log_oof_predictions
            oof_df = pd.DataFrame({
                "record_id": [f"SYNTH_{i:04d}" for i in range(len(y_synth))],
                "oof_proba": proba,
            })
            oof_artifact = log_oof_predictions(oof_df, name="test_track_3a_oof")
            assert oof_artifact == "test_track_3a_oof.csv"
            print("    [PASS] log_oof_predictions logged CSV artifact successfully")

            # Test D2: Schema error handling for log_oof_predictions
            try:
                log_oof_predictions(pd.DataFrame({"invalid_col": [1, 2]}))
                assert False, "log_oof_predictions should raise ValueError when record_id is missing"
            except ValueError:
                pass
            print("    [PASS] log_oof_predictions correctly raised ValueError on missing columns")

            # Test E: save_model_with_metadata
            model_path, meta_path = save_model_with_metadata(
                model=model,
                track="3A",
                feature_list=feat_names,
                shelf_life_version="expiry_remaining_ratio",
                train_period="2023-01-01 to 2024-08-07",
                models_dir=tmp_models_dir,
                name="test_model_artifact",
                overwrite=False,
                log_as_artifact=True,
            )
            assert model_path.exists(), f"Model file missing: {model_path}"
            assert meta_path.exists(), f"Metadata file missing: {meta_path}"
            print("    [PASS] save_model_with_metadata wrote model and JSON beside it")

            # Test E2: Overwrite protection
            try:
                save_model_with_metadata(
                    model=model,
                    track="3A",
                    feature_list=feat_names,
                    shelf_life_version="expiry_remaining_ratio",
                    train_period="2023-01-01 to 2024-08-07",
                    models_dir=tmp_models_dir,
                    name="test_model_artifact",
                    overwrite=False,
                )
                assert False, "save_model_with_metadata should raise FileExistsError when overwrite=False"
            except FileExistsError:
                pass
            print("    [PASS] save_model_with_metadata overwrite protection verified (FileExistsError raised)")

            # Test E3: Overwrite allowed when overwrite=True
            save_model_with_metadata(
                model=model,
                track="3A",
                feature_list=feat_names,
                shelf_life_version="expiry_remaining_ratio",
                train_period="2023-01-01 to 2024-08-07",
                models_dir=tmp_models_dir,
                name="test_model_artifact",
                overwrite=True,
            )
            print("    [PASS] save_model_with_metadata overwrite=True succeeded")

            # Test F: load_model_with_metadata round-trip
            loaded_model, metadata = load_model_with_metadata(model_path)
            assert metadata["feature_list"] == feat_names, "Feature order mismatch on reload!"
            assert metadata["track"] == "3A"
            assert metadata["shelf_life_version"] == "expiry_remaining_ratio"
            assert metadata["train_period"] == "2023-01-01 to 2024-08-07"
            assert metadata["mlflow_run_id"] == run_id
            assert "scikit-learn" in metadata["library_versions"]
            assert "pandas" in metadata["library_versions"]
            assert "numpy" in metadata["library_versions"]
            assert "mlflow" in metadata["library_versions"]

            # Confirm predictions match
            np.testing.assert_allclose(
                loaded_model.predict_proba(X_synth),
                model.predict_proba(X_synth),
                err_msg="Loaded model predictions differ from original model!",
            )
            print("    [PASS] load_model_with_metadata round-trip verified (identical predictions & feature order)")

            # Test F2: missing JSON error
            corrupt_dir = tmp_models_dir / "corrupt_test"
            corrupt_dir.mkdir(parents=True, exist_ok=True)
            corrupt_model_path = corrupt_dir / "orphan_model.joblib"
            import joblib
            joblib.dump(model, corrupt_model_path)
            try:
                load_model_with_metadata(corrupt_model_path)
                assert False, "load_model_with_metadata should raise FileNotFoundError when JSON missing"
            except FileNotFoundError:
                pass
            print("    [PASS] load_model_with_metadata missing JSON raises FileNotFoundError")

            # Test F3: missing feature list error
            corrupt_json_path = corrupt_dir / "orphan_model.json"
            with open(corrupt_json_path, "w", encoding="utf-8") as f:
                json.dump({"track": "3A", "shelf_life_version": "expiry_remaining_ratio"}, f)
            try:
                load_model_with_metadata(corrupt_model_path)
                assert False, "load_model_with_metadata should raise KeyError when feature list absent"
            except KeyError:
                pass
            print("    [PASS] load_model_with_metadata absent feature list raises KeyError")

        # 3. Verify artifact retrievability via MlflowClient
        print("\n[Step 3] Verifying artifact retrievability with MlflowClient...")
        client = MlflowClient()
        artifacts = client.list_artifacts(run_id)
        artifact_paths = {a.path for a in artifacts}
        expected_artifacts = {
            "test_reliability_curve.png",
            "test_confusion_matrix.png",
            "test_shap_summary.png",
            "test_track_3a_oof.csv",
            "test_model_artifact.joblib",
            "test_model_artifact.json",
        }
        for exp_art in expected_artifacts:
            assert exp_art in artifact_paths, f"Expected artifact '{exp_art}' not found in run artifacts: {artifact_paths}"
            print(f"    [RETRIEVED] {exp_art}")

    print("\n[Step 4] Temporary test directory safely cleaned up.")
    print("\n======================================================================")
    print("ALL M2-T13 HELPER TESTS PASSED CLEANLY!")
    print("======================================================================")


if __name__ == "__main__":
    main()
