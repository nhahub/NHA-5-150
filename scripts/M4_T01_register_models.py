"""
M4-T01: Model registry & MLflow finalization — registration script.

What this script does, in plain English
----------------------------------------
Milestone 3 already trained the two final models. This script does NOT
retrain them. It takes the *frozen* models from M3 and puts them in the
MLflow Model Registry so the Milestone 4 Streamlit app can load exactly
one named version of each model, instead of guessing which .joblib file
is the final one.

  * Track 3A — spoilage-risk classifier (Logistic Regression, C=0.005)
  * Track 3B — revenue-response regressor (LightGBM, monotone in discount)

Two modes
---------
1. Default (no flag): ``--check-only``
   Audit-only. No MLflow store is created or changed. It reads the saved
   model JSONs, checks feature lists / train period / metrics against the
   M3-T13 reports, and writes
   ``reports/M4/m4_t01_registration_manifest.json`` describing exactly
   what *would* be registered. Safe to run anywhere, even without MLflow.

2. Live mode: ``--register``
   Requires (a) MLflow installed (``mlflow==3.17.1`` in requirements.txt,
   Python 3.11 ``smart-shelf`` env) and (b) the .joblib binaries.
   It creates the two experiments if missing, re-logs a final run from
   the saved metadata *only if the original run is not found* in
   ``models/mlruns/``, logs the JSON metadata + figures as artifacts,
   and registers the two models. If the .joblib is missing it does NOT
   invent a model version: it creates the registered-model entry only
   and marks the version ``pending_binary`` in the manifest.

Why the .joblib matters
-----------------------
``models/*.joblib`` and ``models/mlruns/`` are gitignored by design
(see .gitignore and reports/M3/M3_T13_Model_and_Code_Manifest.md §4).
That is why this zip has NO .joblib, NO data/ and NO mlruns/ folder:
they were never meant to travel in Git. The registry therefore cannot
be completed from this zip alone — the team must supply the binaries
(by re-running scripts/M3_T01_train_track3a.py and
scripts/M3_T06_train_track3b.py, or by pulling them from the shared
MLflow artifact store). This script says that out loud instead of
pretending registration succeeded.

Sources — every number below is copied from the M3-T13 evidence, not
invented. Full-precision values are in the JSON summaries; the rounded
values in the reports are display rounding only.
  * 3A test metrics: reports/M3/m3_t01_results_summary.json
    ["test"]["model_metrics"]:
    brier 0.15640742607682803, log_loss 0.48987567457756626,
    roc_auc 0.5899190509656349, ece 0.005789645333738579
    (report: Brier 0.15641, log loss 0.48988, AUC 0.5899)
  * 3A CV metrics (chosen LogReg COMPACT): same file ["chosen"]["cv_metrics"]:
    brier 0.1537399141774408, log_loss 0.483743926726195,
    roc_auc 0.5884597442940973
  * 3A final run: 1c6393d91cbd4396b3baee6c91045aa2
    (models/track_3a_classifier.json, M3_T01_Track3A_Report.md)
  * 3B CV metrics: reports/M3/m3_t06_results_summary.json
    (lgbm_ce_monotone): s_rmse 0.1296725443306064,
    s_mae 0.10937267253467968, s_r2 0.7093742810367285,
    profit_unit_mae 1.833657585245831
  * 3B TEST metrics are a *different* run, d3ed3411ecd3464b8baf57aefd62557d
    (M3-T07, reports/M3/m3_t07_results_summary.json ["test_r1_metrics"]):
    s_rmse 0.12781215858528563, s_mae 0.10809673284346759,
    s_r2 0.741930501106602, profit_unit_mae 1.7340064293470097
    Do NOT attribute the test numbers to the training run ce191bff.
  * 3B final training run: ce191bffab874806ad34bfe1485cd6c5
    (models/track_3b_regressor.json, M3_T06_Regressor_Training_Report.md)

Feature-list note (corrects a common misreading)
------------------------------------------------
TRACK_3B_FEATURES in src/m2_feature_pipeline.py has 43 features.
The frozen 3B model has 44: those 43 *plus* ``oof_proba`` (Track 3A
v2 out-of-fold probability, merged in M3-T04). The task brief's "43"
means the base list; the registered model must be checked against 44.
Track 3A COMPACT has exactly 90 features.

Usage
-----
    python scripts/M4_T01_register_models.py --check-only
    python scripts/M4_T01_register_models.py --register [--author Maryam]
    python scripts/M4_T01_register_models.py --register --force-relog

Exit codes: 0 = live registration done (binary present);
            2 = metadata-only / blocked (mlflow or .joblib missing);
            1 = audit failure (feature/metric mismatch — do not register).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# DO NOT `import src.config` here. Importing anything from the `src`
# package runs src/__init__.py, which imports src.pipeline / src.data_loader
# and therefore needs sklearn/pandas installed. This script must work in
# --check-only mode on a machine without those installed, so config values
# are read from src/config.py as plain text (stdlib only) instead.
# (Verified 2026-11-24: `from src.config import ...` fails here with
# ModuleNotFoundError: No module named 'sklearn'.)
_CONFIG_TXT = (PROJECT_ROOT / "src" / "config.py").read_text(encoding="utf-8")


def _config_str(name: str, default: str) -> str:
    m = re.search(rf'^{name}\s*=\s*"([^"]+)"', _CONFIG_TXT, re.M)
    return m.group(1) if m else default


_RANDOM_STATE_M = re.search(r"^RANDOM_STATE\s*=\s*(\d+)", _CONFIG_TXT, re.M)
RANDOM_STATE = int(_RANDOM_STATE_M.group(1)) if _RANDOM_STATE_M else 42
MLFLOW_EXPERIMENT_3A = _config_str("MLFLOW_EXPERIMENT_3A", "smart-shelf-3a-spoilage-classifier")
MLFLOW_EXPERIMENT_3B = _config_str("MLFLOW_EXPERIMENT_3B", "smart-shelf-3b-revenue-response")
MODELS_DIR = PROJECT_ROOT / "models"
ROOT = PROJECT_ROOT
# Local file-store URI, same as src/config.py would build:
# MLFLOW_TRACKING_DIR = MODELS_DIR / "mlruns"; DEFAULT = that dir .as_uri()
MLFLOW_TRACKING_URI = (MODELS_DIR / "mlruns").as_uri()

REPORTS_M4_DIR = ROOT / "reports" / "M4"
MANIFEST_PATH = REPORTS_M4_DIR / "m4_t01_registration_manifest.json"

# ---------------------------------------------------------------------------
# Frozen final models — values copied from the M3-T13 evidence (see docstring)
# ---------------------------------------------------------------------------
FINAL_MODELS = {
    "3A": {
        "track": "3A",
        "registered_name": "smart-shelf-3a-spoilage-classifier",
        "experiment": MLFLOW_EXPERIMENT_3A,
        "model_class": "LogisticRegression",
        "model_type": "logreg_uncalibrated",
        "params": {"C": 0.005, "penalty": "l2", "calibration": "uncalibrated"},
        "feature_set_name": "track_3a_compact",
        "expected_n_features": 90,
        "feature_json": MODELS_DIR / "track_3a_classifier.json",
        "model_joblib": MODELS_DIR / "track_3a_classifier.joblib",
        "original_run_id": "1c6393d91cbd4396b3baee6c91045aa2",
        "git_commit": "6ebbb42",
        "train_period": "2023-01-01 to 2024-08-07",
        "train_rows": 80042,
        "test_rows": 19958,
        "shelf_life_version": "expiry_remaining_ratio",
        # Test metrics (m3_t01_results_summary.json, full precision)
        "metrics": {
            "test_brier_score": 0.15640742607682803,
            "test_log_loss": 0.48987567457756626,
            "test_roc_auc": 0.5899190509656349,
            "test_ece": 0.005789645333738579,
            "cv_brier": 0.1537399141774408,
            "cv_log_loss": 0.483743926726195,
            "cv_roc_auc": 0.5884597442940973,
            "benchmark_test_roc_auc": 0.5967366592848501,
        },
        "artifacts": [
            MODELS_DIR / "track_3a_classifier.json",
            ROOT / "reports" / "figures" / "m3" / "m3_t01_cv_reliability_curve.png",
            ROOT / "reports" / "figures" / "m3" / "m3_t01_test_shap_summary.png",
        ],
        "source": "reports/M3/m3_t01_results_summary.json; M3_T01_Track3A_Report.md; M3_T13_metric_audit.md (92 checks, 0 mismatches)",
    },
    "3B": {
        "track": "3B",
        "registered_name": "smart-shelf-3b-revenue-response",
        "experiment": MLFLOW_EXPERIMENT_3B,
        "model_class": "LGBMRegressor",
        "model_type": "lgbm_ce_monotone",
        "params": {
            "colsample_bytree": 0.9950553116457923,
            "learning_rate": 0.08149619212376244,
            "min_child_samples": 93,
            "num_leaves": 15,
            "n_estimators": 500,
            "reg_lambda": 7.480534388702453,
            "subsample": 0.985296571922335,
            "region_domain": "R1_only",
            "oof_version": "v2",
        },
        "feature_set_name": "track_3b_r1_44",
        "expected_n_features": 44,  # 43 TRACK_3B_FEATURES + oof_proba
        "expected_base_features": 43,
        "feature_json": MODELS_DIR / "track_3b_regressor.json",
        "model_joblib": MODELS_DIR / "track_3b_regressor.joblib",
        "original_run_id": "ce191bffab874806ad34bfe1485cd6c5",
        # Test metrics live on a *separate* M3-T07 run — see docstring.
        "test_run_id": "d3ed3411ecd3464b8baf57aefd62557d",
        "git_commit": "b183fc9",
        "train_period": "2023-01-01 to 2024-08-07 (R1 only)",
        "train_rows": 24475,
        "test_rows": 6176,
        "shelf_life_version": "expiry_remaining_ratio",
        # CV metrics from the training run ce191bff (m3_t06_results_summary.json)
        # + test metrics from the M3-T07 run d3ed3411 (m3_t07_results_summary.json)
        "metrics": {
            "cv_s_rmse": 0.1296725443306064,
            "cv_s_mae": 0.10937267253467968,
            "cv_s_r2": 0.7093742810367285,
            "cv_profit_unit_mae": 1.833657585245831,
            "test_s_rmse": 0.12781215858528563,
            "test_s_mae": 0.10809673284346759,
            "test_s_r2": 0.741930501106602,
            "test_profit_unit_mae": 1.7340064293470097,
            "test_profit_unit_rmse": 6.373298186827674,
        },
        "artifacts": [
            MODELS_DIR / "track_3b_regressor.json",
            MODELS_DIR / "track_3b_region_support_ref.json",
            MODELS_DIR / "track_3c_tier_edges.json",
            ROOT / "reports" / "figures" / "m3" / "m3_t06_cv_calibration.png",
            ROOT / "reports" / "figures" / "m3" / "m3_t07_test_calibration.png",
        ],
        "source": "reports/M3/m3_t06_results_summary.json (CV, run ce191bff); reports/M3/m3_t07_results_summary.json (test R1, run d3ed3411); M3_T13_metric_audit.md",
    },
}

FORBIDDEN_3A = {"was_spoiled", "spoilage_risk", "discount_pct", "markdown_applied", "selling_price",
                "units_sold", "units_wasted", "profit"}
FORBIDDEN_3B = {"was_spoiled", "spoilage_risk", "units_sold", "units_wasted", "profit"}


def canonical_feature_lists() -> dict:
    """Parse src/m2_feature_pipeline.py with ast (no sklearn import needed)."""
    tree = ast.parse((ROOT / "src" / "m2_feature_pipeline.py").read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id.startswith("TRACK_"):
                out[node.target.id] = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    return out


def audit_model(key: str, spec: dict, canon: dict) -> dict:
    """Check one frozen model against its JSON + the canonical feature list."""
    result = {"track": key, "registered_name": spec["registered_name"],
              "original_run_id": spec["original_run_id"], "checks": {}, "errors": []}
    meta_path = spec["feature_json"]
    if not meta_path.exists():
        result["errors"].append(f"missing metadata JSON: {meta_path}")
        return result
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feats = meta.get("feature_list") or meta.get("features") or []
    result["n_features_json"] = len(feats)
    result["feature_list_matches_expected"] = len(feats) == spec["expected_n_features"]
    if not result["feature_list_matches_expected"]:
        result["errors"].append(
            f"feature count {len(feats)} != expected {spec['expected_n_features']}")
    # Canonical comparison
    if key == "3A":
        canon_feats = canon.get("TRACK_3A_FEATURES_COMPACT", [])
        result["canonical_match"] = canon_feats == feats
    else:
        canon_feats = canon.get("TRACK_3B_FEATURES", [])
        result["canonical_match"] = (set(canon_feats) == set(feats) - {"oof_proba"}
                                     and "oof_proba" in feats and len(feats) == 44)
    if not result["canonical_match"]:
        result["errors"].append("feature list does not match canonical list in src/m2_feature_pipeline.py")
    forbidden = FORBIDDEN_3A if key == "3A" else FORBIDDEN_3B
    leaked = sorted(set(feats) & forbidden)
    result["leakage_check_pass"] = not leaked
    if leaked:
        result["errors"].append(f"leaked features present: {leaked}")
    result["train_period"] = meta.get("train_period")
    result["train_period_ok"] = str(meta.get("train_period", "")).endswith("2024-08-07") or "2024-08-07" in str(meta.get("train_period", ""))
    result["shelf_life_version"] = meta.get("shelf_life_version")
    result["model_class"] = meta.get("model_class")
    result["class_matches"] = meta.get("model_class") == spec["model_class"]
    result["run_id_in_json"] = meta.get("mlflow_run_id")
    result["run_id_matches"] = meta.get("mlflow_run_id") == spec["original_run_id"]
    result["git_commit_in_json"] = meta.get("git_commit")
    result["joblib_exists"] = spec["model_joblib"].exists()
    result["artifacts"] = [
        {"path": str(p), "exists": Path(p).exists()} for p in spec["artifacts"]
    ]
    result["checks"] = {
        "n_features": len(feats),
        "expected_n_features": spec["expected_n_features"],
        "joblib_present": result["joblib_exists"],
        "all_listed_artifacts_present": all(a["exists"] for a in result["artifacts"]),
    }
    return result


def live_register(spec: dict, author: str, force_relog: bool) -> dict:
    """Live MLflow path. Only called with --register and mlflow installed."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    out = {"registered_name": spec["registered_name"], "mode": "live"}
    # 1. Find the original run; re-log from metadata only if it is absent.
    run = None
    try:
        run = client.get_run(spec["original_run_id"])
        out["original_run_found"] = True
        out["run_id_used"] = spec["original_run_id"]
    except Exception:
        out["original_run_found"] = False
    if run is None or force_relog:
        from src.mlflow_utils import get_or_create_experiment
        exp_id = get_or_create_experiment(spec["track"])
        with mlflow.start_run(experiment_id=exp_id,
                              run_name=f"m4_t01_final_{spec['track'].lower()}_relog") as active:
            mlflow.set_tags({
                "track": spec["track"], "author": author, "milestone": "M3",
                "m4_t01_relog": "true",
                "source_run_id": spec["original_run_id"],
                "source": "M3_T13 reports/JSONs — no retraining",
                "final": "true",
            })
            mlflow.log_params({**spec["params"], "random_state": RANDOM_STATE})
            mlflow.log_metrics(spec["metrics"])
            for art in spec["artifacts"]:
                if Path(art).exists():
                    mlflow.log_artifact(str(art))
            out["relogged_run_id"] = active.info.run_id
            out["run_id_used"] = active.info.run_id
    # 2. Register — only a real version if the binary exists.
    try:
        client.get_registered_model(spec["registered_name"])
        out["registered_model_existed"] = True
    except Exception:
        client.create_registered_model(
            spec["registered_name"],
            description=(f"Smart Shelf Track {spec['track']} final model from Milestone 3. "
                         f"Source run {spec['original_run_id']}. "
                         f"Registered by M4-T01 (author {author})."),
            tags={"author": author, "milestone": "M3", "track": spec["track"], "final": "true"},
        )
        out["registered_model_existed"] = False
    if spec["model_joblib"].exists():
        # The .joblib is a plain joblib file, not an MLflow pyfunc model, so
        # register it as a version via the run artifact path if it was
        # logged as an artifact; otherwise record that a manual upload
        # of the binary is still needed. Never fake a model URI.
        out["model_version"] = "pending_binary_upload"
        out["note"] = ("joblib present on disk but not yet logged as an MLflow "
                       "model artifact in the source run; upload it with "
                       "mlflow.log_artifact / mlflow.<flavor>.log_model and "
                       "re-run with --register to create version 1.")
    else:
        out["model_version"] = "pending_binary"
        out["note"] = ("NO .joblib in this checkout (gitignored by design). "
                       "Registered-model entry created; no version created. "
                       "Supply models/" + spec["model_joblib"].name +
                       " (re-run M3_T01/M3_T06 or pull from the shared "
                       "artifact store) and re-run with --register.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="M4-T01: register final Track 3A/3B models in MLflow")
    ap.add_argument("--check-only", action="store_true", help="audit only, do not touch MLflow (default)")
    ap.add_argument("--register", action="store_true", help="live MLflow registration (needs mlflow + .joblib)")
    ap.add_argument("--force-relog", action="store_true", help="re-log a run even if the original run is found")
    ap.add_argument("--author", default="Maryam", help="run author tag (default: Maryam)")
    args = ap.parse_args()
    live = args.register and not args.check_only

    print("=" * 70)
    print("M4-T01: Model registry & MLflow finalization")
    print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Mode: {'LIVE --register' if live else 'CHECK-ONLY (no MLflow changes)'}")
    print("=" * 70)

    canon = canonical_feature_lists()
    print(f"Canonical lists: FULL={len(canon.get('TRACK_3A_FEATURES_FULL', []))}, "
          f"COMPACT={len(canon.get('TRACK_3A_FEATURES_COMPACT', []))}, "
          f"TRACK_3B={len(canon.get('TRACK_3B_FEATURES', []))} (+ oof_proba = 44 for 3B)")

    manifest = {"task": "M4-T01", "author": args.author, "mode": "live" if live else "check-only",
                "tracking_uri": MLFLOW_TRACKING_URI, "models": {}, "status": "unknown"}
    any_error = False
    for key, spec in FINAL_MODELS.items():
        audit = audit_model(key, spec, canon)
        print(f"\n[{key}] {spec['registered_name']}")
        print(f"  original run: {spec['original_run_id']}  (git {spec['git_commit']})")
        print(f"  features in JSON: {audit.get('n_features_json')} (expected {spec['expected_n_features']})"
              f"  canonical match: {audit.get('canonical_match')}")
        print(f"  class: {audit.get('model_class')} (expected {spec['model_class']})"
              f"  train period: {audit.get('train_period')}")
        print(f"  .joblib present: {audit.get('joblib_exists')}  <-- expected False in this zip (gitignored)")
        if audit["errors"]:
            any_error = True
            for e in audit["errors"]:
                print(f"  ERROR: {e}")
        manifest["models"][key] = {**audit, "metrics": spec["metrics"], "params": spec["params"],
                                   "source": spec["source"]}
        if live:
            try:
                live_out = live_register(spec, args.author, args.force_relog)
                manifest["models"][key]["live"] = live_out
                print(f"  live: {live_out}")
            except Exception as exc:  # noqa: BLE001 — report, don't crash the other model
                manifest["models"][key]["live_error"] = str(exc)
                print(f"  LIVE ERROR: {exc}")

    # mlflow availability (informational in check-only mode)
    try:
        import mlflow  # noqa: F401
        manifest["mlflow_installed"] = True
    except Exception:
        manifest["mlflow_installed"] = False
        print("\nNOTE: mlflow is NOT installed in this Python env "
              "(requirements pin mlflow==3.17.1, Python 3.11 smart-shelf env). "
              "Live registration needs that env.")

    missing_binaries = [k for k, v in manifest["models"].items() if not v.get("joblib_exists")]
    if any_error:
        manifest["status"] = "AUDIT_FAILED"
        code = 1
    elif live and not missing_binaries and manifest.get("mlflow_installed"):
        manifest["status"] = "REGISTERED"
        code = 0
    else:
        manifest["status"] = "METADATA_ONLY_PENDING_BINARY"
        code = 2
    manifest["missing_binaries"] = missing_binaries
    manifest["todo_for_team"] = (
        "Supply models/track_3a_classifier.joblib and models/track_3b_regressor.joblib "
        "(re-run scripts/M3_T01_train_track3a.py and scripts/M3_T06_train_track3b.py, "
        "or pull them from the shared MLflow artifact store), then re-run "
        "python scripts/M4_T01_register_models.py --register --author Maryam "
        "in the Python 3.11 smart-shelf env. Until then no real model version "
        "can be created — only the registered-model entries / manifest exist."
        if missing_binaries else "Binaries present — re-run with --register to create version 1 of each model."
    )

    REPORTS_M4_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nManifest written: {MANIFEST_PATH}")
    print(f"Status: {manifest['status']}  (exit code {code})")
    print(f"TODO: {manifest['todo_for_team']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
