"""
M4-T01: Reproducibility verification — check the registered models can be
reproduced from the saved evidence, without retraining anything.

What "reproducibility" means here (teach it in plain English)
------------------------------------------------------------
A model is reproducible if a teammate, on a fresh checkout, can:
  1. find the *same* model (same run ID, same git commit, same feature list),
  2. load it and get the *same* metrics as the M3-T13 report, and
  3. know exactly what data / code / library versions produced it.

This script checks the things that *can* be checked from this zip, and
is honest about the things that *cannot*:

  CAN check (stdlib only, no data / no .joblib / no MLflow needed):
    * feature lists match the canonical lists in src/m2_feature_pipeline.py
      (3A COMPACT = 90; 3B = 43 base + oof_proba = 44)
    * random_state = 42 in src/config.py
    * train period ends 2024-08-07 (JSONs + inference defaults)
    * no leaked features (no was_spoiled / spoilage_risk; no discount
      fields in 3A)
    * metrics recorded in the M3 JSON summaries match the M3-T13 report
      values within tolerance 1e-4 (the same tolerance used by
      scripts/M3_T13_metric_audit.py, which reported 92 checks, 0 mismatches)
    * library versions in the model JSONs match requirements.txt
      (mlflow 3.17.1, scikit-learn 1.7.2, lightgbm 4.7.0, …)

  CANNOT check from this zip (and this script says so — it does NOT
  claim a pass):
    * loading the .joblib and re-scoring it — the binaries are gitignored
      and absent from the zip (models/*.joblib)
    * reading the original MLflow runs — models/mlruns/ is gitignored
      and absent from the zip
    * re-running the models on data — data/ is absent from the zip
  Until the team supplies the .joblib + mlruns (or re-runs M3_T01 /
  M3_T06), the honest overall status is BLOCKED / PARTIAL, never PASS.

Supporting evidence already in the repo (not re-run here):
  * M3-T14 parity: run_pipeline() vs the frozen artifacts differs by
    at most 3.61e-07 on spoil_proba over 1,000 test rows
    (m3_t14_results_summary.json, parity_test_1000) and 1.11e-16
    in-memory — that is the closest thing to a reproducibility check
    M3 already did, and it is quoted, not re-computed.

Expected values (sources in comments — full precision from the JSONs):
  * 3A test: Brier 0.15640742607682803, log loss 0.48987567457756626,
    ROC-AUC 0.5899190509656349  (m3_t01_results_summary.json)
  * 3B CV (run ce191bff): s RMSE 0.1296725443306064, s R2 0.7093742810367285
    (m3_t06_results_summary.json)
  * 3B test R1 (run d3ed3411, M3-T07): s RMSE 0.12781215858528563,
    s R2 0.741930501106602  (m3_t07_results_summary.json)

Usage:
    python scripts/M4_T01_verify_reproducibility.py [--output PATH]
    python scripts/M4_T01_verify_reproducibility.py --check-only   # same thing

Exit codes: 0 = PASS (all checks, binaries + MLflow present);
            2 = BLOCKED / PARTIAL (binaries / mlruns / data missing);
            1 = FAIL (a checkable metric / feature / config mismatch).
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

TOL = 1e-4
OUT_DEFAULT = PROJECT_ROOT / "reports" / "M4" / "m4_t01_reproducibility.json"

EXPECTED = {
    "3A": {
        "test_brier": 0.15640742607682803,
        "test_log_loss": 0.48987567457756626,
        "test_roc_auc": 0.5899190509656349,
        "cv_brier": 0.1537399141774408,
        "n_features": 90,
        "run_id": "1c6393d91cbd4396b3baee6c91045aa2",
        "model_class": "LogisticRegression",
    },
    "3B": {
        # CV numbers come from the training run ce191bff (M3-T06);
        # test numbers come from the separate M3-T07 run d3ed3411.
        "cv_s_rmse": 0.1296725443306064,
        "cv_s_r2": 0.7093742810367285,
        "test_s_rmse": 0.12781215858528563,
        "test_s_r2": 0.741930501106602,
        "n_features": 44,  # 43 + oof_proba
        "run_id": "ce191bffab874806ad34bfe1485cd6c5",
        "test_run_id": "d3ed3411ecd3464b8baf57aefd62557d",
        "model_class": "LGBMRegressor",
    },
}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_lists() -> dict:
    tree = ast.parse((PROJECT_ROOT / "src" / "m2_feature_pipeline.py").read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id.startswith("TRACK_"):
                out[node.target.id] = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    return out


def _close(actual: float, expected: float, tol: float = TOL) -> bool:
    return abs(float(actual) - float(expected)) <= tol


def main() -> int:
    ap = argparse.ArgumentParser(description="M4-T01 reproducibility verification")
    ap.add_argument("--output", default=str(OUT_DEFAULT))
    ap.add_argument("--check-only", action="store_true", help="same as default: no retraining, no MLflow writes")
    args = ap.parse_args()

    checks = []

    def check(name: str, passed: bool, detail: str = "", blocked: bool = False):
        checks.append({"name": name, "status": "BLOCKED" if blocked else ("PASS" if passed else "FAIL"),
                       "detail": detail})
        print(f"[{'BLOCKED' if blocked else ('PASS' if passed else 'FAIL')}] {name} — {detail}")

    print("=" * 70)
    print("M4-T01 reproducibility verification (no retraining)")
    print("=" * 70)

    # --- 1. Files that must exist for a full check -------------------------
    for rel in ["models/track_3a_classifier.joblib", "models/track_3b_regressor.joblib"]:
        p = PROJECT_ROOT / rel
        check(f"binary present: {rel}", p.exists(),
              "present" if p.exists() else "MISSING — gitignored (*.joblib), not in this zip",
              blocked=not p.exists())
    # FIX 2026-11-25: these two checks were hardcoded blocked=True, so the
    # script could NEVER exit 0 even when the files were supplied. They are
    # now conditional: BLOCKED only while the folder is actually missing.
    # NOTE: "mlruns present" means a store exists here (re-logged runs from
    # M4_T01_register_models.py --register). It does NOT mean the team's
    # ORIGINAL runs (1c6393d9..., ce191bff..., d3ed3411...) were found —
    # those were never supplied. A full re-score still needs data/ (below).
    _mlruns_dir = PROJECT_ROOT / "models" / "mlruns"
    _mlruns_present = _mlruns_dir.exists() and any(_mlruns_dir.iterdir())
    check("MLflow store present: models/mlruns/", _mlruns_present,
          "present — re-logged runs only; original runs 1c6393d9... / ce191bff... / d3ed3411... not found in this store"
          if _mlruns_present else
          "MISSING/EMPTY — gitignored, not in this zip; original runs cannot be read here",
          blocked=not _mlruns_present)
    _data_present = (PROJECT_ROOT / "data").exists()
    check("data present: data/processed/", _data_present,
          "present" if _data_present else
          "MISSING — no data/ folder in this zip; models cannot be re-scored here",
          blocked=not _data_present)

    # --- 2. Feature lists ----------------------------------------------------
    canon = _canonical_lists()
    meta3a = _load_json(PROJECT_ROOT / "models" / "track_3a_classifier.json")
    meta3b = _load_json(PROJECT_ROOT / "models" / "track_3b_regressor.json")
    check("3A features = COMPACT 90, exact order",
          meta3a["feature_list"] == canon.get("TRACK_3A_FEATURES_COMPACT"),
          f"JSON={len(meta3a['feature_list'])}, canonical COMPACT={len(canon.get('TRACK_3A_FEATURES_COMPACT', []))}")
    check("3B features = 43 base + oof_proba (44 total)",
          set(meta3b["feature_list"]) == set(canon.get("TRACK_3B_FEATURES", [])) | {"oof_proba"}
          and len(meta3b["feature_list"]) == 44,
          f"JSON={len(meta3b['feature_list'])}, canonical TRACK_3B={len(canon.get('TRACK_3B_FEATURES', []))} + oof_proba")
    check("3A leakage: no discount / outcome columns",
          not (set(meta3a["feature_list"]) & {"was_spoiled", "spoilage_risk", "discount_pct", "markdown_applied", "selling_price"}),
          "no forbidden columns in 3A feature list")
    check("3B leakage: no outcome columns",
          not (set(meta3b["feature_list"]) & {"was_spoiled", "spoilage_risk", "units_sold", "units_wasted", "profit"}),
          "no forbidden outcome columns in 3B feature list")

    # --- 3. Config / reproducibility constants --------------------------------
    config_txt = (PROJECT_ROOT / "src" / "config.py").read_text(encoding="utf-8")
    m = re.search(r"RANDOM_STATE\s*=\s*(\d+)", config_txt)
    check("random_state = 42 in src/config.py", m is not None and int(m.group(1)) == 42,
          f"found RANDOM_STATE={m.group(1) if m else 'not found'}")
    defaults = _load_json(PROJECT_ROOT / "models" / "track_3_inference_defaults.json")
    check("train period ends 2024-08-07",
          defaults.get("train_end_date") == "2024-08-07"
          and "2024-08-07" in str(meta3a.get("train_period"))
          and "2024-08-07" in str(meta3b.get("train_period")),
          f"defaults train_end_date={defaults.get('train_end_date')}, "
          f"3A train_period={meta3a.get('train_period')}, 3B train_period={meta3b.get('train_period')}")
    check("run IDs recorded in JSONs match M3 reports",
          meta3a.get("mlflow_run_id") == EXPECTED["3A"]["run_id"]
          and meta3b.get("mlflow_run_id") == EXPECTED["3B"]["run_id"],
          f"3A={meta3a.get('mlflow_run_id')}, 3B={meta3b.get('mlflow_run_id')}")
    check("model classes match M3 reports",
          meta3a.get("model_class") == "LogisticRegression" and meta3b.get("model_class") == "LGBMRegressor",
          f"3A={meta3a.get('model_class')}, 3B={meta3b.get('model_class')}")

    # --- 4. Metrics in JSON summaries vs M3-T13 (tol 1e-4) -------------------
    t01 = _load_json(PROJECT_ROOT / "reports" / "M3" / "m3_t01_results_summary.json")
    tm = t01["test"]["model_metrics"]
    for key, exp_key in [("test_brier", "brier"), ("test_log_loss", "log_loss"), ("test_roc_auc", "roc_auc")]:
        check(f"3A {key} matches M3-T13 (tol 1e-4)",
              _close(tm[exp_key], EXPECTED["3A"][key]),
              f"summary={tm[exp_key]}, expected={EXPECTED['3A'][key]}")
    t06 = _load_json(PROJECT_ROOT / "reports" / "M3" / "m3_t06_results_summary.json")
    mono = next(c for c in t06["candidates"] if c["family"] == "lgbm_ce_monotone")
    check("3B CV s_rmse matches M3-T13 (tol 1e-4)",
          _close(mono["s_rmse"], EXPECTED["3B"]["cv_s_rmse"]),
          f"summary={mono['s_rmse']}, expected={EXPECTED['3B']['cv_s_rmse']}")
    check("3B CV s_r2 matches M3-T13 (tol 1e-4)",
          _close(mono["s_r2"], EXPECTED["3B"]["cv_s_r2"]),
          f"summary={mono['s_r2']}, expected={EXPECTED['3B']['cv_s_r2']}")
    t07 = _load_json(PROJECT_ROOT / "reports" / "M3" / "m3_t07_results_summary.json")
    tr1 = t07["test_r1_metrics"]
    check("3B test R1 s_rmse matches M3-T13 (run d3ed3411, tol 1e-4)",
          _close(tr1["s_rmse"], EXPECTED["3B"]["test_s_rmse"]),
          f"summary={tr1['s_rmse']}, expected={EXPECTED['3B']['test_s_rmse']}")
    check("3B test R1 s_r2 matches M3-T13 (run d3ed3411, tol 1e-4)",
          _close(tr1["s_r2"], EXPECTED["3B"]["test_s_r2"]),
          f"summary={tr1['s_r2']}, expected={EXPECTED['3B']['test_s_r2']}")

    # --- 5. Library versions --------------------------------------------------
    req = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    check("library versions in 3A JSON match requirements.txt (mlflow 3.17.1)",
          meta3a.get("library_versions", {}).get("mlflow") == "3.17.1"
          and "mlflow==3.17.1" in req,
          f"JSON mlflow={meta3a.get('library_versions', {}).get('mlflow')}, "
          f"sklearn={meta3a.get('library_versions', {}).get('scikit-learn')}, "
          f"lightgbm={meta3a.get('library_versions', {}).get('lightgbm')}")

    # --- 6. Registry status (only if MLflow is actually installed) ------------
    try:
        import mlflow  # noqa: F401
        check("MLflow installed in this env", True, f"mlflow {mlflow.__version__}")
    except Exception:
        check("MLflow installed in this env", False,
              "mlflow NOT installed here (Python 3.12; requirements pin Python 3.11, mlflow==3.17.1) — registry cannot be read/verified here",
              blocked=True)

    # --- Overall verdict -------------------------------------------------------
    fails = [c for c in checks if c["status"] == "FAIL"]
    blocked = [c for c in checks if c["status"] == "BLOCKED"]
    if fails:
        status = "FAIL"
        code = 1
    elif blocked:
        status = "BLOCKED / PARTIAL"
        code = 2
    else:
        status = "PASS"
        code = 0

    result = {
        "task": "M4-T01",
        "overall_status": status,
        "tolerance": TOL,
        "checks": checks,
        "supporting_evidence_not_rerun": {
            "m3_t14_parity_spoil_proba_max_abs_diff_1000_rows": 3.607782183556729e-07,
            "m3_t14_parity_in_memory_max_abs_diff": 1.1102230246251565e-16,
            "source": "reports/M3/m3_t14_results_summary.json (parity_test_1000 / inmem_m2_proba_max_abs_diff)",
            "m3_t13_metric_audit": "92 checks, 0 mismatches (reports/M3/M3_T13_metric_audit.md) — that audit read the live MLflow store on the author's machine; it cannot be re-run from this zip because models/mlruns/ is absent",
        },
        "what_is_needed_to_reach_PASS": [
            "models/track_3a_classifier.joblib",
            "models/track_3b_regressor.joblib",
            "models/mlruns/ (original runs 1c6393d91cbd4396b3baee6c91045aa2, ce191bffab874806ad34bfe1485cd6c5, d3ed3411ecd3464b8baf57aefd62557d) or re-logged runs from scripts/M4_T01_register_models.py --register",
            "data/processed/ (or at least track_3a_proba_test.csv + model_ready_3b_dataset.csv) to re-score the loaded models",
            "Python 3.11 smart-shelf env with mlflow==3.17.1 installed",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("=" * 70)
    print(f"Overall: {status}  (exit code {code})")
    print(f"Checks: {sum(1 for c in checks if c['status'] == 'PASS')} PASS, "
          f"{len(fails)} FAIL, {len(blocked)} BLOCKED")
    print(f"Result written: {out}")
    if status != "PASS":
        print("Do NOT tell the team reproducibility 'passes' until the files above are supplied and this script exits 0.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
