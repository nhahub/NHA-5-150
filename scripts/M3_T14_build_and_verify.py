"""
scripts/M3_T14_build_and_verify.py - M3-T14 Build Defaults, Export Snapshot & Verify Pipeline.

Runs end-to-end verification of src/m3_t14_pipeline.py:
  1. Builds training-only defaults -> models/track_3_inference_defaults.json.
  2. Exports latest test date with >= 100 batches -> data/processed/inventory_snapshot.csv.
  3. Parity tests in 'real' mode on (a) 1,000 random test rows and (b) all snapshot rows
     against track_3a_proba_test.csv and optimizer_recs_with_confidence_test.csv.
  4. Hypothetical examples (R0 fresh, R1 short-expiry, Grade C long shelf life,
     all-optional-defaulted, unseen product x store) + handling_score counterfactual sanity check.
  5. Manager View table check, robustness (error) tests, and latency benchmarks.
  6. Logs summary metrics and artifacts to MLflow smart-shelf-3c-discount-optimizer.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    CLEANED_DATASET_PATH,
    INFERENCE_DEFAULTS_PATH,
    INVENTORY_SNAPSHOT_PATH,
    MLFLOW_EXPERIMENT_3C,
    MODEL_READY_M2_DATASET_PATH,
    OPTIMIZER_RECS_CONF_TEST_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TRACK_3A_MODEL_PATH,
    TRACK_3A_PROBA_TEST_PATH,
)
from src.m3_t01_track3a import score_batches
from src.m3_t14_pipeline import (
    build_and_save_defaults,
    export_inventory_snapshot,
    manager_view_table,
    run_pipeline,
    simulate,
    whatif_table,
)
from src.mlflow_utils import load_model_with_metadata, start_tracked_run


def _check_parity(
    raw_subset: pd.DataFrame,
    proba_ref: pd.DataFrame,
    recs_ref: pd.DataFrame,
    label: str,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    res = run_pipeline(raw_subset, mode="real")
    elapsed = time.perf_counter() - t0

    m_proba = res[["record_id", "spoil_proba"]].merge(
        proba_ref, on="record_id", how="inner"
    )
    assert len(m_proba) == len(raw_subset), f"{label}: proba merge length mismatch"
    proba_col = [c for c in proba_ref.columns if "proba" in c.lower()][0]
    proba_diff = np.abs(
        m_proba["spoil_proba"].to_numpy(dtype=float)
        - m_proba[proba_col].to_numpy(dtype=float)
    )

    r1_inf = res[res["region"] == "R1"]
    r0_inf = res[res["region"] == "R0"]
    assert len(r1_inf) + len(r0_inf) == len(res)
    if not r0_inf.empty:
        assert (r0_inf["status"] == "no_markdown_historical_practice").all()
        assert (r0_inf["confidence_level"] == "No recommendation").all()

    m_recs = r1_inf.merge(recs_ref, on="record_id", suffixes=("_inf", "_ref"), how="inner")
    assert len(m_recs) == len(r1_inf), f"{label}: R1 recs merge length mismatch ({len(m_recs)} vs {len(r1_inf)})"

    cat_matches: Dict[str, float] = {}
    for col in ("region", "status", "tier", "confidence_level", "confidence_reasons"):
        s_inf = m_recs[f"{col}_inf"].fillna("NONE").astype(str)
        s_ref = m_recs[f"{col}_ref"].fillna("NONE").astype(str)
        cat_matches[col] = float((s_inf == s_ref).mean())

    # Depth and profit parity on R1 rows
    r1_mask = m_recs["region_ref"] == "R1"
    depth_inf = m_recs.loc[r1_mask, "recommended_depth_inf"].to_numpy(dtype=float)
    depth_ref = m_recs.loc[r1_mask, "recommended_depth_ref"].to_numpy(dtype=float)
    depth_exact_rate = float(np.mean(np.isclose(depth_inf, depth_ref, atol=1e-9)))

    prof_inf = m_recs.loc[r1_mask, "predicted_profit_at_rec_inf"].to_numpy(dtype=float)
    prof_ref = m_recs.loc[r1_mask, "predicted_profit_at_rec_ref"].to_numpy(dtype=float)
    prof_diff = np.abs(prof_inf - prof_ref)

    return {
        "label": label,
        "n_rows": int(len(raw_subset)),
        "n_r1": int(len(r1_inf)),
        "n_r0": int(len(r0_inf)),
        "elapsed_sec": round(float(elapsed), 3),
        "spoil_proba_max_abs_diff": float(np.max(proba_diff)),
        "spoil_proba_mean_abs_diff": float(np.mean(proba_diff)),
        "spoil_proba_within_1e9": bool(np.max(proba_diff) <= 1e-9),
        "spoil_proba_within_1e5": bool(np.max(proba_diff) <= 1e-5),
        "categorical_match_rates": cat_matches,
        "r1_recommended_depth_exact_rate": depth_exact_rate,
        "r1_profit_max_abs_diff": float(np.max(prof_diff)) if len(prof_diff) else 0.0,
        "r1_profit_mean_abs_diff": float(np.mean(prof_diff)) if len(prof_diff) else 0.0,
    }


def main() -> None:
    print("[1/6] Building training-only inference defaults...")
    defaults = build_and_save_defaults(INFERENCE_DEFAULTS_PATH)
    print(f"Saved defaults to {INFERENCE_DEFAULTS_PATH} ({defaults['training_row_count']} training rows)")

    print("[2/6] Exporting inventory snapshot...")
    snap_df, snap_date = export_inventory_snapshot(INVENTORY_SNAPSHOT_PATH)
    print(f"Exported {len(snap_df)} rows for {snap_date} -> {INVENTORY_SNAPSHOT_PATH}")

    print("[3/6] Running parity tests in 'real' mode...")
    cl = pd.read_csv(CLEANED_DATASET_PATH)
    test_cl = cl[cl["split"] == "test"].copy()
    sample_1000 = test_cl.sample(n=1000, random_state=RANDOM_STATE).sort_values("record_id").reset_index(drop=True)

    proba_ref = pd.read_csv(TRACK_3A_PROBA_TEST_PATH)
    recs_ref = pd.read_csv(OPTIMIZER_RECS_CONF_TEST_PATH)

    parity_1000 = _check_parity(sample_1000, proba_ref, recs_ref, "test_random_1000")
    parity_snap = _check_parity(snap_df, proba_ref, recs_ref, f"snapshot_{snap_date}")
    print("Parity 1000:", json.dumps(parity_1000, indent=2))
    print("Parity Snapshot:", json.dumps(parity_snap, indent=2))

    # In-memory parity check avoiding 6-sig-fig CSV mismatch (scoring stored M2 test rows directly)
    m2_test = pd.read_csv(MODEL_READY_M2_DATASET_PATH)
    m2_test = m2_test[m2_test["split"] == "test"].merge(proba_ref[["record_id", "proba"]], on="record_id")
    p_inmem = score_batches(m2_test)
    inmem_proba_max_diff = float(np.max(np.abs(p_inmem - m2_test["proba"].to_numpy(dtype=float))))
    print(f"In-memory M2 -> score_batches max diff (all {len(m2_test)} test rows): {inmem_proba_max_diff:.3e}")

    print("[4/6] Running hypothetical examples & counterfactual sanity check...")
    hypo_cases: List[Dict[str, Any]] = [
        {
            "case": "H1_R0_fresh",
            "input": {
                "category": "Dairy",
                "product_name": "Fresh Milk",
                "store_id": "S001",
                "quality_grade": "A",
                "days_until_expiry": 6,
                "shelf_life_days": 7,
                "initial_quantity": 40,
                "base_price": 4.50,
                "cost_price": 2.40,
            },
        },
        {
            "case": "H2_R1_short_expiry",
            "input": {
                "category": "Seafood",
                "product_name": "Fresh Salmon",
                "store_id": "S001",
                "quality_grade": "B",
                "days_until_expiry": 1,
                "shelf_life_days": 5,
                "initial_quantity": 30,
                "base_price": 12.00,
                "cost_price": 6.50,
            },
        },
        {
            "case": "H3_Grade_C_long_shelf_life",
            "input": {
                "category": "Produce",
                "product_name": "Organic Apples",
                "store_id": "S002",
                "quality_grade": "C",
                "days_until_expiry": 6,
                "shelf_life_days": 7,
                "initial_quantity": 50,
                "base_price": 5.00,
                "cost_price": 2.20,
            },
        },
        {
            "case": "H4_all_optional_defaulted",
            "input": {
                "category": "Bakery",
                "days_until_expiry": 1,
                "initial_quantity": 25,
                "base_price": 3.50,
                "cost_price": 1.40,
            },
        },
        {
            "case": "H5_unseen_product_store",
            "input": {
                "category": "Dairy",
                "product_name": "UNSEEN_ARTISAN_CHEESE",
                "store_id": "S999_NEW",
                "quality_grade": "B",
                "days_until_expiry": 1,
                "shelf_life_days": 10,
                "initial_quantity": 20,
                "base_price": 8.00,
                "cost_price": 4.00,
            },
        },
    ]

    hypo_results: List[Dict[str, Any]] = []
    for hc in hypo_cases:
        res_h = run_pipeline(pd.DataFrame([hc["input"]]), mode="hypothetical").iloc[0]
        hypo_results.append({
            "case": hc["case"],
            "region": str(res_h["region"]),
            "status": str(res_h["status"]),
            "spoil_chance_pct": float(res_h["spoil_chance_pct"]),
            "tier": str(res_h["tier"]),
            "recommended_depth": None if pd.isna(res_h["recommended_depth"]) else float(res_h["recommended_depth"]),
            "predicted_profit_at_rec": None if pd.isna(res_h["predicted_profit_at_rec"]) else float(res_h["predicted_profit_at_rec"]),
            "confidence_level": str(res_h["confidence_level"]),
            "confidence_reasons": str(res_h["confidence_reasons"]),
            "defaults_dominate": bool(res_h["defaults_dominate"]),
            "unknown_entity": bool(res_h["unknown_entity"]),
            "top_drivers_text": str(res_h["top_drivers_text"]),
        })

    # Counterfactual sanity check on handling_score
    model_3a, meta_3a = load_model_with_metadata(TRACK_3A_MODEL_PATH)
    hs_idx = list(meta_3a["feature_list"]).index("handling_score")
    hs_coef = float(model_3a.coef_[0][hs_idx])

    base_cf = hypo_cases[1]["input"]
    t_sim0 = time.perf_counter()
    sim_low = simulate(base_cf, overrides={"handling_score": 0.2}, chosen_depth=0.25)
    sim_latency = time.perf_counter() - t_sim0
    sim_high = simulate(base_cf, overrides={"handling_score": 0.9}, chosen_depth=0.25)

    p_low = float(sim_low["recommendation"]["spoil_proba"])
    p_high = float(sim_high["recommendation"]["spoil_proba"])
    expected_sign = "negative" if hs_coef < 0 else "positive"
    actual_delta = p_high - p_low
    cf_direction_ok = (actual_delta < 0) if hs_coef < 0 else (actual_delta > 0)
    assert cf_direction_ok, f"Counterfactual handling_score direction mismatch: coef={hs_coef}, delta={actual_delta}"

    counterfactual_check = {
        "feature": "handling_score",
        "model_3a_coef": hs_coef,
        "expected_direction_on_increase": expected_sign,
        "spoil_proba_at_0.2": p_low,
        "spoil_proba_at_0.9": p_high,
        "delta": actual_delta,
        "direction_matches_model_coef": bool(cf_direction_ok),
        "chosen_depth_0.25_eval_at_0.9": sim_high["chosen_depth_eval"],
        "simulate_latency_sec": round(float(sim_latency), 3),
    }

    print("[5/6] Checking Manager View & robustness...")
    t_mv0 = time.perf_counter()
    mv = manager_view_table(snap_df, mode="real")
    mv_latency = time.perf_counter() - t_mv0

    # Robustness error tests
    robustness_checks: List[Dict[str, Any]] = []
    bad_inputs = [
        ("empty_df", pd.DataFrame()),
        ("missing_required_col", pd.DataFrame([{"category": "Dairy", "days_until_expiry": 2}])),
        ("negative_days_until_expiry", pd.DataFrame([{
            "category": "Dairy", "days_until_expiry": -1, "initial_quantity": 10, "base_price": 4.0, "cost_price": 2.0
        }])),
        ("unknown_category", pd.DataFrame([{
            "category": "Electronics", "days_until_expiry": 2, "initial_quantity": 10, "base_price": 4.0, "cost_price": 2.0
        }])),
        ("invalid_quality_grade", pd.DataFrame([{
            "category": "Dairy", "days_until_expiry": 2, "initial_quantity": 10, "base_price": 4.0, "cost_price": 2.0, "quality_grade": "Z"
        }])),
    ]
    for name, bad_df in bad_inputs:
        raised = False
        msg = ""
        try:
            run_pipeline(bad_df, mode="hypothetical")
        except ValueError as exc:
            raised = True
            msg = str(exc)
        assert raised, f"Expected ValueError for {name}"
        robustness_checks.append({"case": name, "raised_value_error": raised, "message": msg})

    # Additional robustness checks: single row, all-NaN optional, key-field default P3, duplicate IDs, determinism
    single_row_res = run_pipeline(snap_df.iloc[[0]].copy(), mode="real")
    assert len(single_row_res) == 1
    robustness_checks.append({"case": "single_row", "passed": True, "n_out": int(len(single_row_res))})

    all_nan_opt = pd.DataFrame([{
        "category": "Dairy", "days_until_expiry": 1, "initial_quantity": 20, "base_price": 4.0, "cost_price": 2.0,
        "packaging_score": np.nan, "handling_score": np.nan, "temp_deviation": np.nan,
    }])
    all_nan_res = run_pipeline(all_nan_opt, mode="hypothetical").iloc[0]
    assert bool(all_nan_res["defaults_dominate"]) and all_nan_res["confidence_level"] == "Low"
    robustness_checks.append({
        "case": "all_nan_optional_fields",
        "passed": True,
        "defaults_dominate": bool(all_nan_res["defaults_dominate"]),
        "confidence_level": str(all_nan_res["confidence_level"]),
    })

    # Key-sensitive single-field default (omit only packaging_score on an otherwise complete R1 row)
    full_r1_row = snap_df[mv["raw_results"]["region"] == "R1"].iloc[[0]].copy()
    full_r1_row["packaging_score"] = np.nan
    key_def_res = run_pipeline(full_r1_row, mode="real").iloc[0]
    assert bool(key_def_res["defaults_dominate"]) and key_def_res["confidence_level"] == "Low"
    assert "defaults_dominate" in str(key_def_res["confidence_reasons"])
    robustness_checks.append({
        "case": "p3_key_field_default_packaging_score",
        "passed": True,
        "confidence_level": str(key_def_res["confidence_level"]),
        "confidence_reasons": str(key_def_res["confidence_reasons"]),
    })

    dup_df = pd.concat([snap_df.iloc[[0]], snap_df.iloc[[0]]], ignore_index=True)
    dup_res = run_pipeline(dup_df, mode="real")
    assert len(dup_res) == 2
    robustness_checks.append({"case": "duplicated_record_id", "passed": True, "n_out": int(len(dup_res))})

    det_1 = run_pipeline(snap_df.head(10), mode="real")
    det_2 = run_pipeline(snap_df.head(10), mode="real")
    det_match = bool(np.allclose(det_1["spoil_proba"], det_2["spoil_proba"], atol=0.0, rtol=0.0))
    assert det_match
    robustness_checks.append({"case": "repeated_call_determinism", "passed": det_match})

    # Snapshot statistics (R1 tier & confidence distributions)
    raw_snap_res = mv["raw_results"]
    r1_snap = raw_snap_res[raw_snap_res["region"] == "R1"]
    snapshot_stats = {
        "date": snap_date,
        "n_rows": int(len(snap_df)),
        "n_r0": int((raw_snap_res["region"] == "R0").sum()),
        "n_r1": int(len(r1_snap)),
        "r0_pct": round(float((raw_snap_res["region"] == "R0").mean() * 100.0), 2),
        "r1_tier_counts": r1_snap["tier"].value_counts().to_dict(),
        "r1_confidence_counts": r1_snap["confidence_level"].value_counts().to_dict(),
        "snapshot_csv_columns": list(snap_df.columns),
        "manager_view_columns": list(mv["table"].columns),
    }

    summary = {
        "task": "M3-T14",
        "snapshot_date": snap_date,
        "snapshot_rows": int(len(snap_df)),
        "defaults_training_rows": int(defaults["training_row_count"]),
        "defaults_file_bytes": int(INFERENCE_DEFAULTS_PATH.stat().st_size),
        "defaults_top_level_keys": list(defaults.keys()),
        "inmem_m2_proba_max_abs_diff": inmem_proba_max_diff,
        "parity_test_1000": parity_1000,
        "parity_snapshot": parity_snap,
        "snapshot_stats": snapshot_stats,
        "hypothetical_examples": hypo_results,
        "counterfactual_check": counterfactual_check,
        "manager_view_summary_strip": mv["summary_strip"],
        "manager_view_latency_sec": round(float(mv_latency), 3),
        "robustness_checks": robustness_checks,
    }

    out_dir = REPORTS_DIR / "M3"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "m3_t14_results_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary -> {summary_path}")

    print("[6/6] Logging to MLflow...")
    with start_tracked_run(
        track="3C",
        model_type="inference_pipeline",
        feature_set_name="track_3_end_to_end",
        shelf_life_version="both",
        run_name="M3-T14-inference-pipeline",
        params={
            "snapshot_date": snap_date,
            "snapshot_rows": len(snap_df),
            "parity_sample_rows": 1000,
            "defaults_artifact": str(INFERENCE_DEFAULTS_PATH.name),
        },
        tags={"task_id": "M3-T14", "stage": "inference_pipeline"},
    ):
        mlflow.log_metrics({
            "parity_1000_proba_max_diff": parity_1000["spoil_proba_max_abs_diff"],
            "parity_1000_depth_exact_rate": parity_1000["r1_recommended_depth_exact_rate"],
            "parity_snap_proba_max_diff": parity_snap["spoil_proba_max_abs_diff"],
            "parity_snap_depth_exact_rate": parity_snap["r1_recommended_depth_exact_rate"],
            "simulate_latency_sec": counterfactual_check["simulate_latency_sec"],
            "manager_view_latency_sec": summary["manager_view_latency_sec"],
        })
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(INFERENCE_DEFAULTS_PATH))
    print("M3-T14 verification complete.")


if __name__ == "__main__":
    main()
