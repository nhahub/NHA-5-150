"""
M4-T03: Verify the dashboard Simulator Mode against the frozen M3-T14 results.

The Simulator adds no modelling: it calls src.m3_t14_pipeline.simulate() in
'hypothetical' mode. This script proves that, by checking on the real models:

  1. The constants mirrored in src/m4_t03_simulator.py match the pipeline.
  2. The five documented M3-T14 hypothetical cases (H1-H5), run through the
     Simulator's own build_batch()/run_scenario() path, reproduce the values
     stored in reports/M3/m3_t14_results_summary.json.
  3. The what-if curve has 14 depths and the recommended depth is the
     highest-profit supported depth (ties -> shallower).
  4. R0 batches give no recommendation; setting handling_score higher lowers
     spoilage risk (same direction as the frozen Track 3A coefficient).

Needs the same runtime files as the dashboard (models/*.joblib and the CSVs in
data/processed/). Run from the repository root:

    python scripts/M4_T03_verify_simulator.py

Writes reports/M4/m4_t03_verification.json and exits non-zero on any failure.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import REPORTS_DIR  # noqa: E402
from src import m4_t03_simulator as sim  # noqa: E402
from src import m3_t14_pipeline as pipe  # noqa: E402

M3_SUMMARY = REPORTS_DIR / "M3" / "m3_t14_results_summary.json"
OUT_PATH = REPORTS_DIR / "M4" / "m4_t03_verification.json"

# Same inputs as scripts/M3_T14_build_and_verify.py (H1-H5).
CASES = {
    "H1_R0_fresh": {"category": "Dairy", "product_name": "Fresh Milk", "store_id": "S001", "quality_grade": "A",
                    "days_until_expiry": 6, "shelf_life_days": 7, "initial_quantity": 40, "base_price": 4.50, "cost_price": 2.40},
    "H2_R1_short_expiry": {"category": "Seafood", "product_name": "Fresh Salmon", "store_id": "S001", "quality_grade": "B",
                           "days_until_expiry": 1, "shelf_life_days": 5, "initial_quantity": 30, "base_price": 12.00, "cost_price": 6.50},
    "H3_Grade_C_long_shelf_life": {"category": "Produce", "product_name": "Organic Apples", "store_id": "S002", "quality_grade": "C",
                                   "days_until_expiry": 6, "shelf_life_days": 7, "initial_quantity": 50, "base_price": 5.00, "cost_price": 2.20},
    "H4_all_optional_defaulted": {"category": "Bakery", "days_until_expiry": 1, "initial_quantity": 25, "base_price": 3.50, "cost_price": 1.40},
    "H5_unseen_product_store": {"category": "Dairy", "product_name": "UNSEEN_ARTISAN_CHEESE", "store_id": "S999_NEW", "quality_grade": "B",
                                "days_until_expiry": 1, "shelf_life_days": 10, "initial_quantity": 20, "base_price": 8.00, "cost_price": 4.00},
}

checks = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def close(a, b, tol) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and np.isnan(a):
        return b is None or (isinstance(b, float) and np.isnan(b))
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    # 1) constants mirrored from the pipeline
    check("KEY_SENSITIVE_FIELDS match pipeline", tuple(sim.KEY_SENSITIVE_FIELDS) == tuple(pipe.KEY_DEFAULT_SENSITIVE_FIELDS))
    check("N_OPTIONAL_TRACKED matches pipeline", sim.N_OPTIONAL_TRACKED == len(pipe.ALL_OPTIONAL_TRACKED))
    check("Numeric optional fields match pipeline", set(sim.NUMERIC_OPTIONAL) == set(pipe.OPTIONAL_NUMERIC_FIELDS))

    # 2) parity with the documented M3-T14 hypothetical examples
    ref = {r["case"]: r for r in json.loads(M3_SUMMARY.read_text(encoding="utf-8"))["hypothetical_examples"]}
    for name, raw in CASES.items():
        out = sim.run_scenario(dict(raw))
        rec = out["recommendation"]
        exp = ref[name]
        depth = None if np.isnan(rec["recommended_depth"]) else float(rec["recommended_depth"])
        profit = None if np.isnan(rec["predicted_profit_at_rec"]) else float(rec["predicted_profit_at_rec"])
        ok = (
            rec["region"] == exp["region"]
            and close(rec["spoil_chance_pct"], exp["spoil_chance_pct"], 1e-6)
            and close(depth, exp["recommended_depth"], 1e-9)
            and close(profit, exp["predicted_profit_at_rec"], 1e-6)
            and rec["confidence_level"] == exp["confidence_level"]
            and bool(rec["defaults_dominate"]) == exp["defaults_dominate"]
        )
        check(f"{name} reproduces M3-T14", ok,
              f"region={rec['region']} spoil={rec['spoil_chance_pct']} depth={depth} profit={profit} conf={rec['confidence_level']}")

    # 3) curve structure + optimizer pick consistent with the curve
    out = sim.run_scenario(dict(CASES["H2_R1_short_expiry"]))
    curve, rec = out["whatif_curve"], out["recommendation"]
    check("What-if curve has 14 depths (10%..75%)", len(curve) == 14 and np.isclose(curve["depth"].min(), 0.10) and np.isclose(curve["depth"].max(), 0.75))
    ok_rows = curve[curve["supported"].astype(bool)].sort_values(["batch_profit", "depth"], ascending=[False, True])
    check("Recommended depth = best supported depth on the curve", np.isclose(float(ok_rows.iloc[0]["depth"]), float(rec["recommended_depth"])))
    disp = sim.curve_for_display(curve, 12.0, rec["recommended_depth"])
    check("Display table marks exactly one recommended row", (disp["Optimizer pick"] != "").sum() == 1)

    r0 = sim.run_scenario(dict(CASES["H1_R0_fresh"]))
    check("R0 batch: no recommendation, all depths unsupported",
          r0["recommendation"]["region"] == "R0" and not r0["whatif_curve"]["supported"].any())

    # 4) Simulator form path (defaults omitted) + counterfactual direction
    batch = sim.build_batch(
        {"category": "Seafood", "days_until_expiry": 1, "initial_quantity": 30, "base_price": 12.0, "cost_price": 6.5},
        {"product_name": sim.DEFAULT_CHOICE, "store_id": sim.DEFAULT_CHOICE, "quality_grade": sim.DEFAULT_CHOICE},
    )
    check("build_batch omits 'Default' choices", set(batch) == {"category", "days_until_expiry", "initial_quantity", "base_price", "cost_price"})
    lo = sim.run_scenario({**batch, "handling_score": 0.2})["recommendation"]["spoil_proba"]
    hi = sim.run_scenario({**batch, "handling_score": 0.9})["recommendation"]["spoil_proba"]
    check("Higher handling_score lowers spoilage risk", hi < lo, f"{lo:.4f} -> {hi:.4f}")
    again = sim.run_scenario(dict(batch))["recommendation"]["spoil_proba"]
    check("Repeated run is deterministic", again == sim.run_scenario(dict(batch))["recommendation"]["spoil_proba"])

    n_fail = sum(1 for c in checks if not c["pass"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"task": "M4-T03", "n_checks": len(checks), "n_failed": n_fail, "checks": checks}, indent=2), encoding="utf-8")
    print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed -> {OUT_PATH}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
