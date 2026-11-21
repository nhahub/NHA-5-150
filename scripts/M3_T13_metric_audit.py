"""
scripts/M3_T13_metric_audit.py - M3-T13 Comprehensive Metric Audit.

Verifies headline numbers quoted in M3_T13_Model_Evaluation_Report.md and
M3_T13_Optimizer_Validation_Report.md against reports/M3/*.json and MLflow run metrics.
Documents tolerance rationale (1e-4 default; 1e-2 / 1e-1 for 2-dec / 1-dec display rounding)
and flags any check where looser display rounding could mask a small difference.
Writes reports/M3/M3_T13_metric_audit.md.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MLFLOW_TRACKING_URI, REPORTS_DIR


def _load_json(name: str) -> Dict[str, Any]:
    return json.loads((REPORTS_DIR / "M3" / name).read_text(encoding="utf-8"))


def _get_mlflow_metrics(run_id: str) -> Dict[str, float]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return {k: float(v) for k, v in run.data.metrics.items()}


def main() -> None:
    t01 = _load_json("m3_t01_results_summary.json")
    t02 = _load_json("m3_t02_closeout_summary.json")
    t04 = _load_json("m3_t04_diagnostics_summary.json")
    t05_add = _load_json("m3_t05_addendum_summary.json")
    t05_diag = _load_json("m3_t05_diagnostics_summary.json")
    t06 = _load_json("m3_t06_results_summary.json")
    t07 = _load_json("m3_t07_results_summary.json")
    t08 = _load_json("m3_t08_results_summary.json")
    t09 = _load_json("m3_t09_results_summary.json")
    t10 = _load_json("m3_t10_results_summary.json")
    t11 = _load_json("m3_t11_results_summary.json")
    t12 = _load_json("m3_t12_results_summary.json")
    t14 = _load_json("m3_t14_results_summary.json")

    eval_md = (REPORTS_DIR / "M3" / "M3_T13_Model_Evaluation_Report.md").read_text(encoding="utf-8")
    opt_md = (REPORTS_DIR / "M3" / "M3_T13_Optimizer_Validation_Report.md").read_text(encoding="utf-8")

    ml_3a = _get_mlflow_metrics("1c6393d91cbd4396b3baee6c91045aa2")
    ml_3b_train = _get_mlflow_metrics("ce191bffab874806ad34bfe1485cd6c5")
    ml_3b_test = _get_mlflow_metrics("d3ed3411ecd3464b8baf57aefd62557d")
    ml_3c_opt = _get_mlflow_metrics("63724d725d91460e82b41f37da04825c")
    ml_3c_tier = _get_mlflow_metrics("4774b6f6e1c04181ac7a5441bf22c828")
    ml_3c_val = _get_mlflow_metrics("b62e8893e1b9490db3176f715c576c19")
    ml_3c_conf = _get_mlflow_metrics("2143aa1328194920b3293b121d7ca2d8")
    ml_3c_inf = _get_mlflow_metrics("71eb17c0774145679411040cb27ba251")

    # (statement, doc_txt, rep_str, rep_val, src_lbl, src_val, tol, tol_reason)
    checks: List[Tuple[str, str, str, float, str, float, float, str]] = [
        # --- Track 3A References & Candidates ---
        ("3A Ref Constant Base-Rate CV Brier", eval_md, "0.15606", 0.15606, "m3_t01_results_summary.json", t01["references"]["constant_base_rate_cv"]["brier"], 1e-4, "5-dec display"),
        ("3A Ref M2 LGBM OOF Brier", eval_md, "0.15397", 0.15397, "m3_t01_results_summary.json", t01["references"]["existing_lgbm_oof_baseline"]["brier"], 1e-4, "5-dec display"),
        ("3A Ref spoilage_risk Train AUC", eval_md, "0.5931", 0.5931, "m3_t01_results_summary.json", t01["references"]["spoilage_risk_benchmark_train"]["roc_auc"], 1e-4, "4-dec display"),
        ("3A CV XGBoost FULL Brier", eval_md, "0.15373", 0.15373, "m3_t01_results_summary.json", t01["candidates"][1]["metrics"]["brier"], 1e-4, "5-dec display"),
        ("3A CV LogReg COMPACT Brier", eval_md, "0.15374", 0.15374, "m3_t01_results_summary.json", t01["chosen"]["cv_metrics"]["brier"], 1e-4, "5-dec display"),
        ("3A CV LogReg COMPACT Log Loss", eval_md, "0.48374", 0.48374, "m3_t01_results_summary.json", t01["chosen"]["cv_metrics"]["log_loss"], 1e-4, "5-dec display"),
        ("3A CV LogReg COMPACT ROC-AUC", eval_md, "0.5885", 0.5885, "m3_t01_results_summary.json", t01["chosen"]["cv_metrics"]["roc_auc"], 1e-4, "4-dec display"),
        ("3A CV LogReg COMPACT ECE", eval_md, "0.00167", 0.00167, "m3_t01_results_summary.json", t01["chosen"]["cv_metrics"]["ece"], 1e-4, "5-dec display"),
        ("3A CV RF FULL Brier", eval_md, "0.15389", 0.15389, "m3_t01_results_summary.json", t01["candidates"][6]["metrics"]["brier"], 1e-4, "5-dec display"),
        ("3A CV RF COMPACT Brier", eval_md, "0.15393", 0.15393, "m3_t01_results_summary.json", t01["candidates"][7]["metrics"]["brier"], 1e-4, "5-dec display"),
        # --- Track 3A Test & Closeout ---
        ("3A Test Brier (JSON)", eval_md, "0.15641", 0.15641, "m3_t01_results_summary.json", t01["test"]["model_metrics"]["brier"], 1e-4, "5-dec display"),
        ("3A Test Brier (MLflow)", eval_md, "0.15641", 0.15641, "MLflow 1c6393d9", ml_3a["test_brier_score"], 1e-4, "5-dec display"),
        ("3A Test Log Loss (MLflow)", eval_md, "0.48988", 0.48988, "MLflow 1c6393d9", ml_3a["test_log_loss"], 1e-4, "5-dec display"),
        ("3A Test ROC-AUC (MLflow)", eval_md, "0.5899", 0.5899, "MLflow 1c6393d9", ml_3a["test_roc_auc"], 1e-4, "4-dec display"),
        ("3A Test ECE (MLflow)", eval_md, "0.00579", 0.00579, "MLflow 1c6393d9", ml_3a["test_ece"], 1e-4, "5-dec display"),
        ("3A Benchmark Test Brier (MLflow)", eval_md, "0.15601", 0.15601, "MLflow 1c6393d9", ml_3a["benchmark_test_brier"], 1e-4, "5-dec display"),
        ("3A Benchmark Test ROC-AUC (MLflow)", eval_md, "0.5967", 0.5967, "MLflow 1c6393d9", ml_3a["benchmark_spoilage_risk_auc"], 1e-4, "4-dec display"),
        ("3A Threshold 0.20 Precision", eval_md, "0.2444", 0.2444, "m3_t01_results_summary.json", t01["test"]["thresholds"][0]["precision"], 1e-4, "4-dec display"),
        ("3A Threshold 0.20 Recall", eval_md, "0.5076", 0.5076, "m3_t01_results_summary.json", t01["test"]["thresholds"][0]["recall"], 1e-4, "4-dec display"),
        ("3A Threshold 0.30 Recall", eval_md, "0.0395", 0.0395, "m3_t01_results_summary.json", t01["test"]["thresholds"][1]["recall"], 1e-4, "4-dec display"),
        ("3A OOF v1 vs v2 Pearson r", eval_md, "0.8704", 0.8704, "m3_t02_closeout_summary.json", t02["distributions"]["oof_v1_vs_v2_pearson_corr"], 1e-4, "4-dec display"),
        ("3A Cost Sweep k=5 Recall", eval_md, "0.7399", 0.7399, "m3_t02_closeout_summary.json", t02["cost_asymmetric_sweep_cv_oof_v2"][3]["recall"], 1e-4, "4-dec display"),
        ("3A Support Flagged Test %", eval_md, "14.36%", 14.36, "m3_t01_results_summary.json", t01["test"]["support"]["flagged_pct"], 1e-2, "2-dec % rounding"),
        ("3A Support Flagged Test Brier (MLflow)", eval_md, "0.15310", 0.15310, "MLflow 1c6393d9", ml_3a["test_flagged_brier"], 1e-4, "5-dec display"),
        # --- Track 3B Confounding, Discrepancy Resolutions & Regressor ---
        ("3B R0 Train Rows", eval_md, "55,567", 55567.0, "m3_t05_addendum_summary.json", float(t05_add["regions"]["R0"]["n_rows"]), 1e-4, "exact integer"),
        ("3B R1 Train Rows", eval_md, "24,475", 24475.0, "m3_t05_addendum_summary.json", float(t05_add["regions"]["R1"]["n_rows"]), 1e-4, "exact integer"),
        ("3B R1 s=0 Share % (Corrected)", eval_md, "3.43%", 3.43, "m3_t05_addendum_summary.json", t05_add["regions"]["R1"]["s_eq_0_pct"], 1e-2, "2-dec % rounding"),
        ("3B R1 s=1 Share % (Corrected)", eval_md, "43.41%", 43.41, "m3_t05_addendum_summary.json", t05_add["regions"]["R1"]["s_eq_1_pct"], 1e-2, "2-dec % rounding"),
        ("3B R1 Loss Share %", eval_md, "54.04%", 54.04, "m3_t05_addendum_summary.json", t05_add["regions"]["R1"]["loss_share_pct"], 1e-2, "2-dec % rounding"),
        ("3B R1 Profit/Unit Mean", eval_md, "-$0.7849", -0.7849, "m3_t05_addendum_summary.json", t05_add["regions"]["R1"]["profit_per_unit_mean"], 1e-4, "4-dec display"),
        ("3B Propensity AUC", eval_md, "1.0000", 1.0, "m3_t05_diagnostics_summary.json", t05_diag["propensity"]["propensity_auc"], 1e-4, "exact 1.0"),
        ("3B R1 Depth OOF R2", eval_md, "0.8738", 0.8738, "m3_t05_addendum_summary.json", t05_add["r1_depth_diagnostics"]["depth_oof_r2"], 1e-4, "4-dec display"),
        ("3B R1 Rule E Residual SD", eval_md, "6.10", 6.10, "m3_t05_addendum_summary.json", t05_add["r1_depth_diagnostics"]["residual_sd_pts"], 1e-2, "2-dec display"),
        ("3B R1 Residual OOF R2", eval_md, "-0.0704", -0.0704, "m3_t05_addendum_summary.json", t05_add["r1_depth_diagnostics"]["residual_oof_r2"], 1e-4, "4-dec display"),
        ("3B R1 Partialling-Out Slope", eval_md, "0.2614", 0.2614, "m3_t05_addendum_summary.json", t05_add["r1_local_effect"]["partialling_out_slope"], 1e-4, "4-dec display"),
        ("3B R1 Profit/Unit Min (Corrected)", eval_md, "-$316.80", -316.80, "m3_t06_results_summary.json", t06["extreme_outcomes"]["profit_per_unit_quantiles"]["min"], 1e-2, "2-dec $ rounding"),
        ("3B R1 Profit/Unit P05 (Corrected)", eval_md, "-$7.07", -7.07, "m3_t06_results_summary.json", t06["extreme_outcomes"]["profit_per_unit_quantiles"]["p05"], 1e-2, "2-dec $ rounding"),
        ("3B R1 Profit/Unit P25 (Corrected)", eval_md, "-$1.98", -1.98, "m3_t06_results_summary.json", t06["extreme_outcomes"]["profit_per_unit_quantiles"]["p25"], 1e-2, "2-dec $ rounding"),
        ("3B R1 Profit/Unit P50", eval_md, "-$0.23", -0.23, "m3_t06_results_summary.json", t06["extreme_outcomes"]["profit_per_unit_quantiles"]["p50"], 1e-2, "2-dec $ rounding"),
        ("3B R1 Profit/Unit P75 (Corrected)", eval_md, "+$1.31", 1.31, "m3_t06_results_summary.json", t06["extreme_outcomes"]["profit_per_unit_quantiles"]["p75"], 1e-2, "2-dec $ rounding"),
        ("3B R1 Profit/Unit P95 (Corrected)", eval_md, "+$6.07", 6.07, "m3_t06_results_summary.json", t06["extreme_outcomes"]["profit_per_unit_quantiles"]["p95"], 1e-2, "2-dec $ rounding"),
        ("3B R1 Profit/Unit Max", eval_md, "+$200.56", 200.56, "m3_t06_results_summary.json", t06["extreme_outcomes"]["profit_per_unit_quantiles"]["max"], 1e-2, "2-dec $ rounding"),
        ("3B CV lgbm_ce_monotone s RMSE (MLflow)", eval_md, "0.12967", 0.12967, "MLflow ce191bff", ml_3b_train["sell_through_rmse"], 1e-4, "5-dec display"),
        ("3B CV lgbm_ce_monotone s MAE (MLflow)", eval_md, "0.10937", 0.10937, "MLflow ce191bff", ml_3b_train["sell_through_mae"], 1e-4, "5-dec display"),
        ("3B CV lgbm_ce_monotone s R2 (MLflow)", eval_md, "0.7094", 0.7094, "MLflow ce191bff", ml_3b_train["sell_through_r2"], 1e-4, "4-dec display"),
        ("3B CV profit_unit_mae (MLflow)", eval_md, "1.8337", 1.8337, "MLflow ce191bff", ml_3b_train["profit_unit_mae"], 1e-4, "4-dec display"),
        ("3B CV Model-Implied Slope (MLflow)", eval_md, "+0.2231", 0.2231, "MLflow ce191bff", ml_3b_train["model_implied_slope"], 1e-4, "4-dec display"),
        ("3B Test R1 s RMSE (MLflow)", eval_md, "0.12781", 0.12781, "MLflow d3ed3411", ml_3b_test["sell_through_rmse"], 1e-4, "5-dec display"),
        ("3B Test R1 s MAE (MLflow)", eval_md, "0.10810", 0.10810, "MLflow d3ed3411", ml_3b_test["sell_through_mae"], 1e-4, "5-dec display"),
        ("3B Test R1 s R2 (MLflow)", eval_md, "0.7419", 0.7419, "MLflow d3ed3411", ml_3b_test["sell_through_r2"], 1e-4, "4-dec display"),
        ("3B Test R1 profit_unit_mae (MLflow)", eval_md, "$1.73", 1.73, "MLflow d3ed3411", ml_3b_test["profit_unit_mae"], 1e-2, "2-dec $ rounding"),
        ("3B Test R1 profit_unit_rmse (MLflow)", eval_md, "$6.37", 6.37, "MLflow d3ed3411", ml_3b_test["profit_unit_rmse"], 1e-2, "2-dec $ rounding"),
        ("3B Test R1 Residual Slope (MLflow)", eval_md, "0.2911", 0.2911, "MLflow d3ed3411", ml_3b_test["test_residual_slope"], 1e-4, "4-dec display"),
        ("3B T08 spoilage_sensitivity rho", eval_md, "+0.103", 0.103, "m3_t08_results_summary.json", t08["spoilage_sensitivity_check"]["spearman_sens_vs_model_slope"], 1e-3, "3-dec display"),
        # --- Track 3C Optimizer, Tiers & Confidence ---
        ("3C Test R1 0.15 Depth Share %", opt_md, "48.9%", 48.9, "m3_t09_results_summary.json", t09["test_r1_descriptive"]["depth_shares_pct"]["0.15"], 1e-1, "1-dec % rounding"),
        ("3C Test R1 0.25 Depth Share %", opt_md, "36.6%", 36.6, "m3_t09_results_summary.json", t09["test_r1_descriptive"]["depth_shares_pct"]["0.25"], 1e-1, "1-dec % rounding"),
        ("3C Test R1 Shallowest % (MLflow)", opt_md, "76.2%", 76.2, "MLflow 63724d72", ml_3c_opt["test_r1_at_shallowest_pct"], 1e-1, "1-dec % rounding"),
        ("3C Test R1 flat_curve % (MLflow)", opt_md, "97.0%", 97.0, "MLflow 63724d72", ml_3c_opt["test_r1_flat_curve_pct"], 1e-1, "1-dec % rounding"),
        ("3C Test R1 within_025 % (MLflow)", opt_md, "14.4%", 14.4, "MLflow 63724d72", ml_3c_opt["test_r1_within_025_actual_pct"], 1e-1, "1-dec % rounding"),
        ("3C Test R1 mean rec-actual", opt_md, "-0.1356", -0.1356, "m3_t09_results_summary.json", t09["test_r1_descriptive"]["mean_signed_diff_rec_minus_act"], 1e-4, "4-dec display"),
        ("3C Tier Edge1 (JSON)", opt_md, "0.15", 0.15, "m3_t10_results_summary.json", t10["edges_payload"]["edges"]["edge1"], 1e-4, "exact grid"),
        ("3C Tier Edge2 (JSON)", opt_md, "0.25", 0.25, "m3_t10_results_summary.json", t10["edges_payload"]["edges"]["edge2"], 1e-4, "exact grid"),
        ("3C Tier Edge3 (JSON)", opt_md, "0.35", 0.35, "m3_t10_results_summary.json", t10["edges_payload"]["edges"]["edge3"], 1e-4, "exact grid"),
        ("3C Test Low Tier Share % (MLflow)", opt_md, "53.69%", 53.69, "MLflow 4774b6f6", ml_3c_tier["test_share_low"], 1e-2, "2-dec % rounding"),
        ("3C Test Medium Tier Share % (MLflow)", opt_md, "40.61%", 40.61, "MLflow 4774b6f6", ml_3c_tier["test_share_medium"], 1e-2, "2-dec % rounding"),
        ("3C Test High Tier Share % (MLflow)", opt_md, "5.60%", 5.60, "MLflow 4774b6f6", ml_3c_tier["test_share_high"], 1e-2, "2-dec % rounding"),
        ("3C Test Critical Tier Share % (MLflow)", opt_md, "0.10%", 0.10, "MLflow 4774b6f6", ml_3c_tier["test_share_critical"], 1e-2, "2-dec % rounding"),
        # --- Track 3C Validation (Lane 1 & Lane 2) ---
        ("3C Test R0 Excluded Share % (MLflow)", opt_md, "69.06%", 69.06, "MLflow b62e8893", ml_3c_val["test_r0_excluded_share_pct"], 1e-2, "2-dec % rounding"),
        ("3C Lane 1 Included Strata (MLflow)", opt_md, "8 / 15", 8.0, "MLflow b62e8893", ml_3c_val["lane1_n_included_strata"], 1e-4, "exact integer"),
        ("3C Lane 1 Batches Included (MLflow)", opt_md, "4,841", 4841.0, "MLflow b62e8893", ml_3c_val["lane1_n_batches_included"], 1e-4, "exact integer"),
        ("3C Lane 1 Followed Included (MLflow)", opt_md, "849", 849.0, "MLflow b62e8893", ml_3c_val["lane1_n_followed_included"], 1e-4, "exact integer"),
        ("3C Lane 1 Profit/Unit Diff (MLflow)", opt_md, "-$1.4421", -1.4421, "MLflow b62e8893", ml_3c_val["lane1_profit_per_unit_diff"], 1e-4, "4-dec display"),
        ("3C Lane 1 Sell-Through Diff (MLflow)", opt_md, "-0.1169", -0.1169, "MLflow b62e8893", ml_3c_val["lane1_s_diff"], 1e-4, "4-dec display"),
        ("3C Lane 1 Waste Share Diff (MLflow)", opt_md, "+0.1169", 0.1169, "MLflow b62e8893", ml_3c_val["lane1_waste_share_diff"], 1e-4, "4-dec display"),
        ("3C Lane 1 Batch Profit Diff (MLflow)", opt_md, "-$163.33", -163.33, "MLflow b62e8893", ml_3c_val["lane1_batch_profit_diff"], 1e-2, "2-dec $ rounding"),
        ("3C Lane 1 Sensitivity tol=0.05 Diff", opt_md, "-$1.2846", -1.2846, "m3_t11_results_summary.json", t11["lane1"]["sensitivity_tol_050"]["estimates"]["profit_per_unit"], 1e-4, "4-dec display"),
        ("3C Lane 2 N Batches (MLflow)", opt_md, "5,289", 5289.0, "MLflow b62e8893", ml_3c_val["lane2_n_batches"], 1e-4, "exact integer"),
        ("3C Lane 2 Mean Sim Gain/Unit (MLflow)", opt_md, "+$1.7838", 1.7838, "MLflow b62e8893", ml_3c_val["lane2_mean_gain_unit"], 1e-4, "4-dec display"),
        ("3C Lane 2 Median Sim Gain/Unit (MLflow)", opt_md, "+$0.7026", 0.7026, "MLflow b62e8893", ml_3c_val["lane2_median_gain_unit"], 1e-4, "4-dec display"),
        ("3C Lane 2 Mean Sim Gain/Batch (MLflow)", opt_md, "+$458.87", 458.87, "MLflow b62e8893", ml_3c_val["lane2_mean_gain_batch"], 1e-2, "2-dec $ rounding"),
        ("3C Lane 2 Below CV MAE % (MLflow)", opt_md, "70.09%", 70.09, "MLflow b62e8893", ml_3c_val["lane2_share_below_cv_mae_pct"], 1e-2, "2-dec % rounding"),
        ("3C Lane 2 Actual-Depth MAE (MLflow)", opt_md, "$1.5976", 1.5976, "MLflow b62e8893", ml_3c_val["lane2_actual_depth_mae_unit"], 1e-4, "4-dec display"),
        ("3C Lane 2 Actual-Depth RMSE (JSON)", opt_md, "$5.9696", 5.9696, "m3_t11_results_summary.json", t11["lane2"]["model_error_at_actual_depth"]["rmse_profit_per_unit"], 1e-4, "4-dec display"),
        # --- Confidence & Inference ---
        ("3C Test Low Confidence % (MLflow)", opt_md, "84.12%", 84.12, "MLflow 2143aa13", ml_3c_conf["test_r1_low_pct"], 1e-2, "2-dec % rounding"),
        ("3C Test Moderate Confidence % (MLflow)", opt_md, "15.88%", 15.88, "MLflow 2143aa13", ml_3c_conf["test_r1_moderate_pct"], 1e-2, "2-dec % rounding"),
        ("3C Error-by-Level Low MAE (MLflow)", opt_md, "$1.86", 1.86, "MLflow 2143aa13", ml_3c_conf["test_low_mae_unit"], 1e-2, "2-dec $ rounding"),
        ("3C Error-by-Level Mod MAE (MLflow)", opt_md, "$1.04", 1.04, "MLflow 2143aa13", ml_3c_conf["test_moderate_mae_unit"], 1e-2, "2-dec $ rounding"),
        ("3C Snapshot R0 Share %", opt_md, "73.79%", 73.79, "m3_t14_results_summary.json", t14["snapshot_stats"]["r0_pct"], 1e-2, "2-dec % rounding"),
        ("3C Inference Parity 1000 Proba Max Diff (MLflow)", opt_md, "3.61e-07", 3.61e-7, "MLflow 71eb17c0", ml_3c_inf["parity_1000_proba_max_diff"], 1e-8, "3-sig-fig sci"),
        ("3C Inference Parity Snap Proba Max Diff (MLflow)", opt_md, "2.94e-07", 2.94e-7, "MLflow 71eb17c0", ml_3c_inf["parity_snap_proba_max_diff"], 1e-8, "3-sig-fig sci"),
        ("3C Inference Parity 1000 Depth Exact (MLflow)", opt_md, "100.0%", 1.0, "MLflow 71eb17c0", ml_3c_inf["parity_1000_depth_exact_rate"], 1e-4, "exact 1.0"),
        ("3C Inference In-Memory M2 Max Diff", opt_md, "1.11e-16", 1.11e-16, "m3_t14_results_summary.json", t14["inmem_m2_proba_max_abs_diff"], 1e-12, "machine eps"),
    ]

    rows: List[Dict[str, Any]] = []
    n_mismatches = 0
    for stmt, doc_txt, rep_str, rep_val, src_lbl, src_val, tol, tol_reason in checks:
        in_doc = rep_str in doc_txt
        diff = abs(rep_val - float(src_val))
        ok = in_doc and (diff <= tol)
        if not ok:
            n_mismatches += 1
        looser_flag = "Display-rounded (could hide <=0.05)" if tol >= 1e-1 else ("Display-rounded (could hide <=0.005)" if tol >= 1e-2 else "No")
        rows.append({
            "statement": stmt,
            "report_str": rep_str,
            "in_doc": in_doc,
            "source": src_lbl,
            "source_val": src_val,
            "diff": diff,
            "tol": tol,
            "tol_reason": tol_reason,
            "looser_flag": looser_flag,
            "match": "Yes" if ok else "NO",
        })

    # Prose-only numbers (no JSON/MLflow metric source) & 15-item hand spot-check
    prose_only_notes = [
        "100,000 rows, 42 raw columns, 50 stores, 5 regions, 20 suppliers, 10 categories, 63 products (README.md / M1-T04)",
        "80,042 train / 19,958 test cutoff 2024-08-07 (M1-T04 / M2-T11)",
        "M3-T03 max |sum(phi_j) - logit_diff| = 4.441e-16; max prob-pts sum error = 1.260e-13 (M3_T03_Explainability_Report.md)",
        "M3-T03 top grouped drivers: packaging_score 0.1919, handling_score 0.1203, days_since_receipt 0.0840, days_until_expiry 0.0818, temp_deviation 0.0702 (M3_T03_Explainability_Report.md)",
        "M3-T03 proposal key category x region x store >= 129 rows across 500 cells (M3_T01 / M2-T12)",
        "M3-T06 batch profit RMSE $1,976 -> $1,298 excluding 1% tail (M3_T06_Regressor_Training_Report.md)",
        "M3-T07 Test R1 batch profit RMSE $1,793 ($1,204 excluding 1% tail) (M3_T07_Regressor_Evaluation_Report.md)",
        "M3-T08 4 categories < 8/13 depth bins: Dairy 5/13, Beverages 3/13, Pharmaceuticals 3/13, Frozen_Meals 3/13 (M3_T08_Elasticity_View_Report.md)",
        "M3-T08 10-30% common band range: +1.84 (Frozen_Meals) to +4.29 (Meat); 5/10 in model-free CI (M3_T08_Elasticity_View_Report.md)",
        "M3-T10 representative examples: record_id 25554 (Produce C, 0.15), 18328 (Seafood A, 0.25), 90055 (Bakery C, 0.35), 73650 (Bakery B, 0.50) (M3_T10_Tier_Mapping_Report.md)",
        "M3-T12 Test R1 flag rates: support_edge 77.75%, limited_depth_category 19.28%, low_history_3a 16.35%, critical_tier 0.10%, near_tie 96.97% (M3_T12_Confidence_Propagation_Report.md)",
        "M3-T14 defaults JSON size: 583,751 bytes; snapshot 2024-12-31, 145 rows x 45 cols (M3_T14_Inference_Pipeline_Report.md)",
    ]

    lines = [
        "# M3-T13: Metric Audit Report",
        "",
        f"**Checks Run:** `{len(rows)}` | **Mismatches:** `{n_mismatches}`",
        "",
        "## 1. Automated JSON / MLflow Metric Checks",
        "",
        "| Statement | Number in Report | In Report? | Source File / MLflow Run | Source Value | Abs Diff | Tol | Reason | Looser Tol Flag | Match |",
        "| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- | :--- | :---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['statement']} | `{r['report_str']}` | {r['in_doc']} | `{r['source']}` | `{r['source_val']:.6g}` | `{r['diff']:.2e}` | `{r['tol']:.0e}` | {r['tol_reason']} | {r['looser_flag']} | **{r['match']}** |"
        )

    lines.extend([
        "",
        "## 2. Prose-Only / Earlier-Report Numbers (No Direct MLflow/M3-Summary Check)",
        "",
    ])
    for item in prose_only_notes:
        lines.append(f"- {item}")

    out_path = REPORTS_DIR / "M3" / "M3_T13_metric_audit.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Metric audit complete: {len(rows)} checks, {n_mismatches} mismatches -> {out_path}")


if __name__ == "__main__":
    main()
