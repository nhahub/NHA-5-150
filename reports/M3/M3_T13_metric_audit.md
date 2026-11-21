# M3-T13: Metric Audit Report

**Checks Run:** `92` | **Mismatches:** `0`

## 1. Automated JSON / MLflow Metric Checks

| Statement | Number in Report | In Report? | Source File / MLflow Run | Source Value | Abs Diff | Tol | Reason | Looser Tol Flag | Match |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| 3A Ref Constant Base-Rate CV Brier | `0.15606` | True | `m3_t01_results_summary.json` | `0.156063` | `3.19e-06` | `1e-04` | 5-dec display | No | **Yes** |
| 3A Ref M2 LGBM OOF Brier | `0.15397` | True | `m3_t01_results_summary.json` | `0.153973` | `3.01e-06` | `1e-04` | 5-dec display | No | **Yes** |
| 3A Ref spoilage_risk Train AUC | `0.5931` | True | `m3_t01_results_summary.json` | `0.593095` | `5.44e-06` | `1e-04` | 4-dec display | No | **Yes** |
| 3A CV XGBoost FULL Brier | `0.15373` | True | `m3_t01_results_summary.json` | `0.153726` | `4.39e-06` | `1e-04` | 5-dec display | No | **Yes** |
| 3A CV LogReg COMPACT Brier | `0.15374` | True | `m3_t01_results_summary.json` | `0.15374` | `8.58e-08` | `1e-04` | 5-dec display | No | **Yes** |
| 3A CV LogReg COMPACT Log Loss | `0.48374` | True | `m3_t01_results_summary.json` | `0.483744` | `3.93e-06` | `1e-04` | 5-dec display | No | **Yes** |
| 3A CV LogReg COMPACT ROC-AUC | `0.5885` | True | `m3_t01_results_summary.json` | `0.58846` | `4.03e-05` | `1e-04` | 4-dec display | No | **Yes** |
| 3A CV LogReg COMPACT ECE | `0.00167` | True | `m3_t01_results_summary.json` | `0.00166851` | `1.49e-06` | `1e-04` | 5-dec display | No | **Yes** |
| 3A CV RF FULL Brier | `0.15389` | True | `m3_t01_results_summary.json` | `0.153891` | `9.03e-07` | `1e-04` | 5-dec display | No | **Yes** |
| 3A CV RF COMPACT Brier | `0.15393` | True | `m3_t01_results_summary.json` | `0.153934` | `4.49e-06` | `1e-04` | 5-dec display | No | **Yes** |
| 3A Test Brier (JSON) | `0.15641` | True | `m3_t01_results_summary.json` | `0.156407` | `2.57e-06` | `1e-04` | 5-dec display | No | **Yes** |
| 3A Test Brier (MLflow) | `0.15641` | True | `MLflow 1c6393d9` | `0.156407` | `2.57e-06` | `1e-04` | 5-dec display | No | **Yes** |
| 3A Test Log Loss (MLflow) | `0.48988` | True | `MLflow 1c6393d9` | `0.489876` | `4.33e-06` | `1e-04` | 5-dec display | No | **Yes** |
| 3A Test ROC-AUC (MLflow) | `0.5899` | True | `MLflow 1c6393d9` | `0.589919` | `1.91e-05` | `1e-04` | 4-dec display | No | **Yes** |
| 3A Test ECE (MLflow) | `0.00579` | True | `MLflow 1c6393d9` | `0.00578965` | `3.55e-07` | `1e-04` | 5-dec display | No | **Yes** |
| 3A Benchmark Test Brier (MLflow) | `0.15601` | True | `MLflow 1c6393d9` | `0.15601` | `4.61e-07` | `1e-04` | 5-dec display | No | **Yes** |
| 3A Benchmark Test ROC-AUC (MLflow) | `0.5967` | True | `MLflow 1c6393d9` | `0.596737` | `3.67e-05` | `1e-04` | 4-dec display | No | **Yes** |
| 3A Threshold 0.20 Precision | `0.2444` | True | `m3_t01_results_summary.json` | `0.244398` | `1.56e-06` | `1e-04` | 4-dec display | No | **Yes** |
| 3A Threshold 0.20 Recall | `0.5076` | True | `m3_t01_results_summary.json` | `0.507587` | `1.27e-05` | `1e-04` | 4-dec display | No | **Yes** |
| 3A Threshold 0.30 Recall | `0.0395` | True | `m3_t01_results_summary.json` | `0.0394537` | `4.63e-05` | `1e-04` | 4-dec display | No | **Yes** |
| 3A OOF v1 vs v2 Pearson r | `0.8704` | True | `m3_t02_closeout_summary.json` | `0.87042` | `1.97e-05` | `1e-04` | 4-dec display | No | **Yes** |
| 3A Cost Sweep k=5 Recall | `0.7399` | True | `m3_t02_closeout_summary.json` | `0.739863` | `3.69e-05` | `1e-04` | 4-dec display | No | **Yes** |
| 3A Support Flagged Test % | `14.36%` | True | `m3_t01_results_summary.json` | `14.3602` | `1.56e-04` | `1e-02` | 2-dec % rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3A Support Flagged Test Brier (MLflow) | `0.15310` | True | `MLflow 1c6393d9` | `0.153102` | `2.18e-06` | `1e-04` | 5-dec display | No | **Yes** |
| 3B R0 Train Rows | `55,567` | True | `m3_t05_addendum_summary.json` | `55567` | `0.00e+00` | `1e-04` | exact integer | No | **Yes** |
| 3B R1 Train Rows | `24,475` | True | `m3_t05_addendum_summary.json` | `24475` | `0.00e+00` | `1e-04` | exact integer | No | **Yes** |
| 3B R1 s=0 Share % (Corrected) | `3.43%` | True | `m3_t05_addendum_summary.json` | `3.43207` | `2.07e-03` | `1e-02` | 2-dec % rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3B R1 s=1 Share % (Corrected) | `43.41%` | True | `m3_t05_addendum_summary.json` | `43.4076` | `2.44e-03` | `1e-02` | 2-dec % rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3B R1 Loss Share % | `54.04%` | True | `m3_t05_addendum_summary.json` | `54.0388` | `1.18e-03` | `1e-02` | 2-dec % rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3B R1 Profit/Unit Mean | `-$0.7849` | True | `m3_t05_addendum_summary.json` | `-0.784912` | `1.19e-05` | `1e-04` | 4-dec display | No | **Yes** |
| 3B Propensity AUC | `1.0000` | True | `m3_t05_diagnostics_summary.json` | `1` | `0.00e+00` | `1e-04` | exact 1.0 | No | **Yes** |
| 3B R1 Depth OOF R2 | `0.8738` | True | `m3_t05_addendum_summary.json` | `0.873803` | `3.08e-06` | `1e-04` | 4-dec display | No | **Yes** |
| 3B R1 Rule E Residual SD | `6.10` | True | `m3_t05_addendum_summary.json` | `6.10078` | `7.83e-04` | `1e-02` | 2-dec display | Display-rounded (could hide <=0.005) | **Yes** |
| 3B R1 Residual OOF R2 | `-0.0704` | True | `m3_t05_addendum_summary.json` | `-0.0704204` | `2.04e-05` | `1e-04` | 4-dec display | No | **Yes** |
| 3B R1 Partialling-Out Slope | `0.2614` | True | `m3_t05_addendum_summary.json` | `0.261408` | `8.16e-06` | `1e-04` | 4-dec display | No | **Yes** |
| 3B R1 Profit/Unit Min (Corrected) | `-$316.80` | True | `m3_t06_results_summary.json` | `-316.8` | `0.00e+00` | `1e-02` | 2-dec $ rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3B R1 Profit/Unit P05 (Corrected) | `-$7.07` | True | `m3_t06_results_summary.json` | `-7.07156` | `1.56e-03` | `1e-02` | 2-dec $ rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3B R1 Profit/Unit P25 (Corrected) | `-$1.98` | True | `m3_t06_results_summary.json` | `-1.98` | `0.00e+00` | `1e-02` | 2-dec $ rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3B R1 Profit/Unit P50 | `-$0.23` | True | `m3_t06_results_summary.json` | `-0.229046` | `9.54e-04` | `1e-02` | 2-dec $ rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3B R1 Profit/Unit P75 (Corrected) | `+$1.31` | True | `m3_t06_results_summary.json` | `1.30757` | `2.43e-03` | `1e-02` | 2-dec $ rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3B R1 Profit/Unit P95 (Corrected) | `+$6.07` | True | `m3_t06_results_summary.json` | `6.07` | `0.00e+00` | `1e-02` | 2-dec $ rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3B R1 Profit/Unit Max | `+$200.56` | True | `m3_t06_results_summary.json` | `200.56` | `0.00e+00` | `1e-02` | 2-dec $ rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3B CV lgbm_ce_monotone s RMSE (MLflow) | `0.12967` | True | `MLflow ce191bff` | `0.129673` | `2.54e-06` | `1e-04` | 5-dec display | No | **Yes** |
| 3B CV lgbm_ce_monotone s MAE (MLflow) | `0.10937` | True | `MLflow ce191bff` | `0.109373` | `2.67e-06` | `1e-04` | 5-dec display | No | **Yes** |
| 3B CV lgbm_ce_monotone s R2 (MLflow) | `0.7094` | True | `MLflow ce191bff` | `0.709374` | `2.57e-05` | `1e-04` | 4-dec display | No | **Yes** |
| 3B CV profit_unit_mae (MLflow) | `1.8337` | True | `MLflow ce191bff` | `1.83366` | `4.24e-05` | `1e-04` | 4-dec display | No | **Yes** |
| 3B CV Model-Implied Slope (MLflow) | `+0.2231` | True | `MLflow ce191bff` | `0.223148` | `4.84e-05` | `1e-04` | 4-dec display | No | **Yes** |
| 3B Test R1 s RMSE (MLflow) | `0.12781` | True | `MLflow d3ed3411` | `0.127812` | `2.16e-06` | `1e-04` | 5-dec display | No | **Yes** |
| 3B Test R1 s MAE (MLflow) | `0.10810` | True | `MLflow d3ed3411` | `0.108097` | `3.27e-06` | `1e-04` | 5-dec display | No | **Yes** |
| 3B Test R1 s R2 (MLflow) | `0.7419` | True | `MLflow d3ed3411` | `0.741931` | `3.05e-05` | `1e-04` | 4-dec display | No | **Yes** |
| 3B Test R1 profit_unit_mae (MLflow) | `$1.73` | True | `MLflow d3ed3411` | `1.73401` | `4.01e-03` | `1e-02` | 2-dec $ rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3B Test R1 profit_unit_rmse (MLflow) | `$6.37` | True | `MLflow d3ed3411` | `6.3733` | `3.30e-03` | `1e-02` | 2-dec $ rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3B Test R1 Residual Slope (MLflow) | `0.2911` | True | `MLflow d3ed3411` | `0.29109` | `9.97e-06` | `1e-04` | 4-dec display | No | **Yes** |
| 3B T08 spoilage_sensitivity rho | `+0.103` | True | `m3_t08_results_summary.json` | `0.103344` | `3.44e-04` | `1e-03` | 3-dec display | No | **Yes** |
| 3C Test R1 0.15 Depth Share % | `48.9%` | True | `m3_t09_results_summary.json` | `48.9475` | `4.75e-02` | `1e-01` | 1-dec % rounding | Display-rounded (could hide <=0.05) | **Yes** |
| 3C Test R1 0.25 Depth Share % | `36.6%` | True | `m3_t09_results_summary.json` | `36.6418` | `4.18e-02` | `1e-01` | 1-dec % rounding | Display-rounded (could hide <=0.05) | **Yes** |
| 3C Test R1 Shallowest % (MLflow) | `76.2%` | True | `MLflow 63724d72` | `76.1658` | `3.42e-02` | `1e-01` | 1-dec % rounding | Display-rounded (could hide <=0.05) | **Yes** |
| 3C Test R1 flat_curve % (MLflow) | `97.0%` | True | `MLflow 63724d72` | `96.9722` | `2.78e-02` | `1e-01` | 1-dec % rounding | Display-rounded (could hide <=0.05) | **Yes** |
| 3C Test R1 within_025 % (MLflow) | `14.4%` | True | `MLflow 63724d72` | `14.362` | `3.80e-02` | `1e-01` | 1-dec % rounding | Display-rounded (could hide <=0.05) | **Yes** |
| 3C Test R1 mean rec-actual | `-0.1356` | True | `m3_t09_results_summary.json` | `-0.135623` | `2.34e-05` | `1e-04` | 4-dec display | No | **Yes** |
| 3C Tier Edge1 (JSON) | `0.15` | True | `m3_t10_results_summary.json` | `0.15` | `0.00e+00` | `1e-04` | exact grid | No | **Yes** |
| 3C Tier Edge2 (JSON) | `0.25` | True | `m3_t10_results_summary.json` | `0.25` | `0.00e+00` | `1e-04` | exact grid | No | **Yes** |
| 3C Tier Edge3 (JSON) | `0.35` | True | `m3_t10_results_summary.json` | `0.35` | `0.00e+00` | `1e-04` | exact grid | No | **Yes** |
| 3C Test Low Tier Share % (MLflow) | `53.69%` | True | `MLflow 4774b6f6` | `53.6917` | `1.71e-03` | `1e-02` | 2-dec % rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3C Test Medium Tier Share % (MLflow) | `40.61%` | True | `MLflow 4774b6f6` | `40.6088` | `1.19e-03` | `1e-02` | 2-dec % rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3C Test High Tier Share % (MLflow) | `5.60%` | True | `MLflow 4774b6f6` | `5.60233` | `2.33e-03` | `1e-02` | 2-dec % rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3C Test Critical Tier Share % (MLflow) | `0.10%` | True | `MLflow 4774b6f6` | `0.0971503` | `2.85e-03` | `1e-02` | 2-dec % rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3C Test R0 Excluded Share % (MLflow) | `69.06%` | True | `MLflow b62e8893` | `69.055` | `4.98e-03` | `1e-02` | 2-dec % rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3C Lane 1 Included Strata (MLflow) | `8 / 15` | True | `MLflow b62e8893` | `8` | `0.00e+00` | `1e-04` | exact integer | No | **Yes** |
| 3C Lane 1 Batches Included (MLflow) | `4,841` | True | `MLflow b62e8893` | `4841` | `0.00e+00` | `1e-04` | exact integer | No | **Yes** |
| 3C Lane 1 Followed Included (MLflow) | `849` | True | `MLflow b62e8893` | `849` | `0.00e+00` | `1e-04` | exact integer | No | **Yes** |
| 3C Lane 1 Profit/Unit Diff (MLflow) | `-$1.4421` | True | `MLflow b62e8893` | `-1.44207` | `2.57e-05` | `1e-04` | 4-dec display | No | **Yes** |
| 3C Lane 1 Sell-Through Diff (MLflow) | `-0.1169` | True | `MLflow b62e8893` | `-0.116924` | `2.37e-05` | `1e-04` | 4-dec display | No | **Yes** |
| 3C Lane 1 Waste Share Diff (MLflow) | `+0.1169` | True | `MLflow b62e8893` | `0.116924` | `2.37e-05` | `1e-04` | 4-dec display | No | **Yes** |
| 3C Lane 1 Batch Profit Diff (MLflow) | `-$163.33` | True | `MLflow b62e8893` | `-163.33` | `4.30e-04` | `1e-02` | 2-dec $ rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3C Lane 1 Sensitivity tol=0.05 Diff | `-$1.2846` | True | `m3_t11_results_summary.json` | `-1.28461` | `7.66e-06` | `1e-04` | 4-dec display | No | **Yes** |
| 3C Lane 2 N Batches (MLflow) | `5,289` | True | `MLflow b62e8893` | `5289` | `0.00e+00` | `1e-04` | exact integer | No | **Yes** |
| 3C Lane 2 Mean Sim Gain/Unit (MLflow) | `+$1.7838` | True | `MLflow b62e8893` | `1.78384` | `3.80e-05` | `1e-04` | 4-dec display | No | **Yes** |
| 3C Lane 2 Median Sim Gain/Unit (MLflow) | `+$0.7026` | True | `MLflow b62e8893` | `0.702624` | `2.41e-05` | `1e-04` | 4-dec display | No | **Yes** |
| 3C Lane 2 Mean Sim Gain/Batch (MLflow) | `+$458.87` | True | `MLflow b62e8893` | `458.874` | `3.82e-03` | `1e-02` | 2-dec $ rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3C Lane 2 Below CV MAE % (MLflow) | `70.09%` | True | `MLflow b62e8893` | `70.0889` | `1.14e-03` | `1e-02` | 2-dec % rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3C Lane 2 Actual-Depth MAE (MLflow) | `$1.5976` | True | `MLflow b62e8893` | `1.59759` | `7.00e-06` | `1e-04` | 4-dec display | No | **Yes** |
| 3C Lane 2 Actual-Depth RMSE (JSON) | `$5.9696` | True | `m3_t11_results_summary.json` | `5.96957` | `3.43e-05` | `1e-04` | 4-dec display | No | **Yes** |
| 3C Test Low Confidence % (MLflow) | `84.12%` | True | `MLflow 2143aa13` | `84.1159` | `4.07e-03` | `1e-02` | 2-dec % rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3C Test Moderate Confidence % (MLflow) | `15.88%` | True | `MLflow 2143aa13` | `15.8841` | `4.07e-03` | `1e-02` | 2-dec % rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3C Error-by-Level Low MAE (MLflow) | `$1.86` | True | `MLflow 2143aa13` | `1.86422` | `4.22e-03` | `1e-02` | 2-dec $ rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3C Error-by-Level Mod MAE (MLflow) | `$1.04` | True | `MLflow 2143aa13` | `1.04443` | `4.43e-03` | `1e-02` | 2-dec $ rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3C Snapshot R0 Share % | `73.79%` | True | `m3_t14_results_summary.json` | `73.79` | `0.00e+00` | `1e-02` | 2-dec % rounding | Display-rounded (could hide <=0.005) | **Yes** |
| 3C Inference Parity 1000 Proba Max Diff (MLflow) | `3.61e-07` | True | `MLflow 71eb17c0` | `3.60778e-07` | `2.22e-10` | `1e-08` | 3-sig-fig sci | No | **Yes** |
| 3C Inference Parity Snap Proba Max Diff (MLflow) | `2.94e-07` | True | `MLflow 71eb17c0` | `2.93603e-07` | `3.97e-10` | `1e-08` | 3-sig-fig sci | No | **Yes** |
| 3C Inference Parity 1000 Depth Exact (MLflow) | `100.0%` | True | `MLflow 71eb17c0` | `1` | `0.00e+00` | `1e-04` | exact 1.0 | No | **Yes** |
| 3C Inference In-Memory M2 Max Diff | `1.11e-16` | True | `m3_t14_results_summary.json` | `1.11022e-16` | `2.23e-20` | `1e-12` | machine eps | No | **Yes** |

## 2. Prose-Only / Earlier-Report Numbers (No Direct MLflow/M3-Summary Check)

- 100,000 rows, 42 raw columns, 50 stores, 5 regions, 20 suppliers, 10 categories, 63 products (README.md / M1-T04)
- 80,042 train / 19,958 test cutoff 2024-08-07 (M1-T04 / M2-T11)
- M3-T03 max |sum(phi_j) - logit_diff| = 4.441e-16; max prob-pts sum error = 1.260e-13 (M3_T03_Explainability_Report.md)
- M3-T03 top grouped drivers: packaging_score 0.1919, handling_score 0.1203, days_since_receipt 0.0840, days_until_expiry 0.0818, temp_deviation 0.0702 (M3_T03_Explainability_Report.md)
- M3-T03 proposal key category x region x store >= 129 rows across 500 cells (M3_T01 / M2-T12)
- M3-T06 batch profit RMSE $1,976 -> $1,298 excluding 1% tail (M3_T06_Regressor_Training_Report.md)
- M3-T07 Test R1 batch profit RMSE $1,793 ($1,204 excluding 1% tail) (M3_T07_Regressor_Evaluation_Report.md)
- M3-T08 4 categories < 8/13 depth bins: Dairy 5/13, Beverages 3/13, Pharmaceuticals 3/13, Frozen_Meals 3/13 (M3_T08_Elasticity_View_Report.md)
- M3-T08 10-30% common band range: +1.84 (Frozen_Meals) to +4.29 (Meat); 5/10 in model-free CI (M3_T08_Elasticity_View_Report.md)
- M3-T10 representative examples: record_id 25554 (Produce C, 0.15), 18328 (Seafood A, 0.25), 90055 (Bakery C, 0.35), 73650 (Bakery B, 0.50) (M3_T10_Tier_Mapping_Report.md)
- M3-T12 Test R1 flag rates: support_edge 77.75%, limited_depth_category 19.28%, low_history_3a 16.35%, critical_tier 0.10%, near_tie 96.97% (M3_T12_Confidence_Propagation_Report.md)
- M3-T14 defaults JSON size: 583,751 bytes; snapshot 2024-12-31, 145 rows x 45 cols (M3_T14_Inference_Pipeline_Report.md)
