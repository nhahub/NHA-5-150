# M3-T13: Milestone 3 Optimizer Validation Report (Track 3C & Inference Pipeline)

**Task:** M3-T13 — Compiled Optimizer Validation Report  
**Branch:** `M3-T13-m3-deliverable-reports`  
**Scope:** Strictly compiled from existing M3 task reports (`M3-T09`–`M3-T12`, `M3-T14`), `reports/M3/*.json` summaries, and MLflow runs in `smart-shelf-3c-discount-optimizer`. Lane 1 and Lane 2 are strictly separated and never combined.

---

## 1. Scope Statement

The optimizer evaluates a **single, one-time markdown decision** at the point a batch is inspected (Design Decision D4). It does **not** model a multi-stage markdown trajectory across days. Batches in region `R0` (`expiry_remaining_ratio > 0.40` and `quality_grade != 'C'`) receive **no recommendation** (`status="no_markdown_historical_practice"`, `reason="no_discount_region"`) and are excluded from both validation lanes.

---

## 2. Optimizer Design & Deviations from Proposal/Card (`M3-T09`)

*Sources: `M3_T09_Optimizer_Report.md`, `m3_t09_results_summary.json`, MLflow run `63724d725d91460e82b41f37da04825c`.*

- **D1 (Candidate Grid — Card Deviation):** Instead of `{0, 10, 20, 30, 40, 50}%` across all batches, `R0` receives no markdown, and `R1` batches are scored over `{0.10, 0.15, ..., 0.75}` filtered strictly to depths marked `supported=True` by `discount_support()` (`0` `R1` rows had zero supported depths in train or test).
- **D2 (Objective & Tie-Breaking):** Maximizes expected batch profit `initial_quantity * (s_hat * round(base_price * (1 - d), 2) - cost_price)`; exact ties go to the shallower depth.
- **D3 (`flat_curve` Flag):** `margin_per_unit` = best minus second-best predicted profit/unit; `flat_curve = True` if `margin_per_unit < $1.8337/unit` (frozen Track 3B CV `profit_unit_mae`).
- **Descriptive Output (`Test R1`, `N = 6,176`):** `4.7%` at `0.10`, `48.9%` at `0.15`, `4.0%` at `0.20`, `36.6%` at `0.25`, `0.2%` at `0.30`, `5.4%` at `0.35`, `0.1%` at `0.50` (`76.2%` at shallowest supported depth; **`97.0%` `flat_curve = True`**; median `margin_per_unit = $0.24`). Mean `rec - actual = -0.1356` (`14.4%` within `±0.025`).

---

## 3. Tier Mapping (`M3-T10`)

*Sources: `M3_T10_Tier_Mapping_Report.md`, `m3_t10_results_summary.json`, `models/track_3c_tier_edges.json`, MLflow run `4774b6f6e1c04181ac7a5441bf22c828`.*

- **Pre-Registered Edges (`Train R1`, `N = 24,475`):** `Edge1 = 0.15` ($F = 55.64\%$), `Edge2 = 0.25` ($F = 94.57\%$), `Edge3 = 0.35` ($F = 99.96\%$). `R0` receives `tier = None` (`"No markdown (historical practice)"`).

| Tier | Depth Range | Train `R1` (`N = 24,475`) | Test `R1` (`N = 6,176`) | Representative Test Example (`record_id`) |
| :--- | :---: | :---: | :---: | :--- |
| **Low** | `0.10 – 0.15` | `13,617` (`55.64%`) | `3,316` (`53.69%`) | `25554` (`Produce`, Grade C, `ratio=0.600`, rec `0.15`) |
| **Medium** | `0.20 – 0.25` | `9,529` (`38.93%`) | `2,508` (`40.61%`) | `18328` (`Seafood`, Grade A, `ratio=0.000`, rec `0.25`) |
| **High** | `0.30 – 0.35` | `1,319` (`5.39%`) | `346` (`5.60%`) | `90055` (`Bakery`, Grade C, `ratio=0.000`, rec `0.35`) |
| **Critical** | `0.40 – 0.50` | `10` (`0.04%`) | **`6` (`0.10%`, `< 20`)** | `73650` (`Bakery`, Grade B, `ratio=0.200`, rec `0.50`) |

- **Rare `Critical` Tier:** Contains only `10` train and `6` test batches (`< 20`), flagged by `critical_tier` in confidence propagation.

---

## 4. Confidence Propagation (`M3-T12`)

*Sources: `M3_T12_Confidence_Propagation_Report.md`, `m3_t12_results_summary.json`, MLflow run `2143aa1328194920b3293b121d7ca2d8`.*

- **Levels & Why There Is No `"High"` Level:** `"No recommendation"` (`R0`), `"Low"`, `"Moderate"`. There is **deliberately no `"High"` level** because Lane 1 observed validation was inconclusive (`-$1.44/unit`) and `70.1%` of Lane 2 simulated gains sit below the regressor's CV MAE (`$1.8337/unit`).
- **Weakest-Link Rule (`R1`):** `Low` if **any** of `{F3A low_history_3a (<20 train rows), F3B1 limited_depth_category (<8/13 bins), F3B2 support_edge (at shallowest/deepest supported depth), F3B3 critical_tier}` is `True`; else `Moderate`. `near_tie` (`flat_curve`, `96.97%` test `R1`) is informational only.
- **Test `R1` Shares (`N = 6,176`):** **`Low`: `84.12%` (`5,195`)**, **`Moderate`: `15.88%` (`981`)**. Flag rates: `support_edge` `77.75%`, `limited_depth_category` `19.28%`, `low_history_3a` `16.35%`, `critical_tier` `0.10%`.
- **Error-by-Level Check & Caveats (`Test R1` actual depth):**
  - `Low` (`n = 5,195`): profit/unit MAE **`$1.86`** (95% CI `[$1.69, $2.03]`), RMSE **`$6.92`** (`[$6.11, $7.58]`), `s` MAE `0.1087`.
  - `Moderate` (`n = 981`): profit/unit MAE **`$1.04`** (`[$0.96, $1.13]`), RMSE **`$1.48`** (`[$1.35, $1.60]`), `s` MAE `0.1051`.
  - *Caveats:* Flags measure data support rather than guaranteed accuracy; `low_history_3a` alone showed no calibration gap; `Moderate` remains an advisory estimate.

---

## 5. Two-Lane Validation (`M3-T11`)

*Sources: `M3_T11_Optimizer_Validation_Report.md`, `m3_t11_results_summary.json`, MLflow run `b62e8893e1b9490db3176f715c576c19`.*

### 5.1 Share of Business Not Addressed (`R0`)
- **`13,782 / 19,958` (`69.06%`)** of test batches lie in `R0` and receive no discount recommendation (`0` `R0` rows appear in either lane).

### 5.2 Lane 1 — Observed Evidence (`Followed` vs. `Not Followed`, No Model Grading Itself)
- **Coverage (`tol = 0.025`):** `8 / 15` populated `(expiry × grade × oof_proba)` strata with `>= 20` followed and `>= 20` not-followed batches (`4,841 / 6,176 = 78.38%` of test `R1`; `849` followed, `3,992` not followed).
- **Estimates (`Followed − Not Followed`, 1,000-resample week-grouped bootstrap):**
  - **Observed Profit / Initial Unit:** **`-$1.4421`** (`95% CI [-$3.2437, +$0.3666]`)
  - Observed Sell-Through `s`: **`-0.1169`** (`95% CI [-0.1353, -0.0976]`)
  - Observed Waste Share: **`+0.1169`** (`95% CI [+0.0976, +0.1353]`)
  - Observed Batch Profit: **`-$163.33`** (`95% CI [-$802.82, +$436.42]`)
- **Pre-Registered Verdict:** **`inconclusive`** (CI crosses `$0.00`; sensitivity `tol = 0.050` also inconclusive at `-$1.2846`, CI `[-$2.4503, +$0.2602]`). Plainly: the point estimate is negative, sell-through (`-0.117`) and waste-share (`+0.117`) effects agree with M3-T05/T07 demand slopes, and **the data provide no support for the claim that following the recommendation raised observed profit**.
- **Negative Control:** `200` within-stratum permutations ignoring week clustering and heavy tails gave null mean `+$0.0089` (`SD $0.4381`, `p = 0.000`); not directly comparable to the week-grouped bootstrap CI.

### 5.3 Lane 2 — Simulated / Model-Estimated, Not Observed (`N = 5,289`)
- *All numbers in Lane 2 are simulated / model-estimated, not observed.*
- **Mean Simulated Gain:** **`+$1.7838/unit`** (`95% CI [+$1.6850, +$1.8705]`, median `+$0.7026/unit`; batch mean `+$458.87`, median `+$158.42`).
- **Context Beside Headline Gain:** **`70.09%` (`70.1%`) of simulated gains are smaller than the regressor's CV MAE (`$1.8337/unit`)**, and on these same batches at their actual depth, the regressor's error is **`MAE = $1.5976/unit`**, **`RMSE = $5.9696/unit`**, mean error `+$0.0705/unit`.

> **What Can and Cannot Be Claimed**
> - **CAN Claim:** The optimizer is an **advisory decision-support tool for `R1` batches** (`30.94%` of test); positive demand response inside `R1` is supported by observed data (`M3-T05`/`T07`); and **`R0` (`69.06%`) is outside the optimizer's scope**.
> - **CANNOT Claim:** **Proven profit lift** (Lane 1 is inconclusive with negative point estimate); **any combined improvement percentage** across lanes; or **any evidence on whether to discount at all** (`0%` vs `>0%`).

---

## 6. Inference Pipeline & Inventory Snapshot (`M3-T14`)

*Sources: `M3_T14_Inference_Pipeline_Report.md`, `m3_t14_results_summary.json`, MLflow run `71eb17c0774145679411040cb27ba251`.*

- **Parity (`mode="real"`):** On `test_random_1000` (`2.622 s`) and `snapshot_2024-12-31` (`145` rows, `0.150 s`), `region`, `status`, `tier`, `confidence_level`, `confidence_reasons`, and R1 `recommended_depth` match **`100.0%`**, and R1 `predicted_profit_at_rec` matches to `1.82e-12` / `2.27e-13`. Documented `spoil_proba` tolerance deviation: `3.61e-07` (`test_random_1000`) / `2.94e-07` (`snapshot`) due to 6-significant-digit float serialization in `model_ready_m2_dataset.csv`; scoring stored M2 test rows directly gives **`1.11e-16`** max diff across all `19,958` test rows.
- **Training-Only Defaults (`models/track_3_inference_defaults.json`):** `583,751` bytes, aggregated training medians/modes only (`N = 80,042`).
- **Inventory Snapshot (`data/processed/inventory_snapshot.csv`):** `2024-12-31`, `145` rows × `45` pre-decision columns (all 12 post-sale leakage columns removed during M3-T14 audit). `107` `R0` rows (**`73.79%`**) receive `"No recommendation"`; `38` `R1` rows receive recommendations (`Low: 24, Medium: 12, High: 2`; confidence `Low: 29, Moderate: 9`).
- **P3 Defaulted-Input Rule:** Forces `confidence_level = "Low"` (`"defaults_dominate"`) on R1 when any key sensitive field (`packaging_score`, `handling_score`, `days_until_expiry`, `days_since_receipt`, `temp_deviation`) is defaulted **or** when `> 50%` of the 19 optional fields are defaulted *(the `> 50%` trigger was added during M3-T14 development/audit alongside the 5 key-field triggers)*. Never raises a level.

---

## 7. Limitations

1. **Observed Backtest Inconclusive:** Lane 1 point estimate is negative (`-$1.4421/unit`, CI `[-$3.2437, +$0.3666]`); no proven profit lift.
2. **Narrow Addressable Scope:** `69.06%` of test batches (`73.79%` of snapshot) are `R0` and receive no discount recommendation.
3. **Flat Profit Curves:** `97.0%` of test `R1` recommendations have `margin_per_unit < $1.8337` (`flat_curve = True`).
4. **Heavy Low-Confidence Share & Synthetic Data:** `84.12%` of test `R1` batches are `Low` confidence (`0%` `High`), and all dynamics come from synthetic data.
