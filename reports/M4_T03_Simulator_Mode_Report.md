# M4-T03 Report: Dashboard Simulator Mode

**Task:** M4-T03 - Dashboard: Simulator Mode (depends on M4-T02)
**Branch:** `M4-T03-simulator-mode`

## 1. What was built

A **Simulator Mode** tab in the Streamlit dashboard (`app.py`, next to the M4-T02 Manager View).
The user enters a hypothetical batch and sees:

- the spoilage chance and risk band (Track 3A),
- the what-if scoring for **every** candidate discount from 10% to 75% (Track 3B),
- the optimizer's recommended discount and tier (Track 3C), marked with a star,
- the confidence level and the plain-language reasons behind it (M3-T12 + M3-T14 rules),
- an optional comparison of the manager's own discount against the recommendation.

"Done when" criterion: *the user can input a hypothetical batch (missing fields filled with the M3-T14 documented defaults) and see full what-if scoring across candidate discounts, the optimizer's recommendation, and its confidence.*

## 2. Design

- **No new modelling.** The tab calls the frozen `src.m3_t14_pipeline.simulate()` in `hypothetical` mode. Nothing is re-fitted or re-computed, so results equal the M3 results.
- **Inputs.** Required: category, days until expiry, units, base price, cost price. Optional (product, store, quality grade and 15 more under "More inputs"): anything left on *Default* is omitted from the batch, so the pipeline fills it with the training-only category median/mode from `models/track_3_inference_defaults.json`.
- **Confidence.** Unchanged M3-T14 rules apply: while any key field (`packaging_score`, `handling_score`, `days_since_receipt`, `temp_deviation`) is on default, confidence is capped at Low. The tab lists which key fields are still on default.
- **No-discount batches (R0).** Batches with more than 40% of shelf life left and grade A/B get no recommendation; the tab explains why instead of showing an empty table.
- **Files.** Logic and UI in `src/m4_t03_simulator.py` (paths from `src/config.py`); checks in `scripts/M4_T03_verify_simulator.py`; `app.py` wires the tab in.

## 3. Verification

`python scripts/M4_T03_verify_simulator.py` on the real models and data: **15/15 checks passed** (`reports/M4/m4_t03_verification.json`).
The five documented M3-T14 hypothetical cases (H1-H5) reproduce region, spoil chance, recommended depth (0.25, 0.15, 0.10, 0.25), expected profit, confidence level and `defaults_dominate` exactly. Higher `handling_score` lowers spoilage risk (0.2859 -> 0.2760), matching the frozen Track 3A coefficient direction. The recommended depth equals the best supported depth on the what-if curve.

## 4. Deviations from plan

- `app.py` was restructured into two tabs. The page title changed to "Smart Shelf Dashboard", the sidebar header to "Manager View filters", and `st.stop()` became `return` so a data-loading error in the Manager View no longer blocks the Simulator tab. Manager View logic is otherwise untouched.
- The Simulator shows the `Moderate` confidence level (yellow) because the pipeline returns `Moderate`, not `Medium`. The Manager View colour map has no entry for `Moderate`; left unchanged (suggested follow-up).
- Running the pipeline rewrites `models/track_3b_region_support_ref.json` with floating-point differences around 1e-12 (existing M3 behaviour). That file must not be committed from this task.

## 5. Reviewer checklist

1. Pull the branch, place the model `.joblib` files in `models/` and the CSVs in `data/processed/`.
2. Run `python scripts/M4_T03_verify_simulator.py` (expect 15/15).
3. Run `streamlit run app.py`, open **Simulator Mode**: Seafood with 1 day left should show a starred recommendation; Dairy with 10 days left should show "No discount recommended".

All profit figures are simulated estimates. Advisory estimate: the observed-data backtest was inconclusive; see the validation report.
