# Smart Shelf: Perishable Food Waste Reduction and Dynamic Markdown Optimizer

[![Milestone 1 Status](https://img.shields.io/badge/Milestone%201-Completed%20(36%2F36%20Checks%20Passed)-brightgreen)](#milestone-1-deliverables--task-map)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)](requirements.txt)
[![Code Architecture](https://img.shields.io/badge/Architecture-End--to--End%20Pipeline%20%26%20Data%20Loader-orange)](#3-repository-structure)

An end-to-end, data-driven machine learning decision support system designed to reduce supermarket perishable food waste and maximize recovered revenue through dynamic markdown optimization.

---

## 1. Project Overview & System Architecture

Clearance decisions in retail food chains are traditionally driven by manual staff observation rather than predictive analytics. This leads to two costly operational failures: marking items down too early (sacrificing margin unnecessarily) or marking them down too late (total inventory spoilage).

**Smart Shelf** combines two connected machine learning models into an automated decision engine:

```
                                      [Raw Batch Inputs]
                                               │
                                 ┌─────────────┴─────────────┐
                                 ▼                           ▼
                     [Pre-Sale & Handling Features]     [Candidate Discount Grid]
                     (temp_abuse, handling, lag, etc.)    (0%, 10%, 20%, ..., 75%)
                                 │                           │
                                 ▼                           │
                        ┌──────────────────┐                 │
                        │ Track 3A Model   │                 │
                        │ Spoilage Risk    │                 │
                        └────────┬─────────┘                 │
                                 │                           │
                   Out-of-Fold Spoilage Probability          │
                   + Confounding Controls (is_promoted, etc.) │
                                 │                           │
                                 └─────────────┬─────────────┘
                                               ▼
                                      ┌──────────────────┐
                                      │ Track 3B Model   │
                                      │ Revenue Response │
                                      └────────┬─────────┘
                                               │
                                     Predicted Profit Curve
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │ Track 3C Engine  │
                                      │ Optimal Discount │
                                      └──────────────────┘
```

1. **Track 3A — Spoilage Risk Classifier:** Predicts whether an inventory batch will spoil (`was_spoiled = 1`) before selling out, utilizing pre-sale attributes and concrete handling quality signals (`temp_abuse_events`, `handling_score`, `packaging_score`, `supplier_score`) to power explainable SHAP attributions.
2. **Track 3B — Revenue-Response Regressor:** Learns the relationship between markdown depth (`discount_pct`) and resulting financial outcomes (`profit`, `revenue`, `waste_pct`), adjusting for reverse causality and discounting bias using Track 3A out-of-fold risk scores, `is_promoted`, and trailing sales velocity.
3. **Track 3C — Discount Optimizer:** Evaluates candidate discount depths on an empirical discrete grid (`0%` or `[0.10, 0.75]`) to recommend the profit-maximizing discount tier for each batch. Audited via a two-lane validation framework (**Lane 1: Matched Historical Ground Truth**, **Lane 2: Counterfactual Simulated**).
4. **Milestone 4 — Deployment & Simulator:** A Streamlit application featuring a real-time **Manager Inventory View** and an interactive **Simulator Mode** where store managers input hypothetical batch parameters to test what-if pricing scenarios.

---

## 2. Milestone 1: Deliverables & Task Map

Milestone 1 establishes the verified data foundation, automated leakage boundaries, feature engineering transformers, and exploratory reporting:

| Task ID | Component / Notebook | Status | Description & Key Outcomes |
| :--- | :--- | :---: | :--- |
| **M1-T01** | [`M1-T01.ipynb`](M1-T01.ipynb) | **Done** | **Dataset Ingestion & Schema Audit:** Verified 100,000 transactions, 42 columns, and 0 missing values. Documented schema deviations against the initial proposal and established strict data-leakage boundaries (`spoilage_risk` reserved for benchmark only; all post-sale outcome columns excluded). |
| **M1-T02** | [`M1-T02.ipynb`](M1-T02.ipynb) | **Done** | **Discount & Markdown Integrity Check:** Verified 30,651 markdown rows (30.7%). Proved `markdown_applied == (discount_pct > 0)` in 100% of rows (disproving the proposal's 0-discount contradiction hypothesis). Confirmed `selling_price` matches `base_price * (1 - discount_pct)` within cent rounding. |
| **M1-T03** | [`M1-T03.ipynb`](M1-T03.ipynb) | **Done** | **Exploratory Data Analysis (EDA):** Generated static distributions, boxplots, heatmaps, and temporal charts in [`figures/`](figures/). Confirmed baseline spoilage rate of 19.44% and absence of global multi-year trend or seasonality. |
| **M1-T04** | [`M1-T04.ipynb`](M1-T04.ipynb) | **Done** | **Preprocessing & Feature Encoding:** Built a scikit-learn `ColumnTransformer` (11 `log(1+x)` scaled, 12 standardized, 148 one-hot, 7 binary/cyclical features = 178 total features). Enforced a strict chronological split (Train: `<= 2024-08-07`, 80,042 rows; Test: `> 2024-08-07`, 19,958 rows) with zero lookahead bias. Scalers fit exclusively on training data. |
| **M1-T05** | [`M1-T05.ipynb`](M1-T05.ipynb)<br>[`EDA_Report_Milestone1.md`](EDA_Report_Milestone1.md) | **Done** | **Interactive Visualizations & Synthesis Report:** Created 9 CDN-backed interactive Plotly HTML charts in [`figures_interactive/`](figures_interactive/) with assertion-guarded metrics. Compiled the comprehensive Milestone 1 EDA & Preprocessing Report containing full proposal reconciliations. |
| **M1-T06** | [`M1-T06.ipynb`](M1-T06.ipynb) | **Done** | **Spoilage-Sensitivity Verification:** Proved `spoilage_sensitivity` is a static category-level attribute (9 distinct values, `R² = 1.0` against category one-hot). Cleared it as a legitimate pre-sale feature (`track_3a_use = "allowed"`) while documenting exact collinearity constraints. |

---

## 3. Key Empirical Findings & Proposal Reconciliations

Milestone 1 audits reconciled earlier planning assumptions against the ground truth data:

1. **Markdown Consistency:** The suspected contradiction (`markdown_applied = 1` with `discount_pct = 0`) occurs in **0 rows**. `markdown_applied` is exactly the indicator `discount_pct > 0`.
2. **Empirical Discount Grid:** Real-world discounts take 66 distinct values strictly within `[0.10, 0.75]` (median 0.25). No discounts exist between 0 and 0.10. Track 3C's search space is therefore `{0} ∪ [0.10, 0.75]`.
3. **Date Lag Phenomenon:** `days_remaining_at_purchase` exactly matches the physical date gap. However, `days_until_expiry` is 0 to 3 days lower in **74,191 rows (74.2%)**. Both `shelf_life_used_ratio` (date gap) and `expiry_remaining_ratio` (proposal definition) are exported to preserve full signal.
4. **Sales Velocity Feasibility:** Only **26.2%** of batches have an earlier sale of the same product in the same store in the trailing 7 days. The category × store fallback covers **84.9%** and is designated as the primary velocity proxy for Milestone 2.
5. **Track 3A Signal Ceiling & Probability Calibration:** Individual pre-sale features have low linear correlation with spoilage (max `|r| = 0.091`). Test AUC tops out near `~0.59`. Consequently, Track 3A prioritizes probability calibration (Brier score, reliability curves, log loss) alongside recall.
6. **Pre-Sale Accounting Identity:** `initial_quantity = units_sold + units_wasted` holds in **100.0% of rows**. Pre-sale inputs gain 0.000 predictive power from post-sale outcomes (`check_presale_inputs.py`), verifying `initial_quantity` as the valid pre-sale stock proxy.
7. **Multicollinearity Flagging:** 6 feature pairs exhibit `|r| > 0.90` on scaled train data (`shelf_life_days` ~ `days_remaining_at_purchase` `r = 0.996`; `base_price` ~ `cost_price` `r = 0.991`, etc.). These are bidirectionally documented in [`feature_dictionary.csv`](processed/feature_dictionary.csv) for regularization in linear models.

---

## 4. Repository Structure

```
├── figures/                          # Static EDA PNG figures and large interactive charts
│   ├── 01_target_balance.png
│   ├── 02_numeric_histograms.png
│   ├── 04_outlier_boxplots.png
│   ├── 06_correlation_heatmap.png
│   └── 07_time_patterns.png
├── figures_interactive/              # CDN-backed lightweight Plotly HTML charts (M1-T05)
│   ├── 01_spoilage_by_category.html
│   ├── 02_category_by_grade.html
│   ├── 03_monthly_by_category.html
│   ├── 04_driver_deciles.html
│   ├── 05_spoilage_vs_waste.html
│   ├── 06_markdown_depth.html
│   ├── 07_profit_by_category.html
│   ├── 08_category_region_heatmap.html
│   └── 09_input_correlations.html
├── processed/                        # Audit reports, pre-sale diagnostics, and metadata
│   ├── figures/                      # Preprocessing distribution & skew verification plots
│   ├── check_presale_inputs.py       # Pre-sale independence & accounting identity script
│   ├── feature_dictionary.csv        # Column dictionary with Track 3A/3B use contracts
│   ├── m1_t04_preprocessing_report.json
│   ├── m1_t04_report_text.md
│   ├── m1_t06_report_text.md
│   └── m1_t06_verdict.json
├── src/                              # Reusable Python modules for M2/M3 modeling & M4 simulator
│   ├── __init__.py
│   ├── pipeline.py                   # End-to-end PerishableFeatureEngineer & ColumnTransformer
│   └── data_loader.py                # Confounding-free data loader for Track 3A & Track 3B
├── EDA_Report_Milestone1.md          # Comprehensive Milestone 1 Synthesis & Findings Report
├── M1-T01.ipynb                      # Task 1: Dataset acquisition & schema verification
├── M1-T02.ipynb                      # Task 2: Discount & markdown integrity check
├── M1-T03.ipynb                      # Task 3: Exploratory Data Analysis
├── M1-T04.ipynb                      # Task 4: Preprocessing & feature engineering pipeline
├── M1-T05.ipynb                      # Task 5: Interactive visualizations companion
├── M1-T06.ipynb                      # Task 6: Spoilage-sensitivity verification
├── verify_m1.py                      # Master automated verification suite (36/36 tests)
├── requirements.txt                  # Pinned dependencies
└── README.md                         # Project documentation
```

*Note: Raw datasets (`perishable_goods_management.csv`, `cleaned_dataset.csv`, `model_ready_dataset.csv`) and model binaries (`preprocessor.joblib`) are excluded from Git via `.gitignore` per data governance standards.*

---

## 5. Setup & Verification

### Installation
Clone the repository and install the verified dependencies:
```bash
git clone <repo-url>
cd NHA-5-150
pip install -r requirements.txt
```

### Running Automated Audits
To independently verify the integrity of all Milestone 1 outputs:
```bash
# 1. Run master verification suite (checks shapes, scaling, chronological splits, and redundancies)
python verify_m1.py

# 2. Run pre-sale independence and accounting identity diagnostics
python processed/check_presale_inputs.py
```

### Safe Data Loading for Modeling (Milestones 2 & 3)
To ensure zero treatment confounding in Track 3A and enforce correct treatment isolation in Track 3B, use the built-in loader:
```python
from src.data_loader import load_dataset

# Track 3A: Loads 175 strictly pre-sale features (excludes markdown_applied, discount_pct, selling_price)
X_train_3a, y_train_3a, X_test_3a, y_test_3a = load_dataset(track="3A")

# Track 3B: Loads pre-sale controls + discount_pct as the single treatment
X_train_3b, y_train_3b, X_test_3b, y_test_3b = load_dataset(track="3B")
```

---

## 6. Milestone 2 Transition Roadmap

With the data foundation certified, Milestone 2 will focus on **Advanced Data Analysis & Feature Engineering**:
1. **Statistical Hypothesis Testing:** Conduct formal ANOVA and Chi-Square tests to quantify factor significance across categories and handling parameters.
2. **Sales Velocity & Sell-Through Risk Feature Engineering:**
   - `sales_velocity = (trailing 7-day units sold) / 7` (with category-store fallback)
   - `sell_through_risk = initial_quantity / (sales_velocity * days_until_expiry)`
3. **Store Markdown Frequency:** Build store-level cumulative markdown frequency as an explicit confounding control for Track 3B.
4. **Out-of-Fold Risk Probability Generation:** Implement time-blocked cross-validation inside the training period to produce Track 3A risk probabilities for Track 3B.

---

## 7. Development & Contribution Policies

* **Branching Model:** Never commit directly to `main`. Create feature branches named after the Task ID:
  ```bash
  git checkout -b <Task-ID>-<description>
  ```
* **Commit Conventions:** Prefix all commit messages with the Task ID:
  ```bash
  git commit -m "<Task-ID>: <description>"
  ```
* **Data Hygiene:** Under no circumstances should raw datasets (`*.csv`) or serialized binaries (`*.joblib`, `*.pkl`) be committed to Git.
