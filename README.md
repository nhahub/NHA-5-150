# Smart Shelf: Perishable Food Waste Reduction and Dynamic Markdown Optimizer

[![Milestone 1 Status](https://img.shields.io/badge/Milestone%201-Completed%20(36%2F36%20Checks%20Passed)-brightgreen)](#2-milestone-deliverables--task-map)
[![Milestone 2 Status](https://img.shields.io/badge/Milestone%202-In%20Progress%20(M2--T01%20Complete)-blue)](#2-milestone-deliverables--task-map)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)](requirements.txt)
[![Code Architecture](https://img.shields.io/badge/Architecture-End--to--End%20Pipeline%20%26%20Data%20Loader-orange)](#4-repository-structure)

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

## 2. Milestone Deliverables & Task Map

### Milestone 1: Data Audit, Leakage Boundary & Preprocessing Foundation
| Task ID | Component / Notebook | Status | Description & Key Outcomes |
| :--- | :--- | :---: | :--- |
| **M1-T01** | [`notebooks/M1-T01.ipynb`](notebooks/M1-T01.ipynb) | **Done** | **Dataset Ingestion & Schema Audit:** Verified 100,000 transactions, 42 columns, and 0 missing values. Documented schema deviations against the initial proposal and established strict data-leakage boundaries (`spoilage_risk` reserved for benchmark only; all post-sale outcome columns excluded). |
| **M1-T02** | [`notebooks/M1-T02.ipynb`](notebooks/M1-T02.ipynb) | **Done** | **Discount & Markdown Integrity Check:** Verified 30,651 markdown rows (30.7%). Proved `markdown_applied == (discount_pct > 0)` in 100% of rows (disproving the proposal's 0-discount contradiction hypothesis). Confirmed `selling_price` matches `base_price * (1 - discount_pct)` within cent rounding. |
| **M1-T03** | [`notebooks/M1-T03.ipynb`](notebooks/M1-T03.ipynb) | **Done** | **Exploratory Data Analysis (EDA):** Generated static distributions, boxplots, heatmaps, and temporal charts in [`reports/figures/m1/`](reports/figures/m1/). Confirmed baseline spoilage rate of 19.44% and absence of global multi-year trend or seasonality. |
| **M1-T04** | [`notebooks/M1-T04.ipynb`](notebooks/M1-T04.ipynb) | **Done** | **Preprocessing & Feature Encoding:** Built a scikit-learn `ColumnTransformer` (11 `log(1+x)` scaled, 12 standardized, 148 one-hot, 7 binary/cyclical features = 178 total features). Enforced a strict chronological split (Train: `<= 2024-08-07`, 80,042 rows; Test: `> 2024-08-07`, 19,958 rows) with zero lookahead bias. Scalers fit exclusively on training data. |
| **M1-T05** | [`notebooks/M1-T05.ipynb`](notebooks/M1-T05.ipynb)<br>[`reports/M1/EDA_Report_Milestone1.md`](reports/M1/EDA_Report_Milestone1.md) | **Done** | **Interactive Visualizations & Synthesis Report:** Created 9 CDN-backed interactive Plotly HTML charts in [`reports/figures/interactive/`](reports/figures/interactive/) with assertion-guarded metrics. Compiled the comprehensive Milestone 1 EDA & Preprocessing Report containing full proposal reconciliations. |
| **M1-T06** | [`notebooks/M1-T06.ipynb`](notebooks/M1-T06.ipynb) | **Done** | **Spoilage-Sensitivity Verification:** Proved `spoilage_sensitivity` is a static category-level attribute (9 distinct values, `R² = 1.0` against category one-hot). Cleared it as a legitimate pre-sale feature (`track_3a_use = "allowed"`) while documenting exact collinearity constraints. |

### Milestone 2: Statistical Analysis & Advanced Feature Engineering
| Task ID | Component / Notebook | Status | Description & Key Outcomes |
| :--- | :--- | :---: | :--- |
| **M2-T01** | [`notebooks/M2_T01_statistical_analysis.ipynb`](notebooks/M2_T01_statistical_analysis.ipynb)<br>[`src/m2_t01_analysis.py`](src/m2_t01_analysis.py)<br>[`reports/M2/M2_T01_Spoilage_Drivers_Report.md`](reports/M2/M2_T01_Spoilage_Drivers_Report.md) | **Done** | **Statistical Analysis of Spoilage Drivers:** Completed Chi-Square tests of independence, ANOVA $F$, Welch's $t$, Mann-Whitney $U$, Cohen's $d$, and multiple-testing adjustments (Holm & Bonferroni) across 24 factors. Confirmed `quality_grade` is meaningful ($\chi^2 = 821.74, V = 0.091$), `category` has weak effect ($V = 0.039$), and `region` is independent ($p_{\text{adj}} = 1.0$). Established feature contracts in [`reports/tables/master_factor_ranking.csv`](reports/tables/master_factor_ranking.csv). |

---

## 3. Key Empirical Findings & Proposal Reconciliations

Milestone 1 & 2 audits reconciled earlier planning assumptions against the ground truth data:

1. **Markdown Consistency:** The suspected contradiction (`markdown_applied = 1` with `discount_pct = 0`) occurs in **0 rows**. `markdown_applied` is exactly the indicator `discount_pct > 0`.
2. **Empirical Discount Grid:** Real-world discounts take 66 distinct values strictly within `[0.10, 0.75]` (median 0.25). No discounts exist between 0 and 0.10. Track 3C's search space is therefore `{0} ∪ [0.10, 0.75]`.
3. **Date Lag Phenomenon:** `days_remaining_at_purchase` exactly matches the physical date gap. However, `days_until_expiry` is 0 to 3 days lower in **74,191 rows (74.2%)**. Both `shelf_life_used_ratio` (date gap) and `expiry_remaining_ratio` (proposal definition) are exported to preserve full signal.
4. **Sales Velocity Feasibility:** Only **26.2%** of batches have an earlier sale of the same product in the same store in the trailing 7 days. The category × store fallback covers **84.9%** and is designated as the primary velocity proxy for Milestone 2.
5. **Track 3A Signal Ceiling & Probability Calibration:** Individual pre-sale features have low linear correlation with spoilage (max `|r| = 0.091`). Test AUC tops out near `~0.59`. Consequently, Track 3A prioritizes probability calibration (Brier score, reliability curves, log loss) alongside recall.
6. **Pre-Sale Accounting Identity:** `initial_quantity = units_sold + units_wasted` holds in **100.0% of rows**. Pre-sale inputs gain 0.000 predictive power from post-sale outcomes (`scripts/check_presale_inputs.py`), verifying `initial_quantity` as the valid pre-sale stock proxy.
7. **Multicollinearity Flagging:** 6 feature pairs exhibit `|r| > 0.90` on scaled train data (`shelf_life_days` ~ `days_remaining_at_purchase` `r = 0.996`; `base_price` ~ `cost_price` `r = 0.991`, etc.). These are bidirectionally documented in [`data/processed/feature_dictionary.csv`](data/processed/feature_dictionary.csv) for regularization in linear models.
8. **Factor Significance Hierarchy:** Spoilage is primarily driven by physical handling quality (`packaging_score` Cohen's $d = -0.225$, `handling_score` $d = -0.143$, `quality_grade` $V = 0.091$), while geographic entity IDs (`region`, `store_id`, `supplier_id`) carry zero statistically significant signal after multiplicity adjustment.

---

## 4. Repository Structure

```
├── data/
│   ├── raw/                  # Raw input datasets (perishable_goods_management.csv)
│   └── processed/            # Cleaned, split & scaled datasets, and feature dictionary
├── models/                   # Fitted preprocessor pipeline artifact (preprocessor.joblib)
├── notebooks/                # Milestone analysis notebooks (M1-T01 to M1-T06, M2_T01)
├── reports/
│   ├── M1/                   # Milestone 1 EDA report and preprocessing audit artifacts
│   ├── M2/                   # Milestone 2 Spoilage Drivers Statistical Report
│   ├── figures/              # Visualizations partitioned by milestone
│   │   ├── m1/               # Static PNG figures from Milestone 1 EDA & preprocessing
│   │   ├── m2/               # Static PNG figures from Milestone 2 statistical analysis
│   │   └── interactive/      # Standalone Plotly HTML interactive charts (M1-T03, M1-T05)
│   └── tables/               # Statistical test results and factor ranking tables (M2-T01)
├── scripts/                  # Automated verification and presale diagnostic scripts
├── src/                      # Production Python modules (config, pipeline, data_loader, analysis)
├── .gitignore                # Git ignore rules for data, models, checkpoints, and caches
├── pyrightconfig.json        # Python typing configuration
├── requirements.txt          # Pinned environment dependencies
├── RESTRUCTURE_LOG.md        # Comprehensive file migration & path audit log
└── README.md                 # Project documentation & execution guide
```

*Note: Datasets (`perishable_goods_management.csv`, `cleaned_dataset.csv`, `model_ready_dataset.csv`) and model binaries (`preprocessor.joblib`) are excluded from Git tracking via `.gitignore` per project data governance standards.*

---

## 5. Getting the Data

### 1. Download Raw Dataset
Download the Kaggle **Perishable Goods Management** dataset:
- Place the raw CSV into `data/raw/` with the exact filename:
  ```bash
  data/raw/perishable_goods_management.csv
  ```
- Expected dimensions: 100,000 rows × 42 columns (SHA-256: `de94302b867c9debedfd45c431306623fdfc038f5ed8ca17736339b4460a6674`).

### 2. Generate Processed Datasets
Run the preprocessing notebook or pipeline to generate all artifacts in `data/processed/` and `models/`:
```bash
jupyter nbconvert --execute --inplace notebooks/M1-T04.ipynb
```
This produces:
- `data/processed/cleaned_dataset.csv` (100,000 rows × 57 columns, date-parsed, chronological train/test split)
- `data/processed/model_ready_dataset.csv` (100,000 rows × 189 columns, 178 scaled/encoded features + outcome columns)
- `data/processed/feature_dictionary.csv` (189 rows with transformation metadata & Track 3A/3B use flags)
- `models/preprocessor.joblib` (fitted scikit-learn preprocessing `ColumnTransformer`)
- `reports/M1/m1_t04_preprocessing_report.json`

---

## 6. How to Run

### Installation
Clone the repository and install dependencies in your virtual environment:
```bash
git clone <repo-url>
cd NHA-5-150
pip install -r requirements.txt
```

### Execution Order

1. **Verify Setup & Ingestion Integrity:**
   ```bash
   # Master verification suite (runs all 36 assertion checks)
   python scripts/verify_m1.py

   # Pre-sale feature independence and accounting identity audit
   python scripts/check_presale_inputs.py
   ```

2. **Milestone 1 Analysis & Exploration Notebooks:**
   Notebooks can be run sequentially via Jupyter or headless with `nbconvert`:
   ```bash
   jupyter nbconvert --execute --inplace notebooks/M1-T01.ipynb
   jupyter nbconvert --execute --inplace notebooks/M1-T02.ipynb
   jupyter nbconvert --execute --inplace notebooks/M1-T03.ipynb
   jupyter nbconvert --execute --inplace notebooks/M1-T04.ipynb
   jupyter nbconvert --execute --inplace notebooks/M1-T05.ipynb
   jupyter nbconvert --execute --inplace notebooks/M1-T06.ipynb
   ```

3. **Milestone 2 Statistical Analysis Pipeline:**
   Execute either via the pure Python pipeline or the companion interactive notebook:
   ```bash
   # Run automated statistical test pipeline (generates tables & figures)
   python src/m2_t01_analysis.py

   # Or run notebook
   jupyter nbconvert --execute --inplace notebooks/M2_T01_statistical_analysis.ipynb
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

## 7. Milestone 2 Transition Roadmap

With the data foundation certified, Milestone 2 is delivering **Advanced Data Analysis & Feature Engineering**:
1. **Statistical Hypothesis Testing (M2-T01 - Complete):** Quantified factor significance across categories, regions, and handling parameters with Holm/Bonferroni corrections.
2. **Sales Velocity & Sell-Through Risk Feature Engineering:**
   - `sales_velocity = (trailing 7-day units sold) / 7` (with category-store fallback)
   - `sell_through_risk = initial_quantity / (sales_velocity * days_until_expiry)`
3. **Store Markdown Frequency:** Build store-level cumulative markdown frequency as an explicit confounding control for Track 3B.
4. **Out-of-Fold Risk Probability Generation:** Implement time-blocked cross-validation inside the training period to produce Track 3A risk probabilities for Track 3B.

---

## 8. Development & Contribution Policies

* **Branching Model:** Never commit directly to `main`. Create feature branches named after the Task ID:
  ```bash
  git checkout -b <Task-ID>-<description>
  ```
* **Commit Conventions:** Prefix all commit messages with the Task ID:
  ```bash
  git commit -m "<Task-ID>: <description>"
  ```
* **Data Hygiene:** Under no circumstances should raw datasets (`*.csv`) or serialized binaries (`*.joblib`, `*.pkl`) be committed to Git.
