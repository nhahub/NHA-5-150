# Smart Shelf: Perishable Food Waste Reduction and Dynamic Markdown Optimizer

An end-to-end machine learning decision support system designed to reduce supermarket perishable food waste and maximize recovered revenue through dynamic markdown optimization.

---

## 1. Project Overview & Architecture

Retailers often make clearance decisions through staff observation rather than data, leading to costly outcomes: marking items down too early (sacrificing margin) or marking them down too late (total product loss).

**Smart Shelf** combines two connected machine learning models into an automated decision engine:
1. **Track 3A — Spoilage Risk Classifier:** Predicts whether an inventory batch will spoil before selling out using pre-sale features and handling quality signals (`temp_abuse_events`, `handling_score`, `packaging_score`, `supplier_score`).
2. **Track 3B — Revenue-Response Regressor:** Learns the relationship between markdown depth (`discount_pct`) and resulting profit/revenue, controlling for reverse causality and discounting bias using Track 3A's out-of-fold predicted risk, `is_promoted`, and sales velocity.
3. **Track 3C — Discount Optimizer:** Evaluates candidate discount depths on a discrete grid ($0\%$ or $[0.10, 0.75]$) to recommend the profit-maximizing discount tier for each batch, audited via a two-lane validation framework (Lane 1: Matched Historical, Lane 2: Counterfactual Simulated).
4. **Milestone 4 — Deployment & Simulator:** A Streamlit application featuring a **Manager Inventory View** and an interactive **Simulator Mode** where store managers input raw batch parameters to test what-if scenarios.

---

## 2. Milestone 1: Deliverables & Notebook Map

Milestone 1 establishes the verified data foundation, data leakage boundaries, and preprocessing pipelines:

| Task ID | Component / Notebook | Description & Key Outcomes |
| :--- | :--- | :--- |
| **M1-T01** | `M1-T01.ipynb` | **Dataset Verification & Ingestion:** Confirmed 100,000 transactions, 42 columns, 0 missing values. Documented schema corrections and established strict data leakage exclusion boundaries (`spoilage_risk` reserved for benchmark only; post-sale outcome columns excluded). |
| **M1-T02** | `M1-T02.ipynb` | **Integrity Check (`markdown_applied` vs. `discount_pct`):** Verified 30,651 markdown rows. Disproved potential contradictions (`markdown_applied == 1` with `discount_pct == 0` occurs in 0 rows). Proved prices match base price $\times (1 - \text{discount})$ within cent rounding. |
| **M1-T03** | `M1-T03.ipynb` | **Exploratory Data Analysis (EDA):** Generated distribution, correlation, and temporal figures in `figures/`. Confirmed baseline spoilage rate of $19.44\%$ and absence of global multi-year trend/seasonality. |
| **M1-T04** | `M1-T04.ipynb` | **Preprocessing & Feature Encoding:** Built a `ColumnTransformer` (11 $\log(1+x)$ scaled, 12 standardized, 148 one-hot, 7 binary/cyclical features). Split strictly time-ordered (Train: $\le \text{2024-08-07}$, 80,042 rows; Test: $> \text{2024-08-07}$, 19,958 rows). Scalers fitted on train only. |
| **M1-T05** | *[In Progress]* | **Consolidated EDA & Preprocessing Report:** Active synthesis deliverable documenting data exploration, data-leakage controls, and analytical decisions for stakeholder review. |
| **M1-T06** | `M1-T06.ipynb` | **Spoilage-Sensitivity Verification:** Audited `spoilage_sensitivity`. Proved it is a static category-level attribute (9 distinct values, $R^2 = 1.0$ against category one-hot) and cleared it as a legitimate pre-sale feature (allowed) while noting multicollinearity constraints. |

---

## 3. Repository Structure

```
├── figures/                          # Generated static PNGs and interactive HTML plots
│   ├── 01_target_balance.png
│   ├── 06_correlation_heatmap.png
│   └── interactive_monthly_by_category.html
├── processed/                        # Processed outputs and schema contracts
│   ├── check_presale_inputs.py       # Pre-sale input validation and accounting identity check
│   ├── feature_dictionary.csv        # Feature role, transformation, and Track 3A/3B use flags
│   ├── m1_t04_preprocessing_report.json
│   ├── m1_t04_report_text.md
│   ├── m1_t06_report_text.md
│   └── m1_t06_verdict.json
├── src/                              # Modular Python packages (M2/M3 modeling & M4 simulator)
│   ├── __init__.py
│   ├── pipeline.py                   # End-to-end PerishableFeatureEngineer & ColumnTransformer
│   └── data_loader.py                # Confounding-free data loader for Track 3A & Track 3B
├── M1-T01.ipynb                      # Task 1: Verification & schema mapping
├── M1-T02.ipynb                      # Task 2: Discount & markdown integrity check
├── M1-T03.ipynb                      # Task 3: Exploratory Data Analysis
├── M1-T04.ipynb                      # Task 4: Preprocessing & feature engineering
├── M1-T06.ipynb                      # Task 6: Spoilage sensitivity verification
├── verify_m1.py                      # Master automated verification script
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

*Note: Large datasets (`*.csv`) and serialized model artifacts (`*.joblib`, `*.pkl`) are excluded from version control per project governance.*

---

## 4. Setup & Verification

### Installation
Clone the repository and install the verified dependencies:
```bash
git clone <repo-url>
cd NHA-5-150
pip install -r requirements.txt
```

### Running Automated Audits
To verify the integrity of the Milestone 1 outputs:
```bash
# 1. Run master verification suite (checks shapes, scaling, splits, and redundancies)
python verify_m1.py

# 2. Run pre-sale independence and accounting identity diagnostics
python processed/check_presale_inputs.py
```

---

## 5. Development & Contribution Policies

1. **Branching Model:**
   * Never commit directly to `main`.
   * Create dedicated feature branches named after the Task ID:
     ```bash
     git checkout -b <Task-ID>-<short-description>
     # Example: git checkout -b M1-pipeline-setup
     ```
2. **Commit Conventions:**
   * All commit messages must be prefixed with the active Task ID:
     ```bash
     git commit -m "<Task-ID>: <clear description of changes>"
     ```
3. **Artifact Hygiene:**
   * Under no circumstances should datasets (`*.csv`, `*.xlsx`) or serialized binaries (`*.joblib`, `*.pkl`) be committed to Git.
