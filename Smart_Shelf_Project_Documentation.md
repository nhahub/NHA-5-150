# Smart Shelf: Project Documentation

Graduation project, repository NHA-5-150. Covers project planning, stakeholder analysis, data (database) design and UI/UX design. Status: Milestones 1 and 2 complete, Milestones 3 to 5 in preparation.

## 1. Project Summary

Food retailers lose significant revenue every year because perishable goods spoil. Clearance decisions are usually made by staff observation, which leads to two costly outcomes: marking items down too early and losing margin, or marking them down too late and losing the product entirely.

Smart Shelf is an end-to-end decision-support system. It predicts which inventory batches are at risk of spoiling before they sell out, and a second model learns how revenue responds to different discount depths so the system can recommend the discount expected to recover the most value. The two models are explained transparently and delivered through an interactive dashboard that a store manager could use for daily markdown decisions.

**Data.** The project uses the public Kaggle dataset "Perishable Goods Management": 100,000 synthetic retail transactions, 42 columns, 10 product categories, 63 products, 50 stores, 5 regions and 20 suppliers, from January 2023 to December 2024, with no missing values. Because the data is synthetic, all waste-reduction and revenue-recovery figures in this project are simulated estimates under stated assumptions, not measured real-world outcomes.

**System components.**

| Component | Purpose |
|---|---|
| Track 3A: Spoilage classifier | Predicts the probability that a batch will spoil. Headline metric is calibration (Brier score, log loss, reliability curve), not accuracy. |
| Track 3B: Revenue-response model | Predicts the outcome of a batch (sell-through, from which profit is derived) as a function of the discount applied. |
| Track 3C: Discount optimizer | Scores a set of candidate discounts with the Track 3B model and recommends the one with the highest predicted profit, shown as a Low / Medium / High / Critical tier. |
| Streamlit dashboard | Manager View (inventory with risk flags and recommended discount tier) and Simulator Mode (what-if scoring of a hypothetical batch). |
| MLflow tracking | Records parameters, metrics and artifacts of every training run. |

**Scope boundary.** The system recommends a single, one-time discount decision at the point a batch is inspected. A multi-stage markdown trajectory (a small discount early, a deeper one later) is out of scope and listed as future work.

## 2. Project Planning

### 2.1 Objectives and success criteria

- Build a calibrated spoilage-risk model and a revenue-response model, and combine them into a discount recommendation.
- Keep every claim honest: calibration is the headline for Track 3A, and the final report separates evidence measured on historical data from model-estimated (simulated) results.
- Deliver a working dashboard, reproducible code, experiment tracking, documentation and a final presentation.

The honest ceiling matters for planning. The dataset's own spoilage-risk benchmark reaches a ROC-AUC of about 0.593, and a first baseline model reaches about 0.584 on out-of-fold predictions. The spoilage signal in this data is weak, so the project does not promise high classification accuracy. It promises well-calibrated probabilities and transparent explanations.

### 2.2 Milestones, deliverables and status

| Milestone | Main deliverables | Status |
|---|---|---|
| M1: Data collection, exploration and preprocessing | EDA report, interactive visualizations, preprocessing pipeline, leakage and integrity audits | Complete |
| M2: Advanced analysis and feature engineering | Statistical analysis of spoilage drivers, engineered features (shelf-life ratio, sales velocity, sell-through risk, handling signals, store markdown frequency), enhanced visualizations, MLflow setup, consolidated feature dataset, out-of-fold generator, M2 synthesis report | Complete, with the follow-ups listed in section 6 |
| M3: Model development, optimization and explainability | Track 3A classifier, Track 3B revenue-response model, Track 3C optimizer, SHAP analysis, benchmark comparison, optimizer validation report | Not started (prerequisites done) |
| M4: MLOps, deployment and monitoring | Streamlit dashboard (Manager View and Simulator Mode), MLOps report, monitoring plan, retraining strategy | Not started |
| M5: Final documentation and presentation | Final report with before/after business case, final presentation, worked example | Not started |

Target dates and owners for each task are tracked on the team's Notion board.

### 2.3 Milestone 2 tasks (completed work)

| Task | Result |
|---|---|
| M2-T01 | Statistical tests of 24 spoilage drivers with multiple-testing correction |
| M2-T02 | Shelf-life ratio and past-only sales velocity with a category-by-store fallback |
| M2-T03 | Sell-through risk factor |
| M2-T04 | Discount and markdown history flags (used for analysis only, never as model features) |
| M2-T05 | Data analysis report and feature engineering summary |
| M2-T06 | Quality, handling and promotion signals |
| M2-T07 | Design decision on the out-of-fold spoilage probability used by Track 3B |
| M2-T08 | Store-level historical markdown frequency |
| M2-T09 | MLflow experiment tracking setup (local store per person plus committed run summaries) |
| M2-T10 | Enhanced visualizations |
| M2-T11 | Consolidation of all M2 features into one reproducible dataset, with named feature lists |
| M2-T12 | Out-of-fold generator, shared low-confidence support function, decision checks |
| M2-T13 | MLflow helpers for Milestone 3 (model saving with metadata, calibration curve, confusion matrix, SHAP, OOF logging) |

### 2.4 Milestone 3 work plan

The three tracks can be divided between sub-teams, with these dependencies:

1. **Track 3A first.** A baseline OOF file (`track_3a_oof_train.csv`) is the frozen interface for Track 3B.
2. **Track 3B** starts its confounding diagnosis and target handling against that file while Track 3A is tuned, then swaps in the final OOF probabilities.
3. **Track 3C** builds the what-if scoring function, the discount grid and the two-lane backtest against a placeholder prediction function, then connects the real models.

Design decisions agreed by the team before Milestone 3 (full table in `reports/M2/M2_T11_M3_Design_Decisions.md`):

| Topic | Decision |
|---|---|
| Out-of-fold scheme | Time-blocked 5-fold inside the training period; final model for test rows |
| Feature sets | Track 3A compares a full set against a compact set by Brier score and log loss; Track 3B uses batch features, the OOF probability, discount percentage and confounding controls |
| Shelf-life version | `expiry_remaining_ratio` for both models; the other version only as an ablation |
| Track 3B target | Model sell-through and derive profit with the exact accounting identity |
| Low-confidence flag | Product-by-store cells with fewer than 20 training rows (a documented deviation from the proposal) |
| Model storage | Fixed file names with a JSON metadata file beside each model |
| Sales velocity | Kept as built, limitation documented, ablation planned |

### 2.5 Working method and governance

- One branch per task, a pull request into `main` with at least one reviewer, and no direct commits to `main`.
- Task ID in every notebook, script and report name; all paths come from one configuration module; random seed 42.
- Data and model files are never committed; the pipeline regenerates them from the raw Kaggle file.
- Time-ordered train/test split (training up to 2024-08-07, 80,042 rows; test afterwards, 19,958 rows). The dataset's spoilage-risk column is a benchmark only and never a model feature.
- Deviations from the plan are written down in each task report under "Deviations from plan".
- Task status and deadlines are tracked in Notion. AI coding agents may be used, but the team member is responsible for reviewing and being able to explain every result.

### 2.6 Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Weak spoilage signal in the data | Limited discrimination (ROC-AUC near 0.59) | Focus on calibration, report the benchmark honestly, explain results with SHAP |
| Synthetic data | Results do not transfer directly to real stores | Label every business figure as a simulated estimate; list re-validation as future work |
| Confounding in discount history (heavily marked-down items may differ from others) | Biased discount response | Control for risk, expiry, quality, sales velocity, store markdown frequency and promotion; consider causal methods as a stretch goal |
| Optimizer evaluating its own predictions | Overstated profit gains | Keep two evaluation lanes strictly separate: matched historical evidence versus simulated counterfactuals |
| Data leakage | Invalid models | Enforced checks that reject forbidden columns; past-only features; time-ordered splits |
| Dependencies between tracks | Delays | Frozen interface file; work in parallel against stubs |
| Reproducibility across machines | Results that only work for one person | Pinned requirements, seeded scripts, clean-environment installs tested |
| Schedule pressure | Milestones 3 to 5 contain most of the work | Early definition of interfaces, small pull requests, status tracking in Notion |

## 3. Stakeholder Analysis

The analysis is based on the project's use case and general retail practice. In a real deployment it should be validated with interviews of the people listed below.

### 3.1 Stakeholder register

| Stakeholder | Role | Interest in the system | What they need from it | Influence on success |
|---|---|---|---|---|
| Store managers and inventory staff | Primary users | Make daily markdown decisions with less guesswork and less waste | Simple view of which batches are at risk, a clear recommended discount, a short reason, and a visible warning when data is limited | High |
| Category and merchandising managers | Secondary users | Margin and sales per category | Category-level spoilage trends, discount effects, revenue recovered | Medium |
| Operations and finance leadership | Decision makers | Reduce waste cost, protect margin | Evidence of impact that is credible, not inflated; clear separation of measured and simulated results | High |
| Supply chain and procurement | Upstream partners | Reduce loss before the shelf (storage, handling, packaging, supplier quality) | Insight into which handling and supplier factors relate to spoilage | Medium |
| Sustainability and compliance teams | Interested party | Lower food waste | Waste reduction reporting that is transparent about assumptions | Low to medium |
| Customers | Indirect | Fresh products at fair prices | Not affected negatively: recommendations should not mean selling products past their safe life | Low |
| Suppliers | Indirect | Quality feedback | Fair, transparent use of supplier-level signals | Low |
| Course instructor and evaluators | Reviewers | Rigor, reproducibility, honest reporting | Complete repository, clear documentation, defensible methodology | High |
| Project team | Builders | Deliver a working, well-documented system | Clear task split, stable interfaces, reproducible environment | High |

### 3.2 Influence and interest

| | Low interest | High interest |
|---|---|---|
| **High influence** | Operations and finance leadership (keep informed with a concise business case) | Store managers, instructor, project team (involve closely) |
| **Low influence** | Customers, suppliers (monitor) | Category managers, supply chain, sustainability (keep informed) |

### 3.3 What each key group expects

- **Store managers** want a decision, not a model. The dashboard therefore shows a risk level, a discount tier and a one-line reason first, and the details on demand.
- **Leadership and reviewers** want trustworthy numbers. The project reports calibration as the headline, states the weak signal openly, and never mixes evidence measured on matched historical batches with model-estimated counterfactual results.
- **Everyone** is affected by the confidence flag. Where a product has little history at a store, the dashboard says "limited history for this product at this store". The team found no measurable difference in model error between flagged and unflagged rows, so the wording describes sparse data, not low model accuracy.

### 3.4 Concerns and how the design addresses them

| Concern | Response |
|---|---|
| "Can I trust the recommendation?" | Confidence flag, SHAP explanation per batch, tier mapping documented from the optimizer's actual output |
| "Is the claimed saving real?" | Two separate evaluation lanes, labeled simulated where appropriate; synthetic-data statement in the final report |
| "Will it recommend impossible discounts?" | The optimizer only searches 0% and 10% to 75%, the range that exists in the data, and never extrapolates beyond it |
| "Is it reproducible?" | Pinned dependencies, seeded scripts, experiment tracking, documented regeneration steps |

## 4. Data (Database) Design

The project does not use a database server. Data lives in versioned-by-script CSV files and an experiment-tracking store, so the data design describes the source schema, the entities inside it, the derived datasets and how data flows through the system.

### 4.1 Source dataset and entities

The raw file `perishable_goods_management.csv` has one row per batch transaction (primary key `record_id`). The rows reference five dimensions:

| Entity | Identifier | Size | Notes |
|---|---|---|---|
| Category | `category` | 10 | Dairy, Meat, Seafood, Produce, Bakery, Deli, Frozen Meals, Beverages, Pharmaceuticals, Ready-to-Eat. `spoilage_sensitivity` is constant within each category. |
| Product | `product_name` | 63 | Belongs to a category |
| Region | `region` | 5 | |
| Store | `store_id` | 50 | Each store belongs to exactly one region |
| Supplier | `supplier_id` | 20 | |
| Batch transaction | `record_id` | 100,000 | The fact table: one batch received, priced, sold or wasted |

```
Region (5) 1 ---- N Store (50) 1 ---- N Batch transaction (100,000) N ---- 1 Product (63) N ---- 1 Category (10)
                                                  N
                                                  |
                                                  1
                                            Supplier (20)
```

The dimensions are stored as columns of the single source table (a denormalized, flat layout), which suits model training.

### 4.2 Key columns of the batch transaction table

The complete list with each column's role and its permitted use per track is in `data/processed/feature_dictionary.csv`; the main groups are:

| Group | Columns |
|---|---|
| Identifiers and dates | `record_id`, `transaction_date`, `expiration_date` |
| Product and quality | `product_name`, `category`, `shelf_life_days`, `spoilage_sensitivity`, `quality_grade` |
| Store and supply | `store_id`, `region`, `supplier_id`, `supplier_score` |
| Pricing and discount | `base_price`, `cost_price`, `selling_price`, `discount_pct`, `markdown_applied`, `is_promoted` |
| Quantities and outcomes | `initial_quantity`, `units_sold`, `units_wasted`, `waste_pct`, `revenue`, `waste_cost`, `profit`, `profit_margin_pct` |
| Expiry timing | `days_remaining_at_purchase`, `days_until_expiry` |
| Storage and handling | `storage_temp`, `temp_abuse_events`, `distribution_hours`, `handling_score`, `packaging_score` |
| Demand | `daily_demand`, `demand_variability` |
| Labels and benchmark | `was_spoiled` (target of Track 3A), `spoilage_risk` (benchmark only, never a feature) |

### 4.3 Integrity rules verified on the data

- `initial_quantity = units_sold + units_wasted` in 100% of rows.
- `markdown_applied` equals (`discount_pct` > 0) in all rows; discounts are 0 or between 10% and 75% (66 distinct values).
- `revenue = units_sold x selling_price` and `waste_cost = units_wasted x cost_price` exactly, so profit can be reconstructed from sell-through, price and cost.
- No missing values in the raw file.

### 4.4 Data layers

| Layer | File | Rows x columns | Created by |
|---|---|---|---|
| Raw | `data/raw/perishable_goods_management.csv` | 100,000 x 42 | Downloaded from Kaggle |
| Cleaned | `data/processed/cleaned_dataset.csv` | 100,000 x 57 | Milestone 1 pipeline (parsing, calendar and ratio features, train/test split flag) |
| Model-ready (M1) | `data/processed/model_ready_dataset.csv` | 100,000 x 189 | Milestone 1 preprocessor (178 scaled and encoded features plus targets and outcomes) |
| Model-ready (M2) | `data/processed/model_ready_m2_dataset.csv` | 100,000 x 210 | Milestone 2 feature pipeline |
| Feature dictionaries | `feature_dictionary.csv`, `feature_dictionary_m2.csv` | one row per column | Roles and permitted use per track |
| Out-of-fold predictions | `data/processed/track_3a_oof_train.csv` | 80,042 rows | OOF generator; the interface between Tracks 3A and 3B |
| Fitted transformers | `models/preprocessor.joblib`, `models/preprocessor_m2.joblib` | | Fitted on training rows only |
| Final models | `models/track_3a_classifier.joblib`, `models/track_3b_regressor.joblib`, each with a JSON metadata file | | Milestone 3 |
| Experiment store | `models/mlruns` (local, one per person) | | MLflow |
| Run summaries | `reports/tables/M2_T09_mlflow_runs_<author>.csv` | | Exported from MLflow and committed |

Data files and models are not committed to the repository. A reviewer regenerates them from the raw file by following the setup steps in the README.

### 4.5 Data flow

```
Kaggle raw CSV
   |
   v
Milestone 1 pipeline  --->  cleaned_dataset.csv  --->  preprocessor  --->  model_ready_dataset.csv
                                                                               |
                                                                               v
                                              Milestone 2 feature pipeline  (velocity, sell-through risk,
                                              store markdown frequency, handling scores)
                                                                               |
                                                                               v
                                                               model_ready_m2_dataset.csv
                                                                               |
                                   +-------------------------------------------+
                                   v                                           |
                    Track 3A spoilage classifier                               |
                    (time-blocked OOF probabilities)                           |
                                   |                                           |
                                   v                                           v
                          track_3a_oof_train.csv  ---------->  Track 3B revenue-response model
                                                                               |
                                                                               v
                                                          Track 3C discount optimizer (what-if scoring)
                                                                               |
                                                                               v
                                                            Streamlit dashboard (Manager View, Simulator)

All training runs  --->  MLflow tracking (parameters, metrics, artifacts)  --->  run summaries in the repository
```

### 4.6 Data rules that protect validity

- Time-ordered split: training data up to 2024-08-07, test data afterwards.
- Features built from history (sales velocity, store markdown frequency) use only transactions dated before the batch.
- Transformers are fitted on training rows only.
- Track 3A excludes all discount fields; Track 3B uses the discount percentage as its only discount input.
- Outcome columns and the spoilage-risk benchmark are never features. Code raises an error if a feature list contains them.

## 5. UI/UX Design

The design below is the specification for the Streamlit dashboard built in Milestone 4. Screenshots of the final implementation will be added to the final report.

### 5.1 Users and goals

| User | Goal | Typical session |
|---|---|---|
| Store manager | Decide today's markdowns quickly | Opens the Manager View, scans flagged items, accepts or adjusts the recommended discounts |
| Category or operations analyst | Understand why and test scenarios | Uses Simulator Mode and the explanation panels |

### 5.2 Design principles

1. **Decision first, detail on demand.** Show risk, recommended tier and confidence in one row; open the reasoning only when asked.
2. **Explain every recommendation** with the top factors from the SHAP analysis and the what-if profit curve.
3. **Be honest about uncertainty.** Show the confidence flag next to the recommendation, and label every business estimate as simulated.
4. **Plain language.** Use "chance of spoiling" rather than model terminology; keep the technical details in an expandable section.
5. **Accessible.** Risk tiers use text and icons as well as color; sufficient contrast; no information carried by color alone.

### 5.3 Information architecture

```
Sidebar:  Smart Shelf logo | View selector (Manager View / Simulator Mode) | Store selector | Date | About and data notes

Manager View
   -> Summary strip
   -> Inventory table (filter, sort)
   -> Batch detail panel

Simulator Mode
   -> Batch input form
   -> Results: risk, what-if profit curve, recommendation, confidence
```

### 5.4 Manager View wireframe

```
+------------------------------------------------------------------------------------------+
| Smart Shelf   |  Manager View  |  Store: STORE_014 v   Date: 2024-10-02 v                |
+------------------------------------------------------------------------------------------+
| Batches today: 126  |  High / Critical risk: 18  |  Estimated value at risk: 4,820 (sim.) |
+------------------------------------------------------------------------------------------+
| Filter: Category v  Risk tier v  [ ] Show limited-history only        Sort: Risk (high)  |
+------------------------------------------------------------------------------------------+
| Product        | Days left | Stock | Spoil chance | Recommended discount | Confidence      |
|----------------|-----------|-------|--------------|----------------------|-----------------|
| Fresh Salmon   |     2     |  40   | 34%  [#####] | CRITICAL  50%        | OK              |
| Greek Yogurt   |     3     |  85   | 27%  [####.] | HIGH      30%        | Limited history |
| Whole Milk     |     6     | 120   | 19%  [###..] | MEDIUM    20%        | OK              |
| Canned Beans   |    30     |  60   |  8%  [#....] | LOW        0%        | OK              |
+------------------------------------------------------------------------------------------+
| Selected: Greek Yogurt (batch detail)                                                    |
|  Why: short shelf life left, handling score below average, high stock for the sales rate |
|  [Top factors chart]        [Profit vs discount curve]       Note: limited history for    |
|                                                              this product at this store   |
+------------------------------------------------------------------------------------------+
```

The numbers in the wireframe are illustrative placeholders, not results.

**Elements**

- **Summary strip:** counts and an estimated value at risk, always labeled as a simulated estimate.
- **Inventory table:** one row per batch with days left, stock, spoilage probability with a small bar, the recommended discount tier with its percentage, and the confidence flag. Rows can be filtered by category, tier and limited-history status.
- **Confidence flag:** the text "limited history for this product at this store" when the product-by-store cell has few training rows, plus a flag propagated from the revenue-response model when its estimate is uncertain.
- **Batch detail panel:** top SHAP factors in plain words, the predicted profit at each candidate discount, and the optimizer's choice highlighted.

### 5.5 Simulator Mode wireframe

```
+------------------------------------------------------------------------------------------+
| Smart Shelf   |  Simulator Mode                                                          |
+--------------------------------------+---------------------------------------------------+
| Describe a batch                     | Result                                            |
|  Category      [ Dairy        v ]    |  Chance of spoiling:  27%   (gauge)               |
|  Product       [ Greek Yogurt v ]    |  Confidence: limited history for this product     |
|  Store         [ STORE_014    v ]    |                                                   |
|  Receiving date[ 2024-10-01     ]    |  Expected profit by discount level                |
|  Shelf life    [  14 days       ]    |   |            *                                  |
|  Current stock [  85 units      ]    |   |        *       *                              |
|  Quality grade [ A              ]    |   |    *               *                          |
|  [Advanced inputs v]                 |   |  *                    *                       |
|                                      |   +--0%--10%--20%--30%--40%--50%--60%--70%-->     |
|  [ Run what-if ]                     |  Recommended: 30% (HIGH tier)                     |
|                                      |  Simulated estimate, not observed                 |
+--------------------------------------+---------------------------------------------------+
```

**Elements**

- **Inputs:** the batch description named in the project plan (category, receiving date, shelf life, current stock), plus product, store and quality grade. Advanced inputs (storage and handling values) default to typical values and can be changed.
- **Outputs:** the spoilage probability, the predicted outcome at every candidate discount (0% and 10% to 75%), the optimizer's recommendation with its tier, and the confidence message.
- **Labeling:** every result block carries the text "Simulated estimate, not observed".

### 5.6 Discount tiers

| Tier | Meaning | Visual treatment |
|---|---|---|
| Low | Little or no markdown needed | Neutral color, label "Low" |
| Medium | Moderate markdown | Amber, label "Medium" |
| High | Substantial markdown | Orange, label "High" |
| Critical | Deepest markdown, act today | Red with an alert icon, label "Critical" |

The boundaries between tiers are derived in Milestone 3 from the optimizer's actual output distribution, not assigned by hand, and documented with a representative example batch for each tier.

### 5.7 User flows

1. **Daily check:** open the Manager View, review the summary strip, filter to High and Critical, open a batch, read the reason and confidence, apply or adjust the discount in the store's own system.
2. **What-if test:** open Simulator Mode, enter a hypothetical batch, run the what-if, compare the profit at different discount levels, read the confidence message.

### 5.8 Content and wording rules

- Estimates of savings or profit are always called simulated estimates.
- The confidence flag reads "limited history for this product at this store". It describes data coverage, not a measured loss of accuracy.
- Probabilities are shown as percentages with a short explanation on hover or in an expandable note.
- The optimizer searches only discount levels that exist in the data (0% and 10% to 75%).
- The dashboard does not claim that a recommendation is guaranteed to improve profit.

### 5.9 Technical notes for the Streamlit build

- Wide layout, sidebar navigation between the two views.
- Models and data are loaded once and cached.
- Charts use a single consistent palette; tier colors are paired with icons and text.
- An "About the data" section states that the data is synthetic and what the numbers mean.
- Optional deployment to Streamlit Community Cloud.

## 6. Open Items

| Item | Owner |
|---|---|
| Notebooks of M2-T02, T03, T04, T07 and T08 still contain hardcoded data paths and should read paths from the configuration module | Task owners |
| Four result tables cited in the M2-T10 figure notes were never exported | M2-T10 owner |
| Decision needed on committing result tables in `reports/tables/` | Team |
| Milestones 3 to 5, including dashboard screenshots and the video demonstration | Team |
| Team members and roles are listed in the separate Team Info file | Team |

## 7. Deviations from the Project Proposal and Related Documents

The project proposal (Smart_Shelf_v5) is kept unchanged as the original plan. Decisions and deviations made since then are listed here and in `reports/M2/M2_T11_M3_Design_Decisions.md` and the Milestone 2 task reports.

| # | Proposal said | What the team did | Why |
|---|---|---|---|
| 1 | Low-confidence flag on category x region x store cells | Product x store cells, flagged when fewer than 20 training rows (14.4% of test rows) | All 500 proposal cells have at least 129 training rows, so that flag never triggers. Flagged and unflagged test rows show no measurable difference in model error, so the flag describes sparse data, not low accuracy. |
| 2 | Track 3B: handle extreme negative profit by clipping, a robust loss, or modeling sell-through | Model sell-through and derive profit with the exact identity (a choice among the proposal's options) | The identity holds with zero error on all rows. Profit has extreme tails and 28.7% of batches lose money. |
| 3 | Track 3A feature set includes the Milestone 2 engineered features | Track 3A compares a full set (with them) and a compact set (without them and without other no-signal or duplicate features), choosing by Brier score and log loss | Sales velocity, sell-through risk and store markdown frequency show almost no spoilage signal (ROC-AUC 0.50 to 0.51). They remain confounding controls for Track 3B. |
| 4 | MLflow set up in Milestone 4 | Set up in Milestone 2 (task M2-T09) | Every Milestone 3 training run is tracked from the first run. |
| 5 | Shared MLflow tracking location | Local store per person, with per-author run summaries committed to the repository | GitHub cannot host a tracking server. All results stay reviewable in the repository, and the instructor approved the use of MLflow experiments. |
| 6 | Milestone 2 tasks T01 to T10 | Three tasks added: M2-T11 (consolidated feature dataset), M2-T12 (out-of-fold generator, support function, decision checks), M2-T13 (MLflow helpers) | Prerequisites for Milestone 3 identified in a readiness audit. |

**Clarification on sales velocity.** The proposal already plans a category x store fallback. The measured partition over all 100,000 rows is 25.42% from the product's own 7-day window, 74.01% from the fallback and 0.58% with no history. The figures 26.2% and 84.9% in the Milestone 1 report measure availability of earlier sales, not a partition.

**Related documents**

- Project proposal: Smart_Shelf_v5, stored in the repository next to this document.
- `README.md`: setup and reproduction steps.
- `reports/M1/EDA_Report_Milestone1.md` and `reports/M2/M2_T05_Data_Analysis_Report_and_Feature_Engineering_Summary.md`: analysis reports.
- `reports/M2/M2_T11_M3_Design_Decisions.md`, `reports/M2/M2_T12_Decision_Checks.md` and `reports/M2/M2-T09_MLflow_Setup.md`: design decisions and tracking guide.
- Team Info file (project name, description, members and roles): provided separately.
