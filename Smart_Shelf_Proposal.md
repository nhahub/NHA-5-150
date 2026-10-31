# Smart Shelf

*Perishable Food Waste Reduction and Dynamic Markdown Optimizer*

**Revised Proposal — v5** — refined for instructor review. This version corrects several column names against the actual downloaded dataset (found during Milestone 1 verification) and incorporates additional real columns — quality/handling signals for the classifier, a promotion flag for the confounding check, and a required verification step for spoilage_sensitivity before it is trusted as a feature. Additions were selected for genuine modeling value, not included indiscriminately. Changes are marked **[NEW]** or **[REVISED]** throughout. Version 5 incorporates the results of Milestone 1: it records the outcomes of the data-integrity and spoilage_sensitivity checks, corrects the candidate-discount grid, the remaining-shelf-life and sales-velocity definitions against the verified data, and updates the Track 3A evaluation plan. Changes new in v5 are tagged [v5] and listed in the change log at the end of this document.

## Project Overview

Food retailers and supermarket chains lose significant revenue every year due to the spoilage of perishable goods (fruits, vegetables, dairy, bakery, meat, seafood, and other short-life categories). Clearance decisions are usually made by staff observation rather than data, which leads to two costly outcomes: marking items down too early and losing margin unnecessarily, or marking them down too late and losing the product entirely.

Smart Shelf is an end-to-end data-driven decision support system. It predicts which inventory batches are at risk of spoiling before they sell out, and uses a second model to learn how revenue responds to different discount depths, so it can recommend the discount expected to recover the most value before the product is lost. The system combines two connected machine learning models — a spoilage classifier and a revenue-response optimizer — explained transparently at every step, and is delivered through an interactive dashboard that a store manager could realistically use to make daily markdown decisions.

This project uses the "Perishable Goods Management" dataset, publicly available on Kaggle. It is a synthetic dataset of 100,000 retail transactions tracking perishable products from purchase to sale or waste, covering 42 columns across 10 product categories (Dairy, Meat, Seafood, Produce, Bakery, Deli, Frozen Meals, Beverages, Pharmaceuticals, Ready-to-Eat), 50 stores, 5 regions, and 20 suppliers, with transaction dates spanning January 2023 to December 2024 and no missing values across any column. Confirmed relevant fields, verified directly against the actual downloaded file in Milestone 1 (correcting earlier assumed names), include: product and category identifiers; store, region, and supplier identifiers; transaction_date and expiration_date; shelf_life_days, days_remaining_at_purchase, and days_until_expiry; storage_temp and temp_deviation; base_price and cost_price; daily_demand and demand_variability; spoilage_risk and was_spoiled; quality_grade; markdown_applied (0/1 flag) and discount_pct (the actual discount percentage applied); selling_price, units_sold, units_wasted, and waste_pct; and the resulting revenue, waste_cost, profit, and profit_margin_pct. Since spoilage_risk is derived from the same process as the actual spoilage outcome (was_spoiled), it will be excluded from model input features to avoid data leakage, and used only as a benchmark to evaluate the models built in Milestone 3.

> **Additional columns discovered during verification [NEW]**
>
> Milestone 1's column verification (M1-T01) found the dataset contains several genuinely useful columns beyond what earlier planning assumed, verified directly against the file rather than a screenshot: spoilage_sensitivity, temp_abuse_events, handling_score, packaging_score, supplier_score, is_promoted, initial_quantity, and distribution_hours. Their treatment is specified in Milestones 2 and 3 below rather than added indiscriminately. [v5] Milestone 1 has since verified spoilage_sensitivity as a legitimate pre-sale attribute (constant within each category, not leakage-prone). A separate pre-sale check found no leakage in daily_demand, demand_variability, initial_quantity, temp_abuse_events or distribution_hours: sale outcomes add no information about them beyond the other pre-sale inputs (initial_quantity equals units_sold + units_wasted in every row, which makes it the batch size, not a leak). demand_variability and distribution_hours also carry no measurable signal.

> **Data integrity check — result [REVISED — v5]**
>
> [v5] The data-integrity check planned in v4 is complete. Milestone 1 found no rows where markdown_applied = 1 but discount_pct = 0: across all 100,000 rows markdown_applied equals (discount_pct > 0), with 30,651 markdown rows (30.7%). discount_pct can therefore be trusted as a training signal, and markdown_applied is redundant with it. Discounts take the value 0 or lie between 10% and 75% (66 distinct values), and 21.0% of markdowns are deeper than 50%; this sets the candidate discount grid for the optimizer (Milestone 3).

> **A note on the dataset [NEW]**
>
> This dataset is synthetically generated: 100,000 rows with 0% missing values and a pre-computed risk score are signals of rule-based or generative synthesis rather than a raw operational export. The team will state this explicitly in the Final Report (Milestone 5) and treat all waste-reduction and revenue-recovery figures as simulated estimates under stated assumptions — not measured real-world outcomes. Real deployments would need to re-validate against noisier, messier point-of-sale data (see Future Improvements). [v5] Milestone 1 supports this: was_spoiled behaves like a random draw from the pre-computed spoilage_risk score (observed spoilage rises smoothly from 11.1% to 28.9% across risk deciles), and several columns (demand_variability, distribution_hours) carry no measurable signal. Together these limit what any model can learn from this dataset, which is why Track 3A is judged mainly on calibration (see Milestone 3).

## Milestone 1: Data Collection, Exploration, and Preprocessing

#### Objectives

- Acquire, explore, and preprocess the perishable inventory dataset to prepare it for analysis and model building.

#### Tasks

##### 1. Data Collection

- Acquire the Perishable Goods Management dataset from Kaggle (100,000 transactions, 42 columns, 10 product categories).
- Confirm the presence of every column the project depends on by checking actual column names against the file (not against earlier screenshots or assumptions) — several fields have real names that differ from initial planning (e.g. markdown_applied rather than markdown, demand_variability rather than demand_volatility, days_until_expiry rather than days_until). Correct all downstream references once verified.

- Document the exclusion of the spoilage_risk column from model features, and the rationale (data leakage prevention), for transparency.

##### 2. Data Integrity Check [REVISED — completed in Milestone 1]

- Result: markdown_applied = 1 together with discount_pct = 0 occurs in 0 of 100,000 rows; markdown_applied equals (discount_pct > 0) throughout (30,651 markdown rows, 30.7%).
- Resolution: no rows need to be excluded, flagged or reinterpreted. discount_pct is the discount input for Milestone 3; markdown_applied is redundant with it and is not used as a Track 3B input.

- Document the finding and the resolution in the EDA Report, since the revenue-response model in Milestone 3 depends on discount_pct being a trustworthy signal.

##### 3. Spoilage-Sensitivity Column Verification [NEW — completed in Milestone 1]

- Method: its correlation with was_spoiled (r = 0.037) and with spoilage_risk (r = 0.254) was measured, its structure was examined (it is constant within each product category, with 9 distinct values from 0.30 to 0.95), and a leakage ablation was run on the time-ordered split.
- Verdict: legitimate input feature. It is a fixed, pre-sale property of the category, not derived from outcomes, and it adds essentially nothing beyond category (test AUC changes by +0.0013 when it is added to a model that already has category). It stays in the Milestone 2/3 feature sets; because category fully explains it, linear models should keep only one of the two. spoilage_risk remains excluded from model features and is used only as a benchmark.

- Elasticity (M3-T08): the column carries no information about how demand responds to discounts, so it does not replace the elasticity estimate; it may be used only as a category-level segment or interaction term.

##### 4. Data Exploration

- Conduct exploratory data analysis (EDA) to understand the dataset's structure, distributions, and relationships between features.
- Confirm data quality (the dataset reports 0% missing values across all columns) and check for duplicates or inconsistent entries.

- Analyze historical patterns between product category, storage duration, region, and the percentage of spoiled or wasted items (waste_pct).

##### 5. Preprocessing and Feature Engineering

- Encode categorical fields such as category, quality_grade, region, and supplier_id.
- Normalize or scale numerical features such as shelf_life_days, units_wasted, and revenue.

- Engineer initial temporal features from transaction_date and expiration_date, including days since receipt, day of week, and weekend indicators.

##### 6. Exploratory Data Analysis (EDA)

- Create visualizations (heatmaps, category comparisons, histograms) to detect patterns, correlations, and outliers, particularly around was_spoiled and waste_pct.
- Document key patterns and relationships in the data, such as which categories and storage conditions are associated with higher spoilage.

#### Deliverables

- **EDA Report:** A document summarizing key insights from data exploration and preprocessing decisions, including the data-leakage exclusion note.
- **Interactive Visualizations:** An EDA notebook showcasing visualizations that reveal key patterns and relationships.

- **Cleaned Dataset:** A dataset that is cleaned and prepared for machine learning.

## Milestone 2: Advanced Data Analysis and Feature Engineering

#### Objectives

- Perform deeper data analysis and engineer features that improve the model's ability to predict spoilage risk.

#### Tasks

##### 1. Advanced Data Analysis

- Conduct statistical tests (e.g., Chi-Square, ANOVA) to confirm which factors, such as category, region, and quality_grade, are meaningfully related to the was_spoiled outcome.
- Use correlation analysis to identify the most relevant features for predicting spoilage risk.

##### 2. Feature Engineering [REVISED — formulas specified, column names corrected]

- **Remaining shelf-life ratio:** days_until_expiry / shelf_life_days, clipped to [0, 1]. A value near 0 means the batch is close to or past its expected shelf life; near 1 means it is fresh. [v5] days_until_expiry is 0 to 3 days lower than the date-based gap (expiration_date − transaction_date) in 74.2% of rows, and the meaning of that offset is undocumented. The cleaned dataset therefore carries both versions: expiry_remaining_ratio (this definition) and shelf_life_used_ratio (date-based; one minus days_remaining_at_purchase / shelf_life_days). They correlate at only 0.45 and differ most for short-life items (mean difference 0.40 for shelf lives up to 5 days), so every model must state which version it uses.
- **Sales velocity:** units_sold in a trailing 7-day window ÷ 7, computed per product–store pair. Where fewer than 7 days of history exist for a batch, fall back to the category–store average velocity. The trailing window uses only sales recorded before the batch's own transaction, so a batch's own units_sold never enters its own feature. [v5] Milestone 1 measured the coverage: only 26.2% of batches have an earlier sale of the same product in the same store within 7 days (3,150 product–store pairs, median 31 rows each), while the category–store fallback covers 84.9%. Milestone 2 must therefore define the fallback order explicitly, report coverage for every velocity variant, and compare it with the demand-based alternative below.

- **Sell-through risk factor :** The dataset has no current_stock column, so each batch's starting stock is taken from initial_quantity (the quantity received). The sell-through risk factor is initial_quantity ÷ sales_velocity, giving the estimated "days to sell out" if the whole batch sells at the normal pace for its product–store pair. This is then compared against days_until_expiry: a ratio greater than 1 (days-to-sell-out exceeds days-remaining) flags the batch as at risk of expiring before it sells. Stock is deliberately not derived from outcome columns such as units_sold or units_wasted, since that would leak the batch's own result into the feature. [v5] Because daily_demand is a pre-sale column (Milestone 1 found no information from sale outcomes in it), initial_quantity ÷ daily_demand is an alternative days-to-sell-out measure that avoids the sparse trailing window; Milestone 2 compares the two.
- **Discount and markdown history flags:** clean, per-batch indicators derived from discount_pct (Milestone 1 confirmed that markdown_applied is fully consistent with it, so it adds no information), used directly as model inputs in Milestone 3 rather than pre-aggregated into a single sensitivity score — the actual discount–outcome relationship is learned by the Track 3B regression model, not estimated by hand here. For Track 3B and the optimizer, discount_pct is the only discount input: selling_price and markdown_applied are exact functions of it (and of base_price) and are excluded, so what-if scoring never feeds the model an inconsistent row.

- **Quality and handling signals [NEW]:** temp_abuse_events, handling_score, packaging_score, and supplier_score, used directly as inputs to the Track 3A spoilage classifier (Milestone 3). These were found during Milestone 1 verification and were not part of earlier planning; they give the classifier concrete, explainable handling-quality signal beyond quality_grade alone, and directly strengthen the SHAP explanations in M3-T03 (e.g., “flagged high-risk due to 3 temperature abuse events and low handling score,” rather than a vaguer explanation).
- **Promotion flag [NEW]:** is_promoted, carried forward as its own feature rather than folded into the discount history flags above. It is used specifically in Milestone 3's confounding check (M3-T05) to separate “discounted because promoted” from “discounted because at risk of spoiling” — two different reasons a discount may have been applied historically, which matters for isolating the causal effect of discount depth on outcome.

2.  **Optional enrichment :** distribution_hours may be incorporated into the sell-through risk factor above (e.g., as context for supply-chain lag before a batch reaches the shelf) if time allows. It is lower priority than the features above and should not delay Milestone 2 if time is tight. initial_quantity is no longer optional: it is the required stock input for the sell-through risk factor.

##### 3. Data Visualization

- Create visualizations segmenting items into “High Spoilage Risk” versus “Normal Velocity” categories.
- Build dashboards to illustrate spoilage trends, category-level waste_pct, and feature importance.

#### Deliverables

- **Data Analysis Report:** A comprehensive report on statistical analysis and insights derived from feature analysis.
- **Enhanced Visualizations:** Interactive, insightful visualizations or dashboards that highlight spoilage-related trends and important features.

- **Feature Engineering Summary:** Documentation outlining new features, exact formulas, transformations, and their expected impact on model performance.

## Milestone 3: Machine Learning Model Development, Revenue Optimization, and Explainability

*This milestone is split into three tracks so it can be divided cleanly between sub-teams. Track 3A predicts spoilage risk. Track 3B predicts revenue outcome as a function of discount depth — a second, genuinely distinct machine learning problem, not a lookup table. Track 3C combines both models into the discount optimizer that powers the dashboard. **[REVISED — discount logic upgraded from a manual rule table to a trained regression model, per instructor feedback]***

#### Objectives

- Build and evaluate a classifier that predicts whether a batch will spoil (Track 3A); build a regression model that predicts the revenue outcome of a batch as a function of its discount level, learned from real historical discount variation in the data (Track 3B); and combine both into an optimizer that selects the discount expected to maximize revenue for each batch, mapped to explainable tiers for the dashboard (Track 3C).

### Track 3A — Spoilage Risk Classifier

##### 1. Model Selection

- Choose machine learning models suited for classification (Random Forest, XGBoost, or LightGBM).
- Ensure the models are appropriate for predicting the binary was_spoiled outcome (1 = spoiled, 0 = not spoiled), excluding spoilage_risk from the feature set.

- **Feature set includes:** the Milestone 2 engineered features, plus temp_abuse_events, handling_score, packaging_score, and supplier_score directly from the raw dataset — these give the classifier concrete handling-quality signal beyond quality_grade alone. [v5] The discount treatments (discount_pct, markdown_applied and selling_price) are excluded from Track 3A, which predicts risk before any markdown decision.

##### 2. Model Training

- Split the data into training and test sets, respecting transaction_date order to avoid lookahead bias. [v5] Implemented in Milestone 1: the earliest 80% of days form the training period (2023-01-01 to 2024-08-07) and the rest the test period, giving 80,042 and 19,958 rows with spoilage rates of 19.35% and 19.81%.
- Train models using cross-validation techniques to assess their generalization capabilities.

##### 3. Model Evaluation

- Use evaluation metrics such as Precision, Recall, F1-score, and ROC-AUC, prioritizing recall since missing a spoiling batch is the more costly error. [v5] Calibration is the headline metric (Brier score, log loss and a reliability curve), because Track 3B consumes the predicted probability and because Milestone 1 showed that discrimination is capped: was_spoiled behaves like a draw from the well-calibrated spoilage_risk score, so test ROC-AUC is about 0.587 for honest features against 0.593 for the spoilage_risk benchmark. Recall is reported as a threshold choice on the calibrated probability, not as a measure of model quality, and the evaluation report must state this ceiling.
- Generate confusion matrices to analyze true positives, false positives, true negatives, and false negatives.

- Compare final model performance against the dataset's own pre-computed spoilage_risk column as a reference benchmark.

##### 4. Hyperparameter Tuning

- Use Grid Search or Bayesian Optimization to tune model parameters for improved performance.

##### 5. Explainability

- Apply SHAP (SHapley Additive exPlanations) to explain why a specific batch was flagged as high risk (e.g., short remaining shelf life, category-level spoilage tendency, low quality_grade, elevated temp_abuse_events or low handling_score).

##### 6. Prediction Confidence Flag

- For each prediction, check whether the batch's feature combination (category × region × store, or a rare quality_grade/supplier pairing) is well represented in the training data.
- Where the combination is sparse (below a defined support threshold), attach a “low confidence — limited historical data” flag to the prediction rather than presenting it with the same certainty as a well-supported one. This flag is surfaced directly in the Manager View and Simulator Mode in Milestone 4.

### Track 3B — Revenue-Response Model (discount → outcome) [NEW]

*The core idea: rather than assuming how discounts affect revenue, learn it directly from real variation already present in the data. Manual inspection confirmed that similar batches (same category, similar risk level, similar days_until expiration) received different discount_pct values historically, with different resulting revenue, waste_pct, and profit. That variation is a legitimate training signal, not a simulation.*

##### 1. Scenario Construction

- Build a training set where each historical batch is one example: inputs are its features (category, days_until_expiry, quality_grade, daily_demand, demand_variability, storage_temp/temp_deviation, and spoilage_sensitivity, verified in Milestone 1 as a legitimate pre-sale attribute), plus Track 3A's out-of-fold predicted spoilage probability as the batch's risk input (produced by time-blocked cross-validation inside the training period), plus discount_pct as an explicit input variable (the only discount input: selling_price and markdown_applied are excluded because they are exact functions of it); the target is the resulting outcome (profit as the primary target, waste_pct and revenue as secondary targets). [v5] Because units_sold + units_wasted equals initial_quantity in every row, sales and waste are not independent: the targets must be modeled consistently, for example by modeling sell-through and deriving waste from it rather than fitting waste_pct and revenue as unrelated targets. The dataset's spoilage_risk column is not used as an input to either track: using Track 3A's predicted probability keeps leakage out and lets a hypothetical batch entered in Simulator Mode be scored without a pre-computed risk value.
- This relies on cross-batch comparison — similar batches that historically received different discount_pct values act as natural “what-if” scenarios for each other — rather than an assumed elasticity curve invented without evidence.

##### 2. Confounding and Selection Bias Check [NEW]

- Discounts in this dataset are not randomly assigned: managers discount items that are already struggling or near expiration, which means a naive regressor risks learning that high discounts “cause” poor outcomes, when in fact poor outcomes are why the discount was applied in the first place (reverse causation / confounding).
- As the required baseline approach, include the features that plausibly drive the discounting decision itself — Track 3A's predicted spoilage probability, days_until_expiry, quality_grade, sales_velocity, a store-level historical markdown-frequency feature, and is_promoted — as explicit controls in the regression, so the model must explain profit variation net of “was this batch already heading for a bad outcome regardless of discount, or was this discount a marketing promotion rather than a spoilage response.”

- Diagnose whether this is sufficient: plot predicted profit vs. discount_pct within narrow bins of these control features. If discount still appears to uniformly hurt profit even within similar-risk, similar-timing batches, the simple control is not enough and the confounding has not been adequately addressed.
- Stretch goal, only if time and the diagnosis above warrant it: apply a propensity-style adjustment or a Double Machine Learning-style residualized regressor (e.g., using EconML) to more rigorously separate the effect of discount_pct from the confounding decision to discount at all. This is explicitly optional — the simple control-variable approach is the required deliverable, and the team should not let this block progress on Track 3C.

- Whichever approach is used, state the limitation honestly in the Final Report: this is an observational causal inference problem, and even the improved version is an estimate under stated assumptions, not a proven causal effect.

##### 3. Handling Extreme Outcomes

- Several batches show deeply negative profit (total-loss batches where cost was never recovered). These are meaningful signal, not errors, but left untreated they can dominate training.
- Apply appropriate handling before training — for example, clipping extreme values, using a log-scale or robust-regression target, or training waste_pct as the primary target with profit reconstructed afterward — and document the choice and its effect on model stability.

##### 4. Model Selection and Training

- Train a regression model (Random Forest Regressor, Gradient Boosting Regressor, or XGBoost Regressor) to predict the outcome from batch features plus discount_pct.
- Split by transaction_date, consistent with Track 3A, to avoid lookahead bias.

##### 5. Model Evaluation

- Evaluate using MAE, RMSE, and R² against held-out historical batches.
- Sanity-check the model's learned relationship, using the confounding-controlled version: confirm that, holding risk and timing features fixed, predicted profit behaves sensibly as discount_pct increases (rises as at-risk stock is more likely to sell, then falls once the discount depth erodes margin faster than it recovers units) — a model that does not show this shape, or shows discount uniformly hurting profit even after controls, is not ready to drive pricing decisions and should be revisited before Track 3C uses it.

##### 6. Demand-Response (Elasticity) View

- As a specific, reportable output of this model — not a separate model — extract and document how predicted units_sold changes with discount_pct within each category. This is a simplified, data-derived price-elasticity view, and directly answers the “how sensitive is demand to discount depth” question without requiring a third model.
- For categories with little or no historical discount variation, the model's confidence in this relationship will be low; report this honestly (e.g., via the same support-threshold logic as Track 3A's confidence flag) rather than presenting a flat or extrapolated curve as if it were well-evidenced.

### Track 3C — Discount Optimizer

***Scope statement [NEW]:** the system evaluates and recommends a single, one-time discount decision at the point a batch is inspected (e.g., once per day), based on its features at that moment. It does not model a multi-stage markdown trajectory (for example, a smaller discount several days out followed by a deeper discount on the final day) — real retail markdown policy is often sequential and dynamic in exactly that way, but modeling a full multi-stage decision process is out of scope for this project and is named explicitly in Future Improvements rather than left ambiguous.*

##### 1. What-If Scoring

- For a given batch, use the Track 3B model to predict the outcome at a set of candidate discount_pct values — 0% (no markdown) plus levels between 10% and 75%, for example 10%, 20%, 30%, 40%, 50%, 60% and 70% — holding all other features fixed at that batch's real values. [v5] The grid follows the discounts actually present in the data: nothing exists between 0% and 10%, 21.0% of historical markdowns are deeper than 50%, and support thins above 70% (170 rows between 70% and 75%, 221 at exactly 75%). Candidates outside the range with enough historical support are not scored, and the optimizer never extrapolates past 75%.
- Select the discount_pct that yields the highest predicted profit as the model's recommendation for that batch. This is what makes “optimizer” literally accurate: the system is searching predicted outcomes and picking the best one, not applying a fixed lookup rule.

##### 2. Tier Mapping (for the dashboard)

- Group the optimizer's continuous recommendations into the same Low / Medium / High / Critical display tiers used previously, so the Manager View stays simple and explainable — but the boundaries and typical discount within each tier are now derived from the optimizer's actual output distribution, not assigned by hand.
- Document, for each tier, the range of optimizer outputs it covers and a representative example batch, so the mapping is transparent and auditable rather than another black box.

##### 3. Validation Against Historical Practice [REVISED — split into two evaluation lanes]

- **Lane 1 — Matched Evaluation (real evidence):** identify historical batches where the optimizer's recommended discount happens to closely match the discount that was actually historically applied. For these batches only, compare actual observed profit to actual observed profit — no model is grading its own prediction here, so this is genuine ground-truth evidence of the optimizer's quality, though it necessarily covers a smaller, non-random subset of batches.
- **Lane 2 — Counterfactual Scoring (simulated, self-evaluated):** for batches where the optimizer recommends a discount that was never actually tried historically, the “improved outcome” figure comes entirely from Track 3B's own prediction evaluating its own recommendation. This is expected to look favorable almost by construction, and must never be presented as measured evidence — it is reported separately, clearly labeled “simulated / model-estimated, not observed,” and never blended into the same headline number as Lane 1.

- The Optimizer Validation Report and the Milestone 5 business case must keep these two lanes visibly separate throughout — a single blended “X% profit improvement” figure that mixes matched and counterfactual evidence would overstate the system's proven impact and is exactly the kind of claim a reviewer would be right to challenge.

##### 4. Confidence Propagation

- A batch's optimizer recommendation is only as trustworthy as the two models feeding it. Where either the Track 3A risk prediction or the Track 3B revenue-response estimate is flagged low-confidence, propagate that flag to the final recommendation shown in the dashboard.

#### Deliverables

- **Model Evaluation Report:** A detailed report covering both the spoilage classifier and the revenue-response regression model, including SHAP analysis, the elasticity view, the confounding diagnosis and control approach, the benchmark comparison, and the confidence-flag methodology for both models.
- **Optimizer Validation Report:** A report showing the optimizer's backtest against real historical outcomes, reported as two clearly separated lanes (matched-historical vs. counterfactual-simulated), the tier mapping derivation, and honest reporting of where the optimizer's recommendations would and would not have outperformed historical practice.

- **Model Code:** Python code for the classifier, the revenue-response regressor, the what-if scoring optimizer, and the backtest.
- **Final Models:** The tuned spoilage classifier and revenue-response regressor, both ready for deployment.

## Milestone 4: MLOps, Deployment, and Monitoring

#### Objectives

- Implement MLOps practices and deploy the spoilage prediction and discount recommendation system as an interactive tool.

#### Tasks

##### 1. MLOps Implementation

- Use a tool such as MLflow to manage model experiments, versions, and metrics.
- Log metrics, parameters, and artifacts to ensure reproducibility and traceability.

##### 2. Model Deployment

- Build an interactive dashboard using Streamlit, providing a Manager View showing current inventory with items flagged by spoilage risk and their optimizer-recommended discount tier.
- Surface the confidence flag (propagated from Track 3A and/or 3B in Milestone 3) in the Manager View so low-confidence recommendations are visibly distinguished from well-supported ones.

- Include a Simulator Mode, where a user can input a hypothetical batch (product category, receiving date, shelf life, current stock) and see the full what-if scoring: predicted outcome at each candidate discount level, the optimizer's chosen recommendation, and its confidence.
- Optionally deploy to a cloud platform (e.g., Streamlit Community Cloud) to ensure accessibility.

##### 3. Model Monitoring

- Define monitoring metrics relevant to this use case: prediction drift for both the classifier and the revenue-response model, estimated spoilage reduction percentage, and estimated revenue recovered relative to the dataset's historical waste_pct and revenue figures — labeled clearly as backtested/simulated estimates, consistent with the dataset note above.
- Establish alerts to inform when model performance degrades or when significant shifts occur in inventory patterns.

##### 4. Model Retraining Strategy

- Develop a plan for periodic model retraining based on new data or performance changes.

#### Deliverables

- **Deployed Model:** A fully functional, interactive Streamlit dashboard capable of real-time spoilage predictions and discount recommendations, with confidence flags visible in both views.
- **MLOps Report:** A report detailing the MLOps pipeline, experiment tracking, deployment architecture, and monitoring setup.

- **Monitoring Setup:** Documentation on how model performance would be tracked and how updates or retraining would be triggered.

## Milestone 5: Final Documentation and Presentation

#### Objectives

- Prepare final documentation and create a presentation for stakeholders that showcases the project's results and business impact.

#### Tasks

##### 1. Final Report

- Provide a comprehensive summary of the project, including the problem definition, dataset description, data exploration, model development, and deployment.
- Discuss the business and environmental implications of spoilage prediction, including estimated waste reduction and potential revenue recovery, referenced against the dataset's historical waste_pct and revenue figures and clearly labeled as backtested/simulated rather than measured.

- State the synthetic-data limitation explicitly (see dataset note) and its implications for how confidently the results would transfer to a real deployment.
- Highlight key insights, challenges, and decisions made during the project, including the data-leakage exclusion of spoilage_risk.

##### 2. Before/After Business Case [REVISED — now reports matched vs. simulated evidence separately]

- Present a comparison table built from Lane 1 (matched historical evaluation) as the primary, credible evidence of impact wherever coverage allows, since it reflects real observed outcomes rather than the optimizer scoring its own recommendation.
- Present Lane 2 (counterfactual/simulated scoring) as a secondary, clearly labeled illustration of the optimizer's broader potential, explicitly caveated as model-estimated rather than measured — never merged into a single headline percentage with Lane 1.

- This is the primary business-impact evidence for the presentation and should draw directly on the Milestone 3 optimizer validation backtest rather than a new analysis.

##### 3. Worked Example [NEW]

- Select one representative batch and walk it end-to-end in the presentation: its raw features → the classifier's risk prediction and confidence flag → its SHAP explanation → the revenue-response model's what-if scoring across candidate discounts → the optimizer's chosen recommendation and tier → the backtested outcome. A single concrete example makes the full two-model pipeline legible in a way that aggregate metrics alone do not.

##### 4. Final Presentation

- Prepare a concise, engaging presentation for stakeholders, highlighting the methodology, results, and practical use of the system.
- Demonstrate the deployed model in action with a live demo of the Simulator Mode, including the worked example above.

##### 5. Future Improvements

- Suggest areas for future improvement, such as incorporating real point-of-sale data with true price experimentation (A/B-tested discounts), reframing the discount decision as a contextual bandit or reinforcement-learning problem once live deployment allows the system to observe outcomes it actually caused, or integrating IoT sensors (smart scales or RFID tags) for automatic real-time weight tracking on shelves.
- Note that this project deliberately did not pursue a reinforcement-learning framing: RL requires an agent to learn from consequences of its own actions, which an offline historical backtest cannot provide. This is flagged here as a considered and intentional scope decision, not an oversight.

- Note that this project deliberately modeled a single, static markdown decision rather than a multi-stage markdown trajectory (see Track 3C scope statement); extending to a sequential, multi-day markdown policy — and to a more rigorous causal (Double ML / EconML-based) estimate of the discount effect — are the two clearest next steps for a deeper iteration of this system.
- Note that categories with little historical discount variation (flagged low-confidence in Track 3B) are the clearest next target for real pilot data, since the revenue-response relationship could not be learned with confidence from this dataset alone.

#### Deliverables

- **Final Project Report:** A detailed summary of the project's process, from dataset description to deployment, its business and environmental impact, and the before/after business case.
- **Final Presentation:** A polished presentation for stakeholders, explaining the system's value and usage, including the worked example.

## Final Milestones Summary

| **Milestone**                                                               | **Key Deliverables**                                                           |
|-----------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| 1. Data Collection, Exploration & Preprocessing                            | EDA Report, Interactive Visualizations, Cleaned Dataset                        |
| 2. Advanced Data Analysis & Feature Engineering                            | Data Analysis Report, Enhanced Visualizations, Feature Engineering Summary     |
| 3. Model Development, Revenue Optimization & Explainability (3A + 3B + 3C) | Model Evaluation Report, Optimizer Validation Report, Model Code, Final Models |
| 4. MLOps, Deployment & Monitoring                                          | Deployed Model, MLOps Report, Monitoring Setup                                 |
| 5. Final Documentation & Presentation                                      | Final Project Report (incl. before/after business case), Final Presentation    |

## Conclusion

The Smart Shelf project aims to build a machine learning-driven decision support system that identifies perishable inventory at risk of spoiling and recommends a revenue-optimized markdown to reduce waste. Using the Perishable Goods Management dataset, this step-by-step process combines two connected models — a spoilage classifier and a revenue-response regressor — into a genuine discount optimizer, validated against real historical outcomes rather than manually assigned rules. The process focuses on rigorous data exploration, feature engineering, model development with explainability, honest validation, deployment, and continuous monitoring, to ensure the system remains effective, transparent, and useful to retail decision-makers over time.

## Version 5 change log

The table lists every change from v4 to v5. All figures come from Milestone 1 verification on the final cleaned dataset.

| **Section**                                    | **Change in v5**                                                                                                                                                                                                                     |
|------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Title and tags                                 | Version label changed from v4 to v5. Changes new in v5 are tagged [v5] in the text.                                                                                                                                                |
| Introduction: additional columns               | Records the Milestone 1 verification: spoilage_sensitivity is a legitimate pre-sale attribute, and a pre-sale check found no leakage in daily_demand, demand_variability, initial_quantity, temp_abuse_events or distribution_hours. |
| Introduction: data-quirk callout               | The unverified quirk (and its Salami example) is replaced by the verified result: 0 of 100,000 rows have markdown_applied = 1 with discount_pct = 0, and markdown_applied equals (discount_pct > 0).                                |
| Introduction: dataset note                     | Adds the evidence that the data is synthetic and signal-limited (observed spoilage rises from 11.1% to 28.9% across spoilage_risk deciles).                                                                                          |
| Milestone 1: Data Integrity Check              | Converted from a to-do into a recorded result and resolution.                                                                                                                                                                        |
| Milestone 1: Spoilage-Sensitivity Verification | Records the method and the verdict: legitimate input, constant within each category, adds +0.0013 test AUC beyond category, and carries no elasticity information.                                                                   |
| Milestone 2: Remaining shelf-life ratio        | Documents that days_until_expiry is 0 to 3 days below the date-based gap in 74.2% of rows; both versions are kept and each model must state which it uses.                                                                           |
| Milestone 2: Sales velocity and sell-through   | Adds the measured coverage (26.2% of batches have a 7-day product-store history; the category-store fallback covers 84.9%) and initial_quantity ÷ daily_demand as an alternative.                                                    |
| Milestone 2: Discount flags                    | discount_pct is the only discount input for Track 3B and the optimizer; selling_price and markdown_applied are excluded as exact functions of it.                                                                                    |
| Milestone 3, Track 3A: features and split      | Discount treatments are excluded from Track 3A. The time-ordered split is recorded as implemented (training period to 2024-08-07).                                                                                                   |
| Milestone 3, Track 3A: metrics                 | Calibration (Brier score, log loss, reliability curve) becomes the headline metric. The ROC-AUC ceiling (about 0.59) is stated, and recall is treated as a threshold choice.                                                         |
| Milestone 3, Track 3B                          | Out-of-fold probabilities come from time-blocked cross-validation. Targets must be modeled consistently because units_sold + units_wasted equals initial_quantity. A duplicated phrase in the baseline paragraph is removed.         |
| Milestone 3, Track 3C                          | Candidate discount grid changed from 0 to 50% to 0% plus 10 to 75%, with the observed support limits (21.0% of markdowns exceed 50%; support thins above 70%).                                                                       |
