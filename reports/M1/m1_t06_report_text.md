**Spoilage-Sensitivity Column Verification (M1-T06)**

**What it is.** `spoilage_sensitivity` takes 9 distinct values (0.30 to 0.95) and is constant inside every
category (Frozen_Meals 0.30 ... Seafood 0.95; Meat and Ready_to_Eat both 0.90). It is a fixed, pre-sale property of the category, not a per-batch measurement.

**Correlations.** With `was_spoiled`: r = 0.037 (95% CI 0.031 to 0.043), explaining about 0.1% of the variance.
With `spoilage_risk`: r = 0.254. The column is an ingredient of `spoilage_risk` (adding it raises the R^2 of the risk score from 0.85 to 0.88).

**Leakage tests.** Known before the sale and not derived from the target. Single-column test AUC 0.527 (category alone 0.524, `spoilage_risk` 0.597).
Adding it to a gradient-boosting model that already has `category` changes test AUC by +0.0013, i.e. nothing.

**Verdict: LEGITIMATE input feature (keep).** It stays a feature in the M1-T04 model-ready dataset (no change). It is 100% explained by the `category` one-hot columns, so unregularised linear models should
drop one of the two. `spoilage_risk` remains reserved as an evaluation benchmark only (not an input to either track).

**For M3-T08 (elasticity).** Across the 10 categories the column shows a descriptive association (10 categories only, not a test) with markdown frequency (rho 0.64) and depth (rho 0.56) but no clear
relation with demand response to discount (rho -0.38); it is constant within a category. It covers none of the elasticity estimate and should
be used only as a category-level segment or interaction term.
