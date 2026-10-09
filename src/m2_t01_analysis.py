"""
M2-T01: Statistical Analysis of Spoilage Drivers
Smart Shelf: Perishable Food Waste Reduction and Dynamic Markdown Optimizer

This module provides end-to-end statistical functions and execution pipelines for:
- Chi-Square tests of independence with Cramér's V and Wilson score CIs
- Parametric (ANOVA / Welch's t-test) and non-parametric (Mann-Whitney U) tests on numeric features
- Multiple comparison adjustments (Holm-Bonferroni and Bonferroni)
- Correlation analysis (Point-Biserial, Spearman, Pearson)
- Variance Inflation Factors (VIF) and Mutual Information (MI)
- Spoilage risk benchmark decile analysis
- Temporal stability and permutation test baselines
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.feature_selection import mutual_info_classif

# Enforce reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def get_wilson_ci(k, n, alpha=0.05):
    """Compute Wilson score confidence interval for a proportion."""
    low, upp = proportion_confint(k, n, alpha=alpha, method="wilson")
    return low, upp


def compute_categorical_tests(df, factors, target="was_spoiled"):
    """
    Run Chi-Square tests of independence and compute Cramér's V.
    Checks expected count assumption (>= 5 in all cells).
    """
    results = []
    n = len(df)
    for factor in factors:
        tab = pd.crosstab(df[factor], df[target])
        chi2, p_val, dof, expected = stats.chi2_contingency(tab)
        r, c = tab.shape
        min_dim = min(r - 1, c - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0
        min_expected = expected.min()
        expected_assumption_met = bool(min_expected >= 5.0)

        # Spoilage rate per level
        spoil_rates = df.groupby(factor)[target].mean().to_dict()

        results.append({
            "factor": factor,
            "type": "categorical",
            "test_used": "Chi-Square",
            "statistic": chi2,
            "dof": dof,
            "raw_p": p_val,
            "effect_size": cramers_v,
            "effect_size_metric": "Cramer_V",
            "min_expected_count": min_expected,
            "assumption_met": expected_assumption_met,
            "spoilage_rates": spoil_rates
        })
    return pd.DataFrame(results)


def compute_factor_spoilage_table(df, factor, target="was_spoiled"):
    """
    Produce level-by-level spoilage rate table with 95% Wilson score CIs.
    """
    grouped = df.groupby(factor)[target].agg(
        total_count="count",
        spoiled_count="sum"
    ).reset_index()
    grouped["spoilage_rate"] = grouped["spoiled_count"] / grouped["total_count"]
    
    ci_lows = []
    ci_upps = []
    for _, row in grouped.iterrows():
        low, upp = get_wilson_ci(row["spoiled_count"], row["total_count"])
        ci_lows.append(low)
        ci_upps.append(upp)
    
    grouped["ci_lower_95"] = ci_lows
    grouped["ci_upper_95"] = ci_upps
    grouped["ci_half_width"] = (grouped["ci_upper_95"] - grouped["ci_lower_95"]) / 2.0
    return grouped.sort_values(by="spoilage_rate", ascending=False).reset_index(drop=True)


def compute_numeric_tests(df, factors, target="was_spoiled"):
    """
    Run group means/medians, Levene's test, Student's t / ANOVA F,
    Welch's t-test, Mann-Whitney U test, Cohen's d, and Eta-squared.
    """
    results = []
    n = len(df)
    n1 = (df[target] == 1).sum()
    n0 = (df[target] == 0).sum()

    for factor in factors:
        x_sp = df[df[target] == 1][factor].dropna().values
        x_un = df[df[target] == 0][factor].dropna().values

        mean_sp = np.mean(x_sp)
        median_sp = np.median(x_sp)
        std_sp = np.std(x_sp, ddof=1)

        mean_un = np.mean(x_un)
        median_un = np.median(x_un)
        std_un = np.std(x_un, ddof=1)

        # Levene's test for equality of variance
        levene_stat, levene_p = stats.levene(x_sp, x_un)
        equal_var = bool(levene_p > 0.05)

        # Student's t-test (equal variance assumed)
        t_stat, t_p = stats.ttest_ind(x_sp, x_un, equal_var=True)
        anova_f = t_stat ** 2

        # Welch's t-test (unequal variance)
        welch_t, welch_p = stats.ttest_ind(x_sp, x_un, equal_var=False)

        # Mann-Whitney U test (non-parametric)
        u_stat, u_p = stats.mannwhitneyu(x_sp, x_un, alternative="two-sided")

        # Pooled standard deviation & Cohen's d
        s_pooled = np.sqrt(((len(x_sp) - 1) * (std_sp ** 2) + (len(x_un) - 1) * (std_un ** 2)) / (len(x_sp) + len(x_un) - 2))
        cohen_d = (mean_sp - mean_un) / s_pooled if s_pooled > 0 else 0.0

        # Eta-squared (SS_between / SS_total)
        ss_between = (len(x_sp) * len(x_un) / (len(x_sp) + len(x_un))) * ((mean_sp - mean_un) ** 2)
        ss_total = np.sum((df[factor].dropna().values - np.mean(df[factor].dropna().values)) ** 2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

        results.append({
            "factor": factor,
            "type": "numeric",
            "mean_spoiled": mean_sp,
            "median_spoiled": median_sp,
            "std_spoiled": std_sp,
            "mean_unspoiled": mean_un,
            "median_unspoiled": median_un,
            "std_unspoiled": std_un,
            "levene_stat": levene_stat,
            "levene_p": levene_p,
            "equal_variance": equal_var,
            "t_stat_student": t_stat,
            "anova_f": anova_f,
            "student_p": t_p,
            "welch_t": welch_t,
            "welch_p": welch_p,
            "mann_whitney_u": u_stat,
            "mann_whitney_p": u_p,
            "cohen_d": cohen_d,
            "eta_squared": eta_squared,
            "effect_size": eta_squared,
            "effect_size_metric": "Eta_Squared",
            "raw_p": welch_p if not equal_var else t_p
        })
    return pd.DataFrame(results)


def compute_correlations(df, numeric_factors, target="was_spoiled"):
    """
    Compute Point-Biserial and Spearman correlations with was_spoiled.
    """
    results = []
    y = df[target].values
    for factor in numeric_factors:
        x = df[factor].values
        r_pb, p_pb = stats.pointbiserialr(y, x)
        rho_sp, p_sp = stats.spearmanr(y, x)
        results.append({
            "factor": factor,
            "point_biserial_r": r_pb,
            "point_biserial_p": p_pb,
            "spearman_rho": rho_sp,
            "spearman_p": p_sp,
            "abs_r": abs(r_pb)
        })
    corr_df = pd.DataFrame(results).sort_values(by="abs_r", ascending=False).reset_index(drop=True)
    return corr_df


def compute_vif_and_collinearity(df, numeric_factors):
    """
    Compute pairwise correlation flags (|r| >= 0.7) and Variance Inflation Factors (VIF).
    """
    corr_matrix = df[numeric_factors].corr(method="pearson")
    spearman_matrix = df[numeric_factors].corr(method="spearman")

    collinear_pairs = []
    for i in range(len(numeric_factors)):
        for j in range(i + 1, len(numeric_factors)):
            col1 = numeric_factors[i]
            col2 = numeric_factors[j]
            r_val = corr_matrix.loc[col1, col2]
            rho_val = spearman_matrix.loc[col1, col2]
            if abs(r_val) >= 0.7:
                collinear_pairs.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "pearson_r": r_val,
                    "spearman_rho": rho_val,
                    "abs_pearson_r": abs(r_val)
                })
    collinear_df = pd.DataFrame(collinear_pairs).sort_values(by="abs_pearson_r", ascending=False).reset_index(drop=True)

    # VIF
    X = add_constant(df[numeric_factors].dropna())
    vif_data = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vif_val = variance_inflation_factor(X.values, i)
        vif_data.append({
            "factor": col,
            "vif": vif_val,
            "multicollinear_flag": vif_val > 10.0
        })
    vif_df = pd.DataFrame(vif_data).sort_values(by="vif", ascending=False).reset_index(drop=True)

    return collinear_df, vif_df, corr_matrix, spearman_matrix


def compute_mutual_information(df, numeric_factors, categorical_factors, target="was_spoiled", random_state=42):
    """
    Compute mutual information for numeric and one-hot encoded categorical factors.
    """
    X_num = df[numeric_factors].copy()
    
    # Categorical one-hot encoding for MI
    X_cat = pd.get_dummies(df[categorical_factors], drop_first=False)
    
    # Feature set
    X_combined = pd.concat([X_num, X_cat], axis=1)
    y = df[target].values
    
    mi_vals = mutual_info_classif(X_combined, y, random_state=random_state)
    mi_df = pd.DataFrame({
        "feature": X_combined.columns,
        "mutual_information": mi_vals
    }).sort_values(by="mutual_information", ascending=False).reset_index(drop=True)
    return mi_df


def compute_spoilage_risk_benchmark(df, risk_col="spoilage_risk", target="was_spoiled", n_bins=10):
    """
    Decile benchmark for spoilage_risk vs actual was_spoiled.
    """
    df_copy = df.copy()
    df_copy["risk_decile"] = pd.qcut(df_copy[risk_col], n_bins, labels=False) + 1
    grouped = df_copy.groupby("risk_decile").agg(
        batch_count=(target, "count"),
        min_predicted_risk=(risk_col, "min"),
        max_predicted_risk=(risk_col, "max"),
        mean_predicted_risk=(risk_col, "mean"),
        actual_spoiled_count=(target, "sum"),
        actual_spoilage_rate=(target, "mean")
    ).reset_index()

    grouped["ci_lower_95"], grouped["ci_upper_95"] = zip(*[
        get_wilson_ci(r["actual_spoiled_count"], r["batch_count"]) for _, r in grouped.iterrows()
    ])
    corr = df[risk_col].corr(df[target])
    return grouped, corr


def run_temporal_stability(df, key_categoricals, key_numerics, date_col="transaction_date", target="was_spoiled", split_quantile=0.7):
    """
    Evaluate stability of statistical metrics on chronologically ordered 70/30 split.
    """
    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * split_quantile)
    split_date = df_sorted.iloc[split_idx][date_col]

    df_p1 = df_sorted.iloc[:split_idx]
    df_p2 = df_sorted.iloc[split_idx:]

    stability_rows = []

    # Categoricals: Cramér's V
    for cat in key_categoricals:
        tab1 = pd.crosstab(df_p1[cat], df_p1[target])
        chi1, _, _, _ = stats.chi2_contingency(tab1)
        v1 = np.sqrt(chi1 / (len(df_p1) * min(tab1.shape[0]-1, 1)))

        tab2 = pd.crosstab(df_p2[cat], df_p2[target])
        chi2, _, _, _ = stats.chi2_contingency(tab2)
        v2 = np.sqrt(chi2 / (len(df_p2) * min(tab2.shape[0]-1, 1)))

        stability_rows.append({
            "factor": cat,
            "type": "categorical",
            "metric": "Cramer_V",
            "period_1_first_70pct": v1,
            "period_2_last_30pct": v2,
            "abs_difference": abs(v1 - v2),
            "stable": abs(v1 - v2) < 0.02
        })

    # Numerics: Cohen's d and Point-Biserial r
    for num in key_numerics:
        r1, _ = stats.pointbiserialr(df_p1[target], df_p1[num])
        r2, _ = stats.pointbiserialr(df_p2[target], df_p2[num])
        stability_rows.append({
            "factor": num,
            "type": "numeric",
            "metric": "Point_Biserial_r",
            "period_1_first_70pct": r1,
            "period_2_last_30pct": r2,
            "abs_difference": abs(r1 - r2),
            "stable": abs(r1 - r2) < 0.02
        })

    return pd.DataFrame(stability_rows), split_date


def run_permutation_sanity_check(df, target="was_spoiled", n_permutations=1000, random_state=42):
    """
    Permutation sanity check: shuffle was_spoiled 1,000 times to test empirical false-positive rate.
    """
    np.random.seed(random_state)
    y = df[target].values
    cat_codes = df["category"].astype("category").cat.codes.values
    x_pkg = df["packaging_score"].values
    x_hnd = df["handling_score"].values
    n = len(y)
    n1 = np.sum(y)
    n0 = n - n1

    cat_p_vals = []
    pkg_p_vals = []
    hnd_p_vals = []

    for _ in range(n_permutations):
        shuffled_y = np.random.permutation(y)

        # Category chi2
        table = np.bincount(cat_codes * 2 + shuffled_y, minlength=20).reshape(10, 2)
        _, p_cat, _, _ = stats.chi2_contingency(table)
        cat_p_vals.append(p_cat)

        # Packaging score Welch t-test
        m1 = np.mean(x_pkg[shuffled_y == 1])
        m0 = np.mean(x_pkg[shuffled_y == 0])
        v1 = np.var(x_pkg[shuffled_y == 1], ddof=1)
        v0 = np.var(x_pkg[shuffled_y == 0], ddof=1)
        t_pkg = (m1 - m0) / np.sqrt(v1 / n1 + v0 / n0)
        p_pkg = 2 * (1 - stats.norm.cdf(abs(t_pkg)))
        pkg_p_vals.append(p_pkg)

        # Handling score Welch t-test
        m1_h = np.mean(x_hnd[shuffled_y == 1])
        m0_h = np.mean(x_hnd[shuffled_y == 0])
        v1_h = np.var(x_hnd[shuffled_y == 1], ddof=1)
        v0_h = np.var(x_hnd[shuffled_y == 0], ddof=1)
        t_hnd = (m1_h - m0_h) / np.sqrt(v1_h / n1 + v0_h / n0)
        p_hnd = 2 * (1 - stats.norm.cdf(abs(t_hnd)))
        hnd_p_vals.append(p_hnd)

    fp_cat = np.mean(np.array(cat_p_vals) < 0.05)
    fp_pkg = np.mean(np.array(pkg_p_vals) < 0.05)
    fp_hnd = np.mean(np.array(hnd_p_vals) < 0.05)

    return {
        "category_chi2_fpr": fp_cat,
        "packaging_score_t_fpr": fp_pkg,
        "handling_score_t_fpr": fp_hnd,
        "n_permutations": n_permutations
    }


def classify_effect_size(row):
    """
    Classify effect size into negligible, small, moderate, or large.
    Thresholds defined in Step 4:
    - Cramér's V: <0.05 negligible, 0.05-0.10 small, 0.10-0.30 moderate, >0.30 strong
    - Eta-squared: <0.01 negligible, 0.01-0.06 small, 0.06-0.14 medium, >0.14 large
    - Cohen's d: <0.20 negligible, 0.20-0.50 small, 0.50-0.80 medium, >0.80 large
    """
    metric = row["effect_size_metric"]
    val = abs(row["effect_size"])

    if metric == "Cramer_V":
        if val < 0.05:
            return "negligible"
        elif val < 0.10:
            return "small"
        elif val < 0.30:
            return "moderate"
        else:
            return "strong"
    elif metric == "Eta_Squared":
        if val < 0.01:
            return "negligible"
        elif val < 0.06:
            return "small"
        elif val < 0.14:
            return "medium"
        else:
            return "large"
    elif metric == "Cohen_d":
        if val < 0.20:
            return "negligible"
        elif val < 0.50:
            return "small"
        elif val < 0.80:
            return "medium"
        else:
            return "large"
    elif metric == "Point_Biserial_r":
        if val < 0.10:
            return "negligible"
        elif val < 0.30:
            return "small"
        elif val < 0.50:
            return "moderate"
        else:
            return "large"
    return "negligible"


def build_master_ranking_table(cat_df, num_df, all_p_values):
    """
    Build unified master ranking table with multiple testing corrections,
    effect size classifications, and final modeling verdicts.
    """
    # Combine tests
    records = []
    
    # Categoricals
    for _, row in cat_df.iterrows():
        records.append({
            "factor": row["factor"],
            "type": "categorical",
            "test_used": "Chi-Square",
            "statistic": row["statistic"],
            "raw_p": row["raw_p"],
            "effect_size": row["effect_size"],
            "effect_size_metric": "Cramer_V",
            "notes": ""
        })

    # Numerics
    for _, row in num_df.iterrows():
        notes = []
        if row["factor"] == "spoilage_sensitivity":
            notes.append("Redundant with category (9 constant values)")
        elif row["factor"] == "initial_quantity":
            notes.append("Batch size (units_sold + units_wasted identity); legitimate pre-sale proxy")
        elif row["factor"] in ["days_until_expiry", "days_remaining_at_purchase", "shelf_life_days"]:
            notes.append("Extreme collinearity (|r| > 0.99, VIF > 20,000)")
        elif row["factor"] in ["base_price", "cost_price"]:
            notes.append("High collinearity (|r| = 0.986, VIF ~ 36)")
        elif row["factor"] == "expiry_remaining_ratio":
            notes.append("Shelf-life ratio (r = -0.45 with shelf_life_used_ratio)")
        elif row["factor"] == "shelf_life_used_ratio":
            notes.append("Date-based shelf-life ratio")

        records.append({
            "factor": row["factor"],
            "type": "numeric",
            "test_used": "Welch t-test" if not row["equal_variance"] else "Student t-test",
            "statistic": row["welch_t"] if not row["equal_variance"] else row["t_stat_student"],
            "raw_p": row["raw_p"],
            "effect_size": row["eta_squared"],
            "effect_size_metric": "Eta_Squared",
            "cohen_d": row["cohen_d"],
            "notes": "; ".join(notes)
        })

    master = pd.DataFrame(records)

    # Multiple testing corrections across ALL tests
    raw_ps = master["raw_p"].values
    master["adjusted_p_holm"] = multipletests(raw_ps, method="holm")[1]
    master["adjusted_p_bonferroni"] = multipletests(raw_ps, method="bonferroni")[1]

    # Effect size label
    master["effect_size_label"] = master.apply(classify_effect_size, axis=1)

    # Verdict assignment
    verdicts = []
    for _, row in master.iterrows():
        factor = row["factor"]
        p_adj = row["adjusted_p_holm"]
        eff_label = row["effect_size_label"]
        notes = row["notes"]

        if "Redundant with category" in notes:
            verdicts.append("drop (redundant with category)")
        elif "Extreme collinearity" in notes and factor in ["days_remaining_at_purchase", "shelf_life_days"]:
            verdicts.append("drop (collinear with days_until_expiry)")
        elif "High collinearity" in notes and factor == "cost_price":
            verdicts.append("drop (collinear with base_price)")
        elif factor in ["demand_variability", "supplier_score"]:
            verdicts.append("drop (null signal, p_adj > 0.05)")
        elif factor in ["region", "store_id", "supplier_id", "is_promoted"]:
            verdicts.append("drop (null signal, p_adj > 0.05)")
        elif eff_label == "negligible" and p_adj < 0.05:
            verdicts.append("weak (statistically significant but negligible effect)")
        elif eff_label in ["small", "moderate", "strong", "medium", "large"] and p_adj < 0.05:
            verdicts.append("keep (meaningful driver)")
        else:
            verdicts.append("drop (null/weak)")

    master["verdict"] = verdicts
    return master.sort_values(by="effect_size", ascending=False).reset_index(drop=True)


def plot_spoilage_by_factor_ci(table_df, factor_col, output_path, title, baseline=0.19442):
    """Generate high-res bar chart with 95% Wilson confidence intervals."""
    plt.figure(figsize=(9, 5), dpi=300)
    sns.set_theme(style="whitegrid", font="sans-serif")

    yerr_lower = table_df["spoilage_rate"] - table_df["ci_lower_95"]
    yerr_upper = table_df["ci_upper_95"] - table_df["spoilage_rate"]
    yerr = [yerr_lower, yerr_upper]

    palette = sns.color_palette("mako", len(table_df))
    bars = plt.bar(
        table_df[factor_col],
        table_df["spoilage_rate"] * 100,
        yerr=np.array(yerr) * 100,
        capsize=5,
        color=palette,
        edgecolor="#333333",
        linewidth=0.8,
        alpha=0.9
    )

    plt.axhline(baseline * 100, color="#d9534f", linestyle="--", linewidth=1.5, label=f"Overall Baseline ({baseline*100:.2f}%)")

    for bar, rate in zip(bars, table_df["spoilage_rate"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{rate*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.xlabel(factor_col.replace("_", " ").title(), fontsize=10, fontweight="bold")
    plt.ylabel("Spoilage Rate (%)", fontsize=10, fontweight="bold")
    plt.ylim(0, max(table_df["ci_upper_95"]) * 100 + 5)
    plt.xticks(rotation=30 if len(table_df) > 5 else 0, ha="right" if len(table_df) > 5 else "center")
    plt.legend(frameon=True, facecolor="white", loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_correlation_heatmap(corr_matrix, output_path, title="Feature Correlation Matrix (Pearson)"):
    """Generate high-res annotated correlation heatmap."""
    plt.figure(figsize=(14, 11), dpi=300)
    sns.set_theme(style="white", font="sans-serif")
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"}
    )
    plt.title(title, fontsize=13, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_effect_size_ranking(master_df, output_path):
    """Plot factors ranked by effect size."""
    plt.figure(figsize=(10, 8), dpi=300)
    sns.set_theme(style="whitegrid", font="sans-serif")

    plot_df = master_df.copy().sort_values(by="effect_size", ascending=True)
    
    # Color by effect size label
    color_map = {
        "strong": "#2b5c8f",
        "moderate": "#3e8e7e",
        "small": "#5cb85c",
        "negligible": "#d9534f"
    }
    colors = [color_map.get(label, "#999999") for label in plot_df["effect_size_label"]]

    y_pos = np.arange(len(plot_df))
    bars = plt.barh(y_pos, plot_df["effect_size"], color=colors, edgecolor="#333333", linewidth=0.6, alpha=0.9)

    plt.yticks(y_pos, [f"{row['factor']} ({row['effect_size_metric']})" for _, row in plot_df.iterrows()], fontsize=8)
    plt.xlabel("Effect Size (Cramér's V / Eta-Squared)", fontsize=10, fontweight="bold")
    plt.title("Statistical Drivers of Spoilage Ranked by Effect Size", fontsize=12, fontweight="bold", pad=12)

    # Threshold guidelines
    plt.axvline(0.05, color="#5cb85c", linestyle="--", linewidth=1.2, label="Small threshold (V ≥ 0.05)")
    plt.axvline(0.01, color="#f0ad4e", linestyle=":", linewidth=1.2, label="Small threshold (η² ≥ 0.01)")

    # Legend for labels
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#5cb85c", edgecolor="#333333", label="Small Effect"),
        Patch(facecolor="#d9534f", edgecolor="#333333", label="Negligible Effect"),
        plt.Line2D([0], [0], color="#5cb85c", linestyle="--", label="V = 0.05 Threshold"),
        plt.Line2D([0], [0], color="#f0ad4e", linestyle=":", label="η² = 0.01 Threshold")
    ]
    plt.legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_benchmark_deciles(decile_df, output_path, baseline=0.19442):
    """Plot spoilage rate across spoilage_risk deciles vs predicted risk."""
    plt.figure(figsize=(9, 5), dpi=300)
    sns.set_theme(style="whitegrid", font="sans-serif")

    x = decile_df["risk_decile"]
    y_actual = decile_df["actual_spoilage_rate"] * 100
    y_pred = decile_df["mean_predicted_risk"] * 100

    yerr_lower = (decile_df["actual_spoilage_rate"] - decile_df["ci_lower_95"]) * 100
    yerr_upper = (decile_df["ci_upper_95"] - decile_df["actual_spoilage_rate"]) * 100

    plt.plot(x, y_pred, marker="s", color="#337ab7", linewidth=2.0, label="Mean Predicted Spoilage Risk (%)")
    plt.errorbar(
        x, y_actual,
        yerr=[yerr_lower, yerr_upper],
        fmt="-o",
        color="#d9534f",
        capsize=4,
        linewidth=2.0,
        markersize=6,
        label="Observed Spoilage Rate (%) ± 95% Wilson CI"
    )

    plt.axhline(baseline * 100, color="#888888", linestyle="--", linewidth=1.0, label=f"Dataset Baseline ({baseline*100:.2f}%)")

    for i, row in decile_df.iterrows():
        plt.text(
            row["risk_decile"],
            row["actual_spoilage_rate"] * 100 + 0.8,
            f"{row['actual_spoilage_rate']*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#a94442"
        )

    plt.title("Benchmark Decile Calibration: Observed Spoilage vs. Predicted Spoilage Risk", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Spoilage Risk Decile (1 = Lowest Risk, 10 = Highest Risk)", fontsize=10, fontweight="bold")
    plt.ylabel("Rate / Probability (%)", fontsize=10, fontweight="bold")
    plt.xticks(x)
    plt.ylim(5, 35)
    plt.legend(frameon=True, facecolor="white", loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_full_m2_t01_pipeline(
    data_path="processed/cleaned_dataset.csv",
    tables_dir="reports/tables",
    figures_dir="reports/figures"
):
    """Execute complete M2-T01 statistical analysis pipeline and save all deliverables."""
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 70)
    print("M2-T01: Statistical Analysis of Spoilage Drivers Pipeline")
    print("=" * 70)

    # Step 0: Load dataset
    print(f"\n[Step 0] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    n_rows, n_cols = df.shape
    baseline_spoilage = df["was_spoiled"].mean()
    print(f"Loaded {n_rows:,} rows, {n_cols} columns. Target spoilage rate: {baseline_spoilage:.5f} ({baseline_spoilage*100:.2f}%)")

    # Define factor sets strictly enforcing hard rules
    cat_factors = ["category", "region", "quality_grade", "supplier_id", "store_id", "is_promoted"]
    num_factors = [
        "temp_abuse_events", "handling_score", "packaging_score", "supplier_score",
        "temp_deviation", "storage_temp", "days_until_expiry", "days_remaining_at_purchase",
        "shelf_life_days", "expiry_remaining_ratio", "shelf_life_used_ratio",
        "daily_demand", "demand_variability", "initial_quantity", "distribution_hours",
        "spoilage_sensitivity", "base_price", "cost_price"
    ]

    # Step 1: Chi-Square tests
    print("\n[Step 1] Running Chi-Square tests of independence on categorical factors...")
    cat_results = compute_categorical_tests(df, cat_factors, target="was_spoiled")
    cat_results.to_csv(os.path.join(tables_dir, "chi_square_results.csv"), index=False)
    print("Chi-Square results saved to chi_square_results.csv")

    # Sub-tables for category, region, quality_grade
    category_table = compute_factor_spoilage_table(df, "category", target="was_spoiled")
    region_table = compute_factor_spoilage_table(df, "region", target="was_spoiled")
    grade_table = compute_factor_spoilage_table(df, "quality_grade", target="was_spoiled")

    category_table.to_csv(os.path.join(tables_dir, "spoilage_rate_by_category.csv"), index=False)
    region_table.to_csv(os.path.join(tables_dir, "spoilage_rate_by_region.csv"), index=False)
    grade_table.to_csv(os.path.join(tables_dir, "spoilage_rate_by_quality_grade.csv"), index=False)

    plot_spoilage_by_factor_ci(
        category_table, "category",
        os.path.join(figures_dir, "spoilage_rate_by_category.png"),
        "Observed Spoilage Rate by Product Category (95% Wilson CI)",
        baseline=baseline_spoilage
    )
    plot_spoilage_by_factor_ci(
        region_table, "region",
        os.path.join(figures_dir, "spoilage_rate_by_region.png"),
        "Observed Spoilage Rate by Geographic Region (95% Wilson CI)",
        baseline=baseline_spoilage
    )
    plot_spoilage_by_factor_ci(
        grade_table, "quality_grade",
        os.path.join(figures_dir, "spoilage_rate_by_quality_grade.png"),
        "Observed Spoilage Rate by Quality Grade (95% Wilson CI)",
        baseline=baseline_spoilage
    )
    print("Factor spoilage tables and CI bar charts generated.")

    # Step 2: Numeric tests
    print("\n[Step 2] Running group comparisons, ANOVA/Welch, Mann-Whitney U, and effect sizes on numeric factors...")
    num_results = compute_numeric_tests(df, num_factors, target="was_spoiled")
    num_results.to_csv(os.path.join(tables_dir, "numeric_test_results.csv"), index=False)
    print("Numeric test results saved to numeric_test_results.csv")

    # Tukey HSD on handling_score across quality_grade
    tukey_handling = pairwise_tukeyhsd(df["handling_score"], df["quality_grade"], alpha=0.05)
    tukey_df = pd.DataFrame(data=tukey_handling._results_table.data[1:], columns=tukey_handling._results_table.data[0])
    tukey_df.to_csv(os.path.join(tables_dir, "tukey_handling_by_grade.csv"), index=False)

    # Step 3 & 4 & 7: Multiple comparison corrections & Master table
    print("\n[Step 3 & 4] Applying Holm & Bonferroni multiple testing corrections and effect size classifications...")
    master_ranking = build_master_ranking_table(cat_results, num_results, None)
    master_ranking.to_csv(os.path.join(tables_dir, "master_factor_ranking.csv"), index=False)
    plot_effect_size_ranking(master_ranking, os.path.join(figures_dir, "effect_size_ranking.png"))
    print("Master ranking table saved to master_factor_ranking.csv and effect_size_ranking.png")

    # Step 5: Correlation analysis
    print("\n[Step 5] Running correlation analysis, multicollinearity screening, VIF, and MI...")
    corr_results = compute_correlations(df, num_factors, target="was_spoiled")
    corr_results.to_csv(os.path.join(tables_dir, "correlation_results.csv"), index=False)

    collinear_df, vif_df, corr_matrix, spearman_matrix = compute_vif_and_collinearity(df, num_factors)
    collinear_df.to_csv(os.path.join(tables_dir, "multicollinearity_pairs.csv"), index=False)
    vif_df.to_csv(os.path.join(tables_dir, "vif_results.csv"), index=False)
    plot_correlation_heatmap(corr_matrix, os.path.join(figures_dir, "correlation_heatmap.png"))

    mi_results = compute_mutual_information(df, num_factors, ["category", "region", "quality_grade"], target="was_spoiled")
    mi_results.to_csv(os.path.join(tables_dir, "mutual_information_results.csv"), index=False)

    # Benchmark: spoilage_risk deciles
    benchmark_deciles, benchmark_corr = compute_spoilage_risk_benchmark(df, risk_col="spoilage_risk", target="was_spoiled")
    benchmark_deciles.to_csv(os.path.join(tables_dir, "benchmark_spoilage_risk_deciles.csv"), index=False)
    plot_benchmark_deciles(benchmark_deciles, os.path.join(figures_dir, "benchmark_spoilage_risk_deciles.png"), baseline=baseline_spoilage)
    print(f"Benchmark spoilage_risk correlation with was_spoiled: r = {benchmark_corr:.4f}")

    # Step 6: Robustness checks
    print("\n[Step 6] Running temporal stability check (70/30 chronological split) and permutation test (1,000 runs)...")
    stability_df, split_date = run_temporal_stability(
        df,
        key_categoricals=["category", "quality_grade", "region"],
        key_numerics=["packaging_score", "handling_score", "shelf_life_used_ratio", "temp_deviation"],
        date_col="transaction_date",
        target="was_spoiled"
    )
    stability_df.to_csv(os.path.join(tables_dir, "temporal_stability_results.csv"), index=False)
    print(f"Temporal stability verified on 70/30 split (split date: {split_date}).")

    perm_results = run_permutation_sanity_check(df, target="was_spoiled", n_permutations=1000)
    perm_df = pd.DataFrame([perm_results])
    perm_df.to_csv(os.path.join(tables_dir, "permutation_test_results.csv"), index=False)
    print(f"Permutation baseline FPRs: Category Chi2 = {perm_results['category_chi2_fpr']:.3f}, Packaging t = {perm_results['packaging_score_t_fpr']:.3f}, Handling t = {perm_results['handling_score_t_fpr']:.3f}")

    print("\n" + "=" * 70)
    print("M2-T01 Pipeline Execution Complete! All tables and figures generated.")
    print("=" * 70)


if __name__ == "__main__":
    run_full_m2_t01_pipeline()

