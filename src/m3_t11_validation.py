"""
src/m3_t11_validation.py - Track 3C Two-Lane Optimizer Validation (M3-T11).

Implements strictly separated:
  - Lane 1 (lane1_followed_vs_not): Observed evidence comparing followed vs not-followed
    batches within training-derived strata, with week-grouped bootstrap 95% CI,
    within-stratum permutation negative control, and observed depth-band descriptives.
  - Lane 2 (lane2_counterfactual): Simulated, model-estimated counterfactual gains on
    non-followed batches (never blended with Lane 1), with actual-depth model-error context.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import RANDOM_STATE
from src.m3_t04_scenario import DISCOUNT_BIN_LABELS, assign_discount_bin
from src.m3_t09_optimizer import get_flat_curve_threshold, score_whatif
from src.m3_t10_tiers import assign_tier, load_tier_edges


def derive_train_r1_strata_edges(tr_r1: pd.DataFrame) -> Dict[str, Any]:
    """Derives tercile edges strictly from training R1 rows (N = 24,475)."""
    exp_edges = np.quantile(
        tr_r1["raw_expiry_remaining_ratio"].to_numpy(dtype=float), [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
    )
    oof_edges = np.quantile(
        tr_r1["oof_proba"].to_numpy(dtype=float), [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
    )
    return {"exp_edges": exp_edges, "oof_edges": oof_edges}


def assign_r1_strata(df_r1: pd.DataFrame, strata_edges: Dict[str, Any]) -> pd.DataFrame:
    """Assigns (exp_terc, grade_grp, oof_terc, stratum_id) using training edges."""
    exp_t = np.digitize(
        df_r1["raw_expiry_remaining_ratio"].to_numpy(dtype=float),
        strata_edges["exp_edges"][1:-1],
        right=True,
    ) + 1
    oof_t = np.digitize(
        df_r1["oof_proba"].to_numpy(dtype=float),
        strata_edges["oof_edges"][1:-1],
        right=True,
    ) + 1
    grade_g = np.where(df_r1["quality_grade"] == "C", "C", "AB")
    stratum_id = [f"E{e}_G{g}_R{r}" for e, g, r in zip(exp_t, grade_g, oof_t)]
    return pd.DataFrame({
        "exp_terc": exp_t,
        "grade_grp": grade_g,
        "oof_terc": oof_t,
        "stratum_id": stratum_id,
    }, index=df_r1.index)


def _weighted_stratum_diff(df: pd.DataFrame, followed_col: str, outcome_cols: List[str], included_strata: List[str]) -> Dict[str, float]:
    """Computes stratum-size-weighted mean difference (followed - not_followed)."""
    tot_w = 0.0
    sums = {c: 0.0 for c in outcome_cols}
    for st in included_strata:
        sub = df[df["stratum_id"] == st]
        f_m = sub[followed_col].to_numpy(dtype=bool)
        if f_m.sum() == 0 or (~f_m).sum() == 0:
            continue
        w = float(len(sub))
        tot_w += w
        for c in outcome_cols:
            vals = sub[c].to_numpy(dtype=float)
            sums[c] += w * (float(vals[f_m].mean()) - float(vals[~f_m].mean()))
    if tot_w == 0.0:
        return {c: float("nan") for c in outcome_cols}
    return {c: float(sums[c] / tot_w) for c in outcome_cols}


def lane1_followed_vs_not(
    te_r1_merged: pd.DataFrame,
    tol: float = 0.025,
    min_per_arm: int = 20,
    n_boot: int = 1000,
    n_perm: int = 200,
    run_neg_control: bool = True,
) -> Dict[str, Any]:
    """
    Lane 1 observed evidence: compares followed (|actual - rec| <= tol) vs not-followed
    within strata having >= min_per_arm in both arms, with week-grouped bootstrap 95% CI
    and within-stratum permutation negative control.
    """
    df = te_r1_merged.copy()
    df["followed"] = (df["raw_discount_pct"] - df["recommended_depth"]).abs() <= (tol + 1e-9)
    df["profit_per_unit"] = df["profit"] / df["raw_initial_quantity"]
    df["waste_share"] = df["units_wasted"] / df["raw_initial_quantity"]

    strata_counts = (
        df.groupby("stratum_id", sort=True)
        .agg(
            n_total=("record_id", "count"),
            n_followed=("followed", "sum"),
        )
        .reset_index()
    )
    strata_counts["n_not_followed"] = strata_counts["n_total"] - strata_counts["n_followed"]
    strata_counts["included"] = (
        (strata_counts["n_followed"] >= min_per_arm)
        & (strata_counts["n_not_followed"] >= min_per_arm)
    )

    included_strata = strata_counts.loc[strata_counts["included"], "stratum_id"].tolist()
    inc_df = df[df["stratum_id"].isin(included_strata)].copy()

    n_followed_inc = int(inc_df["followed"].sum())
    n_not_followed_inc = int((~inc_df["followed"]).sum())
    n_inc_batches = int(len(inc_df))

    outcome_cols = ["profit_per_unit", "s", "waste_share", "profit"]
    point_est = _weighted_stratum_diff(inc_df, "followed", outcome_cols, included_strata)

    # Week-grouped bootstrap 95% CI (1,000 resamples)
    weeks = pd.to_datetime(inc_df["transaction_date"]).dt.to_period("W").astype(str).to_numpy()
    unique_weeks = np.unique(weeks)
    week_idx = [np.where(weeks == w)[0] for w in unique_weeks]
    rng = np.random.default_rng(RANDOM_STATE)

    boot_vals = {c: [] for c in outcome_cols}
    for _ in range(n_boot):
        chosen = rng.integers(0, len(week_idx), size=len(week_idx))
        idx = np.concatenate([week_idx[w] for w in chosen])
        b_df = inc_df.iloc[idx]
        est_b = _weighted_stratum_diff(b_df, "followed", outcome_cols, included_strata)
        for c in outcome_cols:
            if not np.isnan(est_b[c]):
                boot_vals[c].append(est_b[c])

    ci95 = {
        c: [float(np.percentile(boot_vals[c], 2.5)), float(np.percentile(boot_vals[c], 97.5))]
        for c in outcome_cols
    }

    # Pre-registered decision rule
    pu_lo, pu_hi = ci95["profit_per_unit"]
    if n_followed_inc < 300:
        verdict = "underpowered"
    elif pu_lo > 0.0:
        verdict = "supports the optimizer"
    elif pu_hi < 0.0:
        verdict = "contradicts"
    else:
        verdict = "inconclusive"

    # Negative control: permute recommended_depth within each stratum 200 times
    neg_ctrl = None
    if run_neg_control:
        rng_perm = np.random.default_rng(RANDOM_STATE)
        perm_est = np.zeros(n_perm, dtype=float)
        stratum_groups = [np.where(df["stratum_id"].to_numpy() == st)[0] for st in df["stratum_id"].unique()]
        rec_orig = df["recommended_depth"].to_numpy(dtype=float)
        act_orig = df["raw_discount_pct"].to_numpy(dtype=float)

        for p_i in range(n_perm):
            rec_perm = rec_orig.copy()
            for g_idx in stratum_groups:
                rec_perm[g_idx] = rng_perm.permutation(rec_orig[g_idx])
            df_p = df.copy()
            df_p["_f_perm"] = np.abs(act_orig - rec_perm) <= (tol + 1e-9)
            sc_p = df_p.groupby("stratum_id")["_f_perm"].agg(tot="count", f="sum")
            inc_p = sc_p[(sc_p["f"] >= min_per_arm) & ((sc_p["tot"] - sc_p["f"]) >= min_per_arm)].index.tolist()
            perm_est[p_i] = _weighted_stratum_diff(df_p[df_p["stratum_id"].isin(inc_p)], "_f_perm", ["profit_per_unit"], inc_p)["profit_per_unit"]

        real_pu = point_est["profit_per_unit"]
        p_two_sided = float(np.mean(np.abs(perm_est) >= abs(real_pu)))
        neg_ctrl = {
            "n_permutations": n_perm,
            "null_mean": float(np.mean(perm_est)),
            "null_sd": float(np.std(perm_est)),
            "null_p025": float(np.percentile(perm_est, 2.5)),
            "null_p975": float(np.percentile(perm_est, 97.5)),
            "real_estimate": float(real_pu),
            "empirical_two_sided_p": p_two_sided,
        }

    # Supporting descriptive view: observed mean profit_per_unit by actual 5-pt depth bin within strata (>=30 rows)
    df["actual_bin"] = assign_discount_bin(df["raw_discount_pct"]).astype(str)
    obs_bin_rows = []
    for (st, b), sub in df.groupby(["stratum_id", "actual_bin"], observed=True):
        if len(sub) >= 30:
            obs_bin_rows.append({
                "stratum_id": st,
                "actual_bin": b,
                "n": int(len(sub)),
                "mean_obs_profit_per_unit": float(sub["profit_per_unit"].mean()),
                "mean_obs_s": float(sub["s"].mean()),
            })

    return {
        "tolerance": tol,
        "n_test_r1_total": int(len(df)),
        "n_included_strata": len(included_strata),
        "n_total_strata": int(len(strata_counts)),
        "n_batches_included": n_inc_batches,
        "coverage_share_pct": float(n_inc_batches / len(df) * 100.0),
        "n_followed_included": n_followed_inc,
        "n_not_followed_included": n_not_followed_inc,
        "verdict": verdict,
        "estimates": point_est,
        "ci95": ci95,
        "strata_table": strata_counts.to_dict(orient="records"),
        "negative_control": neg_ctrl,
        "observed_depth_band_by_stratum": obs_bin_rows,
    }


def lane2_counterfactual(te_r1_merged: pd.DataFrame, tol: float = 0.025, n_boot: int = 1000) -> Dict[str, Any]:
    """
    Lane 2 simulated counterfactual evaluation (NEVER blended with Lane 1).
    Population: test R1 batches where |actual_depth - recommended_depth| > tol.
    """
    df = te_r1_merged[
        (te_r1_merged["raw_discount_pct"] - te_r1_merged["recommended_depth"]).abs() > (tol + 1e-9)
    ].copy().reset_index(drop=True)

    tiers_df = assign_tier(df, load_tier_edges())
    df["tier"] = tiers_df["tier"].values

    long_wf = score_whatif(df)
    supp_wf = long_wf[long_wf["supported"]].copy()

    # For each batch, get supported prediction at recommended_depth and at nearest supported grid point to actual depth
    rec_rows = supp_wf.merge(
        df[["record_id", "recommended_depth"]], on="record_id"
    )
    rec_rows = rec_rows[np.isclose(rec_rows["depth"], rec_rows["recommended_depth"])][
        ["record_id", "batch_profit", "profit_per_unit"]
    ].rename(columns={"batch_profit": "pred_profit_rec", "profit_per_unit": "pred_pu_rec"})

    act_rows = supp_wf.merge(df[["record_id", "raw_discount_pct"]], on="record_id")
    act_rows["snap_dist"] = (act_rows["depth"] - act_rows["raw_discount_pct"]).abs()
    act_rows = act_rows.sort_values(["record_id", "snap_dist", "depth"]).groupby("record_id", as_index=False).first()[
        ["record_id", "depth", "snap_dist", "batch_profit", "profit_per_unit"]
    ].rename(columns={
        "depth": "nearest_supp_actual_depth",
        "batch_profit": "pred_profit_actual",
        "profit_per_unit": "pred_pu_actual",
    })

    eval_df = df.merge(rec_rows, on="record_id", how="inner").merge(act_rows, on="record_id", how="inner")
    assert len(eval_df) == len(df), f"Lane 2 merge mismatch: {len(eval_df)} vs {len(df)}"

    eval_df["gain_batch"] = eval_df["pred_profit_rec"] - eval_df["pred_profit_actual"]
    eval_df["gain_unit"] = eval_df["pred_pu_rec"] - eval_df["pred_pu_actual"]
    eval_df["obs_pu"] = eval_df["profit"] / eval_df["raw_initial_quantity"]

    flat_thr = get_flat_curve_threshold()

    # Week-grouped bootstrap 95% CI on mean gain_unit and mean gain_batch
    weeks = pd.to_datetime(eval_df["transaction_date"]).dt.to_period("W").astype(str).to_numpy()
    week_idx = [np.where(weeks == w)[0] for w in np.unique(weeks)]
    rng = np.random.default_rng(RANDOM_STATE)
    gu_arr = eval_df["gain_unit"].to_numpy(dtype=float)
    gb_arr = eval_df["gain_batch"].to_numpy(dtype=float)
    boot_gu = np.zeros(n_boot, dtype=float)
    boot_gb = np.zeros(n_boot, dtype=float)
    for k in range(n_boot):
        chosen = rng.integers(0, len(week_idx), size=len(week_idx))
        idx = np.concatenate([week_idx[w] for w in chosen])
        boot_gu[k] = gu_arr[idx].mean()
        boot_gb[k] = gb_arr[idx].mean()

    # Model-error context at actual depth
    obs_pu = eval_df["obs_pu"].to_numpy(dtype=float)
    pred_pu_act = eval_df["pred_pu_actual"].to_numpy(dtype=float)
    model_err_ctx = {
        "mae_profit_per_unit": float(mean_absolute_error(obs_pu, pred_pu_act)),
        "rmse_profit_per_unit": float(np.sqrt(mean_squared_error(obs_pu, pred_pu_act))),
        "mean_error_pred_minus_obs": float(np.mean(pred_pu_act - obs_pu)),
        "mean_snap_dist": float(eval_df["snap_dist"].mean()),
        "max_snap_dist": float(eval_df["snap_dist"].max()),
    }

    by_tier = (
        eval_df.groupby("tier", observed=True)
        .agg(n=("record_id", "count"), mean_gain_unit=("gain_unit", "mean"), median_gain_unit=("gain_unit", "median"), mean_gain_batch=("gain_batch", "mean"))
        .reset_index()
        .to_dict(orient="records")
    )
    by_exp_grade = (
        eval_df.groupby(["exp_terc", "grade_grp"], observed=True)
        .agg(n=("record_id", "count"), mean_gain_unit=("gain_unit", "mean"), median_gain_unit=("gain_unit", "median"), mean_gain_batch=("gain_batch", "mean"))
        .reset_index()
        .to_dict(orient="records")
    )
    by_flat = (
        eval_df.groupby("flat_curve", observed=True)
        .agg(n=("record_id", "count"), mean_gain_unit=("gain_unit", "mean"), median_gain_unit=("gain_unit", "median"), mean_gain_batch=("gain_batch", "mean"))
        .reset_index()
        .to_dict(orient="records")
    )

    return {
        "label": "simulated / model-estimated, not observed",
        "n_batches": int(len(eval_df)),
        "mean_gain_unit": float(gu_arr.mean()),
        "median_gain_unit": float(np.median(gu_arr)),
        "ci95_mean_gain_unit": [float(np.percentile(boot_gu, 2.5)), float(np.percentile(boot_gu, 97.5))],
        "mean_gain_batch": float(gb_arr.mean()),
        "median_gain_batch": float(np.median(gb_arr)),
        "ci95_mean_gain_batch": [float(np.percentile(boot_gb, 2.5)), float(np.percentile(boot_gb, 97.5))],
        "share_positive_gain_pct": float((gu_arr > 0).mean() * 100.0),
        "share_below_cv_mae_pct": float((gu_arr < flat_thr).mean() * 100.0),
        "cv_profit_unit_mae": float(flat_thr),
        "model_error_at_actual_depth": model_err_ctx,
        "by_tier": by_tier,
        "by_exp_terc_and_grade": by_exp_grade,
        "by_flat_curve": by_flat,
    }
