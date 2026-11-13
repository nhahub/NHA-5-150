"""
scripts/M3_T01_train_track3a.py - End-to-end Track 3A Modeling, Calibration & Evaluation.

Executes M3-T01:
1. Reference baselines on 5-fold time-blocked CV (constant base-rate, spoilage_risk, M2 LGBM OOF).
2. Candidate tuning (LogisticRegression, RandomForest, XGBoost, LightGBM) on FULL & COMPACT.
3. Leak-free calibration comparison (uncalibrated vs Platt vs isotonic) inside CV.
4. Final model selection (preferring simplest within noise).
5. Conditional regeneration of track_3a_oof_train.csv (only if clearly beating baseline).
6. Single final test evaluation (Brier, log loss, AUC, threshold recall/CM, SHAP, support flag).
7. Save final model & JSON metadata via save_model_with_metadata.
"""

import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FIGURES_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TRACK_3A_OOF_TRAIN_PATH,
)
from src.m3_t01_track3a import (
    FEATURE_SETS,
    compute_threshold_metrics,
    evaluate_calibration_cv,
    evaluate_probs,
    evaluate_reference_baselines,
    load_track_3a_splits,
    make_candidate_factory,
    tune_candidate_cv,
)
from src.mlflow_utils import (
    log_calibration_curve,
    log_confusion_matrix,
    log_figure_artifact,
    log_shap_summary,
    log_track_3a_metrics,
    save_model_with_metadata,
    setup_tracking_environment,
    start_tracked_run,
)
from src.support import compute_support, flag_low_confidence

REPORTS_M3_DIR = REPORTS_DIR / "M3"
FIGURES_M3_DIR = FIGURES_DIR / "m3"
COMPLEXITY_RANK = {"logreg": 1, "lightgbm": 2, "xgboost": 3, "rf": 4}
FEAT_RANK = {"track_3a_compact": 1, "track_3a_full": 2}
CAL_RANK = {"uncalibrated": 1, "sigmoid": 2, "isotonic": 3}


def main() -> None:
    setup_tracking_environment()
    REPORTS_M3_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_M3_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("M3-T01: Track 3A Spoilage Classifier — CV, Calibration & Evaluation")
    print("==================================================================")

    train_df, test_df = load_track_3a_splits()
    y_train = train_df["was_spoiled"].values
    y_test = test_df["was_spoiled"].values
    bench_train_auc = float(roc_auc_score(y_train, train_df["spoilage_risk"].values))
    bench_test_auc = float(roc_auc_score(y_test, test_df["spoilage_risk"].values))

    # ------------------------------------------------------------------
    # Step 1: Reference baselines
    # ------------------------------------------------------------------
    print("\n[Step 1] Computing reference baselines on 5-fold time-blocked CV...")
    refs = evaluate_reference_baselines(train_df, n_blocks=5)
    for k, m in refs.items():
        print(f"  {k:32s} | Brier={m['brier']:.5f} | LogLoss={m['log_loss']:.5f} | AUC={m['roc_auc']:.4f} | ECE={m['ece']:.5f}")

    # ------------------------------------------------------------------
    # Step 2: Candidate tuning (4 families x 2 feature sets = 8 runs)
    # ------------------------------------------------------------------
    print("\n[Step 2] Tuning candidates on 5-fold time-blocked CV...")
    candidates = []
    for family in ["logreg", "rf", "xgboost", "lightgbm"]:
        for fset in ["track_3a_compact", "track_3a_full"]:
            n_trials = 8 if family != "rf" else 6
            res = tune_candidate_cv(
                model_family=family,
                feature_set_name=fset,
                train_df=train_df,
                n_trials=n_trials,
                n_blocks=5,
            )
            m = res["metrics"]
            print(
                f"  {family:9s} | {fset:16s} | Brier={m['brier']:.5f} | LogLoss={m['log_loss']:.5f} "
                f"| AUC={m['roc_auc']:.4f} | ECE={m['ece']:.5f} | params={res['params']}"
            )
            with start_tracked_run(
                track="3A",
                model_type=family,
                feature_set_name=fset,
                shelf_life_version="expiry_remaining_ratio",
                features=FEATURE_SETS[fset],
                run_name=f"m3_t01_cv_{family}_{fset}",
                purpose="cv_candidate",
                params=res["params"],
            ):
                log_track_3a_metrics(
                    brier_score=m["brier"],
                    log_loss_val=m["log_loss"],
                    roc_auc=m["roc_auc"],
                    benchmark_auc=bench_train_auc,
                    extra_metrics={"cv_ece": m["ece"]},
                )
            candidates.append(res)

    # Sort candidates by CV log_loss, then complexity
    candidates_sorted = sorted(
        candidates,
        key=lambda c: (
            round(c["metrics"]["log_loss"], 4),
            round(c["metrics"]["brier"], 4),
            COMPLEXITY_RANK[c["model_family"]],
            FEAT_RANK[c["feature_set_name"]],
        ),
    )
    top_candidates = candidates_sorted[:2]

    # ------------------------------------------------------------------
    # Step 3: Calibration comparison inside CV (no leakage across folds)
    # ------------------------------------------------------------------
    print("\n[Step 3] Evaluating calibration (uncalibrated vs Platt vs isotonic) inside CV...")
    cal_results = []
    for cand in top_candidates:
        family = cand["model_family"]
        fset = cand["feature_set_name"]
        base_factory = make_candidate_factory(
            family, cand["params"], needs_imputation=cand["needs_imputation"]
        )
        for method in ["uncalibrated", "sigmoid", "isotonic"]:
            oof_df, m, rel_df = evaluate_calibration_cv(
                base_factory=base_factory,
                feature_set_name=fset,
                model_family=family,
                train_df=train_df,
                method=method,
                n_blocks=5,
            )
            print(
                f"  {family:9s} | {fset:16s} | {method:12s} | Brier={m['brier']:.5f} "
                f"| LogLoss={m['log_loss']:.5f} | AUC={m['roc_auc']:.4f} | ECE={m['ece']:.5f}"
            )
            with start_tracked_run(
                track="3A",
                model_type=f"{family}_{method}",
                feature_set_name=fset,
                shelf_life_version="expiry_remaining_ratio",
                features=FEATURE_SETS[fset],
                run_name=f"m3_t01_cal_{family}_{fset}_{method}",
                purpose="cv_calibration",
                params={**cand["params"], "calibration": method},
            ):
                log_track_3a_metrics(
                    brier_score=m["brier"],
                    log_loss_val=m["log_loss"],
                    roc_auc=m["roc_auc"],
                    benchmark_auc=bench_train_auc,
                    extra_metrics={"cv_ece": m["ece"]},
                )
                log_calibration_curve(y_train, oof_df["oof_proba"].values, n_bins=10, name=f"cv_rel_{family}_{method}")

            cal_results.append({
                "model_family": family,
                "feature_set_name": fset,
                "params": cand["params"],
                "needs_imputation": cand["needs_imputation"],
                "calibration": method,
                "metrics": m,
                "oof_df": oof_df,
                "rel_df": rel_df,
            })

    # ------------------------------------------------------------------
    # Step 4: Pick final configuration (prefer simplest if within 0.0001 noise)
    # ------------------------------------------------------------------
    best_ll = min(r["metrics"]["log_loss"] for r in cal_results)
    near_best = [r for r in cal_results if r["metrics"]["log_loss"] <= best_ll + 0.0002]
    chosen = sorted(
        near_best,
        key=lambda r: (
            COMPLEXITY_RANK[r["model_family"]],
            FEAT_RANK[r["feature_set_name"]],
            CAL_RANK[r["calibration"]],
            r["metrics"]["log_loss"],
        ),
    )[0]

    print(
        f"\n[Step 4] Chosen Final Config: {chosen['model_family']} | {chosen['feature_set_name']} "
        f"| cal={chosen['calibration']} | CV Brier={chosen['metrics']['brier']:.5f} "
        f"| CV LogLoss={chosen['metrics']['log_loss']:.5f} | CV AUC={chosen['metrics']['roc_auc']:.4f}"
    )

    # Save CV reliability curve figure
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    for r in cal_results:
        if r["model_family"] == chosen["model_family"] and r["feature_set_name"] == chosen["feature_set_name"]:
            ax.plot(
                r["rel_df"]["mean_predicted_prob"],
                r["rel_df"]["observed_spoilage_rate"],
                marker="o",
                label=f"{r['calibration']} (ECE={r['metrics']['ece']:.4f})",
            )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed spoilage rate")
    ax.set_title(f"Track 3A CV Reliability Curve ({chosen['model_family']} / {chosen['feature_set_name']})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    rel_fig_path = FIGURES_M3_DIR / "m3_t01_cv_reliability_curve.png"
    fig.savefig(rel_fig_path, dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Step 5: Conditional OOF regeneration
    # ------------------------------------------------------------------
    baseline_brier = refs["existing_lgbm_oof_baseline"]["brier"]
    baseline_ll = refs["existing_lgbm_oof_baseline"]["log_loss"]
    brier_gain = baseline_brier - chosen["metrics"]["brier"]
    ll_gain = baseline_ll - chosen["metrics"]["log_loss"]
    oof_regenerated = bool(brier_gain > 0.0005 or ll_gain > 0.0010)

    if oof_regenerated:
        chosen["oof_df"].to_csv(TRACK_3A_OOF_TRAIN_PATH, index=False)
        print(f"[Step 5] Regenerated {TRACK_3A_OOF_TRAIN_PATH} (Brier gain={brier_gain:.5f}, LogLoss gain={ll_gain:.5f}).")
    else:
        print(
            f"[Step 5] Kept existing {TRACK_3A_OOF_TRAIN_PATH} (Brier gain={brier_gain:+.5f}, "
            f"LogLoss gain={ll_gain:+.5f} — within noise of M2 baseline)."
        )

    # ------------------------------------------------------------------
    # Step 6 & 7: Final fit on full train, single test eval, SHAP, support, save
    # ------------------------------------------------------------------
    print("\n[Step 6 & 7] Fitting final model on full training set and evaluating ONCE on test...")
    final_features = FEATURE_SETS[chosen["feature_set_name"]]
    X_train_final = train_df[final_features]
    X_test_final = test_df[final_features]

    base_factory = make_candidate_factory(
        chosen["model_family"], chosen["params"], needs_imputation=chosen["needs_imputation"]
    )
    if chosen["calibration"] == "uncalibrated":
        final_model = base_factory()
        final_model.fit(X_train_final, y_train)
    else:
        from sklearn.calibration import CalibratedClassifierCV
        final_model = CalibratedClassifierCV(
            estimator=base_factory(), method=chosen["calibration"], cv=3
        )
        final_model.fit(X_train_final, y_train)

    test_proba = final_model.predict_proba(X_test_final)[:, 1]
    test_metrics = evaluate_probs(y_test, test_proba)
    bench_test_metrics = evaluate_probs(y_test, test_df["spoilage_risk"].values)
    thr_rows = compute_threshold_metrics(y_test, test_proba, thresholds=(0.20, 0.30, 0.50))

    print(f"  Test Model:     Brier={test_metrics['brier']:.5f} | LogLoss={test_metrics['log_loss']:.5f} | AUC={test_metrics['roc_auc']:.4f} | ECE={test_metrics['ece']:.5f}")
    print(f"  Test Benchmark: Brier={bench_test_metrics['brier']:.5f} | LogLoss={bench_test_metrics['log_loss']:.5f} | AUC={bench_test_metrics['roc_auc']:.4f} | ECE={bench_test_metrics['ece']:.5f}")
    for tr in thr_rows:
        print(f"  Threshold {tr['threshold']:.2f}: Recall={tr['recall']:.4f}, Prec={tr['precision']:.4f}, F1={tr['f1']:.4f}, PosRate={tr['pred_pos_rate']:.4f}, CM=[[{tr['tn']},{tr['fp']}],[{tr['fn']},{tr['tp']}]]")

    # Support flag check on test
    supp_table = compute_support(train_df)
    low_conf_flags = flag_low_confidence(test_df, supp_table)
    flagged_mask = low_conf_flags.values
    flagged_metrics = evaluate_probs(y_test[flagged_mask], test_proba[flagged_mask])
    unflagged_metrics = evaluate_probs(y_test[~flagged_mask], test_proba[~flagged_mask])
    print(
        f"  Support Flagged ({int(flagged_mask.sum())} / {len(test_df)} = {flagged_mask.mean()*100:.2f}%): "
        f"Brier={flagged_metrics['brier']:.5f}, AUC={flagged_metrics['roc_auc']:.4f} | "
        f"Unflagged: Brier={unflagged_metrics['brier']:.5f}, AUC={unflagged_metrics['roc_auc']:.4f}"
    )

    # SHAP summary (use underlying tree model or linear explainer)
    with start_tracked_run(
        track="3A",
        model_type=f"{chosen['model_family']}_{chosen['calibration']}",
        feature_set_name=chosen["feature_set_name"],
        shelf_life_version="expiry_remaining_ratio",
        features=final_features,
        run_name="m3_t01_final_track_3a",
        purpose="production",
        params={**chosen["params"], "calibration": chosen["calibration"], "oof_regenerated": oof_regenerated},
    ) as run:
        log_track_3a_metrics(
            brier_score=test_metrics["brier"],
            log_loss_val=test_metrics["log_loss"],
            roc_auc=test_metrics["roc_auc"],
            benchmark_auc=bench_test_auc,
            threshold=0.20,
            precision=thr_rows[0]["precision"],
            recall=thr_rows[0]["recall"],
            f1=thr_rows[0]["f1"],
            extra_metrics={
                "test_ece": test_metrics["ece"],
                "benchmark_test_brier": bench_test_metrics["brier"],
                "benchmark_test_log_loss": bench_test_metrics["log_loss"],
                "test_flagged_brier": flagged_metrics["brier"],
                "test_unflagged_brier": unflagged_metrics["brier"],
            },
        )
        log_calibration_curve(y_test, test_proba, n_bins=10, name="test_reliability_curve")
        for tr in thr_rows:
            log_confusion_matrix(
                y_test,
                (test_proba >= tr["threshold"]).astype(int),
                name=f"test_confusion_matrix_thr_{int(tr['threshold']*100)}",
            )

        # SHAP
        shap_top10: List[Tuple[str, float]] = []
        if chosen["model_family"] in ("lightgbm", "xgboost", "rf"):
            # Unwrap CalibratedClassifierCV or Pipeline if needed
            if chosen["calibration"] != "uncalibrated":
                expl_model = final_model.calibrated_classifiers_[0].estimator
            else:
                expl_model = final_model
            if hasattr(expl_model, "named_steps"):
                expl_model = expl_model.named_steps["clf"]
            shap_dict = log_shap_summary(expl_model, X_test_final, max_rows=2000, name="test_shap_summary")
            shap_top10 = sorted(shap_dict.items(), key=lambda kv: kv[1], reverse=True)[:10]
        else:
            # Linear model SHAP
            expl_model = final_model.calibrated_classifiers_[0].estimator if chosen["calibration"] != "uncalibrated" else final_model
            X_sample = X_test_final.sample(n=min(2000, len(X_test_final)), random_state=RANDOM_STATE)
            if hasattr(expl_model, "named_steps"):
                X_mat = expl_model.named_steps["imputer"].transform(X_sample)
                X_tr_mat = expl_model.named_steps["imputer"].transform(X_train_final.sample(n=2000, random_state=RANDOM_STATE))
                lr_est = expl_model.named_steps["clf"]
            else:
                X_mat = X_sample.values
                X_tr_mat = X_train_final.sample(n=2000, random_state=RANDOM_STATE).values
                lr_est = expl_model
            explainer = shap.LinearExplainer(lr_est, X_tr_mat)
            sv = explainer.shap_values(X_mat)
            mean_abs = np.abs(sv).mean(axis=0)
            shap_dict = {f: float(v) for f, v in zip(final_features, mean_abs)}
            shap_top10 = sorted(shap_dict.items(), key=lambda kv: kv[1], reverse=True)[:10]
            fig, ax = plt.subplots(figsize=(8, 5))
            top_names = [k for k, _ in reversed(shap_top10)]
            top_vals = [v for _, v in reversed(shap_top10)]
            ax.barh(top_names, top_vals)
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title("Track 3A Top-10 SHAP Feature Importances (Test Sample)")
            fig.tight_layout()
            log_figure_artifact(fig, "test_shap_summary.png")
            fig.savefig(FIGURES_M3_DIR / "m3_t01_test_shap_summary.png", dpi=150)
            plt.close(fig)

        model_path, meta_path = save_model_with_metadata(
            model=final_model,
            track="3A",
            feature_list=final_features,
            shelf_life_version="expiry_remaining_ratio",
            train_period="2023-01-01 to 2024-08-07",
            mlflow_run_id=run.info.run_id,
            overwrite=True,
            log_as_artifact=True,
        )
        final_run_id = run.info.run_id

    # Save structured summary JSON for report generation
    summary = {
        "references": refs,
        "candidates": [
            {
                "model_family": c["model_family"],
                "feature_set_name": c["feature_set_name"],
                "params": c["params"],
                "metrics": c["metrics"],
            }
            for c in candidates_sorted
        ],
        "calibration": [
            {
                "model_family": r["model_family"],
                "feature_set_name": r["feature_set_name"],
                "calibration": r["calibration"],
                "metrics": r["metrics"],
                "reliability_bins": r["rel_df"].to_dict(orient="records"),
            }
            for r in cal_results
        ],
        "chosen": {
            "model_family": chosen["model_family"],
            "feature_set_name": chosen["feature_set_name"],
            "calibration": chosen["calibration"],
            "params": chosen["params"],
            "cv_metrics": chosen["metrics"],
            "oof_regenerated": oof_regenerated,
            "brier_gain_vs_m2_baseline": brier_gain,
            "log_loss_gain_vs_m2_baseline": ll_gain,
        },
        "test": {
            "model_metrics": test_metrics,
            "benchmark_metrics": bench_test_metrics,
            "thresholds": thr_rows,
            "support": {
                "flagged_rows": int(flagged_mask.sum()),
                "total_test_rows": int(len(test_df)),
                "flagged_pct": float(flagged_mask.mean() * 100.0),
                "flagged_metrics": flagged_metrics,
                "unflagged_metrics": unflagged_metrics,
            },
            "shap_top10": [{"feature": f, "mean_abs_shap": v} for f, v in shap_top10],
            "mlflow_run_id": final_run_id,
            "model_path": str(model_path),
            "meta_path": str(meta_path),
        },
    }
    summary_path = REPORTS_M3_DIR / "m3_t01_results_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved model: {model_path}")
    print(f"Saved metadata: {meta_path}")
    print(f"Saved summary JSON: {summary_path}")


if __name__ == "__main__":
    main()

