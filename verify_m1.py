"""
verify_m1.py - checks the Milestone 1 outputs against the claims in the reports.

Run it from the folder that contains `processed/` (cleaned_dataset.csv,
model_ready_dataset.csv, feature_dictionary.csv, m1_t04_preprocessing_report.json):

    python verify_m1.py                # uses ./processed
    python verify_m1.py path/to/dir    # custom folder

It prints a short summary (a few dozen lines). Paste that output back; the CSVs
themselves never need to be uploaded or committed.
"""
import json
import sys
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

def _find_dir():
    """Use the argument if given, else the first place that contains cleaned_dataset.csv."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    for cand in (Path.cwd(), Path.cwd() / "processed", Path(__file__).resolve().parent,
                 Path(__file__).resolve().parent / "processed"):
        if (cand / "cleaned_dataset.csv").exists():
            return cand
    sys.exit("Could not find cleaned_dataset.csv. Run: python verify_m1.py <folder with the CSVs>")


P = _find_dir()
print(f"Using data folder: {P.resolve()}\n")
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ->  {detail}" if detail else ""))


cl = pd.read_csv(P / "cleaned_dataset.csv", parse_dates=["transaction_date", "expiration_date"])
mr = pd.read_csv(P / "model_ready_dataset.csv")
fd = pd.read_csv(P / "feature_dictionary.csv")
rep = json.load(open(P / "m1_t04_preprocessing_report.json"))

print(f"cleaned_dataset: {cl.shape} | model_ready_dataset: {mr.shape} | dictionary rows: {len(fd)}\n")

# ---------------------------------------------------------------- structure
print("== Structure ==")
check("cleaned has 100,000 rows, no NaN", len(cl) == 100_000 and cl.isna().sum().sum() == 0)
check("model_ready has 100,000 rows, no NaN", len(mr) == 100_000 and mr.isna().sum().sum() == 0)
check("same record_id order in both files", (cl["record_id"].values == mr["record_id"].values).all())
check("dictionary lists exactly the model_ready columns", list(fd["column"]) == list(mr.columns))
feat = fd.loc[fd["role"] == "feature", "column"].tolist()
check("178 model features (177 + expiry_remaining_ratio)", len(feat) == 178 == rep["n_model_features"], f"found {len(feat)}")

# --- flags added by fixes #1-#3
fdi = fd.set_index("column")
check("fix #1: spoilage_sensitivity is 'allowed' for Track 3A", fdi.loc["spoilage_sensitivity", "track_3a_use"] == "allowed",
      str(fdi.loc["spoilage_sensitivity", "track_3a_use"]))
check("fix #2: spoilage_risk note says benchmark only", "benchmark only" in str(fdi.loc["spoilage_risk", "transformation"]),
      str(fdi.loc["spoilage_risk", "transformation"]))
if "track_3b_use" in fd.columns:
    f3 = fdi["track_3b_use"]
    check("fix #3: selling_price and markdown_applied are 'never' for Track 3B",
          f3["selling_price"].startswith("never") and f3["markdown_applied"].startswith("never"))
    check("fix #3: discount_pct is the 3B treatment", f3["discount_pct"].startswith("treatment"))
    check("fix #3: profit, waste_pct, revenue, units_sold are 3B targets",
          all(f3[c] == "target" for c in ["profit", "waste_pct", "revenue", "units_sold"]))
else:
    check("fix #3: track_3b_use column exists in the dictionary", False, "column missing - re-run T04 after patch 3")

# --- fix #5: expiry_remaining_ratio
check("fix #5: expiry_remaining_ratio is a model feature", "expiry_remaining_ratio" in feat)
if "expiry_remaining_ratio" in cl.columns:
    _er = (cl["days_until_expiry"] / cl["shelf_life_days"]).clip(0, 1)
    check("fix #5: expiry_remaining_ratio == clip(days_until_expiry / shelf_life_days, 0, 1), within [0, 1]",
          np.allclose(cl["expiry_remaining_ratio"], _er, atol=1e-6) and cl["expiry_remaining_ratio"].between(0, 1).all())
else:
    check("fix #5: expiry_remaining_ratio exists in cleaned_dataset", False, "column missing - re-run T04 after patch 5/6")

# --- fix #6: redundant_with
if "redundant_with" in fd.columns:
    rw = fd.set_index("column")["redundant_with"].fillna("")
    structural = ["shelf_life_used_ratio", "quality_grade_ord", "spoilage_sensitivity", "selling_price", "markdown_applied"]
    check("fix #6: structural redundancies are documented", all(rw[c] != "" for c in structural))
    known_pairs = [("shelf_life_days", "days_remaining_at_purchase"), ("base_price", "cost_price"),
                   ("days_remaining_at_purchase", "days_until_expiry"), ("shelf_life_days", "days_until_expiry"),
                   ("base_price", "selling_price"), ("cost_price", "selling_price")]
    missing_pairs = [(a, b) for a, b in known_pairs if b not in rw[a] or a not in rw[b]]
    check("fix #6: the six |r| > 0.9 pairs found by verify_m1 are flagged both ways", not missing_pairs, str(missing_pairs))
    check("fix #6: JSON report lists the redundant pairs", len(rep.get("redundant_pairs_abs_r_gt_0_9", [])) >= 6,
          f"{len(rep.get('redundant_pairs_abs_r_gt_0_9', []))} pairs")
else:
    check("fix #6: redundant_with column exists in the dictionary", False, "column missing - re-run T04 after patch 5/6")

forbidden = set(rep["excluded_for_leakage"])
check("no target/outcome/spoilage_risk among the 178 features", not (forbidden & set(feat)),
      str(sorted(forbidden & set(feat))))

# ---------------------------------------------------------------- split
print("\n== Split (time-ordered since fix #4) ==")
vc = mr["split"].value_counts()
share = vc.get("train", 0) / len(mr)
check("train share is about 80% (time-ordered, whole days)", abs(share - 0.80) < 0.005, f"{vc.to_dict()} -> {share:.2%} train")
dr = cl.assign(split=mr["split"].values).groupby("split")["transaction_date"].agg(["min", "max"])
print("      transaction_date range by split:")
print(dr.to_string().replace("\n", "\n      "))
check("train is strictly earlier than test (no lookahead)", dr.loc["train", "max"] < dr.loc["test", "min"])
rate = mr.groupby("split")["was_spoiled"].mean().round(4).to_dict()
print("      spoilage rate by split:", rate)
check("spoilage rate of train and test within 1 point", abs(rate["train"] - rate["test"]) < 0.01)
check("report JSON records the time-ordered split", rep["split"].get("method", "").startswith("time-ordered"),
      f"last_train_day={rep['split'].get('last_train_day')}")
if rep["split"].get("last_train_day"):
    check("last_train_day in the JSON matches the data",
          str(dr.loc["train", "max"].date()) == rep["split"]["last_train_day"])

# ---------------------------------------------------------------- scaling
print("\n== Scaling (fit on train only) ==")
scaled = rep["log1p_scaled_columns"] + rep["scaled_columns"]
tr = mr[mr["split"] == "train"]
m = tr[scaled].mean().abs().max()
s = (tr[scaled].std(ddof=0) - 1).abs().max()
check("scaled columns: train mean ~0, std ~1", m < 1e-4 and s < 1e-3, f"max|mean|={m:.2e}, max|std-1|={s:.2e}")
oh = [c for c in feat if c.split("_")[0] in ("category", "region", "store", "supplier") or c.startswith("product_name_")]
te = mr[mr["split"] == "test"]
print(f"      test-period drift (info): max |test mean| of a scaled column = {te[scaled].mean().abs().max():.2f} "
      f"(columns are fitted on the earlier period, so a gap here is expected)")
for src in ["category", "region", "product_name", "store_id", "supplier_id"]:
    cols = [c for c in feat if c.startswith(src + "_")]
    check(f"one-hot {src}: exactly one 1 per row", (mr[cols].sum(axis=1) == 1).all(), f"{len(cols)} columns")

# ---------------------------------------------------------------- claims from the reports
print("\n== Claims made in the reports ==")
check("spoilage rate 19.44% overall", abs(cl["was_spoiled"].mean() - 0.1944) < 5e-4, f"{cl['was_spoiled'].mean():.4f}")
md = cl["markdown_applied"] == 1
check("30,651 markdown rows", md.sum() == 30_651, str(int(md.sum())))
check("markdown_applied == (discount_pct > 0)", (md == (cl["discount_pct"] > 0)).all())
dmd = cl.loc[md, "discount_pct"]
check("discounts in [0.10, 0.75]; none in (0, 0.10)", round(dmd.min(), 2) == 0.10 and round(dmd.max(), 2) == 0.75,
      f"min={dmd.min():.3f}, max={dmd.max():.3f}")
gap = (cl["expiration_date"] - cl["transaction_date"]).dt.days
check("days_remaining_at_purchase == date gap everywhere", (gap == cl["days_remaining_at_purchase"]).all())
check("days_until_expiry != date gap in 74,191 rows", int((gap != cl["days_until_expiry"]).sum()) == 74_191)
check("days_since_receipt in [0, shelf_life]", cl["days_since_receipt"].between(0, cl["shelf_life_days"]).all(),
      f"max={cl['days_since_receipt'].max()}")
sp = (cl["base_price"] * (1 - cl["discount_pct"]) - cl["selling_price"]).abs()
print(f"      selling_price vs base_price*(1-discount_pct): max abs diff = {sp.max():.4f}")

# ---------------------------------------------------------------- things I want to settle
print("\n== Open questions for Milestone 2 / 3 ==")

# (1) leakage scan on the final feature matrix
cor = mr[feat].corrwith(mr["was_spoiled"]).abs().sort_values(ascending=False)
print("      top |corr| of any model feature with was_spoiled:")
print("      " + cor.head(5).round(3).to_string().replace("\n", "\n      "))
check("no feature has |corr| > 0.3 with was_spoiled (leak smell test)", cor.iloc[0] < 0.3)

# (1b) fixes #5/#6 diagnostics: is the proposal's remaining-shelf-life ratio already in the matrix,
#      and which model features duplicate each other?
if "shelf_life_used_ratio" in cl.columns:
    rem_a = cl["days_remaining_at_purchase"] / cl["shelf_life_days"]              # remaining ratio via the trustworthy date gap
    rem_b = (cl["days_until_expiry"] / cl["shelf_life_days"]).clip(0, 1)          # proposal / T03 / T06 definition
    check("shelf_life_used_ratio == 1 - days_remaining_at_purchase / shelf_life_days (exact)",
          np.allclose(cl["shelf_life_used_ratio"], 1 - rem_a, atol=1e-6))
    dd = (rem_a - rem_b).abs()
    print(f"      remaining ratio, date-gap version vs days_until_expiry version: corr {rem_a.corr(rem_b):.4f}, "
          f"median abs diff {dd.median():.3f}, max {dd.max():.3f}, rows differing by more than 0.02: {(dd > 0.02).mean():.1%}")
    q = pd.qcut(cl["shelf_life_days"], 4, duplicates="drop")
    print("      mean abs difference by shelf_life_days quartile (offset hurts short-life items most):")
    print("      " + dd.groupby(q, observed=True).mean().round(3).to_string().replace("\n", "\n      "))
num = [c for c in scaled if c in mr.columns]
cm = mr[num].corr().abs()
pairs = [(a, b, cm.loc[a, b]) for i, a in enumerate(num) for b in num[i + 1:] if cm.loc[a, b] > 0.9]
print(f"      highly correlated scaled feature pairs (|r| > 0.9): {len(pairs)}")
for a, b, r in sorted(pairs, key=lambda t: -t[2]):
    print(f"        {a} ~ {b}: {r:.3f}")

# (2) is the sales-velocity feature feasible? (trailing 7 days, product x store, strictly earlier days)
d = cl[["product_name", "store_id", "transaction_date"]].copy()
d["day"] = (d["transaction_date"] - d["transaction_date"].min()).dt.days
d = d.sort_values(["product_name", "store_id", "day"])


def prior_7(g):
    days = g["day"].to_numpy()
    lo = np.searchsorted(days, days - 7, side="left")
    hi = np.searchsorted(days, days, side="left")
    return pd.Series(hi - lo, index=g.index)


d["prior7"] = d.groupby(["product_name", "store_id"], group_keys=False).apply(prior_7)
pair_n = d.groupby(["product_name", "store_id"]).size()
print(f"      product_name x store_id pairs: {len(pair_n):,} | rows per pair: median {pair_n.median():.0f}, "
      f"min {pair_n.min()}, max {pair_n.max()}")
print(f"      share of rows with >=1 earlier sale of the same product in the same store within 7 days: "
      f"{(d['prior7'] >= 1).mean():.1%}")
c = cl[["category", "store_id", "transaction_date"]].copy()
c["day"] = (c["transaction_date"] - c["transaction_date"].min()).dt.days
c = c.sort_values(["category", "store_id", "day"])
c["prior7"] = c.groupby(["category", "store_id"], group_keys=False).apply(prior_7)
print(f"      same check for the category x store fallback: {(c['prior7'] >= 1).mean():.1%}")

# (3) how well does a calibrated probability do? (sets expectations for Track 3A)
print("      observed spoilage rate by spoilage_risk decile (calibration of the reserved benchmark):")
q = pd.qcut(cl["spoilage_risk"], 10, duplicates="drop")
print("      " + cl.groupby(q, observed=True)["was_spoiled"].mean().round(3).to_string().replace("\n", "\n      "))

print(f"\nSUMMARY: {sum(results)}/{len(results)} checks passed")
