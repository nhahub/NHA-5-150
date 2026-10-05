"""
check_presale_inputs.py - fix #8 plus the discount numbers needed for the EDA Report corrections.

    python check_presale_inputs.py                 # finds the folder with cleaned_dataset.csv
    python check_presale_inputs.py <folder>

Part 1  Discount facts for the report (grid for the optimizer).
Part 2  Is each candidate input really known BEFORE the sale? Three tests per column:
          a) how much of it is a product / store level constant (pre-sale attributes usually are),
          b) how well the OUTCOME columns alone predict it (time-ordered split, gradient boosting),
          c) how much predictive power the outcomes ADD on top of the other pre-sale inputs.
        A pre-sale input should gain (almost) nothing from outcomes. A big gain = possible post-sale information.
Part 3  Basic identities between outcome columns and initial_quantity.

Output is about 40 lines. Paste it back; no data leaves your machine.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")


def _find_dir():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    for cand in (Path.cwd(), Path.cwd() / "processed", Path(__file__).resolve().parent,
                 Path(__file__).resolve().parent / "processed"):
        if (cand / "cleaned_dataset.csv").exists():
            return cand
    sys.exit("Could not find cleaned_dataset.csv. Run: python check_presale_inputs.py <folder>")


P = _find_dir()
cl = pd.read_csv(P / "cleaned_dataset.csv")
print(f"Using data folder: {P.resolve()} | rows: {len(cl):,}\n")

# ------------------------------------------------------------------ Part 1: discounts
print("== Part 1: discount facts ==")
md = cl.loc[cl["markdown_applied"] == 1, "discount_pct"]
print(f"markdowns: {len(md):,} ({len(md) / len(cl):.1%} of rows) | min {md.min():.2f}, median {md.median():.2f}, max {md.max():.2f}")
print(f"share of markdowns deeper than 50%: {(md > 0.50).mean():.1%} ({int((md > 0.50).sum()):,} rows)")
print(f"rows with discount exactly 0.75: {int((md.round(2) == 0.75).sum()):,} | in (0.70, 0.75): {int(((md > 0.70) & (md < 0.75)).sum()):,}")
print(f"distinct discount values: {md.round(2).nunique()}")
cat_rate = cl.groupby("category")["was_spoiled"].mean()
print(f"spoilage rate by category: {cat_rate.min():.1%} ({cat_rate.idxmin()}) to {cat_rate.max():.1%} ({cat_rate.idxmax()})")

# ------------------------------------------------------------------ Part 2: pre-sale test
print("\n== Part 2: is each candidate known before the sale? ==")
OUT = ["units_sold", "units_wasted", "waste_pct", "revenue", "waste_cost", "profit", "profit_margin_pct"]
EXCL = set(OUT) | {"was_spoiled", "spoilage_risk", "discount_pct", "selling_price", "markdown_applied",
                   "record_id", "product_id"}
CANDS = [c for c in ["daily_demand", "demand_variability", "initial_quantity", "temp_abuse_events",
                     "distribution_hours"] if c in cl.columns]

cl["_category"] = cl["category"].astype("category").cat.codes
tr, te = cl["split"] == "train", cl["split"] == "test"
numeric = [c for c in cl.select_dtypes("number").columns if c not in EXCL]


def share_explained(s, by):
    m = s.groupby(by).transform("mean")
    return 1 - ((s - m) ** 2).sum() / ((s - s.mean()) ** 2).sum()


def r2(target, cols):
    m = HistGradientBoostingRegressor(max_iter=100, random_state=0)
    m.fit(cl.loc[tr, cols], cl.loc[tr, target])
    return r2_score(cl.loc[te, target], m.predict(cl.loc[te, cols]))


rows = []
for cand in CANDS:
    pre = [c for c in numeric if c != cand and not c.startswith("_")] + ["_category"]
    s = cl[cand].astype(float)
    r_pre, r_out, r_both = r2(cand, pre), r2(cand, OUT), r2(cand, pre + OUT)
    rows.append({
        "column": cand,
        "by_product": round(share_explained(s, cl["product_name"]), 3),
        "by_product_store": round(share_explained(s, [cl["product_name"], cl["store_id"]]), 3),
        "R2_pre_sale_inputs": round(r_pre, 3),
        "R2_outcomes_only": round(r_out, 3),
        "R2_both": round(r_both, 3),
        "gain_from_outcomes": round(r_both - r_pre, 3),
        "flag": "CHECK" if (r_both - r_pre) > 0.05 else "ok",
    })
res = pd.DataFrame(rows).set_index("column")
print(res.to_string())
print("\nReading: 'gain_from_outcomes' > 0.05 means the sale results explain this column beyond what the other "
      "pre-sale inputs do, which would suggest it is partly built from, or recorded after, the sale.\n"
      "Expected exception: initial_quantity flags by ACCOUNTING IDENTITY (units_sold + units_wasted = initial_quantity), "
      "which is not leakage - it is the batch size, known before the sale.")

# ------------------------------------------------------------------ Part 3: identities
print("\n== Part 3: identities between outcomes and initial_quantity ==")
if {"units_sold", "units_wasted", "initial_quantity"} <= set(cl.columns):
    d = cl["initial_quantity"] - cl["units_sold"] - cl["units_wasted"]
    print(f"initial_quantity - units_sold - units_wasted == 0 in {(d == 0).mean():.1%} of rows "
          f"(min {d.min()}, max {d.max()})")
    print(f"units_sold <= initial_quantity in {(cl['units_sold'] <= cl['initial_quantity']).mean():.1%} of rows")
if "daily_demand" in cl.columns:
    print(f"corr(daily_demand, units_sold) = {cl['daily_demand'].corr(cl['units_sold']):.3f} | "
          f"corr(daily_demand, initial_quantity) = {cl['daily_demand'].corr(cl['initial_quantity']):.3f}")
