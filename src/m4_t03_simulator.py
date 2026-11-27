"""
src/m4_t03_simulator.py - M4-T03 Dashboard: Simulator Mode.

Lets a manager type a hypothetical batch and see, for that batch:
  - the spoilage risk (Track 3A),
  - the full what-if scoring across every candidate discount depth (Track 3B),
  - the optimizer's recommended depth + tier (Track 3C),
  - the confidence level and the reasons behind it (M3-T12 + M3-T14 P3 rules).

This module adds NO new modelling. Every number comes from the frozen M3-T14
pipeline (`src.m3_t14_pipeline.simulate`, mode='hypothetical'). Fields the user
leaves on "default" are filled by that pipeline with the training-only category
medians/modes documented in models/track_3_inference_defaults.json (M3-T14 P2).

Layout:
  * Pure helpers (no Streamlit): load_options, build_batch, run_scenario,
    curve_for_display, parse_reason_codes, expiry_ratio, ...  -> unit-testable.
  * render_simulator(): the Streamlit tab, called from app.py.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import CLEANED_DATASET_PATH, INFERENCE_DEFAULTS_PATH

# --------------------------------------------------------------------------- #
# Constants (mirrors of src.m3_t14_pipeline; scripts/M4_T03_verify_simulator.py
# asserts they have not drifted from the pipeline).
# --------------------------------------------------------------------------- #
DEFAULT_CHOICE = "Default (category typical)"

# Optional numeric inputs: field -> (label, help text)
NUMERIC_OPTIONAL: Dict[str, Tuple[str, str]] = {
    "shelf_life_days": ("Shelf life (days)", "Total shelf life of the product."),
    "days_remaining_at_purchase": ("Days remaining at purchase", "Shelf life left when the store received it."),
    "days_since_receipt": ("Days since receipt", "Days the batch has already been in the store. Key input for confidence."),
    "storage_temp": ("Storage temperature", "Average storage temperature."),
    "temp_deviation": ("Temperature deviation", "How far storage drifted from the ideal temperature. Key input for confidence."),
    "temp_abuse_events": ("Temperature abuse events", "Number of recorded temperature excursions."),
    "distribution_hours": ("Distribution hours", "Hours spent in distribution before arrival."),
    "handling_score": ("Handling score", "Handling quality score. Key input for confidence."),
    "packaging_score": ("Packaging score", "Packaging quality score. Key input for confidence."),
    "supplier_score": ("Supplier score", "Supplier reliability score."),
    "spoilage_sensitivity": ("Spoilage sensitivity", "How quickly this product spoils (0-1 scale)."),
    "daily_demand": ("Daily demand (units)", "Typical units sold per day."),
    "demand_variability": ("Demand variability", "How much daily demand fluctuates."),
}
INTEGER_FIELDS = {"temp_abuse_events"}

# Optional categorical inputs that are NOT in the main form (set from "More inputs").
EXTRA_CATEGORICAL: Dict[str, str] = {
    "supplier_id": "Supplier",
    "transaction_date": "Receipt / transaction date",
}

KEY_SENSITIVE_FIELDS: Tuple[str, ...] = (
    "packaging_score",
    "handling_score",
    "days_until_expiry",
    "days_since_receipt",
    "temp_deviation",
)
N_OPTIONAL_TRACKED = 19  # 13 numeric + 6 categorical (M3-T14 P1)

GRADE_CHOICES = ["A", "B", "C"]

# Same colour families as app.py so the Simulator looks like the Manager View.
RISK_STYLE = {
    "Low": "background-color:#fff3e0;color:#111",
    "Medium": "background-color:#ffcc80;color:#111",
    "High": "background-color:#ff8a65;color:#111",
    "Critical": "background-color:#d84315;color:#fff;font-weight:bold",
}
CONF_ICON = {"Moderate": "🟡", "High": "✅", "Medium": "🟡", "Low": "⚠️", "No recommendation": "➖"}
CONF_STYLE = {
    "Moderate": "background-color:#fff59d;color:#111",
    "Low": "background-color:#cfd8dc;color:#111;font-style:italic;font-weight:bold",
    "No recommendation": "background-color:#eceff1;color:#111",
}
REC_ROW_STYLE = "background-color:#bbdefb;color:#111;font-weight:bold"
UNSUPPORTED_ROW_STYLE = "color:#9e9e9e;font-style:italic"


# --------------------------------------------------------------------------- #
# Lazy access to the frozen pipeline (heavy imports: lightgbm, sklearn, ...)
# --------------------------------------------------------------------------- #
def _pipeline():
    from src import m3_t14_pipeline as pipe  # noqa: WPS433 (deliberately lazy)

    return pipe


# --------------------------------------------------------------------------- #
# Options for the form
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_options() -> Dict[str, Any]:
    """Valid choices + category defaults for the form, read from the M3-T14 defaults JSON."""
    defaults = json.loads(INFERENCE_DEFAULTS_PATH.read_text(encoding="utf-8"))
    products_by_cat: Dict[str, List[str]] = {}
    try:
        meta = pd.read_csv(CLEANED_DATASET_PATH, usecols=["category", "product_name"]).drop_duplicates()
        for cat, sub in meta.groupby("category"):
            products_by_cat[str(cat)] = sorted(sub["product_name"].astype(str).unique())
    except Exception:  # cleaned dataset unavailable: offer every product for every category
        products_by_cat = {}
    all_products = sorted(defaults["valid_products"])
    categories = sorted(defaults["valid_categories"])
    return {
        "categories": categories,
        "stores": sorted(defaults["valid_stores"]),
        "suppliers": sorted(defaults["valid_suppliers"]),
        "products_by_category": {c: products_by_cat.get(c, all_products) for c in categories},
        "by_category": defaults["by_category"],
        "store_region_map": defaults["store_region_map"],
        "train_end_date": defaults["train_end_date"],
    }


def category_defaults(category: str) -> Dict[str, Any]:
    """Numeric medians and categorical modes the pipeline will use for this category."""
    return load_options()["by_category"][category]


# --------------------------------------------------------------------------- #
# Building the batch + running the pipeline
# --------------------------------------------------------------------------- #
def _is_unset(value: Any) -> bool:
    return value is None or (isinstance(value, str) and (value == "" or value == DEFAULT_CHOICE))


def build_batch(required: Dict[str, Any], optional: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the raw batch dict for `simulate()`.
    Optional fields that are unset ("Default" / None) are OMITTED so the pipeline
    fills them with training-only category defaults and counts them as defaulted
    (which is what drives the confidence rules).
    """
    batch: Dict[str, Any] = {
        "category": str(required["category"]),
        "days_until_expiry": float(required["days_until_expiry"]),
        "initial_quantity": float(required["initial_quantity"]),
        "base_price": float(required["base_price"]),
        "cost_price": float(required["cost_price"]),
    }
    for key, value in optional.items():
        if _is_unset(value):
            continue
        if key in NUMERIC_OPTIONAL:
            batch[key] = float(value)
        elif key == "transaction_date":
            batch[key] = pd.Timestamp(value).strftime("%Y-%m-%d")
        else:
            batch[key] = str(value)
    return batch


def run_scenario(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Runs the frozen M3-T14 simulator in hypothetical mode (no re-fitting, no extra logic)."""
    return _pipeline().simulate(batch)


def expiry_ratio(batch: Dict[str, Any]) -> float:
    """Same formula as the pipeline: days_until_expiry / shelf_life_days, clipped to [0, 1]."""
    shelf = batch.get("shelf_life_days")
    if shelf is None:
        shelf = category_defaults(batch["category"])["numeric_medians"]["shelf_life_days"]
    return float(np.clip(float(batch["days_until_expiry"]) / max(float(shelf), 1.0), 0.0, 1.0))


def effective_grade(batch: Dict[str, Any]) -> str:
    return batch.get("quality_grade") or category_defaults(batch["category"])["categorical_modes"]["quality_grade"]


def defaulted_key_fields(batch: Dict[str, Any]) -> List[str]:
    """Key sensitive fields the user has NOT set (each one forces Low confidence on R1 batches)."""
    return [f for f in KEY_SENSITIVE_FIELDS if f != "days_until_expiry" and f not in batch]


def not_set_fields(batch: Dict[str, Any]) -> List[str]:
    tracked = list(NUMERIC_OPTIONAL) + ["product_name", "store_id", "region", "supplier_id", "quality_grade", "transaction_date"]
    return [f for f in tracked if f not in batch]


# --------------------------------------------------------------------------- #
# Interpreting the result
# --------------------------------------------------------------------------- #
def parse_reason_codes(raw: Any) -> List[str]:
    """
    Reason codes arrive either as '["a", "b"]|c|d' (JSON list + extra codes) or 'a|b'.
    A regex keeps this robust to both shapes.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    seen: List[str] = []
    for code in re.findall(r"[a-z][a-z0-9_]*", str(raw)):
        if code not in seen:
            seen.append(code)
    return seen


def reason_texts(codes: List[str]) -> List[str]:
    pipe = _pipeline()
    from src.m3_t12_confidence import CONFIDENCE_WORDING

    known = dict(CONFIDENCE_WORDING["reasons"])
    known.update(pipe.EXTRA_CONFIDENCE_REASONS)
    return [known.get(c, c.replace("_", " ").capitalize()) for c in codes]


def pretty_reason(reason: Any) -> str:
    text = "" if reason is None else str(reason)
    if text in ("", "ok", "nan"):
        return ""
    return text.replace("_", " ").capitalize()


def curve_for_display(curve: pd.DataFrame, base_price: float, recommended_depth: Optional[float]) -> pd.DataFrame:
    """Turns the 14-depth what-if curve into the table shown to the manager."""
    depth = curve["depth"].to_numpy(dtype=float)
    is_rec = np.zeros(len(curve), dtype=bool)
    if recommended_depth is not None and not pd.isna(recommended_depth):
        is_rec = np.isclose(depth, float(recommended_depth), atol=1e-6)
    return pd.DataFrame(
        {
            "Discount (%)": np.round(depth * 100.0, 0).astype(int),
            "Selling price": np.round(base_price * (1.0 - depth), 2),
            "Data support": np.where(curve["supported"].to_numpy(dtype=bool), "Supported", "Not supported"),
            "Expected sell-through (%)": np.round(curve["s_hat"].to_numpy(dtype=float) * 100.0, 1),
            "Expected units sold": np.round(curve["expected_units_sold"].to_numpy(dtype=float), 1),
            "Profit per unit": np.round(curve["profit_per_unit"].to_numpy(dtype=float), 2),
            "Expected batch profit": np.round(curve["batch_profit"].to_numpy(dtype=float), 2),
            "Optimizer pick": np.where(is_rec, "★ Recommended", ""),
            "Note": [pretty_reason(r) for r in curve["reason"]],
        }
    )


def style_curve(display: pd.DataFrame):
    """Highlights the recommended row and greys out depths the data does not support."""

    def _row(row: pd.Series) -> List[str]:
        if row["Optimizer pick"]:
            css = REC_ROW_STYLE
        elif row["Data support"] != "Supported":
            css = UNSUPPORTED_ROW_STYLE
        else:
            css = ""
        return [css] * len(row)

    return display.style.apply(_row, axis=1)


def depth_options(curve: pd.DataFrame) -> List[int]:
    """Supported depths (in %) a manager can pick to compare against the recommendation."""
    ok = curve[curve["supported"].astype(bool)]
    return [int(round(d * 100)) for d in ok["depth"]]


def profit_at_depth(curve: pd.DataFrame, depth_pct: int) -> Optional[float]:
    row = curve[np.isclose(curve["depth"].to_numpy(dtype=float), depth_pct / 100.0, atol=1e-6)]
    if row.empty or pd.isna(row.iloc[0]["batch_profit"]):
        return None
    return float(row.iloc[0]["batch_profit"])


def risk_band(spoil_proba: float) -> str:
    return _pipeline()._risk_band(float(spoil_proba))


# --------------------------------------------------------------------------- #
# Streamlit tab
# --------------------------------------------------------------------------- #
def _file_hint(exc: Exception) -> str:
    return (
        f"A file the pipeline needs is missing: `{getattr(exc, 'filename', None) or exc}`. "
        "The Simulator needs `models/*.joblib` and the CSVs in `data/processed/` "
        "(cleaned_dataset, model_ready_m2_dataset, model_ready_3b_dataset). "
        "These are git-ignored, so get them from the team lead (see README)."
    )


def render_simulator() -> None:
    import streamlit as st

    st.subheader("🧪 Simulator Mode")
    st.caption(
        "Enter a hypothetical batch to see its spoilage risk, the what-if result for every candidate "
        "discount, the optimizer's recommendation and how much to trust it. "
        "Anything you leave on **Default** is filled with the training-set typical value for the "
        "category (M3-T14 defaults), and the more defaults are used, the lower the confidence."
    )

    try:
        opts = load_options()
    except FileNotFoundError as exc:
        st.error(_file_hint(exc))
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load simulator options: {exc}")
        return

    # ---- Required inputs ---------------------------------------------------
    st.markdown("##### 1. Batch details")
    c1, c2, c3 = st.columns(3)
    category = c1.selectbox("Category *", opts["categories"], key="sim_category")
    cat_def = opts["by_category"][category]
    modes = cat_def["categorical_modes"]
    meds = cat_def["numeric_medians"]

    product = c2.selectbox(
        "Product",
        [DEFAULT_CHOICE] + opts["products_by_category"][category],
        key=f"sim_product_{category}",
        help=f"Default: {modes['product_name']}. Unknown products would force Low confidence.",
    )
    store = c3.selectbox(
        "Store",
        [DEFAULT_CHOICE] + opts["stores"],
        key="sim_store",
        help=f"Default: {modes['store_id']}. The store sets the region and its sales history.",
    )

    c4, c5, c6, c7, c8 = st.columns(5)
    days_left = c4.number_input("Days until expiry *", min_value=0, max_value=365, value=3, step=1, key="sim_days")
    quantity = c5.number_input("Units in batch *", min_value=1, max_value=100000, value=50, step=1, key="sim_qty")
    base_price = c6.number_input("Base price *", min_value=0.01, value=5.00, step=0.25, format="%.2f", key="sim_base")
    cost_price = c7.number_input("Cost price *", min_value=0.0, value=3.00, step=0.25, format="%.2f", key="sim_cost")
    grade = c8.selectbox(
        "Quality grade",
        [DEFAULT_CHOICE] + GRADE_CHOICES,
        key="sim_grade",
        help=f"Default: {modes['quality_grade']}. Grade C batches are always eligible for a discount.",
    )

    # ---- Optional inputs ---------------------------------------------------
    optional: Dict[str, Any] = {"product_name": product, "store_id": store, "quality_grade": grade}

    with st.expander("2. More inputs (optional) - every field you set yourself raises confidence", expanded=False):
        labels = {k: v[0] for k, v in NUMERIC_OPTIONAL.items()}
        labels.update(EXTRA_CATEGORICAL)
        chosen = st.multiselect(
            "Which extra inputs do you want to set yourself?",
            list(labels),
            format_func=lambda k: labels[k] + ("  (key input)" if k in KEY_SENSITIVE_FIELDS else ""),
            key="sim_extra_fields",
        )
        for field in chosen:
            widget_key = f"sim_x_{field}_{category}"
            if field in NUMERIC_OPTIONAL:
                label, help_text = NUMERIC_OPTIONAL[field]
                default_val = float(meds[field])
                if field in INTEGER_FIELDS:
                    optional[field] = st.number_input(
                        label, min_value=0, value=int(round(default_val)), step=1, help=help_text, key=widget_key
                    )
                else:
                    optional[field] = st.number_input(
                        label, min_value=0.0, value=default_val, step=0.1, help=help_text, key=widget_key
                    )
            elif field == "supplier_id":
                optional[field] = st.selectbox("Supplier", opts["suppliers"], key=widget_key)
            elif field == "transaction_date":
                optional[field] = st.date_input(
                    "Receipt / transaction date", value=pd.Timestamp(opts["train_end_date"]).date(), key=widget_key
                )
        if chosen:
            st.caption("Starting values shown are the category defaults; edit them to your scenario.")

    required = {
        "category": category,
        "days_until_expiry": days_left,
        "initial_quantity": quantity,
        "base_price": base_price,
        "cost_price": cost_price,
    }
    try:
        batch = build_batch(required, optional)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Invalid input: {exc}")
        return

    if st.button("▶ Run simulation", type="primary", key="sim_run"):
        try:
            with st.spinner("Scoring the batch with the frozen Track 3 pipeline..."):
                result = run_scenario(batch)
            st.session_state["sim_last"] = {"batch": batch, "result": result}
        except ValueError as exc:
            st.session_state.pop("sim_last", None)
            st.error(f"Input not accepted: {exc}")
            return
        except FileNotFoundError as exc:
            st.session_state.pop("sim_last", None)
            st.error(_file_hint(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.session_state.pop("sim_last", None)
            st.error(f"Simulation failed: {exc}")
            return

    last = st.session_state.get("sim_last")
    if not last:
        st.info("Fill in the batch above and press **Run simulation**.")
        return
    if last["batch"] != batch:
        st.warning("Inputs changed since the last run. Press **Run simulation** to refresh the results below.")

    _render_result(st, last["batch"], last["result"])


def _render_result(st, batch: Dict[str, Any], result: Dict[str, Any]) -> None:
    rec = result["recommendation"]
    curve = result["whatif_curve"]
    region = str(rec["region"])
    level = str(rec["confidence_level"])
    spoil_pct = float(rec["spoil_chance_pct"])
    band = risk_band(float(rec["spoil_proba"]))
    depth = rec.get("recommended_depth")
    has_rec = region == "R1" and depth is not None and not pd.isna(depth)

    st.markdown("##### 3. Result")
    st.info(result["global_caveat"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Spoilage chance", f"{spoil_pct:.1f}%", band, delta_color="off")
    if has_rec:
        m2.metric("Recommended discount", f"{int(round(float(depth) * 100))}%", f"{rec['tier']} tier", delta_color="off")
        m3.metric("Expected batch profit (simulated)", f"{float(rec['predicted_profit_at_rec']):,.2f}")
    else:
        m2.metric("Recommended discount", "None", str(rec.get("tier_label", "")), delta_color="off")
        m3.metric("Expected batch profit (simulated)", "n/a")
    m4.metric("Confidence", f"{CONF_ICON.get(level, '•')} {level}")

    # ---- Why no recommendation? -------------------------------------------
    if region == "R0":
        ratio = expiry_ratio(batch)
        st.warning(
            "**No discount recommended.** The historical data only supports discounting batches that are "
            "close to expiry (40% or less of their shelf life left) or grade C. "
            f"This batch has about **{ratio:.0%}** of its shelf life left and grade **{effective_grade(batch)}**. "
            "Try fewer days until expiry or grade C to see a recommendation."
        )
    elif not has_rec:
        st.warning(
            "**No discount recommended.** None of the candidate discounts has enough historical support for this batch."
        )

    # ---- What-if table + chart --------------------------------------------
    if region == "R1":
        st.markdown("**What-if scoring for every candidate discount**")
        display = curve_for_display(curve, float(batch["base_price"]), depth if has_rec else None)
        st.dataframe(
            style_curve(display),
            hide_index=True,
            height=(len(display) + 1) * 35 + 3,
            column_config={
                "Discount (%)": st.column_config.NumberColumn("Discount (%)", format="%d%%"),
                "Expected sell-through (%)": st.column_config.NumberColumn("Expected sell-through (%)", format="%.1f%%"),
            },
        )
        st.caption(
            "★ marks the optimizer's pick (highest expected batch profit among the supported depths). "
            "Greyed rows are depths the historical data does not support; the optimizer ignores them."
        )

        supported = display[display["Data support"] == "Supported"]
        if len(supported) >= 2:
            st.markdown("**Expected batch profit by discount (supported depths)**")
            st.line_chart(supported.set_index("Discount (%)")["Expected batch profit"])

        # ---- Compare the manager's own discount ---------------------------
        options = depth_options(curve)
        if has_rec and options:
            st.markdown("**Compare your own discount**")
            rec_pct = int(round(float(depth) * 100))
            pick = st.selectbox(
                "Discount to compare (%)",
                options,
                index=options.index(rec_pct) if rec_pct in options else 0,
                key="sim_compare_depth",
            )
            p_pick = profit_at_depth(curve, pick)
            p_rec = float(rec["predicted_profit_at_rec"])
            if p_pick is not None:
                st.metric(
                    f"Expected batch profit at {pick}%",
                    f"{p_pick:,.2f}",
                    f"{p_pick - p_rec:+,.2f} vs recommended ({rec_pct}%)",
                    delta_color="off",
                )
        if bool(rec.get("flat_curve", False)):
            st.caption("Note: the profit curve is nearly flat here, so neighbouring discounts perform almost the same.")

    # ---- Confidence + drivers ---------------------------------------------
    st.markdown("##### 4. Why this confidence level?")
    codes = parse_reason_codes(rec.get("confidence_reasons"))
    if codes:
        for text in reason_texts(codes):
            st.markdown(f"- {text}")
    elif level == "Moderate":
        st.markdown("- No data-support limits apply: the batch sits inside well-supported ranges.")

    n_def = int(rec.get("n_optional_defaulted", 0))
    st.markdown(
        f"- **{n_def} of {N_OPTIONAL_TRACKED}** optional inputs were filled with category defaults "
        f"({float(rec.get('defaulted_share', 0)):.0%})."
    )
    missing_key = defaulted_key_fields(batch)
    if missing_key and region == "R1":
        pretty = ", ".join(NUMERIC_OPTIONAL[f][0] for f in missing_key)
        st.markdown(
            f"- Key inputs still on default: **{pretty}**. While any key input is on default, "
            "confidence is capped at Low (M3-T14 rule). Set them under *More inputs* to lift the cap."
        )
    with st.expander("Inputs left on default"):
        st.write(", ".join(not_set_fields(batch)) or "None - every optional input was set by you.")

    st.markdown("**Top spoilage drivers:** " + str(rec.get("top_drivers_text", "")))
    if bool(rec.get("limited_history_3a", False)):
        st.warning("Limited history for this product/store: treat the estimate with extra care.")
    st.caption(
        "All profit figures are simulated estimates; no markdown is guaranteed to improve observed profit."
    )
