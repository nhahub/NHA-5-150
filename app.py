"""
app.py - Smart Shelf Track 3 Manager Dashboard (Streamlit).

Built on the M3-T14 inventory snapshot and frozen inference pipeline
(src/m3_t14_pipeline.py). Shows, per batch:
  1. Current inventory (product, category, store, days left, stock)
  2. Spoilage risk flags (spoil chance % + risk band)
  3. Optimizer-recommended discount tier (+ discount %)
  4. Confidence flags, visibly distinguished (icon + colour + reasons)

Run from the repository root:
    streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make `src` importable when launched from anywhere.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import INVENTORY_SNAPSHOT_PATH  # noqa: E402
from src.m3_t14_pipeline import (  # noqa: E402
    EXTRA_CONFIDENCE_REASONS,
    export_inventory_snapshot,
    manager_view_table,
)

# --------------------------------------------------------------------------- #
# Visual encoding: three separate colour families so the three flags never
# get confused with each other.
#   Spoilage risk  -> red/orange scale
#   Discount tier  -> blue scale
#   Confidence     -> green / amber / grey-with-warning (icon + colour)
# --------------------------------------------------------------------------- #
RISK_STYLE = {
    "Low": "background-color:#fff3e0;color:#111",
    "Medium": "background-color:#ffcc80;color:#111",
    "High": "background-color:#ff8a65;color:#111",
    "Critical": "background-color:#d84315;color:#fff;font-weight:bold",
}
TIER_STYLE = {
    "Low": "background-color:#e3f2fd;color:#111",
    "Medium": "background-color:#90caf9;color:#111",
    "High": "background-color:#42a5f5;color:#111",
    "Critical": "background-color:#1565c0;color:#fff;font-weight:bold",
}
CONF_ICON = {"High": "✅", "Medium": "🟡", "Low": "⚠️"}
CONF_STYLE = {
    "High": "background-color:#c8e6c9;color:#111",
    "Medium": "background-color:#fff59d;color:#111",
    "Low": "background-color:#cfd8dc;color:#111;font-style:italic;font-weight:bold",
}
NEUTRAL_STYLE = "background-color:#eceff1;color:#111"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
NO_TIER_LABEL = "No discount evidence (R0)"


def normalize_table(table: pd.DataFrame) -> pd.DataFrame:
    """Fills missing text values so filters, sorting and styling never see NaN."""
    table = table.copy()
    table["recommended_tier"] = table["recommended_tier"].astype(object).where(
        table["recommended_tier"].notna(), NO_TIER_LABEL
    ).astype(str)
    table["confidence_level"] = table["confidence_level"].astype(object).where(
        table["confidence_level"].notna(), "n/a"
    ).astype(str)
    for col in ("confidence_reasons", "top_factors"):
        table[col] = table[col].astype(object).where(table[col].notna(), "").astype(str)
    return table


@st.cache_data(show_spinner="Running the Track 3 inference pipeline...")
def load_dashboard_data():
    """Loads the inventory snapshot (building it if missing) and runs the pipeline."""
    snap_path = Path(INVENTORY_SNAPSHOT_PATH)
    if snap_path.exists():
        snapshot = pd.read_csv(snap_path)
    else:
        snapshot, _ = export_inventory_snapshot()

    result = manager_view_table(snapshot, mode="real")
    snapshot_date = (
        str(snapshot["transaction_date"].iloc[0])
        if "transaction_date" in snapshot.columns and len(snapshot)
        else "n/a"
    )
    return normalize_table(result["table"]), result["summary_strip"], snapshot_date


def build_display_table(df: pd.DataFrame) -> pd.DataFrame:
    """Selects and renames the columns shown to the manager."""
    out = pd.DataFrame(
        {
            "Batch": df["record_id"],
            "Product": df["product_name"],
            "Category": df["category"],
            "Store": df["store_id"],
            "Days left": df["days_left"],
            "Stock (units)": df["stock_units"],
            "Spoil chance (%)": df["spoil_chance_pct"],
            "Risk band": df["risk_band"],
            "Recommended tier": df["recommended_tier"],
            "Recommended discount (%)": df["recommended_discount_pct"],
            "Confidence": df["confidence_level"].map(
                lambda c: f"{CONF_ICON.get(c, '•')} {c}"
            ),
            "Confidence reasons": df["confidence_reasons"].fillna("").str.replace(
                "|", ", ", regex=False
            ),
            "Top factors": df["top_factors"],
        }
    )
    return out


def style_table(df: pd.DataFrame):
    """Applies colour coding to the risk, tier and confidence columns."""

    def _lookup(mapping):
        return lambda v: mapping.get(str(v).split(" ")[-1], NEUTRAL_STYLE)

    styler = df.style
    apply_cell = getattr(styler, "map", None) or styler.applymap  # pandas >= 2.1 / older
    styler = apply_cell(_lookup(RISK_STYLE), subset=["Risk band"])
    apply_cell = getattr(styler, "map", None) or styler.applymap
    styler = apply_cell(_lookup(TIER_STYLE), subset=["Recommended tier"])
    apply_cell = getattr(styler, "map", None) or styler.applymap
    styler = apply_cell(_lookup(CONF_STYLE), subset=["Confidence"])
    return styler


def explain_reasons(reasons: str) -> list:
    """Turns the pipe-separated confidence reason codes into readable lines."""
    lines = []
    for code in [r for r in str(reasons).split("|") if r and r.lower() != "nan"]:
        lines.append(EXTRA_CONFIDENCE_REASONS.get(code, code.replace("_", " ").capitalize()))
    return lines


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Smart Shelf - Manager View", page_icon="🛒", layout="wide")
    st.title("🛒 Smart Shelf - Manager View")

    try:
        table, summary, snapshot_date = load_dashboard_data()
    except Exception as exc:  # show a readable error instead of a raw traceback
        st.error(f"Could not load data or run the pipeline: {exc}")
        st.stop()

    st.caption(f"Inventory snapshot date: **{snapshot_date}**  |  Batches: **{len(table)}**")
    st.info(summary["global_caveat"])

    # ---- KPI strip ---------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Batches today", summary["batches_today"])
    k2.metric("High / Critical spoilage risk", summary["high_or_critical_risk_count"])
    k3.metric("Value at risk (simulated)", f"{summary['estimated_value_at_risk_sim']:,.2f}")
    k4.metric("Batches with discount evidence (R1)", summary["r1_batches_count"])
    st.caption(summary["wording_note"])

    # ---- Sidebar filters ---------------------------------------------------
    st.sidebar.header("Filters")
    sel_cat = st.sidebar.multiselect("Category", sorted(table["category"].dropna().astype(str).unique()))
    sel_store = st.sidebar.multiselect("Store", sorted(table["store_id"].dropna().astype(str).unique()))
    sel_risk = st.sidebar.multiselect(
        "Spoilage risk band", ["Critical", "High", "Medium", "Low"]
    )
    sel_tier = st.sidebar.multiselect(
        "Recommended tier", sorted(table["recommended_tier"].dropna().astype(str).unique())
    )
    sel_conf = st.sidebar.multiselect(
        "Confidence", sorted(table["confidence_level"].dropna().astype(str).unique())
    )
    low_only = st.sidebar.checkbox("Show only low-confidence batches (⚠️)")

    view = table.copy()
    if sel_cat:
        view = view[view["category"].isin(sel_cat)]
    if sel_store:
        view = view[view["store_id"].isin(sel_store)]
    if sel_risk:
        view = view[view["risk_band"].isin(sel_risk)]
    if sel_tier:
        view = view[view["recommended_tier"].astype(str).isin(sel_tier)]
    if sel_conf:
        view = view[view["confidence_level"].astype(str).isin(sel_conf)]
    if low_only:
        view = view[view["confidence_level"] == "Low"]

    # ---- Legend ------------------------------------------------------------
    with st.expander("How to read this table", expanded=False):
        st.markdown(
            "- **Risk band** (orange/red): how likely the batch is to spoil.\n"
            "- **Recommended tier** (blue): the discount tier suggested by the optimizer. "
            "Batches with no discount evidence (region R0) have no discount % shown.\n"
            "- **Confidence** (✅ high / 🟡 medium / ⚠️ low, grey): how much to trust the "
            "recommendation. Low-confidence rows are shown in grey italics; see the reasons column.\n"
            "- All profit and value figures are **simulated estimates**."
        )

    # ---- Main table --------------------------------------------------------
    st.subheader(f"Inventory, risk and recommendations ({len(view)} of {len(table)} batches)")
    if view.empty:
        st.warning("No batches match the selected filters.")
        return

    display = build_display_table(view)
    st.dataframe(
        style_table(display),
        hide_index=True,
        height=520,
        column_config={
            "Spoil chance (%)": st.column_config.ProgressColumn(
                "Spoil chance (%)", min_value=0, max_value=100, format="%.1f%%"
            ),
            "Recommended discount (%)": st.column_config.NumberColumn(
                "Recommended discount (%)", format="%d%%"
            ),
            "Top factors": st.column_config.TextColumn("Top factors", width="large"),
        },
    )
    st.download_button(
        "Download filtered table (CSV)",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="smart_shelf_recommendations.csv",
        mime="text/csv",
    )

    # ---- Summary charts ----------------------------------------------------
    c1, c2, c3 = st.columns(3)
    order_levels = ["Critical", "High", "Medium", "Low"]
    c1.markdown("**Batches by spoilage risk band**")
    c1.bar_chart(view["risk_band"].value_counts().reindex(order_levels).fillna(0))
    c2.markdown("**Batches by recommended tier**")
    c2.bar_chart(view["recommended_tier"].astype(str).value_counts())
    c3.markdown("**Batches by confidence**")
    c3.bar_chart(view["confidence_level"].astype(str).value_counts())

    # ---- Batch detail ------------------------------------------------------
    st.subheader("Batch detail")
    chosen = st.selectbox("Select a batch", view["record_id"].tolist())
    row = view[view["record_id"] == chosen].iloc[0]

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Spoil chance", f"{row['spoil_chance_pct']:.1f}%", row["risk_band"], delta_color="off")
    d2.metric("Days left / stock", f"{int(row['days_left'])} d / {int(row['stock_units'])}")
    disc = row["recommended_discount_pct"]
    d3.metric(
        "Recommended tier",
        str(row["recommended_tier"]),
        "no discount evidence" if pd.isna(disc) else f"{int(disc)}% discount",
        delta_color="off",
    )
    d4.metric(
        "Confidence",
        f"{CONF_ICON.get(row['confidence_level'], '•')} {row['confidence_level']}",
    )

    st.markdown(f"**Top factors:** {row['top_factors']}")
    reasons = explain_reasons(row["confidence_reasons"])
    if reasons:
        st.markdown("**Why this confidence level:**")
        for line in reasons:
            st.markdown(f"- {line}")
    if bool(row["limited_history_flag"]):
        st.warning("Limited history for this product/store: treat the estimate with extra care.")
    st.caption(summary["global_caveat"])


if __name__ == "__main__":
    main()