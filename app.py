"""Mid-term S&OP Streamlit app."""

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd
import streamlit as st

from db import fetch_catalog, fetch_capacity, fetch_history, init_db
from demand import build_demand_plan
from inventory import plan_inventory
from supply import build_supply_plan


@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Ensure the DB is initialized and fetch reference data."""
    init_db()
    history = fetch_history(months=18)
    catalog = fetch_catalog()
    capacity_map = fetch_capacity()
    return history, catalog, capacity_map


def sidebar_controls(catalog: pd.DataFrame, regions: list[str]):
    st.sidebar.header("Planning assumptions")
    horizon = st.sidebar.slider("Planning horizon (months)", min_value=3, max_value=18, value=12, step=1)
    sales_bias = st.sidebar.slider("Sales/marketing bias (%)", min_value=-10.0, max_value=25.0, value=5.0, step=0.5)
    promo_lift = st.sidebar.slider("Near-term promo lift (%)", min_value=0.0, max_value=30.0, value=10.0, step=1.0)
    service_level = st.sidebar.select_slider("Service level target", options=[0.90, 0.95, 0.98], value=0.95)
    cover_months = st.sidebar.slider("Cycle + safety cover (months)", min_value=1.0, max_value=4.0, value=2.0, step=0.5)
    capacity_buffer = st.sidebar.slider("Capacity headroom (%)", min_value=-10.0, max_value=30.0, value=10.0, step=1.0)
    selected_products = st.sidebar.multiselect(
        "Product focus", options=catalog["product"], default=catalog["product"].tolist()
    )
    selected_regions = st.sidebar.multiselect("Regions", options=regions, default=regions)
    st.sidebar.caption("Adjust assumptions to see mid-term demand, inventory, and supply move together.")
    return horizon, sales_bias, promo_lift, service_level, cover_months, capacity_buffer, selected_products, selected_regions


def filter_history(df: pd.DataFrame, products: Iterable[str], regions: Iterable[str]) -> pd.DataFrame:
    filtered = df.copy()
    if products:
        filtered = filtered[filtered["product"].isin(products)]
    if regions:
        filtered = filtered[filtered["region"].isin(regions)]
    return filtered


def kpi_row(forecast_df: pd.DataFrame, inventory_df: pd.DataFrame, supply_df: pd.DataFrame):
    demand_units = int(forecast_df["consensus_units"].sum()) if not forecast_df.empty else 0
    revenue = forecast_df["revenue"].sum() if not forecast_df.empty else 0
    inv_gap = int(inventory_df["projected_gap_units"].clip(lower=0).sum()) if not inventory_df.empty else 0
    util = supply_df["utilization"].max() if not supply_df.empty else 0

    st.subheader("Integrated S&OP pulse")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Consensus demand (units)", f"{demand_units:,}")
    c2.metric("Revenue plan", f"${revenue:,.0f}")
    c3.metric("Projected inv. gap", f"{inv_gap:,} units")
    c4.metric("Peak utilization", f"{util:.0%}")


def demand_section(history: pd.DataFrame, forecast_df: pd.DataFrame):
    st.markdown("### Demand planning")
    if history.empty and forecast_df.empty:
        st.info("No data available for the selected filters.")
        return

    actual_monthly = history.groupby("month")["orders"].sum().reset_index(name="actual_orders")
    forecast_monthly = (
        forecast_df.groupby("month")["consensus_units"].sum().reset_index(name="consensus_forecast")
    )
    timeline = pd.merge(actual_monthly, forecast_monthly, on="month", how="outer").sort_values("month")
    timeline = timeline.set_index("month")
    st.line_chart(timeline)

    by_product = (
        forecast_df.groupby("product")[["consensus_units", "revenue"]]
        .sum()
        .sort_values("consensus_units", ascending=False)
    )
    st.caption("Consensus forecast by product")
    st.dataframe(by_product, use_container_width=True)


def inventory_section(inventory_df: pd.DataFrame):
    st.markdown("### Inventory planning")
    if inventory_df.empty:
        st.info("Inventory plan will appear once demand is available.")
        return

    chart_data = inventory_df.set_index("product")[["on_hand_units", "target_units", "safety_stock_units"]]
    st.bar_chart(chart_data)
    display = inventory_df[
        ["product", "on_hand_units", "target_units", "safety_stock_units", "projected_gap_units", "months_of_cover"]
    ].set_index("product")
    st.dataframe(display, use_container_width=True)


def supply_section(supply_df: pd.DataFrame):
    st.markdown("### Supply planning")
    if supply_df.empty:
        st.info("Supply plan will appear once demand is available.")
        return

    avg_util = (
        supply_df.groupby("site")["utilization"]
        .mean()
        .rename("avg_utilization")
        .sort_values(ascending=False)
    )
    st.caption("Average utilization by site")
    st.bar_chart(avg_util)

    summary = supply_df.copy()
    summary["month"] = summary["month"].dt.strftime("%b %Y")
    summary["utilization"] = (summary["utilization"] * 100).round(1)
    summary = summary[
        ["month", "site", "planned_units", "capacity_units", "feasible_units", "shortfall_units", "utilization"]
    ]
    st.dataframe(summary, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(
        page_title="Mid-term S&OP",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Mid-term Sales & Operations Planning")
    st.caption("Coordinate demand, inventory, and supply over a 3–18 month horizon.")

    history, catalog, capacity_map = load_data()
    regions = sorted(history["region"].unique().tolist())
    (
        horizon,
        sales_bias,
        promo_lift,
        service_level,
        cover_months,
        capacity_buffer,
        products,
        selected_regions,
    ) = sidebar_controls(catalog, regions)
    filtered_history = filter_history(history, products, selected_regions)

    forecast_df = build_demand_plan(
        filtered_history,
        horizon_months=horizon,
        sales_bias_pct=sales_bias,
        promo_lift_pct=promo_lift,
        catalog_df=catalog,
    )
    inventory_df = plan_inventory(
        forecast_df=forecast_df,
        history=filtered_history,
        service_level=service_level,
        cover_months=cover_months,
        catalog_df=catalog,
    )
    supply_df = build_supply_plan(
        forecast_df=forecast_df,
        capacity_buffer_pct=capacity_buffer,
        capacity_by_site=capacity_map,
        catalog_df=catalog,
    )

    kpi_row(forecast_df, inventory_df, supply_df)
    demand_section(filtered_history, forecast_df)
    inventory_section(inventory_df)
    supply_section(supply_df)


if __name__ == "__main__":
    main()
