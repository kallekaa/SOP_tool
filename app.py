"""Mid-term S&OP Streamlit app."""

from __future__ import annotations

from typing import Iterable, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from db import (
    fetch_catalog,
    fetch_capacity,
    fetch_demand_overrides,
    fetch_history,
    fetch_inventory_overrides,
    fetch_supply_overrides,
    init_db,
    upsert_demand_overrides,
    upsert_inventory_overrides,
    upsert_supply_overrides,
)
from demand import apply_overrides, build_demand_plan
from inventory import apply_inventory_overrides, plan_inventory
from supply import apply_supply_overrides, build_supply_plan


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

    presets = {
        "Base": {"horizon": 12, "sales_bias": 5.0, "promo_lift": 10.0, "service_level": 0.95, "cover_months": 2.0},
        "Conservative": {"horizon": 9, "sales_bias": -2.0, "promo_lift": 5.0, "service_level": 0.98, "cover_months": 3.0},
        "Stretch": {"horizon": 15, "sales_bias": 12.0, "promo_lift": 15.0, "service_level": 0.9, "cover_months": 1.5},
    }
    preset_choice = st.sidebar.selectbox("Scenario preset", options=["Custom", *presets.keys()], index=0)
    if preset_choice != "Custom" and st.sidebar.button("Apply preset", use_container_width=True):
        preset = presets[preset_choice]
        st.session_state["horizon"] = preset["horizon"]
        st.session_state["sales_bias"] = preset["sales_bias"]
        st.session_state["promo_lift"] = preset["promo_lift"]
        st.session_state["service_level"] = preset["service_level"]
        st.session_state["cover_months"] = preset["cover_months"]
        st.rerun()

    horizon = st.sidebar.slider(
        "Planning horizon (months)", min_value=3, max_value=18, value=st.session_state.get("horizon", 12), step=1, key="horizon"
    )
    sales_bias = st.sidebar.slider(
        "Sales/marketing bias (%)", min_value=-10.0, max_value=25.0, value=st.session_state.get("sales_bias", 5.0), step=0.5, key="sales_bias"
    )
    promo_lift = st.sidebar.slider(
        "Near-term promo lift (%)", min_value=0.0, max_value=30.0, value=st.session_state.get("promo_lift", 10.0), step=1.0, key="promo_lift"
    )
    service_level = st.sidebar.select_slider(
        "Service level target", options=[0.90, 0.95, 0.98], value=st.session_state.get("service_level", 0.95), key="service_level"
    )
    cover_months = st.sidebar.slider(
        "Cycle + safety cover (months)", min_value=1.0, max_value=4.0, value=st.session_state.get("cover_months", 2.0), step=0.5, key="cover_months"
    )
    capacity_buffer = st.sidebar.slider("Capacity headroom (%)", min_value=-10.0, max_value=30.0, value=10.0, step=1.0)
    freeze_months = st.sidebar.slider("Freeze near-term overrides (months)", min_value=0, max_value=3, value=1, step=1)
    selected_products = st.sidebar.multiselect(
        "Product focus", options=catalog["product"], default=catalog["product"].tolist()
    )
    selected_regions = st.sidebar.multiselect("Regions", options=regions, default=regions)
    st.sidebar.caption("Adjust assumptions to see mid-term demand, inventory, and supply move together.")
    return (
        horizon,
        sales_bias,
        promo_lift,
        service_level,
        cover_months,
        capacity_buffer,
        freeze_months,
        selected_products,
        selected_regions,
    )


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


def demand_section(history: pd.DataFrame, forecast_df: pd.DataFrame, freeze_months: int):
    st.markdown("### Demand planning")
    if history.empty and forecast_df.empty:
        st.info("No data available for the selected filters.")
        return

    actual_monthly = history.groupby("month")["orders"].sum().reset_index(name="actual_orders")
    forecast_monthly = (
        forecast_df.groupby("month")[["stat_forecast_units", "consensus_units"]]
        .sum()
        .reset_index()
        .rename(columns={"stat_forecast_units": "baseline_forecast", "consensus_units": "consensus_forecast"})
    )
    timeline = pd.merge(actual_monthly, forecast_monthly, on="month", how="outer").sort_values("month")
    timeline = timeline.fillna(0)

    timeline_chart = alt.Chart(timeline).encode(x=alt.X("month:T", title="Month"))
    actual_bars = timeline_chart.mark_bar(color="#1f77b4").encode(
        y=alt.Y("actual_orders:Q", title="Units"),
        tooltip=[
            alt.Tooltip("month:T", title="Month", format="%b %Y"),
            alt.Tooltip("actual_orders:Q", title="Actual orders", format=","),
        ],
    )
    baseline_line = timeline_chart.mark_line(color="#7f7f7f", strokeDash=[6, 4], strokeWidth=2).encode(
        y=alt.Y("baseline_forecast:Q", title="Units"),
        tooltip=[
            alt.Tooltip("month:T", title="Month", format="%b %Y"),
            alt.Tooltip("baseline_forecast:Q", title="Baseline", format=","),
        ],
    )
    forecast_line = timeline_chart.mark_line(color="#d62728", strokeWidth=3).encode(
        y=alt.Y("consensus_forecast:Q", title="Units"),
        tooltip=[
            alt.Tooltip("month:T", title="Month", format="%b %Y"),
            alt.Tooltip("consensus_forecast:Q", title="Consensus", format=","),
        ],
    )
    st.altair_chart(actual_bars + baseline_line + forecast_line, use_container_width=True)

    by_product = (
        forecast_df.groupby("product")[["stat_forecast_units", "consensus_units", "revenue"]]
        .sum()
        .sort_values("consensus_units", ascending=False)
    )
    st.caption("Baseline vs consensus forecast by product")
    st.dataframe(by_product, use_container_width=True)

    st.markdown("#### Adjust monthly demand plan")
    plan_df = (
        forecast_df[["month", "product", "region", "stat_forecast_units", "consensus_units"]]
        .rename(columns={"consensus_units": "plan_units"})
        .copy()
    ).sort_values(["month", "product", "region"])
    plan_df["month"] = pd.to_datetime(plan_df["month"])
    plan_df["plan_units"] = pd.to_numeric(plan_df["plan_units"], errors="coerce").fillna(0)
    plan_df["delta_vs_model"] = (plan_df["plan_units"] - plan_df["stat_forecast_units"]).round(1)
    plan_df["delta_pct"] = (
        100 * plan_df["delta_vs_model"] / plan_df["stat_forecast_units"].replace(0, pd.NA)
    ).round(1)

    months_sorted = sorted(plan_df["month"].unique())
    frozen_months = set(months_sorted[:freeze_months])
    if freeze_months > 0:
        first_frozen = pd.to_datetime(months_sorted[0]).strftime("%b %Y") if months_sorted else ""
        st.info(f"Freeze window: first {freeze_months} month(s) (starting {first_frozen}) cannot be overridden.")

    st.caption("Edit plan units below; month, product, and region are locked. Click save to persist.")
    with st.form("demand_plan_form", clear_on_submit=False):
        edited_plan = st.data_editor(
            plan_df,
            hide_index=True,
            use_container_width=True,
            key="demand_plan_editor",
            column_config={
                "month": st.column_config.DatetimeColumn("Month", format="MMM YYYY", disabled=True),
                "product": st.column_config.Column("Product", disabled=True),
                "region": st.column_config.Column("Region", disabled=True),
                "stat_forecast_units": st.column_config.NumberColumn("Model units", disabled=True),
                "plan_units": st.column_config.NumberColumn("Plan units", step=10),
                "delta_vs_model": st.column_config.NumberColumn("Delta vs model", disabled=True),
                "delta_pct": st.column_config.NumberColumn("Delta %", disabled=True, format="0.0"),
            },
        )
        save_clicked = st.form_submit_button("Save demand plan to database", type="primary")

    if save_clicked:
        save_df = edited_plan.copy()
        save_df["plan_units"] = pd.to_numeric(save_df["plan_units"], errors="coerce").fillna(0).clip(lower=0)
        if frozen_months:
            save_df = save_df[~save_df["month"].isin(frozen_months)]
        upsert_demand_overrides(save_df[["month", "product", "region", "plan_units"]])
        st.success("Saved demand plan. Recalculating plans...")
        st.rerun()

    total_delta = plan_df["delta_vs_model"].sum()
    pct_overridden = (plan_df["plan_units"] != plan_df["stat_forecast_units"]).mean() * 100
    st.caption(f"Override impact: {total_delta:,.0f} units vs model; {pct_overridden:.1f}% of rows overridden.")
    st.download_button("Download demand plan (CSV)", forecast_df.to_csv(index=False), file_name="demand_plan.csv")


def inventory_section(inventory_df: pd.DataFrame, forecast_df: pd.DataFrame):
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

    total_gap = int(inventory_df["projected_gap_units"].clip(lower=0).sum())
    avg_cover = inventory_df["months_of_cover"].mean()
    below_cover = (inventory_df["months_of_cover"] < 1).mean() * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Total projected gap", f"{total_gap:,} units")
    c2.metric("Avg months of cover", f"{avg_cover:.2f}")
    c3.metric("Items <1 mo cover", f"{below_cover:.1f}%")

    st.markdown("#### Adjust inventory targets")
    policy_df = inventory_df[
        ["product", "on_hand_units", "recommended_target_units", "target_units"]
    ].copy()
    policy_df = policy_df.rename(columns={"target_units": "planner_target_units"})
    st.caption("Edit planner target units; on-hand and recommended targets are read-only references.")
    with st.form("inventory_plan_form", clear_on_submit=False):
        edited_inventory = st.data_editor(
            policy_df,
            hide_index=True,
            use_container_width=True,
            key="inventory_plan_editor",
            column_config={
                "product": st.column_config.Column("Product", disabled=True),
                "on_hand_units": st.column_config.NumberColumn("On hand", disabled=True),
                "recommended_target_units": st.column_config.NumberColumn("Recommended target", disabled=True),
                "planner_target_units": st.column_config.NumberColumn("Planner target", step=10),
            },
        )
        save_inventory = st.form_submit_button("Save inventory targets", type="primary")

    if save_inventory:
        save_df = edited_inventory.copy()
        save_df["planner_target_units"] = (
            pd.to_numeric(save_df["planner_target_units"], errors="coerce").fillna(0).clip(lower=0)
        )
        upsert_inventory_overrides(
            save_df[["product", "planner_target_units"]].rename(columns={"planner_target_units": "target_units"})
        )
        st.success("Saved inventory targets. Recalculating plans...")
        st.rerun()

    st.markdown("#### Inventory projection simulation")
    products = inventory_df["product"].unique().tolist()
    selected_product = st.selectbox("Product", products)
    product_plan = inventory_df.set_index("product").loc[selected_product]
    product_forecast = (
        forecast_df[forecast_df["product"] == selected_product]
        .groupby("month")["consensus_units"]
        .sum()
        .reset_index()
        .sort_values("month")
    )
    if product_forecast.empty:
        st.info("No forecast available for this product.")
        return

    target_units = int(product_plan["target_units"])
    on_hand = int(product_plan["on_hand_units"])
    records = []
    inventory_level = on_hand
    for _, row in product_forecast.iterrows():
        month = row["month"]
        demand_units = float(row["consensus_units"])
        after_demand = max(inventory_level - demand_units, 0)
        receipt_units = max(target_units - after_demand, 0)
        ending_inventory = after_demand + receipt_units
        records.append(
            {
                "month": month,
                "demand_units": demand_units,
                "receipt_units": receipt_units,
                "ending_inventory": ending_inventory,
                "target_units": target_units,
            }
        )
        inventory_level = ending_inventory

    sim_df = pd.DataFrame(records)
    base_chart = alt.Chart(sim_df).encode(x=alt.X("month:T", title="Month"))
    demand_bars = base_chart.mark_bar(color="#d62728", opacity=0.5).encode(
        y=alt.Y("demand_units:Q", title="Units"),
        tooltip=[alt.Tooltip("month:T", title="Month", format="%b %Y"), alt.Tooltip("demand_units:Q", title="Demand")],
    )
    receipt_bars = base_chart.mark_bar(color="#1f77b4", opacity=0.3).encode(
        y="receipt_units:Q",
        tooltip=[alt.Tooltip("month:T", title="Month", format="%b %Y"), alt.Tooltip("receipt_units:Q", title="Receipts")],
    )
    inventory_line = base_chart.mark_line(color="#2ca02c", strokeWidth=3).encode(
        y=alt.Y("ending_inventory:Q", title="Ending inventory"),
        tooltip=[
            alt.Tooltip("month:T", title="Month", format="%b %Y"),
            alt.Tooltip("ending_inventory:Q", title="Ending inv."),
        ],
    )
    target_rule = alt.Chart(pd.DataFrame({"target_units": [target_units]})).mark_rule(
        color="#888", strokeDash=[6, 4]
    ).encode(y="target_units:Q")
    st.altair_chart(demand_bars + receipt_bars + inventory_line + target_rule, use_container_width=True)
    st.download_button("Download inventory plan (CSV)", inventory_df.to_csv(index=False), file_name="inventory_plan.csv")


def supply_section(supply_df: pd.DataFrame, capacity_map: dict):
    st.markdown("### Supply planning")
    if supply_df.empty:
        st.info("Supply plan will appear once demand is available.")
        return

    capacity_df = pd.DataFrame(
        {"site": list(capacity_map.keys()), "base_capacity": list(capacity_map.values()), "flex_pct": 0}
    )
    st.caption("Adjust temporary flex capacity per site (e.g., overtime or contractors).")
    flex_df = st.data_editor(
        capacity_df,
        hide_index=True,
        key="capacity_flex_editor",
        column_config={
            "site": st.column_config.Column("Site", disabled=True),
            "base_capacity": st.column_config.NumberColumn("Base capacity", disabled=True),
            "flex_pct": st.column_config.NumberColumn("Flex capacity %", step=5),
        },
    )
    adjusted_capacity = {
        row["site"]: row["base_capacity"] * (1 + (row["flex_pct"] or 0) / 100) for _, row in flex_df.iterrows()
    }
    supply_view = supply_df.copy()
    supply_view["capacity_units"] = supply_view["site"].map(adjusted_capacity).fillna(supply_view["capacity_units"])
    supply_view["feasible_units"] = np.minimum(supply_view["planned_units"], supply_view["capacity_units"])
    supply_view["shortfall_units"] = (supply_view["planned_units"] - supply_view["feasible_units"]).clip(lower=0)
    supply_view["utilization"] = np.where(
        supply_view["capacity_units"] > 0, supply_view["planned_units"] / supply_view["capacity_units"], 0
    )

    avg_util = (
        supply_view.groupby("site")["utilization"]
        .mean()
        .rename("avg_utilization")
        .sort_values(ascending=False)
    )
    st.caption("Average utilization by site")
    st.bar_chart(avg_util)

    total_shortfall = int(supply_view["shortfall_units"].sum())
    peak_row = supply_view.loc[supply_view["utilization"].idxmax()] if not supply_view.empty else None
    c1, c2 = st.columns(2)
    c1.metric("Total shortfall", f"{total_shortfall:,} units")
    if peak_row is not None:
        c2.metric("Peak utilization", f"{peak_row['utilization']:.0%}", f"{peak_row['site']} / {peak_row['month'].strftime('%b %Y')}")

    summary = supply_view.copy()
    summary["month"] = summary["month"].dt.strftime("%b %Y")
    summary["utilization"] = (summary["utilization"] * 100).round(1)
    summary = summary[
        ["month", "site", "planned_units", "capacity_units", "feasible_units", "shortfall_units", "utilization"]
    ]
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("#### Adjust supply plan by site and month")
    editable_supply = supply_view.copy()
    editable_supply["planned_units"] = editable_supply["planned_units"].round(0)
    st.caption("Edit planned units; capacity is read-only. Save to persist and recompute feasibility.")
    with st.form("supply_plan_form", clear_on_submit=False):
        edited_supply = st.data_editor(
            editable_supply,
            hide_index=True,
            use_container_width=True,
            key="supply_plan_editor",
            column_config={
                "month": st.column_config.DatetimeColumn("Month", format="MMM YYYY", disabled=True),
                "site": st.column_config.Column("Site", disabled=True),
                "planned_units": st.column_config.NumberColumn("Planned units", step=10),
                "capacity_units": st.column_config.NumberColumn("Capacity", disabled=True),
                "feasible_units": st.column_config.NumberColumn("Feasible", disabled=True),
                "shortfall_units": st.column_config.NumberColumn("Shortfall", disabled=True),
                "utilization": st.column_config.NumberColumn("Utilization", disabled=True, format="0.0%"),
            },
        )
        save_supply = st.form_submit_button("Save supply plan", type="primary")

    if save_supply:
        save_df = edited_supply.copy()
        save_df["planned_units"] = pd.to_numeric(save_df["planned_units"], errors="coerce").fillna(0).clip(lower=0)
        upsert_supply_overrides(save_df[["month", "site", "planned_units"]])
        st.success("Saved supply plan. Recalculating plans...")
        st.rerun()

    st.markdown("#### Supply vs capacity over time")
    sites = sorted(capacity_map.keys())
    selected_site = st.selectbox("Site", sites)
    site_series = supply_view[supply_view["site"] == selected_site].sort_values("month")
    if site_series.empty:
        st.info("No supply plan available for this site.")
        return

    base_chart = alt.Chart(site_series).encode(x=alt.X("month:T", title="Month"))
    planned_line = base_chart.mark_line(color="#1f77b4", strokeWidth=3).encode(
        y=alt.Y("planned_units:Q", title="Units"),
        tooltip=[
            alt.Tooltip("month:T", title="Month", format="%b %Y"),
            alt.Tooltip("planned_units:Q", title="Planned units", format=",.0f"),
        ],
    )
    capacity_line = base_chart.mark_rule(color="#888", strokeDash=[6, 4]).encode(
        y=alt.Y("capacity_units:Q", title="Capacity"),
        tooltip=[alt.Tooltip("capacity_units:Q", title="Capacity", format=",.0f")],
    )
    feasible_area = base_chart.mark_area(color="#2ca02c", opacity=0.2).encode(
        y="feasible_units:Q",
        tooltip=[
            alt.Tooltip("month:T", title="Month", format="%b %Y"),
            alt.Tooltip("feasible_units:Q", title="Feasible units", format=",.0f"),
            alt.Tooltip("shortfall_units:Q", title="Shortfall", format=",.0f"),
        ],
    )
    st.altair_chart(planned_line + capacity_line + feasible_area, use_container_width=True)
    st.download_button("Download supply plan (CSV)", supply_view.to_csv(index=False), file_name="supply_plan.csv")


def main():
    st.set_page_config(
        page_title="Mid-term S&OP",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Mid-term Sales & Operations Planning")
    st.caption("Coordinate demand, inventory, and supply over a 3-18 month horizon.")

    history, catalog, capacity_map = load_data()
    regions = sorted(history["region"].unique().tolist())
    (
        horizon,
        sales_bias,
        promo_lift,
        service_level,
        cover_months,
        capacity_buffer,
        freeze_months,
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
    overrides_df = fetch_demand_overrides(months=horizon)
    forecast_df = apply_overrides(forecast_df, overrides_df)
    inventory_df = plan_inventory(
        forecast_df=forecast_df,
        history=filtered_history,
        service_level=service_level,
        cover_months=cover_months,
        catalog_df=catalog,
    )
    inventory_overrides = fetch_inventory_overrides()
    inventory_df = apply_inventory_overrides(inventory_df, inventory_overrides)
    supply_df = build_supply_plan(
        forecast_df=forecast_df,
        capacity_buffer_pct=capacity_buffer,
        capacity_by_site=capacity_map,
        catalog_df=catalog,
    )
    supply_overrides = fetch_supply_overrides(months=horizon)
    supply_df = apply_supply_overrides(supply_df, supply_overrides)

    kpi_row(forecast_df, inventory_df, supply_df)
    st.divider()

    demand_tab, inventory_tab, supply_tab = st.tabs(["Demand", "Inventory", "Supply"])
    with demand_tab:
        demand_section(filtered_history, forecast_df, freeze_months)
    with inventory_tab:
        inventory_section(inventory_df, forecast_df)
    with supply_tab:
        supply_section(supply_df, capacity_map)


if __name__ == "__main__":
    main()
