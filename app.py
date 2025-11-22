"""Simple Streamlit sales dashboard wireframe with mock data."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import streamlit as st


def make_mock_data(rows: int = 300) -> pd.DataFrame:
    """Create deterministic mock sales data for the demo dashboard."""
    rng = np.random.default_rng(seed=42)
    today = date.today()
    start = today - timedelta(days=120)
    dates = pd.date_range(start=start, end=today, freq="D")

    regions = ["North", "South", "East", "West"]
    products = ["Alpha", "Bravo", "Charlie", "Delta"]

    data = {
        "order_date": rng.choice(dates, size=rows),
        "region": rng.choice(regions, size=rows),
        "product": rng.choice(products, size=rows),
        "units": rng.integers(1, 25, size=rows),
        "unit_price": rng.uniform(20, 150, size=rows).round(2),
    }
    df = pd.DataFrame(data)
    df["revenue"] = (df["units"] * df["unit_price"]).round(2)
    return df.sort_values("order_date")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return make_mock_data()


def filter_data(
    df: pd.DataFrame,
    date_range: Tuple[date, date],
    regions: Iterable[str],
    products: Iterable[str],
) -> pd.DataFrame:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    mask = (
        (df["order_date"] >= start)
        & (df["order_date"] <= end)
        & (df["region"].isin(regions) if regions else True)
        & (df["product"].isin(products) if products else True)
    )
    return df.loc[mask].copy()


def layout_sidebar(df: pd.DataFrame):
    st.sidebar.header("Filters")
    min_date, max_date = df["order_date"].min().date(), df["order_date"].max().date()
    default_start = max_date - timedelta(days=60)
    date_range = st.sidebar.date_input(
        "Order date range", value=(default_start, max_date), min_value=min_date, max_value=max_date
    )
    selected_regions = st.sidebar.multiselect("Region", options=sorted(df["region"].unique()))
    selected_products = st.sidebar.multiselect("Product", options=sorted(df["product"].unique()))
    st.sidebar.caption("Adjust filters to see how KPIs and charts respond.")
    return date_range, selected_regions, selected_products


def kpi_row(df: pd.DataFrame):
    revenue = df["revenue"].sum()
    units = df["units"].sum()
    avg_ticket = revenue / len(df) if len(df) else 0
    st.subheader("Key metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Revenue", f"${revenue:,.0f}")
    col2.metric("Units sold", f"{units:,}")
    col3.metric("Avg order value", f"${avg_ticket:,.0f}")


def charts(df: pd.DataFrame):
    st.subheader("Sales trends")
    daily = df.groupby("order_date")[["revenue", "units"]].sum().reset_index()
    st.line_chart(daily.set_index("order_date"))

    st.subheader("Revenue by region and product")
    by_region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    by_product = df.groupby("product")["revenue"].sum().sort_values(ascending=False)
    left, right = st.columns(2)
    left.bar_chart(by_region)
    right.bar_chart(by_product)


def data_table(df: pd.DataFrame):
    st.subheader("Order detail")
    st.dataframe(
        df.sort_values("order_date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def main():
    st.set_page_config(
        page_title="Sales Insights (Mock)",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Sales Insights")
    st.caption("Mock dashboard showing how sales data might be explored.")

    df = load_data()
    date_range, regions, products = layout_sidebar(df)
    filtered = filter_data(df, date_range, regions, products)

    kpi_row(filtered)
    charts(filtered)
    data_table(filtered)


if __name__ == "__main__":
    main()
