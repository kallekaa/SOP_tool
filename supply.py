"""Supply planning helpers for the S&OP demo."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from db import fetch_capacity, fetch_catalog


def build_supply_plan(
    forecast_df: pd.DataFrame,
    capacity_buffer_pct: float = 0.0,
    capacity_by_site: Optional[Dict[str, int]] = None,
    catalog_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Translate demand to production by site and compare with capacity."""
    if forecast_df.empty:
        return pd.DataFrame()

    catalog = (catalog_df if catalog_df is not None else fetch_catalog())[["product", "site"]].set_index("product")
    site_lookup = catalog["site"].to_dict()

    monthly_product = (
        forecast_df.groupby(["month", "product"])["consensus_units"]
        .sum()
        .reset_index()
    )
    monthly_product["site"] = monthly_product["product"].map(site_lookup)

    base_capacity = capacity_by_site if capacity_by_site is not None else fetch_capacity()
    capacity = {k: v * (1 + capacity_buffer_pct / 100) for k, v in base_capacity.items()}

    plan = (
        monthly_product.groupby(["month", "site"])["consensus_units"]
        .sum()
        .reset_index()
        .rename(columns={"consensus_units": "planned_units"})
    )
    plan["capacity_units"] = plan["site"].map(capacity).fillna(0)
    plan["feasible_units"] = np.minimum(plan["planned_units"], plan["capacity_units"])
    plan["shortfall_units"] = (plan["planned_units"] - plan["feasible_units"]).clip(lower=0)
    plan["utilization"] = np.where(
        plan["capacity_units"] > 0, plan["planned_units"] / plan["capacity_units"], 0
    )
    return plan.sort_values(["month", "site"])


def apply_supply_overrides(supply_df: pd.DataFrame, overrides_df: pd.DataFrame) -> pd.DataFrame:
    """Replace planned units with planner-entered values, recalculating feasibility."""
    if supply_df.empty or overrides_df.empty:
        return supply_df

    df = supply_df.copy()
    overrides = overrides_df.copy()
    overrides["month"] = pd.to_datetime(overrides["month"])
    merged = df.merge(
        overrides[["month", "site", "planned_units"]],
        on=["month", "site"],
        how="left",
        suffixes=("", "_override"),
    )
    merged["planned_units"] = merged["planned_units_override"].combine_first(merged["planned_units"])
    merged = merged.drop(columns=["planned_units_override"])
    merged["feasible_units"] = np.minimum(merged["planned_units"], merged["capacity_units"])
    merged["shortfall_units"] = (merged["planned_units"] - merged["feasible_units"]).clip(lower=0)
    merged["utilization"] = np.where(
        merged["capacity_units"] > 0, merged["planned_units"] / merged["capacity_units"], 0
    )
    return merged.sort_values(["month", "site"])
