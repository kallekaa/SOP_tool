"""Data generation utilities for seeding the S&OP demo database."""

from __future__ import annotations

from datetime import date
from typing import Dict, List

import numpy as np
import pandas as pd

REGIONS: List[str] = ["NA", "EMEA", "APAC"]

PRODUCTS: List[Dict[str, object]] = [
    {
        "product": "Alpha",
        "family": "Core",
        "lead_time_months": 1.5,
        "unit_cost": 42,
        "base_demand": 900,
        "price": 120,
        "site": "Plant A",
    },
    {
        "product": "Bravo",
        "family": "Core",
        "lead_time_months": 2.0,
        "unit_cost": 55,
        "base_demand": 750,
        "price": 135,
        "site": "Plant B",
    },
    {
        "product": "Charlie",
        "family": "Value",
        "lead_time_months": 1.2,
        "unit_cost": 33,
        "base_demand": 650,
        "price": 95,
        "site": "Plant B",
    },
    {
        "product": "Delta",
        "family": "Premium",
        "lead_time_months": 3.0,
        "unit_cost": 78,
        "base_demand": 420,
        "price": 180,
        "site": "Plant C",
    },
]

REGION_FACTORS: Dict[str, float] = {"NA": 1.1, "EMEA": 1.0, "APAC": 0.9}
SEASONALITY = np.array([0.92, 0.94, 0.96, 1.00, 1.05, 1.10, 1.15, 1.12, 1.08, 1.02, 0.98, 0.95])
CAPACITY_BY_SITE: Dict[str, int] = {"Plant A": 1600, "Plant B": 1400, "Plant C": 900}


def product_catalog() -> pd.DataFrame:
    return pd.DataFrame(PRODUCTS)


def make_history(months: int = 18, seed: int = 7) -> pd.DataFrame:
    """Build mock monthly order and shipment history."""
    rng = np.random.default_rng(seed)
    last_month = pd.Period(date.today(), freq="M")
    periods = pd.period_range(end=last_month, periods=months, freq="M")

    records = []
    for idx, period in enumerate(periods):
        month_factor = SEASONALITY[period.month - 1]
        trend = 1 + 0.01 * idx  # 1% monthly growth
        for prod in PRODUCTS:
            for region in REGIONS:
                base = prod["base_demand"] * REGION_FACTORS[region]
                noise = rng.normal(0, base * 0.08)
                orders = max(0, base * month_factor * trend + noise)
                shipments = max(0, orders * rng.uniform(0.96, 1.02))
                price = prod["price"] * (1 + 0.002 * idx)
                records.append(
                    {
                        "month": period.to_timestamp(),
                        "product": prod["product"],
                        "family": prod["family"],
                        "region": region,
                        "orders": round(orders, 0),
                        "shipments": round(shipments, 0),
                        "unit_price": round(price, 2),
                        "lead_time_months": prod["lead_time_months"],
                        "unit_cost": prod["unit_cost"],
                        "site": prod["site"],
                    }
                )
    df = pd.DataFrame(records)
    df["revenue"] = (df["shipments"] * df["unit_price"]).round(2)
    return df.sort_values("month")


def site_capacity() -> pd.DataFrame:
    return pd.DataFrame({"site": list(CAPACITY_BY_SITE.keys()), "capacity_units": list(CAPACITY_BY_SITE.values())})
