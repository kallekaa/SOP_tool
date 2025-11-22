"""Demand planning helpers for the S&OP demo."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from db import fetch_catalog


def _seasonality_index(history: pd.DataFrame) -> Dict[int, float]:
    monthly = history.groupby(history["month"].dt.month)["orders"].mean()
    if monthly.mean() == 0:
        return {m: 1.0 for m in range(1, 13)}
    monthly_index = (monthly / monthly.mean()).to_dict()
    return {int(k): float(v) for k, v in monthly_index.items()}


def build_demand_plan(
    history: pd.DataFrame,
    horizon_months: int,
    sales_bias_pct: float = 0.0,
    promo_lift_pct: float = 0.0,
    catalog_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Combine a simple trend + seasonality model with sales/marketing input."""
    if history.empty:
        return pd.DataFrame()

    catalog = (catalog_df if catalog_df is not None else fetch_catalog()).set_index("product")
    history = history.copy()
    month_index = {m: i for i, m in enumerate(sorted(history["month"].unique()))}
    history["t"] = history["month"].map(month_index)
    seasonality = _seasonality_index(history)
    last_period = pd.Period(history["month"].max(), freq="M")
    future_periods = pd.period_range(start=last_period + 1, periods=horizon_months, freq="M")

    records = []
    for (product, region), grp in history.groupby(["product", "region"]):
        x = grp["t"].to_numpy()
        y = grp["orders"].to_numpy()
        if len(x) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
        else:
            slope, intercept = 0, y.mean() if len(y) else 0

        for step, period in enumerate(future_periods, start=1):
            t_future = len(month_index) + step - 1
            base = max(0, intercept + slope * t_future)
            seasonal_factor = seasonality.get(period.month, 1.0)
            stat_forecast = max(0, base * seasonal_factor)
            marketing = stat_forecast * (1 + sales_bias_pct / 100)
            promo_factor = 1 + promo_lift_pct / 100 if step <= 3 else 1
            consensus_units = (0.65 * stat_forecast + 0.35 * marketing) * promo_factor
            price = float(catalog.loc[product, "price"])
            records.append(
                {
                    "month": period.to_timestamp(),
                    "product": product,
                    "region": region,
                    "stat_forecast_units": round(stat_forecast, 1),
                    "consensus_units": round(consensus_units, 1),
                    "unit_price": price,
                    "revenue": round(consensus_units * price, 2),
                }
            )

    df = pd.DataFrame(records)
    return df.sort_values(["month", "product", "region"])
