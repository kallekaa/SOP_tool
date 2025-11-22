"""Inventory planning helpers for the S&OP demo."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from db import fetch_catalog

Z_MAP: Dict[float, float] = {0.9: 1.28, 0.95: 1.65, 0.98: 2.05}


def plan_inventory(
    forecast_df: pd.DataFrame,
    history: pd.DataFrame,
    service_level: float,
    cover_months: float,
    seed: int = 99,
    catalog_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Estimate inventory targets and projected gaps per product."""
    if forecast_df.empty or history.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    z = Z_MAP.get(round(service_level, 2), 1.65)
    catalog = (catalog_df if catalog_df is not None else fetch_catalog()).set_index("product")

    future_avg = (
        forecast_df.groupby("product")["consensus_units"]
        .mean()
        .reindex(catalog.index)
        .fillna(0)
    )
    demand_std = history.groupby("product")["orders"].std().reindex(catalog.index).fillna(0)
    lead_times = catalog["lead_time_months"]

    safety_stock = (z * demand_std * np.sqrt(lead_times)).round(0)
    cycle_stock = (future_avg * cover_months).round(0)
    target = (cycle_stock + safety_stock).round(0)

    on_hand_noise = rng.uniform(0.7, 1.1, size=len(target))
    on_hand = (target * on_hand_noise).round(0)

    gap = (target - on_hand).round(0)
    months_cover = np.where(future_avg > 0, (on_hand / future_avg), 0).round(2)

    result = pd.DataFrame(
        {
            "product": catalog.index,
            "on_hand_units": on_hand.astype(int),
            "target_units": target.astype(int),
            "safety_stock_units": safety_stock.astype(int),
            "projected_gap_units": gap.astype(int),
            "future_avg_units": future_avg.astype(int),
            "months_of_cover": months_cover,
        }
    )
    return result
