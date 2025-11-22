# Mid-term Sales & Operations Planning tool

This Streamlit-based prototype helps planners balance demand, inventory, and supply over a 3‑ to 18‑month horizon. It layers statistical forecasting with manual overrides, inventory policy tuning, and capacity management so a cross-functional team can make aligned decisions in one place.

## Key features
- **Integrated dashboards**: Demand, inventory, and supply views are separated into tabs with consistent KPI rows, and interactive charts compare actuals, consensus plans, and statistical baselines.
- **Planner controls**: Adjust sliders and scenario presets in the sidebar, then edit demand, inventory, and supply tables directly. Overrides are persisted to SQLite tables so downstream plans (inventory, capacity) respond immediately.
- **Simulations & insights**: Inventory simulation shows receipts and ending balances, and supply visuals track planned vs. feasible vs. capacity with flex (overtime) knobs.
- **Data management**: All override tables support download, and saving actions trigger reruns to refresh all dependent metrics.

## Architecture
- `app.py`: Streamlit UI orchestrating data loading, override forms, and tabs for demand/inventory/supply planning.
- `db.py`: SQLite helpers for catalog/history/capacity plus the new overrides tables (`demand_overrides`, `inventory_overrides`, `supply_overrides`).
- `demand.py`, `inventory.py`, `supply.py`: Core planning logic, each exposing override-aware helpers that merge DB inputs back into the calculated plans.
- `data.py`: Synthetic data generator used to seed the demo DB via `init_db()` in `db.py`.

## Getting started
1. Install dependencies: `pip install -r requirements.txt` (or `pip install altair pandas streamlit numpy`).
2. Run `streamlit run app.py`.  
3. Use the sidebar to choose scenario presets, then edit the demand, inventory, or supply tables in their tabs and save to persist overrides.

## Next steps
- Add scenario snapshots so planners can compare saved cycles.  
- Expand the supply simulation to include priority rules and overtime-cost tradeoffs.  
- Add authentication or role-based guards around override writes if the tool moves beyond demo mode.
