"""SQLite-backed storage for the S&OP demo."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from data import make_history, product_catalog, site_capacity

DB_PATH = Path(__file__).resolve().parent / "sop.db"


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def _create_tables(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS product_catalog (
            product TEXT PRIMARY KEY,
            family TEXT,
            lead_time_months REAL,
            unit_cost REAL,
            base_demand REAL,
            price REAL,
            site TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            month TEXT,
            product TEXT,
            family TEXT,
            region TEXT,
            orders REAL,
            shipments REAL,
            unit_price REAL,
            lead_time_months REAL,
            unit_cost REAL,
            site TEXT,
            revenue REAL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS site_capacity (
            site TEXT PRIMARY KEY,
            capacity_units INTEGER
        );
        """
    )
    conn.commit()


def _seed_catalog(conn: sqlite3.Connection):
    count = conn.execute("SELECT COUNT(*) FROM product_catalog").fetchone()[0]
    if count == 0:
        df = product_catalog()
        df.to_sql("product_catalog", conn, if_exists="append", index=False)
        conn.commit()


def _seed_history(conn: sqlite3.Connection):
    count = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
    if count == 0:
        df = make_history(months=18)
        df = df.assign(month=df["month"].dt.strftime("%Y-%m-%d"))
        df.to_sql("history", conn, if_exists="append", index=False)
        conn.commit()


def _seed_capacity(conn: sqlite3.Connection):
    count = conn.execute("SELECT COUNT(*) FROM site_capacity").fetchone()[0]
    if count == 0:
        df = site_capacity()
        df.to_sql("site_capacity", conn, if_exists="append", index=False)
        conn.commit()


def init_db() -> None:
    """Create tables and seed with mock data if empty."""
    conn = get_connection()
    try:
        _create_tables(conn)
        _seed_catalog(conn)
        _seed_history(conn)
        _seed_capacity(conn)
    finally:
        conn.close()


def fetch_catalog() -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql("SELECT product, family, lead_time_months, unit_cost, price, site FROM product_catalog", conn)
    return df


def fetch_history(months: int | None = None) -> pd.DataFrame:
    query = "SELECT month, product, family, region, orders, shipments, unit_price, lead_time_months, unit_cost, site, revenue FROM history"
    params: Tuple = ()
    if months:
        query += " WHERE month >= ?"
        cutoff = pd.Period(pd.Timestamp.today(), freq="M") - (months - 1)
        cutoff_date = cutoff.to_timestamp().strftime("%Y-%m-%d")
        params = (cutoff_date,)
    query += " ORDER BY month"
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=params)
    df["month"] = pd.to_datetime(df["month"])
    return df


def fetch_capacity(as_dict: bool = True) -> Dict[str, int] | pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql("SELECT site, capacity_units FROM site_capacity", conn)
    return df.set_index("site")["capacity_units"].to_dict() if as_dict else df
