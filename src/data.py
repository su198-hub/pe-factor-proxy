"""Load cached Bloomberg data as wide price and return panels."""
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BBG = ROOT / "data" / "bbg"


def registry():
    return pd.read_csv(ROOT / "config" / "tickers.csv", keep_default_na=False, na_values=[""])  # "NA" is a region, not missing


def panel(name):
    """Wide panel from a cached dataset (a fresh copy; the parquet is read once per process)."""
    return _panel(name).copy()


@lru_cache(maxsize=None)
def _panel(name):
    """Wide panel from a cached dataset (e.g. "px_monthly"): index=date, columns=ticker.

    Monthly data is stamped on each month's calendar end. Bloomberg dates index prices on
    the last business day but economic releases on the calendar day, so without this a month
    ending on a weekend would appear twice.
    """
    df = pd.read_parquet(BBG / f"{name}.parquet")
    if name.endswith("_monthly"):
        df = df.assign(date=df["date"] + pd.offsets.MonthEnd(0))
    return df.pivot_table(index="date", columns="ticker", values="value", aggfunc="last").sort_index()


def prices(freq="monthly"):
    """Wide price panel: index=date, columns=ticker."""
    return panel(f"px_{freq}")


def returns(freq="monthly", tickers=None):
    """Simple returns from index levels. Leading NaNs kept, so each series starts at its own inception."""
    px = prices(freq)
    if tickers is not None:
        px = px[list(tickers)]
    # fill_method=None: never manufacture a zero return across a missing observation
    return px.pct_change(fill_method=None).iloc[1:]


def pert_family():
    """Tickers of each PERT component: {component: {'net': ..., 'long': ..., 'short': ...}}."""
    reg = registry()
    fam = {}
    for _, r in reg[reg["block"].isin(["pert", "pert_long", "pert_short"])].iterrows():
        if r["region"] == "World":
            continue
        key = {"pert": "net", "pert_long": "long", "pert_short": "short"}[r["block"]]
        fam.setdefault(r["region"], {})[key] = r["ticker"]
    return fam
