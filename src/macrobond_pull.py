"""Pull macro and expectations series from Macrobond and cache them to data/macrobond/.

Uses the Macrobond desktop application through its COM API, so Macrobond must be installed,
running and logged in on this machine (`pip install macrobond-data-api`). Nothing is
re-pulled unless --refresh is passed. Output: data/macrobond/series.parquet (long format:
series, date, value) plus a manifest. Licensed data: data/ is git-ignored.

Series are grouped by what they measure, since the macro work is about *expectations*
(markets price expected growth and inflation, not the realised numbers):
    growth expectations   SPF real GDP forecasts, OECD leading indicators, ISM new orders
    inflation expectations SPF CPI forecast, 5y5y breakeven
    policy / conditions   2y Treasury yield, Chicago Fed financial conditions
    commodities           Brent
"""
import argparse
import datetime as dt
import json

import pandas as pd

from data import ROOT

OUT = ROOT / "data" / "macrobond"
SERIES = {
    # code: (short name, description)
    "usfcst1810": ("spf_gdp_q0", "SPF mean real GDP growth, current quarter (AR)"),
    "usfcst1811": ("spf_gdp_q1", "SPF mean real GDP growth, next quarter (AR)"),
    "usfcst1812": ("spf_gdp_q2", "SPF mean real GDP growth, two quarters ahead (AR)"),
    "usfcst1813": ("spf_gdp_q3", "SPF mean real GDP growth, three quarters ahead (AR)"),
    "usfcst1814": ("spf_gdp_q4", "SPF mean real GDP growth, four quarters ahead (AR)"),
    "usfcst1029": ("spf_cpi_next_year", "SPF median CPI inflation, following year annual average"),
    "usfcst1191": ("spf_unemp_next_year", "SPF mean unemployment rate, following year annual average"),
    "eufcst0010": ("ecb_spf_gdp_1y", "ECB SPF mean real GDP growth, one year ahead"),
    "oecd_econ_00079390": ("cli_g7", "OECD composite leading indicator, G7, amplitude adjusted"),
    "oecd_econ_00079540": ("cli_us", "OECD composite leading indicator, US, amplitude adjusted"),
    "oecd_econ_00079417": ("cli_europe4", "OECD composite leading indicator, four big European, amplitude adjusted"),
    "ussurv1056": ("ism_new_orders", "ISM manufacturing new orders, SA"),
    "uslead0010": ("cb_lei", "Conference Board leading economic index, SA"),
    "ussurv0386": ("nfci", "Chicago Fed National Financial Conditions Index (weekly)"),
    "usbkeven5f5": ("be_5y5y", "US 5y5y forward breakeven inflation (daily)"),
    "us02ygov": ("ust_2y", "US 2-year Treasury yield (daily)"),
    "uscaes0302": ("brent", "Brent spot, USD (daily)"),
}


def pull(refresh=False):
    from macrobond_data_api.com import ComClient
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "series.parquet"
    cached = pd.read_parquet(path) if path.exists() and not refresh else pd.DataFrame(columns=["series", "date", "value"])
    todo = [c for c in SERIES if SERIES[c][0] not in set(cached["series"])]
    frames = [cached]
    with ComClient() as api:
        for code in todo:
            s = api.get_one_series(code)
            frames.append(pd.DataFrame({"series": SERIES[code][0], "date": pd.to_datetime(s.dates), "value": s.values}))
            print(f"pulled {code:22s} {SERIES[code][0]:22s} {len(s.values)} obs")
    df = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    df.to_parquet(path, index=False)
    manifest = {"pulled_at": dt.datetime.now().isoformat(timespec="seconds"),
                "series": {v[0]: {"code": k, "description": v[1]} for k, v in SERIES.items()}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return df


def monthly():
    """Wide monthly panel (calendar month-end): last observation in each month."""
    df = pd.read_parquet(OUT / "series.parquet")
    df = df.assign(date=pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0))
    return df.pivot_table(index="date", columns="series", values="value", aggfunc="last").sort_index()


def quarterly():
    """Wide quarterly panel. SPF surveys are dated at the quarter's first day: map to that quarter's end."""
    df = pd.read_parquet(OUT / "series.parquet")
    df = df.assign(date=pd.to_datetime(df["date"]) + pd.offsets.QuarterEnd(0))
    return df.pivot_table(index="date", columns="series", values="value", aggfunc="last").sort_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    d = pull(ap.parse_args().refresh)
    print(d.groupby("series")["date"].agg(["min", "max", "count"]).to_string())
