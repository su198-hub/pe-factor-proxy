"""Pull every ticker in config/tickers.csv from Bloomberg and cache to data/bbg/.

Writes one long-format parquet per dataset (ticker, date, value) plus a manifest.
Datasets: PX_LAST weekly and monthly for every ticker; for the sector and
price-return parent indexes, monthly trailing 12m dividend yield (turns price returns
into approximate total returns) and trailing / forward P/E (valuation decomposition).

Every price series is pulled in USD: tickers quoted in another currency (e.g. MXEU,
in EUR) get Bloomberg's currency override, and are listed in the manifest. Macro
series (yields, spreads, surprise indexes) and local-currency check indexes are
never converted.

Only tickers not already in the cache are pulled, so adding a ticker to the registry
costs only that ticker's data points. --refresh re-pulls everything (e.g. to extend
the history to today).

Usage (project root, terminal running):
    .venv\\Scripts\\python.exe -u src\\pull_bbg.py            # pull tickers missing from the cache
    .venv\\Scripts\\python.exe -u src\\pull_bbg.py --refresh  # re-pull everything
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd
from xbbg import blp

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config" / "tickers.csv"
OUT = ROOT / "data" / "bbg"

START = "2000-01-01"
# name -> (field, Bloomberg periodicity, registry blocks to pull; None = all).
# Weekly for style regressions (avoids asynchronous daily closes across regions),
# monthly for everything else.
DATASETS = {
    "px_weekly": ("PX_LAST", "W", None),
    "px_monthly": ("PX_LAST", "M", None),
    "dy_monthly": ("EQY_DVD_YLD_12M", "M", {"sector", "parent_px"}),
    "pe_monthly": ("PE_RATIO", "M", {"sector", "parent_px"}),
    "fpe_monthly": ("BEST_PE_RATIO", "M", {"sector", "parent_px"}),
}
CHUNK = 20  # tickers per request
CCY = "USD"
NO_CONVERT = {"macro", "parent_local"}  # levels / deliberately local: never currency-convert


def quote_currency(tickers):
    """{ticker: CRNCY}. Series without a currency (yields, spreads) come back blank."""
    df = blp.bdp(tickers, "CRNCY").to_pandas()
    return dict(zip(df["ticker"], df["value"]))


def pull(tickers, field, per, end, convert):
    """convert: tickers needing the USD override (price fields only; yields are ratios)."""
    frames = []
    groups = [([t for t in tickers if t not in convert], {}), ([t for t in tickers if t in convert], {"Currency": CCY})]
    for group, ovr in groups:
        for i in range(0, len(group), CHUNK):
            batch = group[i:i + CHUNK]
            print(f"  {field} {per}{' ' + CCY if ovr else ''}: tickers {i + 1}-{i + len(batch)} of {len(group)}", flush=True)
            df = blp.bdh(batch, field, START, end, Per=per, **ovr).to_pandas()
            frames.append(df[["ticker", "date", "value"]])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-pull even if a cache exists")
    ap.add_argument("--repull", nargs="+", default=[], metavar="TICKER", help="drop these from the caches and pull again")
    args = ap.parse_args()

    reg = pd.read_csv(REGISTRY)
    end = dt.date.today().isoformat()
    OUT.mkdir(parents=True, exist_ok=True)

    mpath = OUT / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest.setdefault("files", {})
    manifest.pop("field", None)  # older single-field manifests
    pulled = False
    ccy = None
    for name, (field, per, blocks) in DATASETS.items():
        tickers = reg["ticker"].tolist() if blocks is None else reg.loc[reg["block"].isin(blocks), "ticker"].tolist()
        path = OUT / f"{name}.parquet"
        cached = pd.read_parquet(path) if path.exists() and not args.refresh else None
        if cached is not None and args.repull:
            cached = cached[~cached["ticker"].isin(args.repull)]
        todo = tickers if cached is None else [t for t in tickers if t not in set(cached["ticker"])]
        if not todo:
            print(f"{path.name}: all {len(tickers)} tickers cached - nothing to pull")
            continue
        convert = set()
        if field == "PX_LAST":
            ccy = ccy or quote_currency(reg["ticker"].tolist())
            keep = set(reg.loc[reg["block"].isin(NO_CONVERT), "ticker"])
            convert = {t for t in todo if ccy.get(t) not in (CCY, "", None) and t not in keep}
        new = pull(todo, field, per, end, convert)
        df = new if cached is None else pd.concat([cached, new], ignore_index=True)
        df.to_parquet(path, index=False)
        cov = df.groupby("ticker")["date"].agg(["min", "max", "count"])
        missing = sorted(set(tickers) - set(cov.index))
        prev = manifest["files"].get(path.name, {}) if cached is not None else {}
        manifest["files"][path.name] = {"field": field, "per": per, "start": START, "end": end,
                                        "pulled_at": dt.datetime.now().isoformat(timespec="seconds"),
                                        "rows": len(df), "missing": missing}
        if field == "PX_LAST":
            conv = {t: c for t, c in prev.get("converted_to_usd", {}).items() if t not in args.repull}
            conv.update({t: ccy[t] for t in convert})
            manifest["files"][path.name]["converted_to_usd"] = dict(sorted(conv.items()))
        print(f"wrote {path.name}: {len(df):,} rows, {len(cov)} tickers, missing: {missing or 'none'}")
        cov.to_csv(OUT / f"coverage_{name}.csv")
        pulled = True

    if pulled:
        mpath.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
