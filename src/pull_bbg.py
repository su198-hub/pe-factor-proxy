"""Pull every ticker in config/tickers.csv from Bloomberg and cache to data/bbg/.

Writes one long-format parquet per frequency (ticker, date, value) plus a manifest.
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
# Bloomberg periodicity -> file suffix. Weekly for style regressions (avoids
# asynchronous daily closes across regions), monthly for everything else.
FREQS = {"W": "weekly", "M": "monthly"}
CHUNK = 20  # tickers per request


def pull(tickers, per, end):
    frames = []
    for i in range(0, len(tickers), CHUNK):
        batch = tickers[i:i + CHUNK]
        print(f"  {per}: tickers {i + 1}-{i + len(batch)} of {len(tickers)}", flush=True)
        df = blp.bdh(batch, "PX_LAST", START, end, Per=per).to_pandas()
        frames.append(df[["ticker", "date", "value"]])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-pull even if a cache exists")
    args = ap.parse_args()

    reg = pd.read_csv(REGISTRY)
    tickers = reg["ticker"].tolist()
    end = dt.date.today().isoformat()
    OUT.mkdir(parents=True, exist_ok=True)

    manifest = {"pulled_at": dt.datetime.now().isoformat(timespec="seconds"), "start": START, "end": end,
                "field": "PX_LAST", "files": {}}
    for per, name in FREQS.items():
        path = OUT / f"px_{name}.parquet"
        cached = pd.read_parquet(path) if path.exists() and not args.refresh else None
        todo = tickers if cached is None else [t for t in tickers if t not in set(cached["ticker"])]
        if not todo:
            print(f"{path.name}: all {len(tickers)} tickers cached - nothing to pull")
            continue
        new = pull(todo, per, end)
        df = new if cached is None else pd.concat([cached, new], ignore_index=True)
        df.to_parquet(path, index=False)
        cov = df.groupby("ticker")["date"].agg(["min", "max", "count"])
        missing = sorted(set(tickers) - set(cov.index))
        manifest["files"][path.name] = {"rows": len(df), "missing": missing}
        print(f"wrote {path.name}: {len(df):,} rows, {len(cov)} tickers, missing: {missing or 'none'}")
        cov.to_csv(OUT / f"coverage_{name}.csv")

    if manifest["files"]:
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
