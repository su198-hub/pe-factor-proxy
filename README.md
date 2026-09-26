# pe-factor-proxy

Factor-based public proxy for a PE portfolio, following the MSCI PE Return Tracker (PERT) methodology.

**Milestone 1:** map macro drivers onto PERT (public data only).

## Layout

```
config/     ticker registry and settings (committed)
data/       raw vendor pulls - read-only, never committed
interim/    cleaned parquet cache - never committed
output/     tables, charts, exports - never committed
notebooks/  exploration only; logic lives in src/
src/        pull, construction, attribution and scenario code
```

Vendor data (Bloomberg, MSCI) is licensed and must not be pushed to any remote.

## Setup (Windows)

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Bloomberg Terminal must be running and logged in on the same machine (Desktop API, localhost:8194).
Check with `.venv\Scripts\python.exe -u src\bbg_smoke_test.py`.
If console printing fails with `UnicodeEncodeError`, set `PYTHONIOENCODING=utf-8`.
