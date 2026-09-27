"""Reported PE (Preqin closed-end fund indexes) vs the listed trackers.

Preqin indexes (quarterly TWR, USD, net of fees, appraisal-based) are exported by hand from
Preqin Pro into data/preqin/ (git-ignored). Each is paired with the tracker that represents
the same slice of PE, and with the public market:

    Preqin Global PE & VC, Global PE   MSCI PERT (World), Proxy PERT        World IMI
    Preqin North America PE            MSCI NA buyout + VC trackers          NA IMI
    Preqin North America PE - Growth   MSCI NA VC tracker (MSCI groups growth with VC)
    Preqin Europe PE, Europe Buyout    MSCI Europe & ME buyout (+ VC) trackers   Europe IMI
    Preqin Asia PE                     MSCI Pacific trackers (developed only; Preqin Asia
                                       also holds emerging Asia, so a loose match)  Pacific IMI

Regional blends use PERT's average component weights (NA 62/38, Europe 85/15 buyout/VC).

Questions answered, on a common sample (Q1 2007 - Q1 2026):
    1. Return level: how much of reported PE's lead over the market does the tracker capture?
    2. Smoothing: persistence (AR1), volatility as reported vs desmoothed (Geltner), drawdown.
    3. Timing: how a tracker move passes into reported returns over the next quarters.
    4. Nowcasting, out of sample: predict each quarter's reported return from the tracker's
       current and lagged returns, fitted only on earlier quarters; compare with using the
       plain market (IMI) instead, and with naive forecasts.
Output: output/preqin/.
"""
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm

from data import ROOT, pert_family, returns

OUT = ROOT / "output" / "preqin"
SRC = ROOT / "data" / "preqin"
START, END = "2007-01-01", "2026-03-31"
LAGS = 3
NOWCAST_FROM = "2012-03-31"


def load_preqin():
    out = {}
    for f in sorted(SRC.glob("*PerformanceHistory*.xlsx")):
        name = re.sub(r"^Preqin_Preqin|Closed-EndFundIndex.*$", "", f.stem)
        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("&", " & ").replace("-", " - ")
        p = pd.read_excel(f, sheet_name="Performance")
        s = p.set_index(pd.to_datetime(p["Date"]))["Performance Since Inception"].sort_index().pct_change().dropna()
        s.index = s.index + pd.offsets.QuarterEnd(0)
        out[f"Preqin {re.sub(r'\\s+', ' ', name).strip()}"] = s
    return pd.DataFrame(out)


def trackers():
    rm = returns("monthly")
    fam = pert_family()
    q = lambda s: (1 + s).resample("QE").prod(min_count=3) - 1
    w = pd.read_csv(ROOT / "output" / "replica_proxy" / "world_monthly.csv", index_col=0, parse_dates=True)
    c = {k: q(rm[v["net"]]) for k, v in fam.items()}
    return pd.DataFrame({
        "MSCI PERT": q(rm["MXWOPERT Index"]), "Proxy PERT": q(w["Proxy PERT static"]), "World IMI": q(rm["M1WOIM Index"]),
        "NA trackers": 0.62 * c["NA_BO"] + 0.38 * c["NA_VC"], "NA VC tracker": c["NA_VC"], "NA IMI": q(rm["M1NAIM Index"]),
        "Europe trackers": 0.85 * c["EME_BO"] + 0.15 * c["EME_VC"], "Europe buyout tracker": c["EME_BO"],
        "Europe IMI": q(rm["MIMUEURN Index"]),
        "Pacific trackers": 0.5 * c["PAC_BO"] + 0.5 * c["PAC_VC"], "Pacific IMI": q(rm["M1PCIM Index"]),
    })


PAIRS = [  # (Preqin index, tracker, public market)
    ("Preqin Global Private Equity & Venture Capital", "MSCI PERT", "World IMI"),
    ("Preqin Global Private Equity", "MSCI PERT", "World IMI"),
    ("Preqin Global Private Equity & Venture Capital", "Proxy PERT", "World IMI"),
    ("Preqin North America Private Equity", "NA trackers", "NA IMI"),
    ("Preqin North America Private Equity - Growth", "NA VC tracker", "NA IMI"),
    ("Preqin Europe Private Equity", "Europe trackers", "Europe IMI"),
    ("Preqin Europe Private Equity - Buyout", "Europe buyout tracker", "Europe IMI"),
    ("Preqin Asia Private Equity", "Pacific trackers", "Pacific IMI"),
]


def ann(s):
    return ((1 + s).prod() ** (4 / len(s)) - 1) * 100


def lag_fit(y, x, lags=LAGS):
    X = pd.concat({f"lag{k}": x.shift(k) for k in range(lags + 1)}, axis=1)
    z = pd.concat([y.rename("y"), X], axis=1).dropna()
    return sm.OLS(z["y"], sm.add_constant(z.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": 2}), z


def nowcast(y, x, lags=2):
    """Expanding-window out-of-sample prediction of y_t from x_t..x_{t-lags}."""
    X = pd.concat({f"lag{k}": x.shift(k) for k in range(lags + 1)}, axis=1)
    z = pd.concat([y.rename("y"), X], axis=1).dropna()
    preds = {}
    for t in z.index[z.index >= NOWCAST_FROM]:
        tr = z.loc[z.index < t]
        m = sm.OLS(tr["y"], sm.add_constant(tr.drop(columns="y"))).fit()
        preds[t] = float(m.predict(sm.add_constant(z.drop(columns="y"), has_constant="add").loc[[t]]).iloc[0])
    p = pd.Series(preds)
    act = z.loc[p.index, "y"]
    hist_mean = pd.Series({t: z.loc[z.index < t, "y"].mean() for t in p.index})
    rmse = lambda e: np.sqrt((e ** 2).mean()) * 100
    return {"OOS R2 vs hist. mean": 1 - ((act - p) ** 2).sum() / ((act - hist_mean) ** 2).sum(),
            "RMSE %": rmse(act - p), "RMSE, last quarter's return %": rmse(act - y.shift(1).loc[p.index]),
            "RMSE, hist. mean %": rmse(act - hist_mean), "corr(pred, actual)": p.corr(act), "quarters": len(p)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 260)
    pd.set_option("display.float_format", "{:.2f}".format)
    P, T = load_preqin(), trackers()
    D = P.join(T, how="inner").loc[START:END]
    print(f"Common sample {D.index[0].date()}..{D.index[-1].date()} ({len(D)} quarters). Preqin series: {list(P.columns)}")

    lvl, smooth, lag, nc = [], [], [], []
    for pe, tr, mk in PAIRS:
        d = D[[pe, tr, mk]].dropna()
        a_pe, a_tr, a_mk = ann(d[pe]), ann(d[tr]), ann(d[mk])
        row = {"PE index": pe, "tracker": tr, "market": mk, "PE %/yr": a_pe, "tracker %/yr": a_tr, "market %/yr": a_mk,
               "PE - market": a_pe - a_mk, "tracker - market": a_tr - a_mk,
               "capture %": (a_tr - a_mk) / (a_pe - a_mk) * 100 if abs(a_pe - a_mk) > 0.3 else np.nan}
        for lab, lo, hi in [("07-14", "2007", "2014"), ("15-21", "2015", "2021"), ("22-26", "2022", "2026")]:
            x = d.loc[lo:hi]
            row[f"PE - tracker {lab}"] = ann(x[pe]) - ann(x[tr])
        lvl.append(row)

        lam = d[pe].autocorr(1)
        ds = (d[pe] - lam * d[pe].shift(1)) / (1 - lam)
        dd = lambda s: (((1 + s).cumprod() / (1 + s).cumprod().cummax()) - 1).min() * 100
        smooth.append({"PE index": pe, "AR(1) PE": lam, "AR(1) tracker": d[tr].autocorr(1), "vol PE, reported %": d[pe].std() * 200,
                       "vol PE, desmoothed %": ds.std() * 200, "vol tracker %": d[tr].std() * 200,
                       "max DD PE %": dd(d[pe]), "max DD tracker %": dd(d[tr])})

        m, _ = lag_fit(d[pe], d[tr])
        lag.append({"PE index": pe, "tracker": tr, **{f"lag {k}": m.params[f"lag{k}"] for k in range(LAGS + 1)},
                    "pass-through within a year": m.params[[f"lag{k}" for k in range(LAGS + 1)]].sum(), "R2": m.rsquared})

        n_tr = nowcast(d[pe], d[tr])
        n_mk = nowcast(d[pe], d[mk])
        nc.append({"PE index": pe, "tracker": tr, **{f"{k} (tracker)": v for k, v in n_tr.items() if k != "quarters"},
                   "OOS R2 (market instead)": n_mk["OOS R2 vs hist. mean"], "RMSE % (market instead)": n_mk["RMSE %"],
                   "quarters": n_tr["quarters"]})

    for name, rows in [("levels", lvl), ("smoothing", smooth), ("lags", lag), ("nowcast", nc)]:
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"\n== {name}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
