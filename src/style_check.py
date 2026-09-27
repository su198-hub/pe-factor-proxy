"""Are the style tilts' small (slightly negative) contributions to PERT's lead believable?

Three checks:
1. Each tilt's own return (long leg minus the PE-matched Core), its beta to the Core, and
   its beta-adjusted ("pure style") return, by period. A long-only style index is not
   market-neutral: min vol has beta < 1, small caps > 1, so in a bull market part of a
   tilt's return is a hidden market bet.
2. The same periods in Fama-French's published factor returns (Ken French data library):
   value (HML), size (SMB), profitability (RMW, the Quality side of our leverage proxy),
   investment (CMA) and momentum (WML/UMD), for the US, North America, Europe, Developed.
3. The tilts' contribution to MSCI PERT's lead over World IMI (Proxy PERT dynamic weights),
   split into the beta part (weight x beta x Core) and the pure-style part.

Output: output/style_check/.
"""
import io
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm

from blocks import TILTS
from data import ROOT
from pe_core import component_blocks

OUT = ROOT / "output" / "style_check"
FF = ROOT / "data" / "ff"
PERIODS = [("2007-2026", "2007-01", None), ("2009-2026", "2009-01", None), ("2007-2014", "2007-01", "2014-12"),
           ("2015-2021", "2015-01", "2021-12"), ("2022-2026", "2022-01", None)]
FF_FILES = {"US": ("F-F_Research_Data_5_Factors_2x3_CSV", "F-F_Momentum_Factor_CSV"),
            "North America": ("North_America_5_Factors_CSV", "North_America_Mom_Factor_CSV"),
            "Europe": ("Europe_5_Factors_CSV", "Europe_Mom_Factor_CSV"),
            "Developed": ("Developed_5_Factors_CSV", "Developed_Mom_Factor_CSV")}


def read_ff(name):
    """Monthly table from a Ken French zip (percent -> decimal)."""
    with zipfile.ZipFile(FF / f"{name}.zip") as z:
        txt = z.read(z.namelist()[0]).decode("latin-1")
    lines = txt.splitlines()
    is_row = lambda l: (p := [x.strip() for x in l.split(",")]) and len(p) > 1 and p[0].isdigit() and len(p[0]) == 6
    first = next(i for i, l in enumerate(lines) if is_row(l))  # first monthly data row
    header = next(lines[j] for j in range(first - 1, -1, -1) if "," in lines[j])
    rows = []
    for l in lines[first:]:
        if not is_row(l):
            break
        rows.append([x.strip() for x in l.split(",")])
    cols = [c.strip() for c in header.split(",")][1:len(rows[0])]
    df = pd.DataFrame(rows, columns=["date"] + cols).set_index("date").astype(float) / 100
    df.index = pd.to_datetime(df.index, format="%Y%m") + pd.offsets.MonthEnd(0)
    return df


def ff_table():
    out = {}
    for reg, (f5, fm) in FF_FILES.items():
        d = read_ff(f5).join(read_ff(fm), how="inner")
        d.columns = [c.replace("WML", "Mom").strip() for c in d.columns]
        out[reg] = d
    return out


def ann(s):
    return s.mean() * 12 * 100


def tilt_stats(blocks, comp):
    b = blocks[comp]
    rows = []
    for lab, lo, hi in PERIODS:
        x = b.loc[lo:hi].dropna()
        for t in TILTS:
            m = sm.OLS(x[t], sm.add_constant(x["core"])).fit()
            rows.append({"component": comp, "period": lab, "tilt": t, "tilt return %/yr": ann(x[t]),
                         "beta to core": m.params["core"], "pure style %/yr": m.params["const"] * 12 * 100,
                         "core %/yr": ann(x["core"])})
    return pd.DataFrame(rows)


def contributions(blocks):
    """Tilt contributions to MSCI PERT's lead over World IMI, split into beta and pure-style parts."""
    W = pd.read_csv(ROOT / "output" / "replica_proxy" / "weights_dynamic.csv", header=[0, 1], index_col=0, parse_dates=True)
    comps = [c for c in W.columns.get_level_values(0).unique() if c in blocks]
    rows = []
    for t in TILTS:
        beta_part = pure_part = 0.0
        tot = 0.0
        for c in comps:
            b = blocks[c].reindex(W.index)
            x = b.dropna()
            m = sm.OLS(x[t], sm.add_constant(x["core"])).fit()
            w = W[("pert_cw", c)] * W[(c, t)]
            contrib = (w * b[t]).dropna()
            bpart = (w * m.params["core"] * b["core"]).dropna()
            tot += contrib.sum()
            beta_part += bpart.sum()
        yrs = len(W) / 12
        rows.append({"tilt": t, "contribution %/yr": tot / yrs * 100, "of which beta (market) part": beta_part / yrs * 100,
                     "of which pure style part": (tot - beta_part) / yrs * 100})
    return pd.DataFrame(rows).set_index("tilt")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:+.2f}".format)
    blocks = component_blocks("proxy", "monthly")
    ts = pd.concat([tilt_stats(blocks, c) for c in ["NA_BO", "EME_BO"]])
    ts.to_csv(OUT / "tilt_stats.csv", index=False)
    for c in ["NA_BO", "EME_BO"]:
        print(f"\n{c}: tilt returns (long leg minus PE-matched Core), beta to Core, beta-adjusted return")
        sub = ts[ts["component"] == c]
        print(sub.pivot(index="tilt", columns="period", values="tilt return %/yr")[[p for p, _, _ in PERIODS]].to_string())
        print("beta to core (2007-2026):", sub[sub["period"] == "2007-2026"].set_index("tilt")["beta to core"].round(2).to_dict())
        print(sub.pivot(index="tilt", columns="period", values="pure style %/yr")[[p for p, _, _ in PERIODS]].add_prefix("pure ").to_string())

    ff = ff_table()
    rows = []
    for reg, d in ff.items():
        for lab, lo, hi in PERIODS:
            x = d.loc[lo:hi]
            rows.append({"region": reg, "period": lab, **{k: ann(x[k]) for k in ["SMB", "HML", "RMW", "CMA", "Mom"] if k in x}})
    ffs = pd.DataFrame(rows).set_index(["region", "period"])
    ffs.to_csv(OUT / "fama_french.csv")
    print("\nFama-French factor returns, %/yr (long-short, published): SMB size, HML value, RMW profitability, CMA investment, Mom momentum")
    print(ffs.to_string())

    con = contributions(blocks)
    con.to_csv(OUT / "tilt_contributions.csv")
    print("\nTilt contributions to MSCI PERT's lead over World IMI (Proxy PERT dynamic weights, Dec 2008-Aug 2026), %/yr")
    print(con.to_string())
    print(con.sum().to_frame("total").T.to_string())


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Leverage proxy diagnostic
# ---------------------------------------------------------------------------
AQR = {"BAB": "Betting-Against-Beta-Equity-Factors-Monthly", "QMJ": "Quality-Minus-Junk-Factors-Monthly"}


def read_aqr(name, col="USA"):
    d = pd.read_excel(FF / f"{AQR[name]}.xlsx", sheet_name=0, header=18)
    d = d[pd.to_datetime(d["DATE"], errors="coerce").notna()]
    s = pd.Series(d[col].astype(float).values, index=pd.to_datetime(d["DATE"]) + pd.offsets.MonthEnd(0), name=name)
    return s


def leverage_check(start="2007-01", end=None):
    """What do the candidate leverage tilts load on? Returns by period and factor loadings."""
    from data import returns
    rm = returns("monthly")
    cands = {
        "ours: NA IMI - USA Quality": rm["M1NAIM Index"] - rm["M1USQU Index"],
        "Barra: NA IMI - USA Barra Low Leverage": rm["M1NAIM Index"] - rm["M00JUSSO Index"],
        "JPM: high - low leverage (L/S)": -rm["JPSFLEVU Index"],
        "GS: weak - strong balance sheet (L/S)": -rm["GSPRLEVR Index"],
    }
    ff = ff_table()["US"]
    F = pd.concat([ff[["Mkt-RF", "SMB", "HML", "Mom"]], read_aqr("QMJ"), read_aqr("BAB")], axis=1)
    rows, load = [], []
    for name, s in cands.items():
        row = {"candidate": name}
        for lab, lo, hi in PERIODS:
            row[lab] = ann(s.loc[lo:hi].dropna())
        rows.append(row)
        z = pd.concat([s.rename("y"), F], axis=1).loc[start:end].dropna()
        m = sm.OLS(z["y"], sm.add_constant(z[F.columns])).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        load.append({"candidate": name, **{k: f"{m.params[k]:+.2f} ({m.tvalues[k]:+.1f})" for k in F.columns},
                     "alpha %/yr": m.params["const"] * 1200, "R2": m.rsquared, "months": int(m.nobs)})
    facs = pd.DataFrame({lab: {k: ann(F[k].loc[lo:hi].dropna()) for k in ["QMJ", "BAB"]} for lab, lo, hi in PERIODS}).T
    return pd.DataFrame(rows).set_index("candidate"), pd.DataFrame(load).set_index("candidate"), facs


# ---------------------------------------------------------------------------
# Reconciliation to MSCI's published risk/return table (Rethinking Access to PE, p.14)
# ---------------------------------------------------------------------------
def reconcile(start="2006-12-31", end="2025-03-31"):
    from data import returns
    rm = returns("monthly")
    w = pd.read_csv(ROOT / "output" / "replica_proxy" / "world_monthly.csv", index_col=0, parse_dates=True)
    S = pd.DataFrame({"MSCI PERT": rm["MXWOPERT Index"], "Proxy PERT (fixed weights)": w["Proxy PERT static"],
                      "Proxy PERT (rebalanced, from Dec 2008)": w["Proxy PERT dynamic"], "World IMI (net)": rm["M1WOIM Index"]})
    S = S.loc[pd.Timestamp(start) + pd.offsets.MonthEnd(1):end]
    end_t = S.index[-1]
    rows = {}
    for c in S:
        s = S[c].dropna()
        r = {}
        for lab, yrs in [("1Y", 1), ("3Y", 3), ("5Y", 5), ("10Y", 10)]:
            x = s.loc[end_t - pd.DateOffset(years=yrs) + pd.offsets.MonthEnd(0) + pd.offsets.MonthEnd(1):]
            r[lab] = ((1 + x).prod() ** (12 / len(x)) - 1) * 100 if len(x) >= yrs * 12 - 1 else np.nan
        r["Full"] = ((1 + s).prod() ** (12 / len(s)) - 1) * 100
        q = (1 + s).resample("QE").prod() - 1
        r["Risk (quarterly)"] = q.std() * 200
        r["Risk (monthly)"] = s.std() * np.sqrt(12) * 100
        r["Return/Risk"] = r["Full"] / r["Risk (quarterly)"]
        wealth = (1 + s).cumprod()
        r["Max DD"] = (wealth / wealth.cummax() - 1).min() * 100
        r["Starts"] = str(s.index[0].date())
        rows[c] = r
    return pd.DataFrame(rows).T
