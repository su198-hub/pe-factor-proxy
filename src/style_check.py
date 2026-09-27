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
