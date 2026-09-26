"""Macro sensitivity of PERT, GIC PERT, World IMI, the components and the NA blocks.

Two kinds of driver, each at its natural frequency:

A. Market-priced drivers (monthly, from attribution.MACRO): real yield, breakeven, HY
   spread, economic surprise, dollar. Same-month changes; multivariate OLS; Newey-West
   (HAC) t-stats.

B. Economic data (quarterly): US real GDP growth (QoQ annualised), change in US CPI
   inflation (YoY, change over the quarter), change in the ISM manufacturing index.
   1. Lead-lag: correlation of the quarter-t return with GDP growth / inflation change in
      quarter t+k, k = -4..+4. k > 0 means returns move BEFORE the data (markets price
      expected growth); k < 0 means after.
   2. Regressions, HAC t-stats:
        same quarter : r_t = a + b_g * GDP_t          + b_pi * dCPI_t + e
        forward      : r_t = a + b_g * mean(GDP_t+1..t+4) + b_pi * dCPI_t + e
      The forward spec uses realised future growth as a stand-in for what the market
      expected; it is not a forecast (it uses hindsight).
   3. Regimes: each quarter classified by whether GDP YoY growth and CPI YoY inflation rose
      or fell over the quarter; average annualised return in each of the four regimes.

Caveats: macro data are the latest revised vintage, not what was known at the time; ~78
quarters and a handful of independent cycles; US drivers applied to all regions.
These are descriptive co-movements for stress testing, not causal estimates and not
inputs to expected returns (those come from the structural build, brief section 7.3).

Needs the replica's world_monthly.csv for the chosen Core (run replica.py --core X first).
Output: output/macro_sens/ (IMI Core) or output/macro_sens_{core}/.
"""
import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

from attribution import MACRO, macro_changes, sensitivities
from blocks import BLOCKS
from data import ROOT, pert_family, prices, returns
from pe_core import VARIANTS, component_blocks

OUT = ROOT / "output" / "macro_sens"
START = "2006-12-01"
LAGS = range(-4, 5)


def load_series(core):
    """Monthly returns: World series, six components, and GIC PERT's NA buyout / VC blocks."""
    rm = returns("monthly")
    rep = ROOT / "output" / ("replica" if core == "imi" else f"replica_{core}") / "world_monthly.csv"
    world = pd.read_csv(rep, index_col=0, parse_dates=True)
    # static replica: covers the full period incl. the GFC (dynamic starts Dec 2008)
    s = {k: world[k] for k in ["MSCI PERT", "GIC PERT static", "MSCI World IMI"]}
    s.update({c: rm[t["net"]] for c, t in pert_family().items()})
    blocks = component_blocks(core, "monthly")
    for comp in ["NA_BO", "NA_VC"]:
        s.update({f"{comp} {b}": blocks[comp][b] for b in BLOCKS})
    return pd.DataFrame(s).loc[START:]


def quarterly(monthly):
    """Compound monthly returns to calendar quarters (full quarters only)."""
    q = (1 + monthly).resample("QE").prod(min_count=3) - 1
    return q


def econ_quarterly():
    px = prices("monthly")
    q = pd.DataFrame({
        "gdp": px["GDP CQOQ Index"].resample("QE").last(),                     # % QoQ annualised
        "gdp_yoy": px["GDP CYOY Index"].resample("QE").last(),
        "cpi_yoy": px["CPI YOY Index"].resample("QE").last(),
        "ism": px["NAPMPMI Index"].resample("QE").last(),
    })
    q["d_cpi"] = q["cpi_yoy"].diff()                                           # pp change in inflation
    q["d_ism"] = q["ism"].diff()
    q["gdp_fwd4"] = q["gdp"].shift(-1).rolling(4).mean().shift(-3)             # mean growth t+1..t+4
    q["growth_up"] = q["gdp_yoy"].diff() > 0
    q["infl_up"] = q["d_cpi"] > 0
    return q


def lead_lag(qr, x):
    return pd.DataFrame({k: qr.apply(lambda s: s.corr(x.shift(-k))) for k in LAGS})


def regress(qr, econ, xcols):
    rows = []
    for name, y in qr.items():
        d = pd.concat([y.rename("y"), econ[xcols]], axis=1).dropna()
        m = sm.OLS(d["y"], sm.add_constant(d[xcols])).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
        rows.append({"series": name, "n": int(m.nobs), "r2": m.rsquared,
                     **{f"{x}": f"{m.params[x] * 100:+.1f} ({m.tvalues[x]:+.1f})" for x in xcols}})
    return pd.DataFrame(rows).set_index("series")


def regimes(qr, econ):
    lab = np.select([econ["growth_up"] & ~econ["infl_up"], econ["growth_up"] & econ["infl_up"],
                     ~econ["growth_up"] & econ["infl_up"], ~econ["growth_up"] & ~econ["infl_up"]],
                    ["growth up, inflation down", "growth up, inflation up",
                     "growth down, inflation up", "growth down, inflation down"], default="")
    lab = pd.Series(lab, index=econ.index)
    d = qr.join(lab.rename("regime"), how="inner")
    d = d[d["regime"] != ""]
    ann = d.groupby("regime").mean() * 4 * 100
    ann.insert(0, "quarters", d.groupby("regime").size())
    return ann.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", choices=VARIANTS, default="gic")
    core = ap.parse_args().core
    out = OUT if core == "imi" else OUT.parent / f"macro_sens_{core}"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:+.2f}".format)
    m = load_series(core)
    print(f"Core: {core}")
    rm = returns("monthly")

    # A. market-priced drivers, monthly
    X = macro_changes(rm)
    world = ["MSCI PERT", "GIC PERT static", "MSCI World IMI"]
    world_dyn = pd.read_csv(ROOT / "output" / ("replica" if core == "imi" else f"replica_{core}") / "world_monthly.csv",
                            index_col=0, parse_dates=True)["GIC PERT dynamic"]
    sens = sensitivities({**{k: m[k] for k in world}, "GIC PERT dynamic (from Dec 2008)": world_dyn}, X)
    sens.to_csv(out / "market_drivers_world.csv")
    print("A. Market-priced drivers: % return per shock, same month (HAC t). Shocks: "
          + "; ".join(f"{k} = {v[2]}" for k, v in MACRO.items()))
    print(pd.DataFrame({k: [f"{b * 100:+.1f} ({t:+.1f})" for b, t in zip(sens[k], sens[f't_{k}'])] for k in MACRO},
                       index=sens.index).assign(r2=sens["r2"].round(2), n=sens["n"]).to_string())

    # B. economic data, quarterly
    qr = quarterly(m)
    econ = econ_quarterly()
    qr, econ = qr.loc[qr.index.intersection(econ.index)], econ
    show = world + ["NA_BO", "NA_VC", "EME_BO", "EME_VC"] + [f"NA_BO {b}" for b in BLOCKS] + ["NA_VC core", "NA_VC growth"]

    print(f"\nB1. Lead-lag: corr(return in quarter t, macro in quarter t+k). k>0: returns move before the data. "
          f"{qr.index[0].date()}..{qr.index[-1].date()}")
    for key, label in [("gdp", "US real GDP growth (QoQ annualised)"), ("d_cpi", "Change in US CPI inflation"),
                       ("d_ism", "Change in ISM manufacturing")]:
        ll = lead_lag(qr[show], econ[key])
        ll.to_csv(out / f"leadlag_{key}.csv")
        print(f"\n{label}")
        print(ll.round(2).to_string())

    print("\nB2. Quarterly regressions: % quarterly return per +1pp (HAC t)")
    for label, cols in [("same quarter: GDP growth_t, change in CPI inflation_t", ["gdp", "d_cpi"]),
                        ("forward: mean GDP growth over next 4 quarters, change in CPI inflation_t", ["gdp_fwd4", "d_cpi"])]:
        r = regress(qr[show], econ, cols)
        r.to_csv(out / f"regress_{cols[0]}.csv")
        print(f"\n{label}")
        print(r.to_string())

    reg = regimes(qr[show], econ)
    reg.to_csv(out / "regimes.csv")
    print("\nB3. Growth / inflation regimes (quarter's change in GDP YoY and CPI YoY): average return, % annualised")
    print(reg.round(1).to_string())


if __name__ == "__main__":
    main()
