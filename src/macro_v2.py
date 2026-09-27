"""Macro sensitivities v2: returns vs changes in *expectations*, not realised macro data.

Markets price expected growth and inflation, so realised GDP in the same quarter says
little (shown for contrast). Drivers, as same-month changes (monthly, Jan 2007 - Aug 2026):

    growth expectations   revision of MSCI World 12m-forward EPS (%), OECD G7 leading
                          indicator (change), ISM new orders (change)
    inflation expectations 5y5y forward breakeven (change, pp)
    rates                 10y real yield, 2y nominal yield (policy path) (change, pp)
    credit / conditions   high-yield OAS (pp), Chicago Fed NFCI (change)
    other                 US dollar index (%), Brent (%)

Series explained: MSCI PERT, Proxy PERT, World IMI, MSCI PERT minus World IMI (what makes
PE-like equity behave differently from the market), and the NA/Europe buyout and VC
components.

Methods (sensitivities are per one-standard-deviation monthly move of the driver, so their
sizes are comparable; t-stats are Newey-West):
    A. univariate: each driver alone (beta, correlation)
    B. multivariate: a parsimonious set together (growth-expectations revision, 5y5y,
       real yield, 2y, HY spread, dollar, oil)
    C. quarterly: SPF expected real GDP growth over the next four quarters and SPF next-year
       CPI - revisions vs realised same-quarter GDP
    D. regimes: months classified by the 3-month direction of growth expectations (G7 leading
       indicator) and inflation expectations (5y5y): average annualised returns

Co-movement, not causation: spreads widen partly because equities fall. OECD leading
indicators are published with about a one-month lag and revised; SPF is quarterly.
Output: output/macro_v2/.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import macrobond_pull as mb
from data import ROOT, panel, pert_family, prices, returns

OUT = ROOT / "output" / "macro_v2"
START, END = "2007-01-01", None
MULTI = ["fwd_eps_rev", "be_5y5y", "real_yield", "ust_2y", "hy_spread", "dollar", "oil"]
LABEL = {"fwd_eps_rev": "World fwd EPS revision (%)", "cli_g7": "G7 leading indicator", "ism_new_orders": "ISM new orders",
         "be_5y5y": "5y5y breakeven (pp)", "real_yield": "US 10y real yield (pp)", "ust_2y": "US 2y yield (pp)",
         "hy_spread": "US HY OAS (pp)", "nfci": "Chicago Fed NFCI", "dollar": "US dollar index (%)", "oil": "Brent (%)"}


def drivers():
    px = prices("monthly")
    mbm = mb.monthly()
    fpe = panel("fpe_monthly")["MXWO Index"]
    fwd_eps = (px["MXWO Index"] / fpe).where(fpe > 0)
    d = pd.DataFrame({
        "fwd_eps_rev": np.log(fwd_eps).diff() * 100,
        "cli_g7": mbm["cli_g7"].diff(),
        "ism_new_orders": mbm["ism_new_orders"].diff(),
        "be_5y5y": mbm["be_5y5y"].diff(),
        "real_yield": px["USGGT10Y Index"].diff(),
        "ust_2y": mbm["ust_2y"].diff(),
        "hy_spread": px["LF98OAS Index"].diff(),
        "nfci": mbm["nfci"].diff(),
        "dollar": px["DXY Index"].pct_change(fill_method=None) * 100,
        "oil": mbm["brent"].pct_change(fill_method=None) * 100,
    })
    return d


def targets():
    rm = returns("monthly")
    fam = pert_family()
    w = pd.read_csv(ROOT / "output" / "replica_proxy" / "world_monthly.csv", index_col=0, parse_dates=True)
    t = pd.DataFrame({"MSCI PERT": rm["MXWOPERT Index"], "Proxy PERT": w["Proxy PERT static"], "World IMI": rm["M1WOIM Index"]})
    t["PERT minus IMI"] = t["MSCI PERT"] - t["World IMI"]
    for c in ["NA_BO", "NA_VC", "EME_BO", "EME_VC"]:
        t[c] = rm[fam[c]["net"]]
    return t


def hac(y, X):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})


def univariate(T, D):
    rows = []
    for d in D:
        x = D[d] / D[d].std()
        for s in T:
            z = pd.concat([T[s].rename("y"), x.rename("x")], axis=1).dropna()
            m = hac(z["y"], z[["x"]])
            rows.append({"driver": LABEL[d], "series": s, "beta": m.params["x"] * 100, "t": m.tvalues["x"], "corr": z["y"].corr(z["x"])})
    return pd.DataFrame(rows)


def multivariate(T, D):
    X = D[MULTI] / D[MULTI].std()
    out = {}
    for s in T:
        z = pd.concat([T[s].rename("y"), X], axis=1).dropna()
        m = hac(z["y"], z[MULTI])
        out[s] = {**{LABEL[k]: f"{m.params[k] * 100:+.2f} ({m.tvalues[k]:+.1f})" for k in MULTI}, "R2": round(m.rsquared, 2), "n": int(m.nobs)}
    return pd.DataFrame(out).T


def quarterly(T):
    q = (1 + T).resample("QE").prod(min_count=3) - 1
    q["PERT minus IMI"] = q["MSCI PERT"] - q["World IMI"]
    mq = mb.quarterly()
    exp_g = mq[["spf_gdp_q1", "spf_gdp_q2", "spf_gdp_q3", "spf_gdp_q4"]].mean(axis=1)
    px = prices("monthly")
    X = pd.DataFrame({"SPF expected growth, revision": exp_g.diff(), "SPF next-year CPI, revision": mq["spf_cpi_next_year"].diff(),
                      "realised GDP growth, same quarter": px["GDP CQOQ Index"].resample("QE").last()})
    rows = []
    for s in ["MSCI PERT", "Proxy PERT", "World IMI", "PERT minus IMI", "NA_BO", "NA_VC"]:
        for spec in [["SPF expected growth, revision", "SPF next-year CPI, revision"], ["realised GDP growth, same quarter"]]:
            z = pd.concat([q[s].rename("y"), X[spec]], axis=1).loc[START:].dropna()
            m = sm.OLS(z["y"], sm.add_constant(z[spec])).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
            rows.append({"series": s, **{k: f"{m.params[k] * 100:+.2f} ({m.tvalues[k]:+.1f})" for k in spec}, "R2": round(m.rsquared, 2), "n": int(m.nobs)})
    return pd.DataFrame(rows).set_index("series")


def regimes(T):
    mbm = mb.monthly()
    g = mbm["cli_g7"].diff(3) > 0
    i = mbm["be_5y5y"].diff(3) > 0
    lab = pd.Series(np.select([g & ~i, g & i, ~g & i, ~g & ~i],
                              ["growth exp. up, inflation exp. down", "growth exp. up, inflation exp. up",
                               "growth exp. down, inflation exp. up", "growth exp. down, inflation exp. down"], ""), index=mbm.index)
    z = T.join(lab.rename("regime"), how="inner").loc[START:]
    z = z[z["regime"] != ""]
    out = (z.groupby("regime").mean() * 12 * 100).T
    out.loc["months"] = z.groupby("regime").size()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 260)
    pd.set_option("display.float_format", "{:+.2f}".format)
    T = targets().loc[START:END]
    D = drivers().reindex(T.index)
    print(f"Sample {T.index[0].date()}..{T.index[-1].date()}; driver s.d. (1-sd monthly move):")
    print((D.std()).rename(index=LABEL).round(2).to_string())

    u = univariate(T, D)
    u.to_csv(OUT / "univariate.csv", index=False)
    print("\nA. Univariate: % return per 1-sd monthly move in the driver (HAC t)")
    tab = u.assign(v=u.apply(lambda r: f"{r['beta']:+.2f} ({r['t']:+.1f})", axis=1)).pivot(index="driver", columns="series", values="v")
    print(tab[list(T.columns)].to_string())

    m = multivariate(T, D)
    m.to_csv(OUT / "multivariate.csv")
    print("\nB. Multivariate: % return per 1-sd move, all drivers together (HAC t)")
    print(m.T.to_string())

    qd = quarterly(T)
    qd.to_csv(OUT / "quarterly_spf.csv")
    print("\nC. Quarterly: % quarterly return per 1pp change (HAC t). SPF revisions vs realised same-quarter GDP")
    print(qd.to_string())

    rg = regimes(T)
    rg.to_csv(OUT / "regimes.csv")
    print("\nD. Regimes by 3-month direction of growth expectations (G7 leading indicator) and inflation expectations (5y5y): % annualised")
    print(rg.round(1).to_string())


if __name__ == "__main__":
    main()
