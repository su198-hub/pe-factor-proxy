"""Historical attribution of the PERT components: by building block and by macro driver.

1. Block attribution (monthly returns, full-sample weights from style_decomp):

       r_comp,t = b * r_market,t + sum_i w_i * tilt_i,t + unexplained_t

   Summed over a period this is exactly additive (sums of monthly returns, close to log
   returns; not compounded). The market term is split further, using the standard
   regional index's valuation data:

       r_market = income + earnings growth (local ccy) + currency + multiple change + IMI gap

   income   = net minus price return of the standard index (dividends after withholding)
   earnings = change in implied trailing EPS (price / trailing P/E), local currency
   currency = USD net minus local-currency net return
   multiple = change in trailing P/E
   IMI gap  = regional IMI (the block) minus the standard index (small caps, mostly)

   The log price change splits exactly into ln EPS + ln P/E; the three log pieces are
   scaled by (simple / log) price return so they add up to the simple return.

2. Macro sensitivities (monthly, same-period changes, HAC t-stats):

       r_t = a + sum_k beta_k * dmacro_k,t + e_t

   for every block and every component. These are co-movements for stress testing,
   not forecasts, and not causal: spreads widen partly because equities fall. The
   component's beta minus sum_b w_b * beta_b is the macro exposure the blocks miss.

Output: output/attribution/.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from blocks import BLOCKS, TILTS, block_returns
from data import ROOT, panel, pert_family, prices, returns
from sectors import NET, parent_ticker
from style_decomp import full_sample

OUT = ROOT / "output" / "attribution"
START = "2006-12-01"
LOCAL = {"NA": "NDDLNA Index", "EME": "NDDLE15 Index", "PAC": "NDDLP Index"}
EPISODES = [("GFC", "2007-11", "2009-02"), ("Recovery and QE", "2009-03", "2019-12"),
            ("COVID and stimulus", "2020-01", "2021-12"), ("2022 rate shock", "2022-01", "2022-12"),
            ("2023 on", "2023-01", None)]
MARKET_PARTS = ["income", "earnings", "currency", "multiple", "imi_gap"]
# macro driver -> (ticker, transform, unit label)
MACRO = {
    "real_yield": ("USGGT10Y Index", "diff", "+100bp US 10y real yield"),
    "breakeven": ("USGGBE10 Index", "diff", "+100bp US 10y breakeven"),
    "hy_spread": ("LF98OAS Index", "diff", "+100bp US HY spread"),
    "surprise": ("CESIG10 Index", "diff_10", "+10pt G10 economic surprise"),
    "dollar": ("DXY Index", "pct", "+1% US dollar index"),
}


def market_split(region, r):
    """Monthly split of the standard regional index's USD net return, plus the IMI gap."""
    px_t, net_t = parent_ticker(region), NET[region]
    px = prices("monthly")[px_t]
    pe = panel("pe_monthly")[px_t].reindex(px.index).ffill()
    r_px, r_net, r_loc = r[px_t], r[net_t], r[LOCAL[region]]
    l_px = np.log1p(r_px)
    l_pe = np.log(pe).diff().reindex(r.index)
    l_ccy = np.log1p(r_net) - np.log1p(r_loc)
    l_eps = l_px - l_pe - l_ccy
    k = (r_px / l_px).where(l_px.abs() > 1e-10, 1.0)
    out = pd.DataFrame({"income": r_net - r_px, "earnings": k * l_eps, "currency": k * l_ccy,
                        "multiple": k * l_pe})
    return out, net_t


def block_attribution(fam, weights, r, blocks):
    """{component: DataFrame[month x parts]} of monthly contributions."""
    splits = {}
    out = {}
    for comp, t in fam.items():
        region = comp.split("_")[0]
        if region not in splits:
            parts, net_t = market_split(region, r)
            parts["imi_gap"] = blocks[region]["core"] - r[net_t]
            splits[region] = parts
        w = weights.loc[comp]
        d = pd.concat([r[t["net"]].rename("total"), blocks[region]], axis=1).loc[START:].dropna()
        c = pd.DataFrame(index=d.index)
        c["total"] = d["total"]
        for p in MARKET_PARTS:
            c[p] = w["core"] * splits[region][p].reindex(d.index)
        for tl in TILTS:
            c[tl] = w[tl] * d[tl]
        c["unexplained"] = c["total"] - c[MARKET_PARTS + TILTS].sum(axis=1)
        out[comp] = c
    return out


def periods_table(contrib):
    rows = []
    for name, lo, hi in EPISODES:
        s = contrib.loc[lo:hi]
        rows.append({"period": f"{name} ({lo}..{(hi or str(s.index[-1].date())[:7])})", **s.sum()})
    yrs = len(contrib) / 12
    rows.append({"period": f"Full sample, per year ({str(contrib.index[0].date())[:7]}..)", **(contrib.sum() / yrs)})
    return pd.DataFrame(rows).set_index("period")


def macro_changes(r):
    px = prices("monthly")
    x = {}
    for k, (t, how, _) in MACRO.items():
        s = px[t]
        x[k] = {"diff": s.diff(), "diff_10": s.diff() / 10,
                "pct": s.pct_change(fill_method=None) * 100}[how]
    return pd.DataFrame(x).reindex(r.index)


def sensitivities(series, X):
    """{name: series} -> DataFrame of betas and t-stats."""
    rows = []
    for name, y in series.items():
        d = pd.concat([y.rename("y"), X], axis=1).loc[START:].dropna()
        m = sm.OLS(d["y"], sm.add_constant(d[list(MACRO)])).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        rows.append({"series": name, "n": int(m.nobs), "r2": m.rsquared,
                     **{k: m.params[k] for k in MACRO}, **{f"t_{k}": m.tvalues[k] for k in MACRO}})
    return pd.DataFrame(rows).set_index("series")


def fmt_beta(df):
    """'beta (t)' strings, betas in % return per unit shock."""
    return pd.DataFrame({k: [f"{b * 100:+.1f} ({t:+.1f})" for b, t in zip(df[k], df[f"t_{k}"])] for k in MACRO},
                        index=df.index).assign(r2=df["r2"].round(2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:+.1f}".format)
    fam = pert_family()
    weights, _ = full_sample(returns("weekly"), block_returns("weekly")[0], fam)
    r = returns("monthly")
    blocks, _ = block_returns("monthly")

    # sanity: local-currency series really are local (USD minus local tracks the dollar)
    X = macro_changes(r)
    print("Check: corr(USD net - local net, % change in DXY), monthly - should be strongly negative for EME/PAC")
    print({reg: round((r[NET[reg]] - r[LOCAL[reg]]).corr(X["dollar"]), 2) for reg in LOCAL})

    contrib = block_attribution(fam, weights, r, blocks)
    tables = {}
    for comp, c in contrib.items():
        t = periods_table(c) * 100
        tables[comp] = t
        t.to_csv(OUT / f"blocks_{comp}.csv")
        c.to_csv(OUT / f"blocks_{comp}_monthly.csv")
    cols = ["total"] + MARKET_PARTS + TILTS + ["unexplained"]
    for label, suffix in [("BUYOUT", "_BO"), ("VENTURE", "_VC")]:
        print(f"\n===== {label}: return attribution, % (sums of monthly returns; last row per year) =====")
        for comp in [c for c in fam if c.endswith(suffix)]:
            print(f"\n{comp}  (weights: " + ", ".join(f"{b} {weights.loc[comp, b]:.2f}" for b in BLOCKS) + ")")
            print(tables[comp][cols].to_string())

    # macro sensitivities
    series = {}
    for region, b in blocks.items():
        for blk in BLOCKS:
            series[f"{region} {blk}"] = b[blk]
    for comp, t in fam.items():
        series[comp] = r[t["net"]]
    sens = sensitivities(series, X)
    sens.to_csv(OUT / "macro_sensitivities.csv")
    implied = {}
    for comp in fam:
        region = comp.split("_")[0]
        implied[comp] = sum(weights.loc[comp, b] * sens.loc[f"{region} {b}", list(MACRO)] for b in BLOCKS)
    implied = pd.DataFrame(implied).T
    gap = sens.loc[list(fam), list(MACRO)] - implied
    gap.to_csv(OUT / "macro_gap.csv")

    print("\n===== Macro sensitivities: % return per shock, same month (HAC t-stat) =====")
    print("Shocks: " + "; ".join(f"{k} = {v[2]}" for k, v in MACRO.items()))
    for label, suffix in [("BUYOUT", "_BO"), ("VENTURE", "_VC")]:
        print(f"\n{label} components")
        print(fmt_beta(sens.loc[[c for c in fam if c.endswith(suffix)]]).to_string())
    for region in blocks:
        print(f"\nBlocks, {region}")
        print(fmt_beta(sens.loc[[f"{region} {b}" for b in BLOCKS]]).to_string())
    print("\nComponent beta minus weight-implied beta (macro exposure the blocks miss), % per shock")
    print((gap * 100).to_string())


if __name__ == "__main__":
    main()
