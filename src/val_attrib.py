"""Valuation attribution of MSCI PERT and World IMI: dividends, earnings, P/E, currency, tilts.

Extends attribution.py from the plain regional market to the PE sector-matched Core. Each
regional GICS sector's monthly USD return is split, as for the market:

    r_sector = income + earnings growth (local ccy) + currency + P/E change

    income    trailing 12m dividend yield / 12 x k (withholding factor, sectors.calibrate)
    P/E       change in trailing P/E; earnings = price change - P/E change - currency
    currency  the region's USD-minus-local return, applied to every sector in the region
    The three log pieces are scaled by (simple / log) price return so they add up exactly.
    Months where a sector's trailing P/E is missing, negative or above 150 (earnings near
    zero, e.g. energy in 2020) keep earnings and P/E together as "price, unsplit".

Core parts = sum_s Core weight_s x sector parts_s (Pacific: regional market parts, since
Proxy PERT keeps the plain IMI Core there). MSCI PERT is then attributed with Proxy PERT's
quarterly-rebalanced weights:

    MSCI PERT = sum_j cw_j * b_j * Core parts_j + tilts + unexplained
    World IMI = sum_r cw_r * market parts_r (+ small-cap gap)

Sums of monthly returns (additive, not compounded). Output: output/val_attrib/.
"""
import numpy as np
import pandas as pd

from attribution import LOCAL, market_split
from blocks import TILTS, block_returns
from data import ROOT, panel, pert_family, returns
from pe_core import IMI_CORE_REGIONS, component_blocks, core_weights
from sectors import NET, PREFIX, SECTORS, accrual, calibrate, sector_ticker

OUT = ROOT / "output" / "val_attrib"
PARTS = ["income", "earnings", "currency", "multiple", "unsplit"]
PERIODS = [("Dec 2008 - Aug 2026, per year", None, None), ("2009-2014", "2009-01", "2014-12"),
           ("2015-2021", "2015-01", "2021-12"), ("2022-Aug 2026", "2022-01", None)]
PE_MAX = 150


def sector_parts(region, rm, k):
    """{sector: DataFrame[month x PARTS]} for one region."""
    pe = panel("pe_monthly")
    idx = rm.index
    l_ccy = (np.log1p(rm[NET[region]]) - np.log1p(rm[LOCAL[region]])).reindex(idx)
    out = {}
    for s in SECTORS:
        t = sector_ticker(region, s)
        r_px = rm[t]
        inc = k * accrual([t], idx, "monthly")[t]
        p = pe[t].reindex(idx).ffill() if t in pe else pd.Series(np.nan, index=idx)
        ok = (p > 0) & (p < PE_MAX) & (p.shift(1) > 0) & (p.shift(1) < PE_MAX)
        l_px = np.log1p(r_px)
        l_pe = np.log(p.where(p > 0)).diff().where(ok)
        kk = (r_px / l_px).where(l_px.abs() > 1e-10, 1.0)
        df = pd.DataFrame(index=idx)
        df["income"] = inc
        df["currency"] = kk * l_ccy
        df["multiple"] = (kk * l_pe).fillna(0.0)
        df["earnings"] = (kk * (l_px - l_ccy - l_pe)).where(ok, 0.0)
        df["unsplit"] = (kk * (l_px - l_ccy)).where(~ok, 0.0)
        out[s] = df
    rl_start = rm[sector_ticker(region, "RL")].first_valid_index()
    before = out["RL"].index < rl_start  # Real Estate sat in Financials before 2016
    out["RL"] = out["RL"].copy()
    out["RL"].loc[before] = out["FN"].loc[before].values
    return out


def core_parts(rm):
    """{component: DataFrame[month x PARTS + imi_gap]} for the Proxy PERT Core."""
    k = calibrate("monthly")
    w = core_weights("proxy", "monthly")
    imi = block_returns("monthly")[0]
    sp = {reg: sector_parts(reg, rm, k[reg]) for reg in PREFIX}
    out = {}
    for comp in pert_family():
        region = comp.split("_")[0]
        if region in IMI_CORE_REGIONS["proxy"]:
            p, net_t = market_split(region, rm)
            p = p.assign(unsplit=0.0)
            p["imi_gap"] = imi[region]["core"] - rm[net_t]
        else:
            wc = w[comp].reindex(rm.index)
            p = sum(sp[region][s].mul(wc[s], axis=0) for s in SECTORS)
            p["imi_gap"] = 0.0
        out[comp] = p[PARTS + ["imi_gap"]]
    return out


def market_parts(rm):
    imi = block_returns("monthly")[0]
    out = {}
    for region in PREFIX:
        p, net_t = market_split(region, rm)
        p = p.assign(unsplit=0.0)
        p["imi_gap"] = imi[region]["core"] - rm[net_t]
        out[region] = p[PARTS + ["imi_gap"]]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:+.2f}".format)
    rm = returns("monthly")
    fam = pert_family()
    blocks = component_blocks("proxy", "monthly")
    cp = core_parts(rm)
    mp = market_parts(rm)

    # check: Core parts add up to the Core return
    for comp in ["NA_BO", "EME_VC"]:
        gap = (cp[comp][PARTS].sum(axis=1) - blocks[comp]["core"]).loc["2007":].abs().max()
        print(f"check {comp}: max |Core - sum of parts| = {gap:.2e}")

    W = pd.read_csv(ROOT / "output" / "replica_proxy" / "weights_dynamic.csv", header=[0, 1], index_col=0, parse_dates=True)
    idx = W.index
    cols = PARTS + ["imi_gap"] + TILTS
    pert = pd.DataFrame(0.0, index=idx, columns=cols)
    for comp in fam:
        cw, b = W[("pert_cw", comp)], W[(comp, "core")]
        for p in PARTS + ["imi_gap"]:
            pert[p] += cw * b * cp[comp][p].reindex(idx)
        for t in TILTS:
            pert[t] += cw * W[(comp, t)] * blocks[comp][t].reindex(idx)
    pert.insert(0, "total", rm["MXWOPERT Index"].reindex(idx))
    pert["unexplained"] = pert["total"] - pert[cols].sum(axis=1)

    imi = pd.DataFrame(0.0, index=idx, columns=cols)
    for region in PREFIX:
        for p in PARTS + ["imi_gap"]:
            imi[p] += W[("imi_cw", region)] * mp[region][p].reindex(idx)
    imi.insert(0, "total", rm["M1WOIM Index"].reindex(idx))
    imi["unexplained"] = imi["total"] - imi[cols].sum(axis=1)

    rows = []
    for lab, lo, hi in PERIODS:
        a, b_ = pert.loc[lo:hi], imi.loc[lo:hi]
        yrs = len(a) / 12
        for name, df in [("MSCI PERT", a), ("World IMI", b_), ("Difference", a - b_)]:
            rows.append({"period": lab, "series": name, **(df.sum() / yrs * 100)})
    res = pd.DataFrame(rows).set_index(["period", "series"])
    res["dividends"] = res["income"]
    res["P/E change"] = res["multiple"]
    res["earnings growth"] = res["earnings"]
    res["price, unsplit"] = res["unsplit"]
    res["style tilts"] = res[TILTS].sum(axis=1)
    show = ["total", "dividends", "earnings growth", "P/E change", "currency", "price, unsplit", "imi_gap", "style tilts", "unexplained"]
    res.to_csv(OUT / "world.csv")
    print("\nMSCI PERT vs World IMI, % per year (sums of monthly returns / years); PERT attributed with Proxy PERT weights")
    print(res[show].to_string())

    # components: PE Core vs regional IMI, valuation split
    rows = []
    for comp in ["NA_BO", "NA_VC", "EME_BO", "EME_VC"]:
        region = comp.split("_")[0]
        for name, df in [(f"{comp} PE Core", cp[comp]), (f"{region} market (IMI)", mp[region])]:
            x = df.loc["2007-01":]
            rows.append({"series": name, **(x.sum() / (len(x) / 12) * 100)})
    comp_t = pd.DataFrame(rows).set_index("series")
    comp_t.to_csv(OUT / "components.csv")
    print("\nPE-matched Core vs regional market, Jan 2007 - Aug 2026, % per year")
    print(comp_t[PARTS + ["imi_gap"]].assign(total=comp_t[PARTS + ["imi_gap"]].sum(axis=1)).to_string())


if __name__ == "__main__":
    main()
