"""Pilot: rebuild MSCI's leverage block stock by stock on the PE sector-matched universe (NA, 2021-26).

MSCI's leverage long leg applies the MSCI Quality methodology, with the quality score based
only on debt-to-equity, to the Core universe (the regional IMI reweighted to the PE
category's sector mix); the short leg is the Core. This pilot reproduces that for North
America over 2021 - Aug 2026, where Proxy PERT misses MSCI PERT most:

    universe     S&P 1500 members at each rebalance (point in time, delisted names kept)
    Core weight  market cap x (PE sector weight / universe sector weight), multiplier <= 25x
                 (PE sector weights: Proxy PERT's Core for NA buyout / NA VC, already lagged)
    score        debt-to-equity known at least 90 days before the rebalance; z-score across
                 non-financial names, winsorised at +-3; score = 1+z (z>0) or 1/(1-z) (z<0),
                 as in MSCI Quality. Built both ways, because MSCI's wording is ambiguous:
                 "high" (high D/E scores high) and "low" (low D/E scores high, the usual
                 Quality direction). Negative equity counts as the highest leverage.
    long leg     top 25% of scored names; weight = Core weight x score, issuer cap 5%;
                 rebalanced at calendar quarter-ends (MSCI: Feb/May/Aug/Nov) and held a quarter
    tilt         long leg total return minus the PE-matched Core

Stock returns are gross total returns (Core is net), which biases the tilt up by roughly
0.3%/yr. Names that stop trading mid-quarter drop out and the rest are renormalised.

Evaluation: monthly regressions of MSCI's NA buyout and NA VC indexes on the Proxy PERT
blocks, swapping only the leverage block, Jan 2021 - Aug 2026.
Bloomberg pulls are cached in data/pilot/ (git-ignored); re-runs cost nothing.
"""
import time

import numpy as np
import pandas as pd

from data import ROOT, pert_family, returns
from pe_core import component_blocks, core_weights
from sectors import SECTORS

OUT = ROOT / "output" / "lev_pilot"
CACHE = ROOT / "data" / "pilot"
UNIVERSE = "SPR Index"
REBAL = pd.date_range("2020-12-31", "2026-06-30", freq="QE")
LAG_DAYS = 90
TOP = 0.25
ISSUER_CAP = 0.05
MULT_CAP = 25
CHUNK = 50

GICS_NAME = {"Energy": "EN", "Materials": "MT", "Industrials": "IN", "Consumer Discretionary": "CD",
             "Consumer Staples": "CS", "Health Care": "HC", "Financials": "FN", "Information Technology": "IT",
             "Communication Services": "TC", "Utilities": "UT", "Real Estate": "RL"}
# Bloomberg industry group -> GICS sector, for delisted names without a GICS sector
BICS_GROUP = {
    "Pharmaceuticals": "HC", "Biotechnology": "HC", "Healthcare-Products": "HC", "Healthcare-Services": "HC",
    "Food": "CS", "Beverages": "CS", "Agriculture": "CS", "Cosmetics/Personal Care": "CS", "Household Products/Wares": "CS",
    "Retail": "CD", "Auto Manufacturers": "CD", "Auto Parts&Equipment": "CD", "Apparel": "CD", "Home Builders": "CD",
    "Home Furnishings": "CD", "Leisure Time": "CD", "Lodging": "CD", "Entertainment": "CD", "Toys/Games/Hobbies": "CD",
    "Housewares": "CD", "Textiles": "CD", "Distribution/Wholesale": "IN", "Commercial Services": "IN",
    "Software": "IT", "Computers": "IT", "Semiconductors": "IT", "Electronics": "IT", "Office/Business Equip": "IT",
    "Telecommunications": "TC", "Media": "TC", "Internet": "TC", "Advertising": "TC",
    "Oil&Gas": "EN", "Oil&Gas Services": "EN", "Pipelines": "EN", "Coal": "EN", "Energy-Alternate Sources": "EN",
    "Chemicals": "MT", "Mining": "MT", "Iron/Steel": "MT", "Forest Products&Paper": "MT", "Packaging&Containers": "MT",
    "Building Materials": "MT",
    "Aerospace/Defense": "IN", "Airlines": "IN", "Transportation": "IN", "Machinery-Diversified": "IN",
    "Machinery-Constr&Mining": "IN", "Miscellaneous Manufactur": "IN", "Electrical Compo&Equip": "IN",
    "Engineering&Construction": "IN", "Environmental Control": "IN", "Hand/Machine Tools": "IN", "Metal Fabricate/Hardware": "IN",
    "Trucking&Leasing": "IN", "Shipbuilding": "IN",
    "Electric": "UT", "Gas": "UT", "Water": "UT",
    "REITS": "RL", "Real Estate": "RL",
    "Banks": "FN", "Diversified Finan Serv": "FN", "Insurance": "FN", "Savings&Loans": "FN", "Investment Companies": "FN",
    "Private Equity": "FN", "Closed-end Funds": "FN",
}


def _blp():
    from xbbg import blp
    return blp


def members():
    f = CACHE / "members.parquet"
    if f.exists():
        return pd.read_parquet(f)
    blp, rows = _blp(), []
    for d in REBAL:
        m = blp.bds(UNIVERSE, "INDX_MWEIGHT_HIST", END_DATE_OVERRIDE=d.strftime("%Y%m%d")).to_pandas()
        rows.append(pd.DataFrame({"rebal": d, "ticker": m["Index Member"].astype(str) + " Equity"}))
        print(f"members {d.date()}: {len(m)}", flush=True)
        time.sleep(0.2)
    df = pd.concat(rows, ignore_index=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(f, index=False)
    return df


def sectors(tickers):
    f = CACHE / "sectors.parquet"
    if f.exists():
        return pd.read_parquet(f)
    blp, frames = _blp(), []
    for i in range(0, len(tickers), 200):
        d = blp.bdp(tickers[i:i + 200], ["GICS_SECTOR_NAME", "INDUSTRY_GROUP"]).to_pandas()
        frames.append(d.pivot(index="ticker", columns="field", values="value"))
        print(f"sectors {i + len(tickers[i:i + 200])}/{len(tickers)}", flush=True)
    df = pd.concat(frames)
    df["gics"] = df["GICS_SECTOR_NAME"].map(GICS_NAME)
    df["gics"] = df["gics"].fillna(df["INDUSTRY_GROUP"].map(BICS_GROUP))
    df.reset_index().to_parquet(f, index=False)
    return df.reset_index()


def history(tickers, field, per, start, end, name):
    """Long-format bdh, cached by name; chunked, with per-ticker retry on chunk failures."""
    f = CACHE / f"{name}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    blp, frames = _blp(), []
    for i in range(0, len(tickers), CHUNK):
        batch = tickers[i:i + CHUNK]
        try:
            frames.append(blp.bdh(batch, field, start, end, Per=per).to_pandas()[["ticker", "date", "value"]])
        except Exception:
            for t in batch:
                try:
                    frames.append(blp.bdh(t, field, start, end, Per=per).to_pandas()[["ticker", "date", "value"]])
                except Exception:
                    pass
        print(f"{name}: {min(i + CHUNK, len(tickers))}/{len(tickers)}", flush=True)
    df = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(f, index=False)
    return df


def msci_score(x):
    z = (x - x.mean()) / x.std()
    z = z.clip(-3, 3)
    return np.where(z > 0, 1 + z, 1 / (1 - z))


def build_legs(mem, sec, mcap, de, comps=("NA_BO", "NA_VC")):
    """{(component, direction): DataFrame[rebal x ticker] of long-leg weights}."""
    cw = core_weights("proxy", "monthly")
    mcap = mcap.pivot_table(index="date", columns="ticker", values="value")
    mcap.index = mcap.index + pd.offsets.QuarterEnd(0)
    de = de.sort_values("date")
    gics = sec.set_index("ticker")["gics"]
    legs = {(c, d): {} for c in comps for d in ("high", "low")}
    for t in REBAL:
        univ = mem.loc[mem["rebal"] == t, "ticker"]
        mc = mcap.reindex(index=[t]).T.iloc[:, 0].reindex(univ).dropna()
        s = gics.reindex(mc.index)
        mc, s = mc[s.notna()], s[s.notna()]
        known = de[de["date"] <= t - pd.Timedelta(days=LAG_DAYS)].groupby("ticker")["value"].last()
        univ_w = mc.groupby(s).sum() / mc.sum()
        for c in comps:
            pe_w = cw[c].loc[:t].iloc[-1]
            mult = (pe_w / univ_w).reindex(SECTORS).fillna(0).clip(upper=MULT_CAP)
            core_w = mc * s.map(mult).fillna(0)
            elig = core_w[(core_w > 0) & (s != "FN")].index
            x = known.reindex(elig).dropna()
            x = x.where(x >= 0, x.max())  # negative equity = most levered
            x = x.clip(upper=x.quantile(0.99))
            for direction, sign in (("high", 1), ("low", -1)):
                score = pd.Series(msci_score(sign * x), index=x.index)
                top = score.nlargest(int(len(score) * TOP)).index
                w = core_w[top] * score[top]
                w = w / w.sum()
                for _ in range(10):  # issuer cap with redistribution
                    over = w > ISSUER_CAP
                    if not over.any():
                        break
                    excess = (w[over] - ISSUER_CAP).sum()
                    w[over] = ISSUER_CAP
                    w[~over] += excess * w[~over] / w[~over].sum()
                legs[(c, direction)][t] = w
    return {k: pd.DataFrame(v).T.fillna(0.0) for k, v in legs.items()}


def leg_returns(W, tr):
    """Monthly long-leg returns: weights set at quarter-end t, applied to the next three months."""
    rets = tr.pivot_table(index="date", columns="ticker", values="value").sort_index()
    rets.index = rets.index + pd.offsets.MonthEnd(0)
    rets = rets.pct_change(fill_method=None)
    out = {}
    for t in W.index:
        w = W.loc[t]
        w = w[w > 0]
        for m in pd.date_range(t + pd.offsets.MonthEnd(1), periods=3, freq="ME"):
            if m not in rets.index:
                continue
            r = rets.loc[m].reindex(w.index)
            ok = r.notna()
            out[m] = (w[ok] * r[ok]).sum() / w[ok].sum() if ok.any() else np.nan
    return pd.Series(out).sort_index()


def evaluate(tilts):
    from style_decomp import fit
    rm = returns("monthly")
    fam = pert_family()
    blocks = component_blocks("proxy", "monthly")
    rows, yearly = [], []
    for comp in ["NA_BO", "NA_VC"]:
        b = blocks[comp].loc["2021-01":"2026-08"].copy()
        y = rm[fam[comp]["net"]].reindex(b.index)
        variants = {
            "market minus Quality (old)": (rm["M1NAIM Index"] - rm["M1USQU Index"]).reindex(b.index),
            "J.P. Morgan leverage L/S (current)": b["leverage"],
            "D/E, high leverage (new, MSCI-style)": tilts[(comp, "high")].reindex(b.index),
            "D/E, low leverage (new, MSCI-style)": tilts[(comp, "low")].reindex(b.index),
            "no leverage block": pd.Series(0.0, index=b.index),
        }
        for name, lev in variants.items():
            X = b.assign(leverage=lev)
            d = pd.concat([y.rename("y"), X], axis=1).dropna()
            coef, r2, resid = fit(d["y"].values, d[list(X.columns)].values)
            rows.append({"component": comp, "leverage block": name, "months": len(d),
                         "unexplained %/yr": coef[0] * 1200, "R2": r2, "tracking error %": resid.std() * np.sqrt(12) * 100,
                         "leverage weight": coef[1 + list(X.columns).index("leverage")],
                         "leverage tilt %/yr": lev.reindex(d.index).mean() * 1200})
            gap = pd.Series(d["y"].values - (d[list(X.columns)].values @ coef[1:]), index=d.index)
            yearly.append({"component": comp, "leverage block": name,
                           **{str(yr): g.sum() * 100 for yr, g in gap.groupby(gap.index.year)}})
    return pd.DataFrame(rows).set_index(["component", "leverage block"]), pd.DataFrame(yearly).set_index(["component", "leverage block"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:.2f}".format)
    mem = members()
    tickers = sorted(mem["ticker"].unique())
    print(f"{len(tickers)} unique members across {len(REBAL)} rebalances")
    sec = sectors(tickers)
    print("sector coverage:", f"{sec['gics'].notna().mean():.1%}", "| unmapped groups:",
          sec.loc[sec["gics"].isna(), "INDUSTRY_GROUP"].value_counts().head(8).to_dict())
    mcap = history(tickers, "CUR_MKT_CAP", "Q", "2020-10-01", "2026-06-30", "mcap_q")
    de = history(tickers, "TOT_DEBT_TO_TOT_EQY", "Q", "2019-10-01", "2026-06-30", "de_q")
    legs = build_legs(mem, sec, mcap, de)
    for k, W in legs.items():
        print(f"long leg {k}: {int((W > 0).sum(axis=1).mean())} names on average; top holding {W.max(axis=1).max():.1%}")
    sel = sorted(set().union(*[set(W.columns[(W > 0).any()]) for W in legs.values()]))
    print(f"{len(sel)} unique selected names -> pulling monthly total returns")
    tr = history(sel, "TOT_RETURN_INDEX_GROSS_DVDS", "M", "2020-12-01", "2026-08-31", "tr_m")
    blocks = component_blocks("proxy", "monthly")
    tilts = {k: leg_returns(W, tr) - blocks[k[0]]["core"] for k, W in legs.items()}
    pd.DataFrame({f"{c}_{d}": s for (c, d), s in tilts.items()}).to_csv(OUT / "tilts.csv")
    res, yearly = evaluate(tilts)
    res.to_csv(OUT / "evaluation.csv")
    yearly.to_csv(OUT / "yearly_gap.csv")
    print("\nMSCI NA buyout / VC indexes on Proxy PERT blocks, monthly Jan 2021 - Aug 2026, swapping the leverage block")
    print(res.to_string())
    print("\nUnexplained return by calendar year (MSCI component minus fitted blocks), %")
    print(yearly.to_string())


if __name__ == "__main__":
    main()
