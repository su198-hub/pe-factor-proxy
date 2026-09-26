"""PE-matched Core Factor and component-specific building blocks.

MSCI's Core Factor for each component reweights the regional IMI to the PE category's
sector (and country) mix; every style tilt is "style index minus Core". Until now the
replica used the plain regional IMI as the Core. This module builds a PE-matched Core:

    core_j(t) = sum_s w_js(t - lag) * r_region(j),s(t)

w_js     sector mix of PE category j (region x strategy), from Cambridge Associates'
         NAV-weighted sector breakdowns (ca_sectors.py), lagged LAG_Q quarters like
         MSCI's PE data; held at the earliest observation before it starts
r_s      regional GICS sector returns (price + dividend accrual, sectors.py)

Variants (the `core` argument):
    "imi"       plain regional IMI (the original replica)
    "ca"        CA sector weights
    "ca_msci"   CA weights rescaled, sector by sector, so their mix at August 2025 matches
                MSCI's own PE universe snapshot (Rethinking Access to Private Equity, Oct
                2025, data/msci/, all strategies per region). CA supplies the history, MSCI
                the level. Sectors MSCI does not list keep their CA weight.
    "gic"       the GIC PERT choice: "ca_msci" for North America and Europe & Middle East,
                plain IMI for Pacific (CA's ex US weights are mostly European and fit Pacific
                worse than the plain market; Pacific is 2-5% of PERT)

Mapping of CA series to components: US PE -> NA buyout, US VC -> NA VC, developed ex US
PE / VC (combined PE/VC before 2024) -> Europe & Middle East and Pacific.

Tilts: long - Core wherever the tilt's short leg was the regional IMI; the leverage proxy
(IMI - Quality) and backfilled World spreads are unchanged. The small-cap long leg is the
plain regional small-cap index (MSCI reweights it to PE sectors; not replicated).
"""
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from blocks import BLOCKS, block_returns, definitions
from ca_sectors import GICS, quarterly
from data import ROOT, pert_family, returns
from pert_structure import constrained_weights
from sectors import SECTORS, market_return, sector_returns

LAG_Q = 2
SRC = {"NA_BO": ("US", "PE"), "NA_VC": ("US", "VC"), "EME_BO": ("XUS", "PE"),
       "EME_VC": ("XUS", "VC"), "PAC_BO": ("XUS", "PE"), "PAC_VC": ("XUS", "VC")}
MSCI_SNAPSHOT = ROOT / "data" / "msci" / "pe_fund_sector_weights_2025-08.csv"
SNAPSHOT_DATE = "2025-08-29"
VARIANTS = ["imi", "ca", "ca_msci", "gic"]
IMI_CORE_REGIONS = {"gic": {"PAC"}}  # regions kept on the plain IMI Core under each variant
assert SECTORS == GICS


def market_sector_weights(freq="weekly"):
    """Average sector mix of each regional market: its return regressed on its sectors (w >= 0)."""
    sect, _ = sector_returns(freq)
    out = {}
    for region in sect:
        d = pd.concat([market_return(region, freq).rename("y"), sect[region]], axis=1).dropna()
        w = lsq_linear(d[SECTORS].values, d["y"].values, bounds=(0, np.inf)).x
        out[region] = pd.Series(w / w.sum(), index=SECTORS)
    return out


def _ca_weights(mw):
    """{component: DataFrame[quarter-end x GICS]} of CA weights, 'Other' split by the region's market."""
    out = {}
    for comp, (src, strat) in SRC.items():
        region = comp.split("_")[0]
        q = quarterly({"US": mw["NA"], "XUS": mw[region if region != "NA" else "EME"]})
        out[comp] = q[(src, strat)]
    return out


def _msci_adjustment(ca, freq="weekly"):
    """Per-region sector multipliers so CA's strategy blend matches MSCI's Aug 2025 region mix."""
    snap = pd.read_csv(MSCI_SNAPSHOT, keep_default_na=False)
    r = returns(freq)
    fam = pert_family()
    d = r[["MXWOPERT Index"] + [fam[c]["net"] for c in fam]].loc[:SNAPSHOT_DATE].dropna().iloc[-156:]
    cw, _, _ = constrained_weights(d.iloc[:, 0].values, d.iloc[:, 1:].values)
    cw = pd.Series(cw, index=list(fam))
    adj = {}
    for region in ["NA", "EME", "PAC"]:
        bo, vc = cw[f"{region}_BO"], cw[f"{region}_VC"]
        at = lambda c: ca[c].loc[:SNAPSHOT_DATE].iloc[-1]
        blend = (bo * at(f"{region}_BO") + vc * at(f"{region}_VC")) / (bo + vc) if bo + vc > 0 else at(f"{region}_BO")
        m = snap[snap["region"] == region].set_index("sector")["pe_fund_index"]
        named = list(m.index)
        target = blend.copy()
        target[named] = m / m.sum() * blend[named].sum()  # MSCI mix across the sectors it lists
        adj[region] = (target / blend).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    return adj


def core_weights(core="ca", freq="weekly"):
    """{component: DataFrame[date x GICS]} of Core sector weights on the `freq` return dates."""
    mw = market_sector_weights(freq)
    ca = _ca_weights(mw)
    if core in ("ca_msci", "gic"):
        adj = _msci_adjustment(ca, freq)
        ca = {c: (w * adj[c.split("_")[0]]).div((w * adj[c.split("_")[0]]).sum(axis=1), axis=0) for c, w in ca.items()}
    idx = returns(freq).index
    out = {}
    for comp, w in ca.items():
        w = w.shift(LAG_Q).bfill()  # PE data arrive ~2 quarters late; hold the earliest mix before it
        out[comp] = w.reindex(w.index.union(idx)).ffill().bfill().reindex(idx)
    return out


def core_returns(core="ca", freq="weekly"):
    sect, _ = sector_returns(freq)
    w = core_weights(core, freq)
    return {c: (w[c] * sect[c.split("_")[0]][SECTORS]).sum(axis=1, min_count=len(SECTORS)) for c in SRC}


def component_blocks(core="imi", freq="weekly"):
    """{component: DataFrame[date x BLOCKS]} with the chosen Core; tilts re-pointed at it."""
    base, _ = block_returns(freq)
    if core == "imi":
        return {c: base[c.split("_")[0]] for c in SRC}
    r = returns(freq)
    pe = core_returns(core, freq)
    defs = definitions()
    out = {}
    for comp in SRC:
        region = comp.split("_")[0]
        if region in IMI_CORE_REGIONS.get(core, set()):
            out[comp] = base[region]
            continue
        rd = defs[defs["region"] == region]
        imi_t = rd.loc[rd["block"] == "core", "long"].iloc[0]
        b = base[region].copy()
        b["core"] = pe[comp]
        for _, d in rd[rd["block"] != "core"].iterrows():
            if d["short"] != imi_t:
                continue  # leverage proxy (IMI - Quality): not a "style minus Core" spread
            primary = r[d["long"]] - r[d["short"]]
            start = primary.first_valid_index()
            repointed = r[d["long"]] - pe[comp]
            b[d["block"]] = repointed.where(repointed.index >= start, b[d["block"]])  # backfill untouched
        out[comp] = b[BLOCKS]
    return out


def compare(freq="weekly", start="2006-12-01"):
    """Full-sample style decomposition of each component under each Core variant."""
    from style_decomp import PPY, fit
    r = returns(freq)
    fam = pert_family()
    rows = []
    for core in VARIANTS:
        blocks = component_blocks(core, freq)
        for comp, t in fam.items():
            d = pd.concat([r[t["net"]].rename("y"), blocks[comp]], axis=1).loc[start:].dropna()
            coef, r2, resid = fit(d["y"].values, d[BLOCKS].values)
            se = resid.std() / np.sqrt(len(resid))
            rows.append({"component": comp, "variant": core, "unexplained_%": coef[0] * PPY * 100,
                         "t": coef[0] / se, "r2": r2, "te_%": resid.std() * np.sqrt(PPY) * 100,
                         **dict(zip(BLOCKS, coef[1:]))})
    return pd.DataFrame(rows).set_index(["component", "variant"]).sort_index()


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.2f}".format)
    w = core_weights("ca_msci")
    print("Core sector weights (CA rescaled to MSCI Aug 2025), latest vs first:")
    print(pd.DataFrame({c: w[c].iloc[-1] for c in w}).T.mul(100).round(1).to_string())
    print(pd.DataFrame({c: w[c].loc["2007-06"].iloc[0] for c in w}).T.mul(100).round(1).to_string())
    res = compare()
    (ROOT / "output" / "pe_core").mkdir(parents=True, exist_ok=True)
    res.to_csv(ROOT / "output" / "pe_core" / "compare.csv")
    print("\nStyle decomposition by Core variant (weekly, full sample):")
    print(res.to_string())
