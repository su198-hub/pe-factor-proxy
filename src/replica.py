"""GIC PERT: our factor replica of MSCI PERT, compared with MSCI PERT and MSCI World IMI.

Replica, for each component j (region x strategy) and for the World index:

    GIC_j     = b_j * core_region(j) + sum_i w_ij * tilt_i,region(j)      (no intercept)
    GIC_World = sum_j cw_j * GIC_j

Two versions:
    static   one set of weights estimated on the full weekly sample (in-sample: uses
             hindsight; the best fit these blocks can give)
    dynamic  re-estimated at each MSCI review date (last business day of Feb, May, Aug,
             Nov) on the trailing 3 years of weekly data (at least 2), and held until the
             next review. Uses only data available at the time, so it is the honest
             replica. It starts once two years of history exist (first review Nov 2008), so it
             misses most of the GFC; a full-period table (from Dec 2006) compares PERT,
             the static replica and World IMI, and checks our data against MSCI's
             published Dec 2006 - Dec 2024 figures.

Block weights: bounded least squares as in style_decomp (tilts >= 0, size <= 0.4).
World weights: PERT World on its six components, weights >= 0 summing to 1.
World IMI gets the same treatment (its regional mix drifts too): World IMI on the three
regional IMIs, weights >= 0 summing to 1, re-estimated at each review.

Attribution (monthly, additive sums of monthly returns), common format for all series:
    market   = sum_j cw_j * b_j * [income, earnings, currency, multiple, IMI gap]_region(j)
    tilts    = sum_j cw_j * w_ij * tilt_i,region(j), summed across regions per style
    residual = series - (market + tilts): for PERT this is the unexplained part; for the
               replica it is zero by construction; for World IMI it is the part of World
               outside the three regions (Israel) plus estimation error.

Output: output/replica/.
"""
import numpy as np
import pandas as pd

from attribution import MARKET_PARTS, market_split
from blocks import BLOCKS, TILTS, block_returns
from data import ROOT, pert_family, returns
from pert_structure import constrained_weights
from style_decomp import fit

OUT = ROOT / "output" / "replica"
PERT, WORLD_IMI = "MXWOPERT Index", "M1WOIM Index"
REGIONAL_IMI = {"NA": "M1NAIM Index", "EME": "MIMUEURN Index", "PAC": "M1PCIM Index"}
REVIEW_MONTHS = (2, 5, 8, 11)
WINDOW, MIN_WEEKS = 156, 104
START = "2006-12-01"


def weights_at(rw, bw, fam, end=None, window=None):
    """Block weights per component, PERT World and World IMI regional weights, from weekly data up to `end`."""
    out = {}
    for comp, t in fam.items():
        region = comp.split("_")[0]
        d = pd.concat([rw[t["net"]].rename("y"), bw[region]], axis=1).loc[START:end].dropna()
        d = d.iloc[-window:] if window else d
        if len(d) < MIN_WEEKS:
            return None
        coef, _, _ = fit(d["y"].values, d[BLOCKS].values)
        out.update({(comp, b): c for b, c in zip(BLOCKS, coef[1:])})
    for name, y_t, xs in [("pert", PERT, [fam[c]["net"] for c in fam]), ("imi", WORLD_IMI, list(REGIONAL_IMI.values()))]:
        d = rw[[y_t] + xs].loc[START:end].dropna()
        d = d.iloc[-window:] if window else d
        cw, _, _ = constrained_weights(d.iloc[:, 0].values, d.iloc[:, 1:].values)
        keys = list(fam) if name == "pert" else list(REGIONAL_IMI)
        out.update({(f"{name}_cw", k): w for k, w in zip(keys, cw)})
    return pd.Series(out)


def weight_path(rw, bw, fam, months):
    """Monthly weights: static (one row repeated) and dynamic (set at each review, applied from the next month)."""
    static = weights_at(rw, bw, fam)
    reviews = [m for m in months if m.month in REVIEW_MONTHS]
    dyn = {}
    for d in reviews:
        w = weights_at(rw, bw, fam, end=d, window=WINDOW)
        if w is not None:
            dyn[d] = w
    dyn = pd.DataFrame(dyn).T
    dyn.index = dyn.index.map(lambda d: months[months.get_loc(d) + 1] if months.get_loc(d) + 1 < len(months) else pd.NaT)
    dyn = dyn[dyn.index.notna()].reindex(months).ffill().dropna()
    stat = pd.DataFrame([static] * len(months), index=months)
    return {"static": stat, "dynamic": dyn}


def contributions(W, fam, rm, bm, parts, series):
    """Monthly contributions for PERT-style weights W (rows = months)."""
    idx = W.index
    c = pd.DataFrame(0.0, index=idx, columns=MARKET_PARTS + TILTS)
    for comp in fam:
        region = comp.split("_")[0]
        cw = W[("pert_cw", comp)]
        for p in MARKET_PARTS:
            c[p] += cw * W[(comp, "core")] * parts[region][p].reindex(idx)
        for t in TILTS:
            c[t] += cw * W[(comp, t)] * bm[region][t].reindex(idx)
    replica = c.sum(axis=1)
    c.insert(0, "total", rm[series].reindex(idx) if series else replica)
    c["residual"] = c["total"] - replica
    return c, replica


def imi_contributions(W, rm, parts):
    idx = W.index
    c = pd.DataFrame(0.0, index=idx, columns=MARKET_PARTS + TILTS)
    for region in REGIONAL_IMI:
        for p in MARKET_PARTS:
            c[p] += W[("imi_cw", region)] * parts[region][p].reindex(idx)
    c.insert(0, "total", rm[WORLD_IMI].reindex(idx))
    c["residual"] = c["total"] - c[MARKET_PARTS].sum(axis=1)
    return c


def component_replicas(W, fam, bm):
    out = {}
    for comp in fam:
        region = comp.split("_")[0]
        out[comp] = sum(W[(comp, b)] * bm[region][b].reindex(W.index) for b in BLOCKS)
    return pd.DataFrame(out)


def stats(s, ref=None):
    s = s.dropna()
    wealth = (1 + s).cumprod()
    out = {"cagr": wealth.iloc[-1] ** (12 / len(s)) - 1, "vol": s.std() * np.sqrt(12),
           "max_dd": (wealth / wealth.cummax() - 1).min()}
    if ref is not None:
        d = pd.concat([s, ref], axis=1, sort=True).dropna()
        out["corr_pert"] = d.iloc[:, 0].corr(d.iloc[:, 1])
        out["te_pert"] = (d.iloc[:, 0] - d.iloc[:, 1]).std() * np.sqrt(12)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    fam = pert_family()
    rw, rm = returns("weekly"), returns("monthly")
    bw, bm = block_returns("weekly")[0], block_returns("monthly")[0]
    parts = {}
    for region in REGIONAL_IMI:
        p, net_t = market_split(region, rm)
        p["imi_gap"] = bm[region]["core"] - rm[net_t]
        parts[region] = p
    months = rm.loc[START:].index
    paths = weight_path(rw, bw, fam, months)
    for k, W in paths.items():
        W.to_csv(OUT / f"weights_{k}.csv")

    common = paths["dynamic"].index  # dynamic starts later; compare on its sample
    series, attrib = {}, {}
    c_pert, rep_dyn = contributions(paths["dynamic"], fam, rm, bm, parts, PERT)
    _, rep_sta = contributions(paths["static"], fam, rm, bm, parts, PERT)
    c_gic, _ = contributions(paths["dynamic"], fam, rm, bm, parts, None)
    c_imi = imi_contributions(paths["dynamic"], rm, parts)
    allm = pd.DataFrame({"MSCI PERT": rm[PERT], "GIC PERT dynamic": rep_dyn.reindex(months),
                         "GIC PERT static": rep_sta, "MSCI World IMI": rm[WORLD_IMI]}).loc[START:]
    allm.to_csv(OUT / "world_monthly.csv")  # full history; dynamic is empty before its first review
    series = allm.loc[common]

    print(f"Sample: {common[0].date()} .. {common[-1].date()} ({len(common)} months). "
          "Dynamic weights set at each Feb/May/Aug/Nov review from trailing 3y weekly data.")
    st = pd.DataFrame({k: stats(v, series["MSCI PERT"] if k != "MSCI PERT" else None) for k, v in series.items()}).T
    print("\nWorld level: returns (monthly data, annualised; cagr = compound annual return)")
    print((st * 100).round(1).to_string())

    full = pd.DataFrame({"MSCI PERT": rm[PERT], "GIC PERT static": rep_sta, "MSCI World IMI": rm[WORLD_IMI]}).loc[START:].dropna()
    for label, lo, hi in [("Full period", START, None), ("MSCI published window (PERT 9.9%, World IMI 7.2%)", START, "2024-12-31")]:
        f = full.loc[lo:hi]
        st_f = pd.DataFrame({k: stats(v, f["MSCI PERT"] if k != "MSCI PERT" else None) for k, v in f.items()}).T
        print(f"\n{label}: {f.index[0].date()} .. {f.index[-1].date()} ({len(f)} months)")
        print((st_f * 100).round(1).to_string())

    yearly = (1 + series).groupby(series.index.year).prod() - 1
    yearly = yearly[series.groupby(series.index.year).size() == 12]
    yearly["PERT - IMI"] = yearly["MSCI PERT"] - yearly["MSCI World IMI"]
    yearly["PERT - GIC dyn"] = yearly["MSCI PERT"] - yearly["GIC PERT dynamic"]
    yearly.to_csv(OUT / "world_yearly.csv")
    print("\nCalendar-year returns, %")
    print((yearly * 100).round(1).to_string())

    yrs = len(common) / 12
    att = pd.DataFrame({"MSCI PERT": c_pert.sum() / yrs, "GIC PERT dynamic": c_gic.sum() / yrs,
                        "MSCI World IMI": c_imi.sum() / yrs}).T * 100
    att.to_csv(OUT / "world_attribution.csv")
    print("\nWorld attribution, % per year (sum of monthly contributions / years). PERT uses the dynamic GIC PERT weights,")
    print("so its residual is the unexplained part; the replica's residual is zero by construction.")
    print(att.round(2).to_string())

    # bridge: PERT - World IMI
    W = paths["dynamic"]
    core = {r: bm[r]["core"].reindex(common) for r in REGIONAL_IMI}
    pert_core = sum(W[("pert_cw", j)] * core[j.split("_")[0]] for j in fam)
    imi_core = sum(W[("imi_cw", r)] * core[r] for r in REGIONAL_IMI)
    exposure = sum(W[("pert_cw", j)] * (W[(j, "core")] - 1) * core[j.split("_")[0]] for j in fam)
    bridge = pd.Series({
        "MSCI PERT": rm[PERT].loc[common].sum(),
        "less MSCI World IMI": -rm[WORLD_IMI].loc[common].sum(),
        "= PERT minus IMI": (rm[PERT] - rm[WORLD_IMI]).loc[common].sum(),
        "regional mix (PERT's regional weights vs World IMI's)": (pert_core - imi_core).sum(),
        "market exposure above/below 1": exposure.sum(),
        "style tilts": c_pert[TILTS].sum().sum(),
        "unexplained (PERT minus GIC PERT)": c_pert["residual"].sum(),
        "World IMI outside the three regions / fit error": -c_imi["residual"].sum(),
    }) / yrs * 100
    bridge.to_csv(OUT / "bridge_pert_vs_imi.csv")
    print("\nBridge: why PERT differs from World IMI, % per year")
    print(bridge.round(2).to_string())

    # regional and component weights over time
    cw = W[[("pert_cw", j) for j in fam]].droplevel(0, axis=1)
    iw = W[[("imi_cw", r) for r in REGIONAL_IMI]].droplevel(0, axis=1)
    snap = [d for d in cw.index if d.month == 3][::3] + [cw.index[-1]]
    print("\nDynamic weights: PERT component weights (rows = dates weights applied from)")
    print(cw.loc[snap].round(2).to_string())
    print("\nDynamic weights: World IMI regional weights")
    print(iw.loc[snap].round(2).to_string())
    print("\nDynamic block weights, NA_BO and NA_VC")
    for j in ["NA_BO", "NA_VC"]:
        print(j); print(W[[(j, b) for b in BLOCKS]].droplevel(0, axis=1).loc[snap].round(2).to_string())

    # components
    rows = []
    reps = {k: component_replicas(paths[k], fam, bm) for k in paths}
    for j, t in fam.items():
        region = j.split("_")[0]
        ref = rm[t["net"]].loc[common]
        for name, s in [("MSCI component", ref), ("GIC PERT dynamic", reps["dynamic"][j].loc[common]),
                        ("GIC PERT static", reps["static"][j].loc[common]), ("regional IMI", rm[REGIONAL_IMI[region]].loc[common])]:
            rows.append({"component": j, "series": name, **stats(s, ref if name != "MSCI component" else None)})
    comp = pd.DataFrame(rows).set_index(["component", "series"])
    comp.to_csv(OUT / "components.csv")
    print("\nComponents: returns, % (same sample)")
    print((comp * 100).round(1).to_string())


if __name__ == "__main__":
    main()
