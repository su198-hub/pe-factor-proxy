"""Sector-matched market block: replace the plain regional index with a PE sector mix.

PERT's Core Factor is the regional market reweighted to PE's sector mix. MSCI does not
publish those weights, so they are estimated here from returns:

    r_component = alpha + sum_s w_s * r_sector_s  [+ sum_t v_t * tilt_t] + e

w_s >= 0 (the sector mix; sum w_s is the net market exposure), v_t >= 0 as in
style_decomp, Low Size <= 0.4. Three fits per component, weekly, full sample:

    baseline  : regional IMI + tilts (style_decomp as before)
    sectors   : sectors only
    sect+tilt : sectors + tilts (the candidate replacement)

Method check: the same sectors-only regression on the regional market itself should
recover its sector weights with R^2 near 1. The sector mix of each fit is reported
next to the market's, and re-estimated on two halves of the sample for stability.

Output: output/sector_core/.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import lsq_linear

from blocks import BLOCKS, TILTS, block_returns
from data import ROOT, pert_family, returns
from sectors import SECTORS, market_return, sector_returns
from style_decomp import PPY, SIZE_CAP, fit as fit_baseline

OUT = ROOT / "output" / "sector_core"
FREQ = "weekly"
START = "2006-12-01"
HALVES = [("2006-12-01", "2016-06-30"), ("2016-07-01", None)]


def fit(y, S, T=None):
    """Bounded LS: free intercept, sector weights >= 0, tilt weights >= 0, size <= cap."""
    cols = [np.ones(len(y)), S] + ([T] if T is not None else [])
    A = np.column_stack(cols)
    ns, nt = S.shape[1], 0 if T is None else T.shape[1]
    lo = [-np.inf] + [0.0] * (ns + nt)
    hi = [np.inf] + [np.inf] * ns + ([SIZE_CAP if t == "size" else np.inf for t in TILTS] if nt else [])
    x = lsq_linear(A, y, bounds=(lo, hi)).x
    resid = y - A @ x
    return x[0], x[1:1 + ns], x[1 + ns:], 1 - resid.var() / y.var(), resid


def alpha_t(alpha, resid):
    """t-stat of alpha: mean of (alpha + residual), Newey-West SE (4 lags). Slopes treated as known."""
    ols = sm.OLS(alpha + resid, np.ones(len(resid))).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    return ols.tvalues[0]


def summarise(label, comp, alpha, w, v, r2, resid, n):
    row = {"component": comp, "fit": label, "n": n, "mkt_exposure": w.sum(), "alpha_ann": alpha * PPY,
           "alpha_t": alpha_t(alpha, resid), "r2": r2, "te_ann": resid.std() * np.sqrt(PPY)}
    row.update({s: x for s, x in zip(SECTORS, w / w.sum())})
    row.update({t: x for t, x in zip(TILTS, v)} if len(v) else {})
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.width", 250)
    r = returns(FREQ)
    sect, splices = sector_returns(FREQ)
    blocks, _ = block_returns(FREQ)
    fam = pert_family()

    rows, halves = [], []
    # method check: regional market on its own sectors
    for region in sect:
        d = pd.concat([market_return(region, FREQ).rename("y"), sect[region]], axis=1).loc[START:].dropna()
        rows.append(summarise("market", region, *fit(d["y"].values, d[SECTORS].values), len(d)))

    for comp, t in fam.items():
        region = comp.split("_")[0]
        d = pd.concat([r[t["net"]].rename("y"), sect[region].add_prefix("s_"), blocks[region]], axis=1)
        d = d.loc[START:].dropna()
        y, S, T = d["y"].values, d[[f"s_{s}" for s in SECTORS]].values, d[TILTS].values

        coef, r2, resid = fit_baseline(y, d[BLOCKS].values)
        base = {"component": comp, "fit": "baseline", "n": len(y), "mkt_exposure": coef[1],
                "alpha_ann": coef[0] * PPY, "alpha_t": alpha_t(coef[0], resid), "r2": r2, "te_ann": resid.std() * np.sqrt(PPY),
                **dict(zip(TILTS, coef[2:]))}
        rows.append(base)
        rows.append(summarise("sectors", comp, *fit(y, S), len(y)))
        rows.append(summarise("sect+tilt", comp, *fit(y, S, T), len(y)))

        for lo, hi in HALVES:
            h = d.loc[lo:hi]
            hf = fit(h["y"].values, h[[f"s_{s}" for s in SECTORS]].values, h[TILTS].values)
            halves.append(summarise(f"{lo[:4]}-{(hi or str(h.index[-1].date()))[:4]}", comp, *hf, len(h)))

    res = pd.DataFrame(rows).set_index(["component", "fit"])
    hv = pd.DataFrame(halves).set_index(["component", "fit"])
    res.to_csv(OUT / "fits.csv")
    hv.to_csv(OUT / "halves.csv")
    splices.to_csv(OUT / "splices.csv", index=False)

    summ = ["n", "mkt_exposure", "alpha_ann", "alpha_t", "r2", "te_ann"]
    print("Method check - regional market on its own sectors (should be R^2 ~ 1):")
    print(res.xs("market", level="fit")[summ + SECTORS].to_string())
    for label, suffix in [("BUYOUT", "_BO"), ("VENTURE", "_VC")]:
        comps = [c for c in fam if c.endswith(suffix)]
        sub = res.loc[comps]
        print(f"\n===== {label}: fit comparison =====")
        print(sub[summ + TILTS].to_string())
        print(f"\n{label}: sector mix (share of market exposure) vs regional market")
        mix = pd.concat([sub.xs("sect+tilt", level="fit")[SECTORS].assign(fit="PE (sect+tilt)"),
                         sub.xs("sectors", level="fit")[SECTORS].assign(fit="PE (sectors only)"),
                         res.xs("market", level="fit")[SECTORS].rename(
                             index={c.split('_')[0]: c for c in comps}).loc[comps].assign(fit="market")])
        print(mix.reset_index().set_index(["component", "fit"]).sort_index().to_string())
        print(f"\n{label}: sect+tilt fit on each half of the sample")
        print(hv.loc[comps][summ + SECTORS + TILTS].to_string())


if __name__ == "__main__":
    main()
