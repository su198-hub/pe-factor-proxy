"""Returns-based style decomposition of each PERT component into building blocks.

    r_component = alpha + b_core * r_core + sum_i w_i * tilt_i + e

Constraints follow the PERT methodology: tilt weights >= 0, Low Size <= 0.4.
b_core is left free in [0, 3] (it is the net market exposure; the leg analysis put it
at ~0.9-1.2).

Also reports unconstrained OLS with HAC t-stats, so it is visible where the
non-negativity constraints bind and which tilts are statistically distinguishable.

Buyout and VC are printed separately. Output: output/style_decomp/.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import lsq_linear

from blocks import BLOCKS, TILTS, block_returns
from data import ROOT, pert_family, returns

OUT = ROOT / "output" / "style_decomp"
FREQ = "weekly"
PPY = 52                 # periods per year
WINDOW = 156             # rolling window: 3 years of weeks
STEP = 13                # re-estimate each quarter
SIZE_CAP = 0.4           # MSCI's Low Size guardrail
CORE_BOUNDS = (0.0, 3.0)


def fit(y, X):
    """Bounded least squares with a free intercept. X columns in BLOCKS order."""
    A = np.column_stack([np.ones(len(y)), X])
    lo = [-np.inf, CORE_BOUNDS[0]] + [0.0] * len(TILTS)
    hi = [np.inf, CORE_BOUNDS[1]] + [SIZE_CAP if t == "size" else np.inf for t in TILTS]
    res = lsq_linear(A, y, bounds=(lo, hi))
    resid = y - A @ res.x
    return res.x, 1 - resid.var() / y.var(), resid


def component_data(r, blocks, comp, tick):
    region = comp.split("_")[0]
    d = pd.concat([r[tick].rename("y"), blocks[region]], axis=1).dropna()
    return d["y"].values, d[BLOCKS].values, d.index


def full_sample(r, blocks, fam):
    rows, ols_rows = [], []
    for comp, t in fam.items():
        y, X, idx = component_data(r, blocks, comp, t["net"])
        coef, r2, resid = fit(y, X)
        core_only = sm.OLS(y, sm.add_constant(X[:, :1])).fit().rsquared
        rows.append({"component": comp, "start": idx[0].date(), "n": len(y),
                     **dict(zip(BLOCKS, coef[1:])), "alpha_ann": coef[0] * PPY,
                     "r2": r2, "r2_core_only": core_only, "te_ann": resid.std() * np.sqrt(PPY)})
        ols = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        ols_rows.append({"component": comp,
                         **{f"{b}": f"{c:.2f} ({tv:.1f})" for b, c, tv in zip(BLOCKS, ols.params[1:], ols.tvalues[1:])}})
    return pd.DataFrame(rows).set_index("component"), pd.DataFrame(ols_rows).set_index("component")


def rolling(r, blocks, fam):
    rows = []
    for comp, t in fam.items():
        y, X, idx = component_data(r, blocks, comp, t["net"])
        for end in range(WINDOW, len(y) + 1, STEP):
            sl = slice(end - WINDOW, end)
            coef, r2, _ = fit(y[sl], X[sl])
            rows.append({"component": comp, "date": idx[end - 1], **dict(zip(BLOCKS, coef[1:])),
                         "alpha_ann": coef[0] * PPY, "r2": r2})
    return pd.DataFrame(rows)


def show(title, df):
    print(f"\n{title}")
    print(df.to_string())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.width", 200)
    fam = pert_family()
    r = returns(FREQ)
    blocks, splices = block_returns(FREQ)

    show("Backfilled blocks (primary tilt series starts later than PERT history):", splices)
    for region, b in blocks.items():
        show(f"Tilt correlations, {region} (weekly, 2006-12 on):", b.loc["2006-12":, TILTS].corr())

    fs, ols = full_sample(r, blocks, fam)
    roll = rolling(r, blocks, fam)
    fs.to_csv(OUT / "full_sample.csv")
    ols.to_csv(OUT / "ols_unconstrained.csv")
    roll.to_csv(OUT / "rolling.csv", index=False)
    splices.to_csv(OUT / "splices.csv", index=False)

    for label, suffix in [("BUYOUT", "_BO"), ("VENTURE", "_VC")]:
        comps = [c for c in fs.index if c.endswith(suffix)]
        show(f"===== {label}: constrained weights, full sample (weekly) =====", fs.loc[comps])
        show(f"{label}: unconstrained OLS, coef (HAC t-stat)", ols.loc[comps])
        med = roll[roll["component"].isin(comps)].groupby("component")[BLOCKS + ["r2"]]
        show(f"{label}: rolling 3y weights - 10th pct", med.quantile(0.1))
        show(f"{label}: rolling 3y weights - median", med.median())
        show(f"{label}: rolling 3y weights - 90th pct", med.quantile(0.9))


if __name__ == "__main__":
    main()
