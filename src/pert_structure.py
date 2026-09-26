"""How is PERT assembled from its published pieces?

Three checks, all on published index returns (no estimation of PE anything):

1. Leg decomposition. Each component is long a basket and short its Core Factor, so
   r_component ~= a * r_long - b * r_short. a and b are the gross long and short
   exposures; a - b should be ~1 (fully invested) and b is the total overlay weight.
2. Short leg vs regional IMI. If the short leg is the sector-reweighted Core Factor,
   it should track the regional IMI closely but not exactly.
3. Top-level weights. MXWOPERT = sum_j w_j * component_j, with w >= 0 and sum w = 1.

Writes CSVs to output/pert_structure/ and prints a summary.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data import pert_family, returns

OUT = __import__("data").ROOT / "output" / "pert_structure"
PARENT = {"NA": "M1NAIM Index", "EME": "MIMUEURN Index", "PAC": "M1PCIM Index"}
FREQ = "weekly"
WINDOW = 52  # rolling window, weeks


def ols(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    r2 = 1 - resid.var() / y.var()
    return coef, r2, resid


def leg_decomposition(r, fam):
    rows, rolling = [], {}
    for comp, t in fam.items():
        d = r[[t["net"], t["long"], t["short"]]].dropna()
        y, L, S = d.iloc[:, 0].values, d.iloc[:, 1].values, d.iloc[:, 2].values
        coef, r2, resid = ols(y, np.column_stack([L, -S]))
        rows.append({"component": comp, "start": d.index[0].date(), "n": len(d), "alpha_ann": coef[0] * 52,
                     "long_a": coef[1], "short_b": coef[2], "net_a_minus_b": coef[1] - coef[2],
                     "r2": r2, "te_ann": resid.std() * np.sqrt(52)})
        roll = []
        for end in range(WINDOW, len(d) + 1):
            w = d.iloc[end - WINDOW:end]
            c, _, _ = ols(w.iloc[:, 0].values, np.column_stack([w.iloc[:, 1].values, -w.iloc[:, 2].values]))
            roll.append((w.index[-1], c[1], c[2]))
        rolling[comp] = pd.DataFrame(roll, columns=["date", "long_a", "short_b"]).set_index("date")
    return pd.DataFrame(rows).set_index("component"), rolling


def short_vs_parent(r, fam):
    rows = []
    for comp, t in fam.items():
        parent = PARENT[comp.split("_")[0]]
        d = r[[t["short"], parent]].dropna()
        coef, r2, resid = ols(d.iloc[:, 0].values, d.iloc[:, 1].values)
        rows.append({"component": comp, "parent": parent, "corr": d.corr().iloc[0, 1], "beta": coef[1],
                     "te_ann": resid.std() * np.sqrt(52),
                     "short_ret_ann": (1 + d.iloc[:, 0]).prod() ** (52 / len(d)) - 1,
                     "parent_ret_ann": (1 + d.iloc[:, 1]).prod() ** (52 / len(d)) - 1})
    return pd.DataFrame(rows).set_index("component")


def constrained_weights(y, X):
    """min ||y - Xw||^2  s.t. w >= 0, sum w = 1."""
    k = X.shape[1]
    res = minimize(lambda w: ((y - X @ w) ** 2).sum(), np.full(k, 1 / k), method="SLSQP",
                   bounds=[(0, 1)] * k, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    resid = y - X @ res.x
    return res.x, 1 - resid.var() / y.var(), resid


def top_level_weights(r, fam):
    comps = list(fam)
    d = r[["MXWOPERT Index"] + [fam[c]["net"] for c in comps]].dropna()
    y, X = d.iloc[:, 0].values, d.iloc[:, 1:].values
    w, r2, resid = constrained_weights(y, X)
    full = pd.Series(w, index=comps, name="full_sample")
    roll = []
    for end in range(WINDOW, len(d) + 1, 13):  # step a quarter at a time
        win = d.iloc[end - WINDOW:end]
        wr, _, _ = constrained_weights(win.iloc[:, 0].values, win.iloc[:, 1:].values)
        roll.append(pd.Series(wr, index=comps, name=win.index[-1]))
    return full, r2, resid.std() * np.sqrt(52), pd.DataFrame(roll)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fam = pert_family()
    r = returns(FREQ)
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 160)

    legs, rolling = leg_decomposition(r, fam)
    legs.to_csv(OUT / "leg_decomposition.csv")
    pd.concat(rolling, axis=1, sort=True).to_csv(OUT / "leg_decomposition_rolling.csv")
    print("\n1. r_component ~ a*r_long - b*r_short (weekly, full sample)")
    print(legs.to_string())
    print("\n   rolling 52w exposures, min / median / max:")
    for comp, df in rolling.items():
        q = df.quantile([0, 0.5, 1])
        print(f"   {comp:7s} long a {q['long_a'].iloc[0]:.2f} / {q['long_a'].iloc[1]:.2f} / {q['long_a'].iloc[2]:.2f}"
              f"   short b {q['short_b'].iloc[0]:.2f} / {q['short_b'].iloc[1]:.2f} / {q['short_b'].iloc[2]:.2f}")

    sp = short_vs_parent(r, fam)
    sp.to_csv(OUT / "short_leg_vs_parent.csv")
    print("\n2. short leg vs regional IMI (weekly)")
    print(sp.to_string())

    full, r2, te, roll = top_level_weights(r, fam)
    roll.to_csv(OUT / "top_level_weights_rolling.csv")
    print(f"\n3. MXWOPERT on six components, w>=0, sum=1 (weekly, full sample): R2={r2:.4f}, TE={te:.4f}")
    print(full.to_string())
    print("\n   rolling 52w weights, sampled yearly:")
    print(roll.iloc[::4].to_string())


if __name__ == "__main__":
    main()
