"""Regional GICS sector returns, as approximate net total returns.

Net-return sector indexes are not available to us for North America and Pacific
(M1NA0xx / M1PC0xx resolve but return no history), so every region uses the price
indexes (MX..0xx) plus a dividend accrual:

    r_net ~ r_price + k_region * dy_12m / periods_per_year

dy_12m is the index's trailing dividend yield as of the prior month end, and k_region
(< 1, net of withholding) is calibrated so the regional price index plus accrual matches
the regional net index on average. validate() reports how close that gets.

Real Estate became its own GICS sector in 2016; before then it sat in Financials, so
Real Estate's return is set to Financials' before its series starts.
"""
import pandas as pd

from data import BBG, returns

SECTORS = ["EN", "MT", "IN", "CD", "CS", "HC", "FN", "IT", "TC", "UT", "RL"]
PREFIX = {"NA": "MXNA", "EME": "MXEU", "PAC": "MXPC"}
NET = {"NA": "M1NA Index", "EME": "NDDUE15 Index", "PAC": "NDDUP Index"}
PPY = {"weekly": 52, "monthly": 12}


def sector_ticker(region, code):
    return f"{PREFIX[region]}0{code} Index"


def parent_ticker(region):
    return f"{PREFIX[region]} Index"


def dividend_yield():
    """Trailing 12m dividend yield (decimal), monthly, wide."""
    df = pd.read_parquet(BBG / "dy_monthly.parquet")
    return df.pivot(index="date", columns="ticker", values="value").sort_index() / 100


def accrual(tickers, index, freq):
    """Per-period dividend accrual, using the yield known at the previous month end."""
    dy = dividend_yield()[list(tickers)].shift(1)          # month m's return uses m-1's yield
    dy = dy.reindex(dy.index.union(index)).ffill().reindex(index)
    return dy / PPY[freq]


def calibrate(freq="weekly", start="2006-12-01"):
    """k per region: mean(net - price) / mean(dy accrual) over the overlap."""
    r = returns(freq)
    k = {}
    for region in PREFIX:
        px, net = parent_ticker(region), NET[region]
        acc = accrual([px], r.index, freq)[px]
        d = pd.concat([r[net] - r[px], acc], axis=1, keys=["gap", "acc"]).loc[start:].dropna()
        k[region] = d["gap"].mean() / d["acc"].mean()
    return k


def sector_returns(freq="weekly", k=None):
    """{region: DataFrame[date x SECTORS]} of approximate net total returns, plus the splice log."""
    k = k or calibrate(freq)
    r = returns(freq)
    out, splices = {}, []
    for region in PREFIX:
        tk = [sector_ticker(region, c) for c in SECTORS]
        df = r[tk] + k[region] * accrual(tk, r.index, freq)
        df.columns = SECTORS
        start = r[sector_ticker(region, "RL")].first_valid_index()
        df.loc[df.index < start, "RL"] = df.loc[df.index < start, "FN"]
        splices.append({"region": region, "block": "sector RL", "splice_date": start.date(),
                        "backfill": "Financials (Real Estate was part of it before GICS 2016)"})
        out[region] = df
    return out, pd.DataFrame(splices)


def market_return(region, freq="weekly", k=None):
    """Regional (standard, non-IMI) market as price + accrual, on the same basis as the sectors."""
    k = k or calibrate(freq)
    r = returns(freq)
    px = parent_ticker(region)
    return r[px] + k[region] * accrual([px], r.index, freq)[px]


def validate(freq="weekly", start="2006-12-01"):
    """How well price + calibrated accrual reproduces true net returns."""
    k = calibrate(freq, start)
    r = returns(freq)
    ppy = PPY[freq]
    rows = []
    pairs = [(reg, parent_ticker(reg), NET[reg]) for reg in PREFIX]
    pairs += [("EME", sector_ticker("EME", c), f"NDRU{n} Index")
              for c, n in [("IT", "IT"), ("FN", "FNCL"), ("HC", "HC"), ("UT", "UTI")]]
    for region, px, net in pairs:
        approx = r[px] + k[region] * accrual([px], r.index, freq)[px]
        d = pd.concat([approx, r[net], r[px]], axis=1, keys=["approx", "net", "price"]).loc[start:].dropna()
        err = d["approx"] - d["net"]
        rows.append({"region": region, "price": px, "net": net, "k": k[region],
                     "gap_price_ann": (d["price"] - d["net"]).mean() * ppy,
                     "gap_approx_ann": err.mean() * ppy, "te_ann": err.std() * ppy ** 0.5,
                     "corr": d["approx"].corr(d["net"])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(validate().to_string(index=False))
