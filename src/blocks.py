"""Building-block return series, as defined in config/blocks.csv.

core  = regional parent index return
tilt  = long index return - short index return   (a self-financing style tilt)

Where a block has a backfill pair, dates before the primary pair starts use the
backfill tilt instead. The splice date is reported so it is never silent.
"""
import pandas as pd

from data import ROOT, returns

BLOCKS = ["core", "value", "growth", "momentum", "minvol", "leverage", "size"]
TILTS = BLOCKS[1:]


def definitions():
    return pd.read_csv(ROOT / "config" / "blocks.csv", keep_default_na=False, na_values=[""])  # "NA" is a region


def block_returns(freq="weekly"):
    """Returns ({region: DataFrame[date x block]}, splice log)."""
    r = returns(freq)
    out, splices = {}, []
    for region, defs in definitions().groupby("region", sort=False):
        cols = {}
        for _, d in defs.iterrows():
            if d["block"] == "core":
                cols["core"] = r[d["long"]]
                continue
            s = r[d["long"]] - r[d["short"]]
            if pd.notna(d["backfill_long"]):
                back = r[d["backfill_long"]] - r[d["backfill_short"]]
                start = s.first_valid_index()
                s = s.where(s.index >= start, back)
                splices.append({"region": region, "block": d["block"], "splice_date": start.date(),
                                "backfill": f"{d['backfill_long']} - {d['backfill_short']}"})
            cols[d["block"]] = s
        out[region] = pd.DataFrame(cols)[BLOCKS]
    return out, pd.DataFrame(splices)
