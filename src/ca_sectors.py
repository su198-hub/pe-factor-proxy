"""PE and VC sector weights (by NAV) from Cambridge Associates' public benchmark commentaries.

MSCI builds PERT's Core Factor by reweighting the regional IMI to the PE category's
country-sector mix, taken from its private capital database, which we do not have.
Cambridge Associates (CA) publishes, every six months, the GICS sector breakdown by market
value (NAV) of its US PE index, US VC index and developed ex US PE/VC indexes, next to a
public index. This module downloads those commentaries and extracts the weights.

    discover()   find each edition's PDF (publishedresearch.cambridgeassociates.com)
    download()   cache PDFs to data/ca/ (git-ignored; CA material is copyrighted)
    parse()      extract every sector chart -> data/ca/sector_weights.csv (long format)
    quarterly()  GICS weights per (region, strategy), quarter-ends, interpolated

How a chart is read. Chart labels come out of the PDF text as "Sector, value" pairs, each
sector twice (one bar per series), followed by the y axis (0 10 20 ... 100) and a legend.
The pairs are the contiguous run of known sector labels just before the axis (plus any
label printed after the axis whose partner is in the run). The first occurrence of a
label belongs to the first series in the legend, the second to the second series. The
legend also says which index each series is (CA vs Russell 2000 / Nasdaq / MSCI) and
whether it is a historical comparison (CA 2007 or 2010 vs today).

Coverage: US PE and US VC from mid-2019, plus end-2007 (in the end-2019 edition) and
end-2010 (end-2020 edition) snapshots; developed ex US from mid-2019, one combined PE/VC
index until 2023 and separate PE and VC indexes after. "Other" (and any sector CA does not
name) is split across the missing GICS sectors in proportion to the regional market.
"""
import re
import time
import urllib.request

import pandas as pd

from data import ROOT

OUT = ROOT / "data" / "ca"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
SITE = "https://publishedresearch.cambridgeassociates.com/insight/{series}-benchmark-commentary-{half}-{year}/"
SERIES = {"US": "us-pe-vc", "XUS": "global-ex-us-pe-vc"}
FIRST_YEAR = 2019

GICS = ["EN", "MT", "IN", "CD", "CS", "HC", "FN", "IT", "TC", "UT", "RL"]
LABELS = {  # CA label (lower case, single spaces) -> GICS code
    "information technology": "IT", "healthcare": "HC", "health care": "HC",
    "industrials": "IN", "consumer discretionary": "CD", "con disc": "CD", "cons disc": "CD",
    "financials": "FN", "communication services": "TC", "comm srvs": "TC", "comm services": "TC",
    "materials": "MT", "consumer staples": "CS", "cons stap": "CS", "con stap": "CS",
    "energy": "EN", "utilities": "UT", "real estate": "RL", "other": "Other",
}
_WORDS = sorted(LABELS, key=len, reverse=True)
PAIR = re.compile(r"(?<![A-Za-z])(" + "|".join(r"\s+".join(map(re.escape, w.split())) for w in _WORDS)
                  + r")\s*,\s*(\d{1,2}(?:\.\s?\d)?)(?!\d)", re.I)
AXIS = re.compile(r"(?:^|\n)0\s*\n10\s*\n20\s*\n30")
DATE_GLUED = re.compile(r"([A-Za-z]+)(\d+),(\d{4})")  # "December31,2021" once spaces are removed
# matched on legend text with ALL whitespace removed (some PDFs split words: "R ussell", "In dex")
LEGEND = re.compile(r"(Russell2000|NasdaqComposite|NASDAQComposite|MSCI[A-Za-z]+?(?=Index)|"
                    r"(?:CA|CambridgeAssociatesLLC)[A-Za-z/]+?Index)"
                    r"(?:[^()]{0,40}\(SectorWeightsasof([A-Za-z]+\d+,\d{4})\))?")


def _get(url):
    try:
        return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
    except Exception:
        return None


def discover(last_year=None):
    """{tag: pdf_url}; tag like US_2024H1 (as of June) / US_2024H2 (as of December)."""
    last_year = last_year or pd.Timestamp.today().year
    found = {}
    for region, series in SERIES.items():
        for year in range(FIRST_YEAR, last_year + 1):
            for half, h in [("first-half", "H1"), ("calendar-year", "H2")]:
                page = _get(SITE.format(series=series, half=half, year=year))
                if not page:
                    continue
                pdfs = re.findall(rb"https://[a-z.]*cambridgeassociates\.com/wp-content/uploads/[^\"' ]+\.pdf", page)
                if pdfs:
                    found[f"{region}_{year}{h}"] = pdfs[0].decode()
                time.sleep(0.3)
    return found


def download(found):
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, url in found.items():
        f = OUT / f"{tag}.pdf"
        if not f.exists():
            data = _get(url)
            if data and data[:4] == b"%PDF":
                f.write_bytes(data)
                time.sleep(0.3)


def _code(label):
    return LABELS[re.sub(r"\s+", " ", label).strip().lower()]


def _legend(text):
    """Series in legend order: [(who 'CA'|'public', kind 'PE'|'VC'|'PEVC'|'EM'|None, date or None)]."""
    text = re.sub(r"\s+", "", text)
    out = []
    for m in LEGEND.finditer(text):
        name = m.group(1)
        date = re.sub(DATE_GLUED, lambda g: f"{g[1]} {g[2]}, {g[3]}", m.group(2)) if m.group(2) else None
        who = "public" if re.match(r"Russell|Nasdaq|NASDAQ|MSCI", name) else "CA"
        if "Emerging" in name:
            kind = "EM"
        elif re.search(r"VentureCapital|Nasdaq|NASDAQ", name):
            kind = "VC"
        elif re.search(r"PrivateEquity|Russell", name):
            kind = "PE"
        elif re.search(r"Developed|exUS|EAFE", name):
            kind = "PEVC"
        else:
            kind = None
        out.append((who, kind, date))
    return out[:2]


def _run_before(text, end):
    """Contiguous run of sector pairs ending just before `end` (only whitespace between them)."""
    run = []
    for m in reversed(list(PAIR.finditer(text, max(0, end - 3000), end))):
        if text[m.end():(run[0].start() if run else end)].strip():
            break
        run.insert(0, m)
    return run


def parse_pdf(path):
    from pypdf import PdfReader
    tag = path.stem
    region, when = tag.split("_")
    asof = pd.Timestamp(f"{when[:4]}-{'06-30' if when[4:] == 'H1' else '12-31'}")
    text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    rows = []
    for ax in AXIS.finditer(text):
        run = _run_before(text, ax.start())
        if len(run) < 8:
            continue
        after = text[ax.end(): ax.end() + 400]
        leg = _legend(after)
        if len(leg) < 2 or any(k == "EM" for _, k, _ in leg):
            continue
        pairs = [(_code(m.group(1)), float(m.group(2).replace(" ", ""))) for m in run]
        counts = {}
        for lab, _ in pairs:
            counts[lab] = counts.get(lab, 0) + 1
        for m in PAIR.finditer(after):  # labels printed after the axis (one or both bars of a sector)
            lab = _code(m.group(1))
            if counts.get(lab, 0) < 2:
                pairs.append((lab, float(m.group(2).replace(" ", ""))))
                counts[lab] = counts.get(lab, 0) + 1
        title = re.sub(r"\s+", " ", text[max(0, run[0].start() - 500): run[0].start()])
        kinds = {k for _, k, _ in leg if k}
        if "VC" in kinds:
            strat = "VC"
        elif "PE" in kinds:
            strat = "PE"
        elif region == "XUS" and ("PEVC" in kinds or re.search(r"PE/VC", title)):
            strat = "PEVC"
        else:
            strat = "PE"
        first, second = {}, {}
        for lab, v in pairs:
            (second if lab in first else first).setdefault(lab, v)
        for (who, _, date), ser in zip(leg, [first, second]):
            # legend dates only mark historical comparisons (CA 2007 / 2010); ignore typos near the edition date
            d = pd.Timestamp(date) if date and pd.Timestamp(date).year <= asof.year - 5 else asof
            for lab, v in ser.items():
                rows.append({"edition": tag, "region": region, "strategy": strat, "series": who,
                             "asof": d, "label": lab, "weight": v})
    return rows


def parse():
    rows = []
    for f in sorted(OUT.glob("*.pdf")):
        rows += parse_pdf(f)
    df = pd.DataFrame(rows)
    # one observation per (region, strategy, series, date, label): the latest edition wins
    df = df.sort_values("edition").drop_duplicates(["region", "strategy", "series", "asof", "label"], keep="last")
    df.to_csv(OUT / "sector_weights.csv", index=False)
    return df


def load():
    return pd.read_csv(OUT / "sector_weights.csv", parse_dates=["asof"])


def to_gics(row, market):
    """Spread 'Other' (and any unnamed sector) across the missing GICS sectors pro rata to `market`."""
    named = row.drop("Other", errors="ignore").dropna()
    w = named.reindex(GICS).fillna(0.0)
    other = max(0.0, 100.0 - named.sum())
    missing = [s for s in GICS if s not in named.index]
    if missing and other > 0:
        m = market.reindex(missing).fillna(0.0)
        w[missing] = other * m / m.sum() if m.sum() > 0 else other / len(missing)
    return w / w.sum()


def quarterly(market_weights, start="2006-12-31", end=None):
    """{(region, strategy): DataFrame[quarter-end x GICS]} of CA sector weights (sum to 1).

    market_weights: {"US": Series, "XUS": Series} (GICS -> weight) of the public market used to
    split "Other" across the sectors CA does not name.
    Weights are interpolated linearly between observations, held at the earliest one before
    it (end-2007 US, end-2010 ex US) and at the latest one after. Ex US uses the separate PE
    or VC index where published and the combined PE/VC index otherwise.
    """
    ca = load()
    ca = ca[ca["series"] == "CA"]
    end = end or ca["asof"].max()
    qidx = pd.date_range(start, end, freq="QE")
    out = {}
    for region in ["US", "XUS"]:
        for strat in ["PE", "VC"]:
            sub = ca[(ca["region"] == region) & (ca["strategy"].isin([strat, "PEVC"]))]
            if sub.empty:
                continue
            sub = sub.assign(pri=(sub["strategy"] == strat).astype(int)).sort_values("pri")
            sub = sub.drop_duplicates(["asof", "label"], keep="last")
            wide = sub.pivot(index="asof", columns="label", values="weight")
            g = pd.DataFrame({d: to_gics(r.dropna(), market_weights[region]) for d, r in wide.iterrows()}).T
            g = g.reindex(g.index.union(qidx)).interpolate(method="time").ffill().bfill().reindex(qidx)
            out[(region, strat)] = g
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    download(discover())
    df = parse()
    chk = df.groupby(["region", "strategy", "series", "asof"])["weight"].agg(["sum", "count"]).unstack("series")
    print("Charts parsed (each series should sum to ~100):")
    print(chk.round(1).to_string())
    ca = df[df["series"] == "CA"].pivot_table(index=["region", "strategy", "asof"], columns="label", values="weight")
    print("\nCA index sector weights, % of NAV")
    print(ca.round(1).to_string())
