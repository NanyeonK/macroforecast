"""B1 Medeiros (2021) data prep — faithful port of the author 01_get_fred_data.R.

Author pipeline: fbi::fredmd(current.csv) tcode transforms -> drop Prices-group
tcode==6 price indices -> re-add them as 100*diff(log) (tcode-5 override) ->
filter date>=1960-01-01 -> select_if(no NA) (balanced-panel completeness screen).

Fable gate decision: current.csv is lost; use the cached FRED-MD 2016-01 vintage as its
proxy, but apply the AUTHOR ORDER (transform first, then completeness screen). The
112 vs 122 series delta is recorded as a vintage-timing GAP, not a hard gate.
"""
from __future__ import annotations
import io, zipfile
import numpy as np
import pandas as pd

ZIP = "/home/nanyeon99/project/macroforecast_replication_cache/fred_md/historical/historical-vintages-of-fred-md-2015-01-to-2024-12.zip"
VINTAGE = "2016-01.csv"
START, END = "1960-01-01", "2015-12-01"

# McCracken-Ng Group 7 (Prices), all tcode==6 — override to 100*dlog (verified vs
# appendix group-7 block and the 2016-01 tcode row; monetary/wage tcode-6 series excluded).
PRICE_OVERRIDE = {
    "WPSFD49207","WPSFD49502","WPSID61","WPSID62","OILPRICEx","PPICMM","PPICRM",
    "PPIFCG","PPIFGS","PPIITM","CPIAUCSL","CPIAPPSL","CPITRNSL","CPIMEDSL",
    "CUSR0000SAC","CUSR0000SAD","CUSR0000SAS","CPIULFSL","CUSR0000SA0L2",
    "CUSR0000SA0L5","CUUR0000SA0L2","CUUR0000SAD","PCEPI","DDURRG3M086SBEA",
    "DNDGRG3M086SBEA","DSERRG3M086SBEA",
}

def _transform(x: pd.Series, tcode: int) -> pd.Series:
    lx = np.log(x.where(x > 0))
    return {
        1: x,
        2: x.diff(),
        3: x.diff().diff(),
        4: lx,
        5: lx.diff(),
        6: lx.diff().diff(),
        7: (x / x.shift(1) - 1.0).diff(),
    }[int(tcode)]

def prepare() -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(zipfile.ZipFile(ZIP).read(VINTAGE)))
    tcodes = raw.iloc[0, 1:].astype(float).astype(int)
    data = raw.iloc[1:].copy()
    data = data[data["sasdate"].notna()]
    data["sasdate"] = pd.to_datetime(data["sasdate"])
    data = data.set_index("sasdate").sort_index()
    out = {}
    for col in data.columns:
        series = pd.to_numeric(data[col], errors="coerce")
        tcode = 5 if col in PRICE_OVERRIDE else int(tcodes[col])   # price override -> dlog
        t = _transform(series, tcode)
        if col in PRICE_OVERRIDE:
            t = 100.0 * t                                          # author: 100*diff(log)
        out[col] = t
    panel = pd.DataFrame(out, index=data.index)
    panel = panel.loc[(panel.index >= START) & (panel.index <= END)]
    complete = panel.columns[panel.notna().all(axis=0)]
    panel = panel[complete]
    return panel

if __name__ == "__main__":
    panel = prepare()
    print("panel shape:", panel.shape, "(months x series)")
    print("months:", len(panel), "| complete series:", panel.shape[1], "(paper=122)")
    print("CPIAUCSL present:", "CPIAUCSL" in panel.columns)
    cpi = panel["CPIAUCSL"]
    s1 = cpi.loc["1990-01":"2000-12"]; s2 = cpi.loc["2001-01":"2015-12"]
    print(f"CPI monthly inflation  Sd(1990-2000)={s1.std():.3f}  Sd(2001-2015)={s2.std():.3f}"
          f"  (paper Table 1: 0.17 / 0.32)")
    print(f"CPI inflation mean(1990-2000)={s1.mean():.3f} mean(2001-2015)={s2.mean():.3f}")
    panel.to_parquet("qa/medeiros_panel.parquet")
    print("saved -> qa/medeiros_panel.parquet")
