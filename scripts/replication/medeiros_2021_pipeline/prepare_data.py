"""B1 Medeiros (2021) data prep — faithful port of the author 01_get_fred_data.R.

Author pipeline: fbi::fredmd(current.csv) tcode transforms -> drop Prices-group
tcode==6 price indices -> re-add them as 100*diff(log) (tcode-5 override) ->
filter date>=1960-01-01 -> select_if(no NA) (balanced-panel completeness screen).

Fable gate decision: current.csv is lost; use the cached FRED-MD 2016-01 vintage as its
proxy, but apply the AUTHOR ORDER (transform first, then completeness screen). The
112 vs 122 series delta is recorded as a vintage-timing GAP, not a hard gate.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

#: Defaults only. Every one is overridable on the command line so a fresh clone on
#: someone else's machine can run this -- the previous hard-coded absolute path
#: could not (external review, 2026-08-09).
DEFAULT_ARCHIVE = Path("data/fred_md_historical_vintages.zip")
DEFAULT_VINTAGE = "2016-01.csv"
DEFAULT_OUTPUT = Path("runs/medeiros_2021/panel.parquet")
START, END = "1960-01-01", "2015-12-01"

#: Where to obtain the archive. It is NOT redistributed with this repository.
ARCHIVE_SOURCE = (
    "FRED-MD historical vintages, 2015-01 to 2024-12, from the McCracken-Ng "
    "archive at https://www.stlouisfed.org/research/economists/mccracken/fred-databases/"
)

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

def prepare(archive: 'Path | str' = None, vintage: str = None) -> pd.DataFrame:
    archive = Path(archive) if archive is not None else DEFAULT_ARCHIVE
    vintage = vintage or DEFAULT_VINTAGE
    raw = pd.read_csv(io.BytesIO(zipfile.ZipFile(archive).read(vintage)))
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

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Medeiros (2021) panel from a FRED-MD vintage archive.",
        epilog=f"Archive source: {ARCHIVE_SOURCE}",
    )
    parser.add_argument(
        "--archive", type=Path, default=DEFAULT_ARCHIVE,
        help=f"zip of FRED-MD historical vintages (default: {DEFAULT_ARCHIVE})",
    )
    parser.add_argument(
        "--vintage", default=DEFAULT_VINTAGE,
        help=f"member to read from the archive (default: {DEFAULT_VINTAGE})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"where to write the panel (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if not args.archive.exists():
        parser.error(
            f"archive not found: {args.archive}\n"
            f"  This repository does not redistribute it. Obtain it from:\n"
            f"  {ARCHIVE_SOURCE}\n"
            f"  then pass --archive /path/to/that.zip"
        )

    print(f"archive : {args.archive}")
    print(f"sha256  : {_sha256(args.archive)}")
    print(f"member  : {args.vintage}")

    panel = prepare(archive=args.archive, vintage=args.vintage)
    print("panel shape:", panel.shape, "(months x series)")
    print("months:", len(panel), "| complete series:", panel.shape[1], "(paper=122)")
    print("CPIAUCSL present:", "CPIAUCSL" in panel.columns)
    cpi = panel["CPIAUCSL"]
    s1 = cpi.loc["1990-01":"2000-12"]; s2 = cpi.loc["2001-01":"2015-12"]
    print(f"CPI monthly inflation  Sd(1990-2000)={s1.std():.3f}  Sd(2001-2015)={s2.std():.3f}"
          f"  (paper Table 1: 0.17 / 0.32)")
    print(f"CPI inflation mean(1990-2000)={s1.mean():.3f} mean(2001-2015)={s2.mean():.3f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.output)
    print(f"saved   -> {args.output}")
    print(f"sha256  : {_sha256(args.output)}")


if __name__ == "__main__":
    main()
