"""Build Hounyo-Li (2026) macro CSV inputs from the author Excel files.

The author reproducibility package ships legacy ``.xls`` workbooks.  This
converter reads the extracted macro panel and inflation target from
``qa/hounyo_li_matlab`` and writes normalized CSV files under ``qa/``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
AUTHOR_DIR = (
    REPO_ROOT
    / "qa"
    / "hounyo_li_matlab"
    / "Reproducibility package"
    / "Empirical"
    / "Macro"
    / "Inflation_results"
)
PANEL_XLS = AUTHOR_DIR / "Macrodataset.xls"
INFLATION_XLS = AUTHOR_DIR / "USinflation.xls"
PANEL_CSV = REPO_ROOT / "qa" / "hounyo_li_macro_panel.csv"
INFLATION_CSV = REPO_ROOT / "qa" / "hounyo_li_inflation.csv"
MERGED_CSV = REPO_ROOT / "qa" / "hounyo_li_macro_inflation_panel.csv"
MANIFEST_JSON = REPO_ROOT / "qa" / "hounyo_li_author_data_manifest.json"


def _read_workbook(path: Path) -> pd.DataFrame:
    """Read ``path`` with pandas, using LibreOffice conversion if xlrd is absent."""

    try:
        return pd.read_excel(path, sheet_name=0, header=0)
    except ImportError as exc:
        if "xlrd" not in str(exc).lower():
            raise
        soffice = shutil.which("libreoffice") or shutil.which("soffice")
        if soffice is None:
            raise RuntimeError(
                "Reading .xls requires xlrd or LibreOffice/soffice on PATH."
            ) from exc
        with tempfile.TemporaryDirectory(prefix="hl2026_xlsx_") as tmp:
            outdir = Path(tmp)
            subprocess.run(
                [
                    soffice,
                    "--headless",
                    "--convert-to",
                    "xlsx",
                    "--outdir",
                    str(outdir),
                    str(path),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            converted = outdir / f"{path.stem}.xlsx"
            return pd.read_excel(converted, sheet_name=0, header=0, engine="openpyxl")


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out = out.rename(columns={out.columns[0]: "date"})
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.to_period("M").dt.to_timestamp()
    for column in out.columns:
        if column != "date":
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def build_data() -> dict[str, object]:
    if not PANEL_XLS.exists():
        raise FileNotFoundError(f"missing author macro panel: {PANEL_XLS}")
    if not INFLATION_XLS.exists():
        raise FileNotFoundError(f"missing author inflation target: {INFLATION_XLS}")

    panel = _normalize_frame(_read_workbook(PANEL_XLS))
    inflation = _normalize_frame(_read_workbook(INFLATION_XLS))
    if "CPIAUCSL" not in inflation.columns:
        raise ValueError("USinflation.xls did not contain expected CPIAUCSL column")

    merged = panel.merge(inflation[["date", "CPIAUCSL"]], on="date", how="inner")
    numeric_panel = [column for column in panel.columns if column != "date"]
    numeric_inflation = [column for column in inflation.columns if column != "date"]
    numeric_merged = [column for column in merged.columns if column != "date"]
    panel.loc[:, numeric_panel] = panel.loc[:, numeric_panel].astype(float)
    inflation.loc[:, numeric_inflation] = inflation.loc[:, numeric_inflation].astype(float)
    merged.loc[:, numeric_merged] = merged.loc[:, numeric_merged].astype(float)

    PANEL_CSV.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_CSV, index=False)
    inflation.to_csv(INFLATION_CSV, index=False)
    merged.to_csv(MERGED_CSV, index=False)

    h1 = merged[(merged["date"] >= "1973-03-01") & (merged["date"] <= "2023-03-01")]
    manifest = {
        "source_panel": str(PANEL_XLS.relative_to(REPO_ROOT)),
        "source_inflation": str(INFLATION_XLS.relative_to(REPO_ROOT)),
        "panel_csv": str(PANEL_CSV.relative_to(REPO_ROOT)),
        "inflation_csv": str(INFLATION_CSV.relative_to(REPO_ROOT)),
        "merged_csv": str(MERGED_CSV.relative_to(REPO_ROOT)),
        "rows": int(merged.shape[0]),
        "predictor_columns": int(panel.shape[1] - 1),
        "merged_columns": int(merged.shape[1]),
        "date_start": merged["date"].min().strftime("%Y-%m-%d"),
        "date_end": merged["date"].max().strftime("%Y-%m-%d"),
        "h1_full_sample_rows": int(h1.shape[0]),
        "h1_full_sample_start": "1973-03-01",
        "h1_full_sample_end": "2023-03-01",
        "h1_rolling_window": 240,
        "h1_expected_oos_origins": 361,
        "notes": [
            "Author Excel h=1 full-sample range is B122:B722 / B122:DW722.",
            "The merged CSV keeps all author workbook dates; the smoke runner slices the h=1 full-sample range.",
        ],
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    manifest = build_data()
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
