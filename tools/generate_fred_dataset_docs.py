from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "fred_dataset"
SOURCE_DIR = ROOT / "build" / "fred_dataset_sources"
CACHE_ROOT = Path.home() / ".macrocast" / "raw"

FRED_MD_CURRENT_URL = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv"
)
FRED_QD_CURRENT_URL = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/quarterly/current.csv"
)
FRED_SD_LANDING_URL = "https://www.stlouisfed.org/research/economists/owyang/fred-sd"
FRED_SD_SERIES_URL_TEMPLATE = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-sd/series/series-{vintage}.xlsx"
)
FRED_MD_APPENDIX_URL = "https://files.stlouisfed.org/files/htdocs/fred-databases/Appendix_Tables_Update.pdf"
FRED_QD_APPENDIX_URL = "https://files.stlouisfed.org/files/htdocs/fred-databases/FRED-QD_appendix_v6.pdf"

TCODE_LABELS = {
    "1": "No transformation",
    "2": "First difference",
    "3": "Second difference",
    "4": "Log level",
    "5": "First log difference",
    "6": "Second log difference",
    "7": "First difference of percent change",
    "8": "Quarterly volatility transform",
}

STATE_NAMES = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}


@dataclass(frozen=True)
class FredColumn:
    position: int
    mnemonic: str
    tcode: str
    tcode_label: str
    description: str = ""
    group: str = ""
    factor_flag: str = ""


def _request(url: str) -> Request:
    return Request(url, headers={"User-Agent": "macrocast docs generator"})


def _download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target
    with urlopen(_request(url), timeout=30) as src:
        target.write_bytes(src.read())
    return target


def _download_optional(url: str, target: Path) -> Path | None:
    try:
        return _download(url, target)
    except (OSError, URLError):
        return None


def _cached_or_download(cache_path: Path, url: str, target: Path) -> Path:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    return _download(url, target)


def _pdf_text(pdf_path: Path) -> str:
    txt_path = pdf_path.with_suffix(".txt")
    subprocess.run(["pdftotext", "-layout", str(pdf_path), str(txt_path)], check=True)
    return txt_path.read_text(errors="replace")


def _clean_pdf_text(value: str) -> str:
    value = value.replace("\ufb01", "fi").replace("\ufb02", "fl")
    value = value.replace("…", "fi")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _match_known_mnemonic(line: str, known: list[str]) -> tuple[str, re.Match[str]] | None:
    for mnemonic in known:
        pattern = r"(?<!\S)" + re.escape(mnemonic) + r"(?!\S)"
        match = re.search(pattern, line)
        if match and re.search(r"\d", line[: match.start()]):
            return mnemonic, match
    return None


def _append_continuation(lines: list[str], start: int) -> str:
    parts: list[str] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped or stripped == "\x0c":
            continue
        if "Group " in stripped:
            break
        if re.match(r"^\d+\s+\d+", stripped):
            break
        if re.search(r"\bid\s+.*tcode\b", stripped, re.I):
            break
        if re.match(r"^\d+$", stripped):
            continue
        parts.append(re.sub(r"^\d+\s{2,}", "", stripped))
        if len(parts) >= 2:
            break
    return " ".join(parts)


def _description_from_tail(tail: str, *, qd: bool) -> str:
    parts = [part.strip() for part in re.split(r"\s{2,}", tail.strip()) if part.strip()]
    if not parts:
        return ""
    if qd:
        return parts[-1]
    return parts[0]


def _parse_appendix_descriptions(text: str, known_columns: list[str], *, qd: bool) -> dict[str, tuple[str, str]]:
    known = sorted(known_columns, key=len, reverse=True)
    rows: dict[str, tuple[str, str]] = {}
    group = ""
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        group_match = re.search(r"Group\s+\d+:\s*(.+)", line)
        if group_match:
            group = _clean_pdf_text(group_match.group(1))
            continue
        match = _match_known_mnemonic(line, known)
        if not match:
            continue
        mnemonic, mnemonic_match = match
        tail = line[mnemonic_match.end() :]
        desc = _description_from_tail(tail, qd=qd)
        continuation = _append_continuation(lines, idx)
        if continuation and qd:
            desc = f"{desc} {continuation}"
        rows[mnemonic.lower()] = (group, _clean_pdf_text(desc))
    return rows


def _read_fred_csv(path: Path, *, qd: bool) -> tuple[list[dict[str, str]], str]:
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    header_idx = None
    tcode_idx = None
    factor_idx = None
    for idx, value in raw.iloc[:, 0].items():
        label = str(value).strip().lower()
        if label in {"sasdate", "sasqdate"}:
            header_idx = int(idx)
        elif label in {"transform", "transform:"}:
            tcode_idx = int(idx)
        elif label == "factors":
            factor_idx = int(idx)
    if header_idx is None or tcode_idx is None:
        raise RuntimeError(f"could not locate header/transform rows in {path}")
    header = [str(value).strip() for value in raw.iloc[header_idx].tolist()]
    tcodes = [str(value).strip() for value in raw.iloc[tcode_idx].tolist()]
    factors = [str(value).strip() for value in raw.iloc[factor_idx].tolist()] if factor_idx is not None else []
    data_start = max(x for x in [header_idx, tcode_idx, factor_idx] if x is not None) + 1
    dates = pd.to_datetime(raw.iloc[data_start:, 0], errors="coerce").dropna()
    data_through = dates.iloc[-1].strftime("%Y-%m") if not dates.empty else "unknown"

    rows: list[dict[str, str]] = []
    for idx, mnemonic in enumerate(header[1:], start=1):
        tcode = tcodes[idx] if idx < len(tcodes) and tcodes[idx] else "1"
        row = {
            "position": str(idx),
            "mnemonic": mnemonic,
            "tcode": tcode,
            "tcode_label": TCODE_LABELS.get(tcode, "Unknown"),
            "factor_flag": factors[idx] if qd and idx < len(factors) else "",
        }
        rows.append(row)
    return rows, data_through


def _fred_link(mnemonic: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_]+", mnemonic) and not mnemonic.endswith("x"):
        return f"[FRED](https://fred.stlouisfed.org/series/{mnemonic})"
    return "constructed / appendix"


def _md_table(rows: list[FredColumn]) -> str:
    out = ["| # | Column | T-code | Transform | Group | Definition | Source |", "|---:|---|---:|---|---|---|---|"]
    for row in rows:
        out.append(
            "| "
            + " | ".join(
                [
                    str(row.position),
                    f"`{row.mnemonic}`",
                    row.tcode,
                    row.tcode_label,
                    _escape_cell(row.group or "-"),
                    _escape_cell(row.description or "See official appendix / FRED source page."),
                    _fred_link(row.mnemonic),
                ]
            )
            + " |"
        )
    return "\n".join(out)


def _qd_table(rows: list[FredColumn]) -> str:
    out = [
        "| # | Column | T-code | Transform | SW factor | Group | Definition | Source |",
        "|---:|---|---:|---|---:|---|---|---|",
    ]
    for row in rows:
        out.append(
            "| "
            + " | ".join(
                [
                    str(row.position),
                    f"`{row.mnemonic}`",
                    row.tcode,
                    row.tcode_label,
                    row.factor_flag or "-",
                    _escape_cell(row.group or "-"),
                    _escape_cell(row.description or "See official appendix / FRED source page."),
                    _fred_link(row.mnemonic),
                ]
            )
            + " |"
        )
    return "\n".join(out)


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _load_fred_sd_workbook(path: Path) -> tuple[list[dict[str, str]], list[str], str, Counter[str]]:
    sheets = pd.read_excel(path, sheet_name=None, index_col=0, engine="openpyxl")
    rows: list[dict[str, str]] = []
    states_seen: set[str] = set()
    data_through_values: list[pd.Timestamp] = []
    frequency_counts: Counter[str] = Counter()
    for variable, frame in sorted(sheets.items()):
        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index, errors="coerce")
            frame = frame[frame.index.notna()]
        for state in sorted(str(col) for col in frame.columns):
            series = pd.to_numeric(frame[state], errors="coerce").dropna()
            if not series.empty:
                data_through_values.append(pd.Timestamp(series.index[-1]))
            frequency = _infer_frequency(series)
            frequency_counts[frequency] += 1
            states_seen.add(state)
            rows.append(
                {
                    "column": f"{variable}_{state}",
                    "sd_variable": str(variable),
                    "state": state,
                    "state_name": STATE_NAMES.get(state, state),
                    "native_frequency": frequency,
                    "observed_start": series.index[0].strftime("%Y-%m-%d") if not series.empty else "",
                    "observed_end": series.index[-1].strftime("%Y-%m-%d") if not series.empty else "",
                    "non_missing": str(int(series.shape[0])),
                }
            )
    data_through = max(data_through_values).strftime("%Y-%m") if data_through_values else "unknown"
    return rows, sorted(states_seen), data_through, frequency_counts


def _infer_frequency(series: pd.Series) -> str:
    if series.shape[0] < 2:
        return "unknown"
    periods = pd.DatetimeIndex(series.index).to_period("M")
    deltas = [
        int(right.ordinal - left.ordinal)
        for left, right in zip(periods[:-1], periods[1:], strict=False)
        if right.ordinal > left.ordinal
    ]
    if not deltas:
        return "unknown"
    delta, _ = Counter(deltas).most_common(1)[0]
    return {1: "monthly", 3: "quarterly", 12: "annual"}.get(delta, "irregular")


def _sd_variable_table(rows: list[dict[str, str]]) -> str:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[row["sd_variable"]][row["native_frequency"]] += 1
    out = ["| FRED-SD variable / workbook sheet | Generated state columns | Native-frequency profile |", "|---|---:|---|"]
    for variable in sorted(grouped):
        counts = ", ".join(f"{freq}: {count}" for freq, count in sorted(grouped[variable].items()))
        out.append(f"| `{variable}` | {sum(grouped[variable].values())} | {counts} |")
    return "\n".join(out)


def _sd_column_table(rows: list[dict[str, str]]) -> str:
    out = [
        "| Column | FRED-SD variable | State | Native frequency | Observed window | Non-missing obs |",
        "|---|---|---|---|---|---:|",
    ]
    for row in rows:
        window = f"{row['observed_start']} to {row['observed_end']}" if row["observed_start"] else "-"
        out.append(
            "| "
            + " | ".join(
                [
                    f"`{row['column']}`",
                    f"`{row['sd_variable']}`",
                    f"`{row['state']}` ({_escape_cell(row['state_name'])})",
                    row["native_frequency"],
                    window,
                    row["non_missing"],
                ]
            )
            + " |"
        )
    return "\n".join(out)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def _build_md() -> tuple[list[FredColumn], str]:
    csv_path = _cached_or_download(
        CACHE_ROOT / "fred_md" / "current" / "raw.csv",
        FRED_MD_CURRENT_URL,
        SOURCE_DIR / "fred_md_current.csv",
    )
    raw_rows, data_through = _read_fred_csv(csv_path, qd=False)
    appendix = _download_optional(FRED_MD_APPENDIX_URL, SOURCE_DIR / "fred_md_appendix.pdf")
    descriptions = {}
    if appendix is not None:
        descriptions = _parse_appendix_descriptions(
            _pdf_text(appendix),
            [row["mnemonic"] for row in raw_rows],
            qd=False,
        )
    rows = []
    for row in raw_rows:
        group, description = descriptions.get(row["mnemonic"].lower(), ("", ""))
        rows.append(
            FredColumn(
                position=int(row["position"]),
                mnemonic=row["mnemonic"],
                tcode=row["tcode"],
                tcode_label=row["tcode_label"],
                group=group,
                description=description,
            )
        )
    return rows, data_through


def _build_qd() -> tuple[list[FredColumn], str]:
    csv_path = _cached_or_download(
        CACHE_ROOT / "fred_qd" / "current" / "raw.csv",
        FRED_QD_CURRENT_URL,
        SOURCE_DIR / "fred_qd_current.csv",
    )
    raw_rows, data_through = _read_fred_csv(csv_path, qd=True)
    appendix = _download_optional(FRED_QD_APPENDIX_URL, SOURCE_DIR / "fred_qd_appendix.pdf")
    descriptions = {}
    if appendix is not None:
        descriptions = _parse_appendix_descriptions(
            _pdf_text(appendix),
            [row["mnemonic"] for row in raw_rows],
            qd=True,
        )
    rows = []
    for row in raw_rows:
        group, description = descriptions.get(row["mnemonic"].lower(), ("", ""))
        rows.append(
            FredColumn(
                position=int(row["position"]),
                mnemonic=row["mnemonic"],
                tcode=row["tcode"],
                tcode_label=row["tcode_label"],
                factor_flag=row["factor_flag"],
                group=group,
                description=description,
            )
        )
    return rows, data_through


def _latest_fred_sd_vintage() -> str:
    with urlopen(_request(FRED_SD_LANDING_URL), timeout=30) as src:
        html = src.read().decode("utf-8", errors="ignore")
    vintages = re.findall(r"/series/series-(\d{4}-\d{2})\.xlsx", html)
    if not vintages:
        return "current"
    return sorted(vintages)[-1]


def _cached_fred_sd_source() -> tuple[str, str] | None:
    manifest_path = CACHE_ROOT / "manifest" / "raw_artifacts.jsonl"
    if not manifest_path.exists():
        return None
    records = []
    for line in manifest_path.read_text(errors="replace").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    fallback: tuple[str, str] | None = None
    for record in reversed(records):
        if record.get("dataset") != "fred_sd" or record.get("file_format") != "xlsx":
            continue
        source_url = str(record.get("source_url") or FRED_SD_LANDING_URL)
        vintage_match = re.search(r"series-(\d{4}-\d{2})\.xlsx", source_url)
        if vintage_match:
            vintage = vintage_match.group(1)
            return source_url, vintage
        data_through = str(record.get("data_through") or "")
        if re.fullmatch(r"\d{4}-\d{2}", data_through):
            fallback = (
                FRED_SD_SERIES_URL_TEMPLATE.format(vintage=data_through),
                data_through,
            )
    return fallback


def _build_sd() -> tuple[list[dict[str, str]], list[str], str, Counter[str], str, str]:
    cached = CACHE_ROOT / "fred_sd" / "current" / "raw.xlsx"
    if cached.exists() and cached.stat().st_size > 0:
        xlsx_path = cached
        cached_source = _cached_fred_sd_source()
        if cached_source is None:
            source_url, vintage = FRED_SD_LANDING_URL, "current-cache"
        else:
            source_url, vintage = cached_source
    else:
        vintage = _latest_fred_sd_vintage()
        source_url = FRED_SD_SERIES_URL_TEMPLATE.format(vintage=vintage)
        xlsx_path = _download(source_url, SOURCE_DIR / f"fred_sd_series_{vintage}.xlsx")
    rows, states, data_through, frequency_counts = _load_fred_sd_workbook(xlsx_path)
    return rows, states, data_through, frequency_counts, vintage, source_url


def main() -> None:
    md_rows, md_through = _build_md()
    qd_rows, qd_through = _build_qd()
    sd_rows, sd_states, sd_through, sd_freq_counts, sd_vintage, sd_source_url = _build_sd()
    generated = date.today().isoformat()

    _write(
        OUT_DIR / "index.md",
        f"""# 5. FRED-Dataset

- Previous: [4. Detail (code): Full](../detail/index.md)
- Current: FRED-Dataset
- Next: [API Reference](../api/index.md)

This section is the dataset dictionary for macrocast's official FRED-backed
source panels. It is separate from Layer 1 because Layer 1 should decide the
source contract, target y, predictor x universe, and timing rules. The raw
dataset definitions belong here.

Generated: `{generated}` from current official FRED-MD/FRED-QD CSV files and
the current FRED-SD by-series workbook.

## Current Snapshot

| Dataset | macrocast `dataset` value | Frequency | Current source count | Data through | Column definition |
|---|---|---|---:|---|---|
| FRED-MD | `fred_md` | monthly | {len(md_rows)} columns | {md_through} | one column per official current CSV mnemonic |
| FRED-QD | `fred_qd` | quarterly | {len(qd_rows)} columns | {qd_through} | one column per official current CSV mnemonic |
| FRED-SD | `fred_sd` | mixed state monthly/quarterly | {len(sd_rows)} generated columns | {sd_through} | `{{sd_variable}}_{{state}}` from by-series workbook sheets and state columns |

## How This Connects To Layer 1

- `dataset=fred_md` activates the FRED-MD monthly panel.
- `dataset=fred_qd` activates the FRED-QD quarterly panel.
- `dataset=fred_sd` activates the FRED-SD state-level panel and then requires
  `frequency` plus optional FRED-SD state/series scope choices.
- `dataset=fred_md+fred_sd` and `dataset=fred_qd+fred_sd` combine a national
  MD/QD panel with selected FRED-SD state-level columns.
- `variable_universe` is a FRED-MD/QD predictor-universe axis. FRED-SD uses
  State Scope/List and Series Scope/List before the source frame is loaded.

## Pages

```{{toctree}}
:maxdepth: 1

fred_md
fred_qd
fred_sd
```
""",
    )

    _write(
        OUT_DIR / "fred_md.md",
        f"""# 5.1 FRED-MD

- Parent: [5. FRED-Dataset](index.md)
- Current dataset: FRED-MD

FRED-MD is the monthly national macro panel used by `dataset=fred_md`.
macrocast downloads the official current CSV from:

`{FRED_MD_CURRENT_URL}`

Generated: `{generated}`. Current data through: `{md_through}`. Current column
count excluding the date index: `{len(md_rows)}`.

## Column Contract

- Date column: `sasdate`, parsed as the monthly date index.
- Data columns: one column per FRED-MD mnemonic in the official current CSV.
- Transform row: `Transform:` gives the official FRED-MD T-code for each data
  column.
- Description source: official FRED-MD appendix when the mnemonic is present;
  otherwise the FRED series page or the official appendix/change log.

## All Current Columns

{_md_table(md_rows)}
""",
    )

    _write(
        OUT_DIR / "fred_qd.md",
        f"""# 5.2 FRED-QD

- Parent: [5. FRED-Dataset](index.md)
- Current dataset: FRED-QD

FRED-QD is the quarterly national macro panel used by `dataset=fred_qd`.
macrocast downloads the official current CSV from:

`{FRED_QD_CURRENT_URL}`

Generated: `{generated}`. Current data through: `{qd_through}`. Current column
count excluding the date index: `{len(qd_rows)}`.

## Column Contract

- Date column: `sasdate`, parsed as the quarterly date index.
- Data columns: one column per FRED-QD mnemonic in the official current CSV.
- `factors` row: `1` means the series is in the Stock-Watson-style factor
  construction set; `0` means it is not.
- Transform row: `transform` gives the official FRED-QD T-code for each data
  column.
- Description source: official FRED-QD appendix when the mnemonic is present;
  otherwise the FRED series page or the official appendix/change log.

## All Current Columns

{_qd_table(qd_rows)}
""",
    )

    state_rows = "\n".join(f"- `{code}`: {STATE_NAMES.get(code, code)}" for code in sd_states)
    freq_summary = ", ".join(f"{freq}: {count}" for freq, count in sorted(sd_freq_counts.items()))
    _write(
        OUT_DIR / "fred_sd.md",
        f"""# 5.3 FRED-SD

- Parent: [5. FRED-Dataset](index.md)
- Current dataset: FRED-SD

FRED-SD is the state-level panel used by `dataset=fred_sd` and by composite
routes such as `fred_md+fred_sd` or `fred_qd+fred_sd`.

macrocast uses the official **Data by Series** workbook. The workbook vintage
used for this generated page is `{sd_vintage}`:

`{sd_source_url}`

Generated: `{generated}`. Current data through: `{sd_through}`. Current
generated column count: `{len(sd_rows)}`. Native-frequency counts:
{freq_summary}.

## Column Contract

- Workbook sheets are FRED-SD variables.
- Sheet columns are state abbreviations.
- macrocast generated columns use `{{sd_variable}}_{{state}}`.
- Example: sheet `UR`, state `CA` becomes `UR_CA`.
- Layer 1 owns state/series source selection. Layer 2 owns any mixed-frequency
  representation after the source frame exists.

## FRED-SD Variables / Workbook Sheets

{_sd_variable_table(sd_rows)}

## States

{state_rows}

## All Current Generated Columns

{_sd_column_table(sd_rows)}
""",
    )


if __name__ == "__main__":
    main()
