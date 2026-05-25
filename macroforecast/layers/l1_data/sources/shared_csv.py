from __future__ import annotations

from pathlib import Path

import pandas as pd


def parse_fred_csv(filepath: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    path = Path(filepath)
    raw = pd.read_csv(path, header=None, dtype=str, na_values=["", ".", " "])
    if raw.shape[0] < 3 or raw.shape[1] < 2:
        raise ValueError(f"file does not look like a FRED CSV: {path}")

    header_idx: int | None = None
    tcodes_idx: int | None = None
    for idx, value in raw.iloc[:, 0].items():
        label = str(value).strip().lower()
        if label == "nan":
            label = ""
        if label in {"sasdate", "sasqdate"} and header_idx is None:
            header_idx = int(idx)
        elif label in {"transform", "transform:"} and tcodes_idx is None:
            tcodes_idx = int(idx)
        if header_idx is not None and tcodes_idx is not None:
            break

    if header_idx is None:
        # Historical/current FRED-MD fixtures put t-codes in row 0 and the
        # sasdate/sasqdate header in row 1.
        fallback_header = str(raw.iloc[1, 0]).strip().lower()
        if fallback_header in {"sasdate", "sasqdate"}:
            header_idx = 1
            tcodes_idx = 0
        else:
            raise ValueError(f"missing sasdate/sasqdate header row in {path}")
    elif tcodes_idx is None:
        # Older MD/QD files use an unlabeled row immediately before or after
        # the header. Newer QD files include an intervening factors row, so
        # prefer explicit labels when present and use this only as a fallback.
        if header_idx > 0:
            tcodes_idx = header_idx - 1
        else:
            tcodes_idx = header_idx + 1

    header_row = raw.iloc[header_idx].tolist()
    tcodes_row = raw.iloc[tcodes_idx].tolist()
    data_start = max(header_idx, tcodes_idx) + 1

    columns = [str(x).strip() for x in header_row]
    if not columns or columns[0].lower() not in {"sasdate", "sasqdate"}:
        raise ValueError(f"missing sasdate/sasqdate header row in {path}")

    tcodes: dict[str, int] = {}
    for name, value in zip(columns[1:], tcodes_row[1:], strict=False):
        try:
            tcodes[name] = int(float(str(value)))
        except (TypeError, ValueError):
            tcodes[name] = 1

    data = raw.iloc[data_start:].copy()
    data.columns = columns
    date_col = columns[0]
    data[date_col] = pd.to_datetime(data[date_col], errors="raise")
    data.set_index(date_col, inplace=True)

    for col in columns[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data.index.name = "date"
    data.sort_index(inplace=True)
    return data, tcodes
