from __future__ import annotations

from pathlib import Path

import pandas as pd


def _write_table(df: pd.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == '.csv':
        df.to_csv(out, index=False)
    else:
        df.to_parquet(out, index=False)
    return out


def write_forecast_table(df: pd.DataFrame, path: str | Path) -> Path:
    return _write_table(df, path)


def write_eval_table(df: pd.DataFrame, path: str | Path) -> Path:
    return _write_table(df, path)


def write_test_table(df: pd.DataFrame, path: str | Path) -> Path:
    return _write_table(df, path)


def write_interpretation_table(df: pd.DataFrame, path: str | Path) -> Path:
    return _write_table(df, path)


def write_failure_log(df: pd.DataFrame, path: str | Path) -> Path:
    return _write_table(df, path)
