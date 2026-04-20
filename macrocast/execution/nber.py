"""NBER business-cycle reference dates (US monthly frequency).

Source: NBER Business Cycle Dating Committee (https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions).
Peak (last month of expansion) / Trough (last month of recession) — our representation
uses the *recession interval* as [peak_next_month, trough]. A date falls in a recession
when it is >= peak_next_month and <= trough.

The list below covers all NBER-dated US recessions from 1948 onward (the FRED-MD
coverage window); older recessions can be added if future datasets extend further
back. Dates are frozen at the committee's most recent determination.
"""
from __future__ import annotations

from typing import Final

import pandas as pd


# Each tuple is (first month of recession, last month of recession).
# The pair is INCLUSIVE on both ends.
NBER_RECESSIONS: Final[tuple[tuple[pd.Timestamp, pd.Timestamp], ...]] = tuple(
    (pd.Timestamp(start), pd.Timestamp(end))
    for start, end in (
        ("1948-12-01", "1949-10-01"),
        ("1953-08-01", "1954-05-01"),
        ("1957-09-01", "1958-04-01"),
        ("1960-05-01", "1961-02-01"),
        ("1970-01-01", "1970-11-01"),
        ("1973-12-01", "1975-03-01"),
        ("1980-02-01", "1980-07-01"),
        ("1981-08-01", "1982-11-01"),
        ("1990-08-01", "1991-03-01"),
        ("2001-04-01", "2001-11-01"),
        ("2008-01-01", "2009-06-01"),
        ("2020-03-01", "2020-04-01"),
    )
)


def is_recession(date: pd.Timestamp | str) -> bool:
    """Return True if `date` falls within any NBER-dated US recession interval."""
    ts = pd.Timestamp(date)
    for start, end in NBER_RECESSIONS:
        if start <= ts <= end:
            return True
    return False


def is_expansion(date: pd.Timestamp | str) -> bool:
    """Return True if `date` falls OUTSIDE any NBER-dated US recession interval."""
    return not is_recession(date)


def filter_origins_by_regime(
    origin_plan: "list[tuple[int, int, int]]",
    *,
    index: pd.DatetimeIndex,
    regime: str,
) -> "list[tuple[int, int, int]]":
    """Filter an origin plan list to keep only origins matching `regime`.

    `origin_plan` entries are (origin_idx, start_idx, effective_origin_idx).
    `regime` is one of {"recession_only_oos", "expansion_only_oos"}.
    The check uses `index[origin_idx]` (the forecast origin date, not the target date).
    """
    if regime == "recession_only_oos":
        predicate = is_recession
    elif regime == "expansion_only_oos":
        predicate = is_expansion
    else:
        raise ValueError(f"unsupported regime={regime!r}")
    return [item for item in origin_plan if predicate(index[item[0]])]
