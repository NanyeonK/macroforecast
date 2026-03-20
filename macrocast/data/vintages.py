"""Vintage management for real-time FRED-MD, FRED-QD, and FRED-SD data.

Utilities for listing available vintages and loading a collection of
vintage snapshots (a "real-time panel"). The ``RealTimePanel`` class is
a lightweight wrapper around a dict of MacroFrame objects.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from macrocast.data.schema import MacroFrame

# Earliest vintage available on the FRED website
_FRED_MD_FIRST_VINTAGE = "1999-01"
_FRED_QD_FIRST_VINTAGE = "2005-01"
_FRED_SD_FIRST_VINTAGE = "2005-01"

_KNOWN_DATASETS = {"fred_md", "fred_qd", "fred_sd"}


def list_available_vintages(
    dataset: str,
    start: str | None = None,
    end: str | None = None,
) -> list[str]:
    """Generate a list of expected vintage identifiers for a dataset.

    This function does *not* make network requests; it enumerates all
    calendar months between *start* and *end* (or sensible defaults).
    Not every generated vintage necessarily has a file on the FRED server.

    Parameters
    ----------
    dataset : str
        One of ``"fred_md"``, ``"fred_qd"``, or ``"fred_sd"``.
    start : str or None
        Start vintage in ``"YYYY-MM"`` format (inclusive).
    end : str or None
        End vintage in ``"YYYY-MM"`` format (inclusive). Defaults to
        the current month.

    Returns
    -------
    list of str
        Sorted list of ``"YYYY-MM"`` vintage strings.

    Raises
    ------
    ValueError
        If *dataset* is not recognised.
    """
    defaults = {
        "fred_md": _FRED_MD_FIRST_VINTAGE,
        "fred_qd": _FRED_QD_FIRST_VINTAGE,
        "fred_sd": _FRED_SD_FIRST_VINTAGE,
    }
    if dataset not in defaults:
        raise ValueError(
            f"Unknown dataset: '{dataset}'. Use one of {sorted(defaults.keys())}."
        )

    start_dt = _parse_vintage(start or defaults[dataset])
    end_dt = _parse_vintage(end) if end else datetime.now().replace(day=1)

    vintages: list[str] = []
    current = start_dt
    while current <= end_dt:
        vintages.append(current.strftime("%Y-%m"))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    return vintages


def load_vintage_panel(
    dataset: str,
    vintages: list[str],
    target: str | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, MacroFrame]:
    """Load multiple vintage snapshots of a dataset.

    Parameters
    ----------
    dataset : str
        One of ``"fred_md"``, ``"fred_qd"``, or ``"fred_sd"``.
    vintages : list of str
        List of ``"YYYY-MM"`` vintage strings to load.
    target : str or None
        Variable of interest. When provided, each MacroFrame is
        restricted to that variable (useful for extracting real-time
        revisions of a single series). Not yet implemented; currently
        ignored.
    cache_dir : str or Path, optional
        Override default cache directory.

    Returns
    -------
    dict of str -> MacroFrame
        Mapping from vintage string to the corresponding MacroFrame.

    Raises
    ------
    ValueError
        If *dataset* is not recognised.
    """
    if dataset == "fred_md":
        from macrocast.data.fred_md import load_fred_md as _loader
    elif dataset == "fred_qd":
        from macrocast.data.fred_qd import load_fred_qd as _loader
    elif dataset == "fred_sd":
        from macrocast.data.fred_sd import load_fred_sd as _loader
    else:
        raise ValueError(
            f"Unknown dataset: '{dataset}'. Use one of {sorted(_KNOWN_DATASETS)}."
        )

    panel: dict[str, MacroFrame] = {}
    for vintage in vintages:
        panel[vintage] = _loader(vintage=vintage, cache_dir=cache_dir)

    return panel


class RealTimePanel:
    """Lightweight wrapper around a dict of vintage MacroFrames.

    Wraps the output of ``load_vintage_panel`` and provides minimal
    convenience methods for accessing vintage snapshots.

    Parameters
    ----------
    panel : dict[str, MacroFrame]
        Mapping from vintage string to MacroFrame.
    """

    def __init__(self, panel: dict[str, MacroFrame]) -> None:
        self._panel = dict(panel)

    @property
    def vintages(self) -> list[str]:
        """Sorted list of vintage identifiers in the panel."""
        return sorted(self._panel.keys())

    def __getitem__(self, vintage: str) -> MacroFrame:
        return self._panel[vintage]

    def __len__(self) -> int:
        return len(self._panel)

    def __repr__(self) -> str:
        n = len(self._panel)
        first = self.vintages[0] if n > 0 else "N/A"
        last = self.vintages[-1] if n > 0 else "N/A"
        return f"RealTimePanel(n_vintages={n}, range={first} to {last})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_vintage(vintage: str) -> datetime:
    """Parse a ``"YYYY-MM"`` vintage string to a datetime."""
    try:
        return datetime.strptime(vintage, "%Y-%m")
    except ValueError as exc:
        raise ValueError(
            f"Invalid vintage format: '{vintage}'. Expected 'YYYY-MM'."
        ) from exc
