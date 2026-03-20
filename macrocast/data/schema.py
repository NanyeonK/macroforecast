"""MacroFrame: the central data container for macrocast.

MacroFrame wraps a pandas DataFrame together with dataset metadata
(variable descriptions, transformation codes, groupings, vintage
identifier). It provides a fluent interface for the most common
pre-processing steps while remaining immutable: every method that
transforms data returns a new MacroFrame rather than modifying the
original.
"""

from __future__ import annotations

import importlib.resources
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from macrocast.data.missing import classify_missing, handle_missing
from macrocast.data.transforms import apply_tcodes

# ---------------------------------------------------------------------------
# Metadata dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VariableMetadata:
    """Metadata for a single macroeconomic variable.

    Parameters
    ----------
    name : str
        FRED mnemonic (e.g. ``"INDPRO"``).
    description : str
        Human-readable label.
    group : str
        Group key (e.g. ``"output_income"``).
    tcode : int
        McCracken-Ng transformation code (1-7).
    frequency : str
        Observation frequency: ``"monthly"``, ``"quarterly"``, or
        ``"state_monthly"``.
    """

    name: str
    description: str
    group: str
    tcode: int
    frequency: str = "monthly"


@dataclass
class MacroFrameMetadata:
    """Dataset-level metadata for a MacroFrame.

    Parameters
    ----------
    dataset : str
        Source dataset identifier (e.g. ``"FRED-MD"``).
    vintage : str or None
        Vintage identifier in ``"YYYY-MM"`` format, or ``None`` for the
        current release.
    frequency : str
        Data frequency.
    variables : dict[str, VariableMetadata]
        Metadata keyed by variable name.
    groups : dict[str, str]
        Mapping from group key to display label.
    is_transformed : bool
        Whether the data has already been stationarity-transformed.
    download_date : str or None
        ISO date (``"YYYY-MM-DD"``) on which the current-release file was
        cached locally. ``None`` when *vintage* is explicitly specified,
        since the vintage identifier already pins the data version.
    data_through : str or None
        Last observation date in ``"YYYY-MM"`` format, read directly from
        the data. For ``current.csv`` this identifies the FRED release
        content (e.g. ``"2024-12"`` means data runs through December 2024).
        Set for both current and vintage loads.
    """

    dataset: str
    vintage: str | None
    frequency: str
    variables: dict[str, VariableMetadata] = field(default_factory=dict)
    groups: dict[str, str] = field(default_factory=dict)
    is_transformed: bool = False
    download_date: str | None = None
    data_through: str | None = None


# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------


def _load_spec(dataset: str) -> dict[str, Any]:
    """Load the bundled JSON spec for *dataset*.

    Parameters
    ----------
    dataset : str
        One of ``"fred_md"``, ``"fred_qd"``, ``"fred_sd"``.

    Returns
    -------
    dict
        Parsed JSON spec dictionary.

    Raises
    ------
    FileNotFoundError
        If the spec file does not exist under ``data/specs/``.
    """
    # Try importlib.resources first (installed package)
    try:
        pkg = importlib.resources.files("macrocast.data.specs")
        spec_file = pkg.joinpath(f"{dataset}.json")
        with spec_file.open("r") as fh:
            return json.load(fh)
    except (TypeError, FileNotFoundError, ModuleNotFoundError):
        pass

    # Fall back to filesystem path relative to this file
    specs_dir = Path(__file__).parent / "specs"
    spec_path = specs_dir / f"{dataset}.json"
    if not spec_path.exists():
        raise FileNotFoundError(
            f"Spec file not found: {spec_path}. "
            "Run macrocast.data.generate_specs() to generate it."
        )
    with open(spec_path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# MacroFrame
# ---------------------------------------------------------------------------


class MacroFrame:
    """Immutable container for macroeconomic panel data.

    Parameters
    ----------
    data : pd.DataFrame
        Panel with a DatetimeIndex and variable columns.
    metadata : MacroFrameMetadata
        Dataset metadata.
    tcodes : dict[str, int], optional
        Transformation code for each variable. Defaults to 1 (level)
        for variables not listed.

    Notes
    -----
    MacroFrame uses **composition**, not inheritance, so ``data`` is
    always accessible as an attribute. All methods that modify the data
    return a **new** MacroFrame; the original is never mutated.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        metadata: MacroFrameMetadata,
        tcodes: dict[str, int] | None = None,
    ) -> None:
        self._data = data.copy()
        self._metadata = metadata
        self._tcodes: dict[str, int] = tcodes.copy() if tcodes else {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> pd.DataFrame:
        """Underlying DataFrame (read-only view)."""
        return self._data

    @property
    def metadata(self) -> MacroFrameMetadata:
        """Dataset metadata."""
        return self._metadata

    @property
    def tcodes(self) -> dict[str, int]:
        """Transformation code map (variable -> tcode)."""
        return dict(self._tcodes)

    @property
    def vintage(self) -> str | None:
        """Vintage identifier, or None for the current release."""
        return self._metadata.vintage

    # ------------------------------------------------------------------
    # Fluent transformation methods (return new MacroFrame)
    # ------------------------------------------------------------------

    def transform(
        self,
        override: dict[str, int] | None = None,
    ) -> MacroFrame:
        """Apply stationarity transformations using stored tcodes.

        Parameters
        ----------
        override : dict[str, int], optional
            Per-variable tcode overrides. Unspecified variables use the
            default tcode from the spec (or 1 if absent).

        Returns
        -------
        MacroFrame
            New MacroFrame with transformed data. The metadata flag
            ``is_transformed`` is set to True.
        """
        effective_tcodes = dict(self._tcodes)
        if override:
            effective_tcodes.update(override)

        transformed = apply_tcodes(self._data, effective_tcodes)
        new_meta = MacroFrameMetadata(
            dataset=self._metadata.dataset,
            vintage=self._metadata.vintage,
            frequency=self._metadata.frequency,
            variables=self._metadata.variables,
            groups=self._metadata.groups,
            is_transformed=True,
            download_date=self._metadata.download_date,
            data_through=self._metadata.data_through,
        )
        return MacroFrame(transformed, new_meta, effective_tcodes)

    def group(self, group_name: str) -> MacroFrame:
        """Return a MacroFrame restricted to variables in *group_name*.

        Parameters
        ----------
        group_name : str
            Group key (e.g. ``"labor"``).

        Returns
        -------
        MacroFrame
            Subset MacroFrame.

        Raises
        ------
        KeyError
            If no variables belong to *group_name*.
        """
        cols = [
            name
            for name, vmeta in self._metadata.variables.items()
            if vmeta.group == group_name and name in self._data.columns
        ]
        if not cols:
            raise KeyError(
                f"No variables found for group '{group_name}'. "
                f"Available groups: {list(self._metadata.groups.keys())}"
            )
        sub_vars = {k: v for k, v in self._metadata.variables.items() if k in cols}
        new_meta = MacroFrameMetadata(
            dataset=self._metadata.dataset,
            vintage=self._metadata.vintage,
            frequency=self._metadata.frequency,
            variables=sub_vars,
            groups=self._metadata.groups,
            is_transformed=self._metadata.is_transformed,
            download_date=self._metadata.download_date,
            data_through=self._metadata.data_through,
        )
        return MacroFrame(self._data[cols], new_meta, self._tcodes)

    def trim(
        self,
        start: str | None = None,
        end: str | None = None,
        min_obs_pct: float | None = None,
    ) -> MacroFrame:
        """Restrict the sample period and optionally drop sparse variables.

        Parameters
        ----------
        start : str, optional
            Start date in ``"YYYY-MM"`` or any pandas-parseable format.
        end : str, optional
            End date (inclusive).
        min_obs_pct : float, optional
            Drop variables with fewer than *min_obs_pct* non-missing
            observations within the (trimmed) sample.

        Returns
        -------
        MacroFrame
            Trimmed MacroFrame.
        """
        df = self._data
        if start is not None:
            df = df.loc[start:]
        if end is not None:
            df = df.loc[:end]

        if min_obs_pct is not None:
            obs_frac = df.notna().mean()
            keep = obs_frac[obs_frac >= min_obs_pct].index
            dropped = set(df.columns) - set(keep)
            if dropped:
                import warnings

                warnings.warn(
                    f"trim: dropped {len(dropped)} sparse variable(s): "
                    f"{sorted(dropped)}",
                    stacklevel=2,
                )
            df = df[keep]

        sub_vars = {
            k: v for k, v in self._metadata.variables.items() if k in df.columns
        }
        # Recompute data_through after trimming — end date may have changed
        new_data_through = df.index[-1].strftime("%Y-%m") if len(df) > 0 else None
        new_meta = MacroFrameMetadata(
            dataset=self._metadata.dataset,
            vintage=self._metadata.vintage,
            frequency=self._metadata.frequency,
            variables=sub_vars,
            groups=self._metadata.groups,
            is_transformed=self._metadata.is_transformed,
            download_date=self._metadata.download_date,
            data_through=new_data_through,
        )
        return MacroFrame(df, new_meta, self._tcodes)

    def handle_missing(self, method: str, **kwargs: object) -> MacroFrame:
        """Apply a missing-value treatment.

        Parameters
        ----------
        method : str
            One of ``"trim_start"``, ``"drop_vars"``, ``"interpolate"``,
            ``"forward_fill"``, ``"em"``.
        **kwargs
            Forwarded to :func:`macrocast.data.missing.handle_missing`.

        Returns
        -------
        MacroFrame
            New MacroFrame after treatment.
        """
        cleaned = handle_missing(self._data, method, **kwargs)
        sub_vars = {
            k: v for k, v in self._metadata.variables.items() if k in cleaned.columns
        }
        new_meta = MacroFrameMetadata(
            dataset=self._metadata.dataset,
            vintage=self._metadata.vintage,
            frequency=self._metadata.frequency,
            variables=sub_vars,
            groups=self._metadata.groups,
            is_transformed=self._metadata.is_transformed,
            download_date=self._metadata.download_date,
            data_through=self._metadata.data_through,
        )
        return MacroFrame(cleaned, new_meta, self._tcodes)

    # ------------------------------------------------------------------
    # Reporting / inspection
    # ------------------------------------------------------------------

    def missing_report(self) -> pd.DataFrame:
        """Return a per-variable missing value summary.

        Returns
        -------
        pd.DataFrame
            Output of :func:`macrocast.data.missing.classify_missing`.
        """
        return classify_missing(self._data)

    def outlier_flag(
        self,
        method: str = "iqr",
        threshold: float = 10.0,
    ) -> pd.DataFrame:
        """Return a boolean mask of outlier cells.

        Parameters
        ----------
        method : str
            Detection method. Currently only ``"iqr"`` is supported:
            flags values more than *threshold* IQR widths from the median.
        threshold : float
            Number of IQR units beyond which a value is flagged.

        Returns
        -------
        pd.DataFrame
            Boolean DataFrame with True where an outlier is detected.

        Raises
        ------
        ValueError
            If *method* is unsupported.
        """
        if method != "iqr":
            raise ValueError(f"Unsupported outlier method: '{method}'. Use 'iqr'.")

        df = self._data
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        median = df.median()
        flags = (df - median).abs() > threshold * iqr
        return flags

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_numpy(self) -> np.ndarray:
        """Return the data as a 2-D NumPy array (rows=time, cols=vars).

        Returns
        -------
        np.ndarray
            Array with shape ``(T, N)`` in float64.
        """
        return self._data.to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_t, n_n = self._data.shape
        start = self._data.index[0] if n_t > 0 else "N/A"
        end = self._data.index[-1] if n_t > 0 else "N/A"
        transformed = "transformed" if self._metadata.is_transformed else "levels"
        if self._metadata.vintage is not None:
            version_str = f"vintage={self._metadata.vintage!r}"
        else:
            # current release: show what data is actually in the file
            parts = []
            if self._metadata.data_through is not None:
                parts.append(f"data_through={self._metadata.data_through!r}")
            if self._metadata.download_date is not None:
                parts.append(f"download_date={self._metadata.download_date!r}")
            version_str = ", ".join(parts) if parts else "vintage='current'"
        return (
            f"MacroFrame("
            f"dataset={self._metadata.dataset!r}, "
            f"{version_str}, "
            f"T={n_t}, N={n_n}, "
            f"period={start} to {end}, "
            f"status={transformed})"
        )

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: str | list[str]) -> pd.DataFrame | pd.Series:
        """Column selection forwarded to the underlying DataFrame."""
        return self._data[key]
