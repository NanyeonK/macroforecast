"""Multi-dataset panel merging for MacroFrame objects.

merge_macro_frames combines two or more MacroFrame objects into a single
tidy pandas DataFrame suitable for passing to ForecastExperiment.

Frequency alignment rules
-------------------------
* target_freq="ME"  — Monthly basis.  Frames with quarterly frequency are
  upsampled via forward-fill (ffill).  A UserWarning is emitted.
* target_freq="QE"  — Quarterly basis.  Frames with monthly frequency are
  downsampled by taking the last observation of each quarter.  A
  UserWarning is emitted.

Column conflict rules
---------------------
When the same column name appears in more than one frame, the version from
the frame whose native frequency matches target_freq is kept.  If multiple
frames share target_freq, the first one encountered wins.  A UserWarning
is emitted for every conflict.

Date alignment
--------------
The merged DataFrame covers the intersection of all frames' date ranges
(inner join on DatetimeIndex).
"""

from __future__ import annotations

import warnings

import pandas as pd

from macrocast.data.schema import MacroFrame

# ---------------------------------------------------------------------------
# Internal frequency normalisation
# ---------------------------------------------------------------------------

# Maps various user-supplied strings to canonical pandas resample aliases
_FREQ_ALIASES: dict[str, str] = {
    "monthly": "ME",
    "quarterly": "QE",
    "state_monthly": "ME",
    # pandas aliases — common variants
    "me": "ME",
    "qe": "QE",
    "m": "ME",
    "q": "QE",
    "ms": "ME",
    "qs": "QE",
}


def _normalise_freq(freq: str) -> str:
    """Normalise *freq* to one of ``'ME'`` or ``'QE'``.

    Parameters
    ----------
    freq : str
        User-supplied frequency string (e.g. ``"monthly"``, ``"ME"``, ``"Q"``).

    Returns
    -------
    str
        ``"ME"`` for monthly, ``"QE"`` for quarterly.

    Raises
    ------
    ValueError
        If *freq* cannot be mapped to a supported alias.
    """
    normalised = _FREQ_ALIASES.get(freq.lower())
    if normalised is None:
        raise ValueError(
            f"Unknown target_freq '{freq}'. "
            "Use 'ME' / 'monthly' for monthly or 'QE' / 'quarterly' for quarterly."
        )
    return normalised


def _frame_native_freq(frame: MacroFrame) -> str:
    """Return the normalised pandas frequency of *frame*."""
    return _normalise_freq(frame.metadata.frequency)


def _resample_to_target(df: pd.DataFrame, native: str, target: str) -> pd.DataFrame:
    """Resample *df* from *native* to *target* frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Panel with DatetimeIndex.
    native : str
        Current frequency (``"ME"`` or ``"QE"``).
    target : str
        Target frequency (``"ME"`` or ``"QE"``).

    Returns
    -------
    pd.DataFrame
        Resampled panel.
    """
    if native == target:
        return df
    if native == "QE" and target == "ME":
        # Quarterly → Monthly: forward-fill within each quarter
        return df.resample("ME").ffill()
    if native == "ME" and target == "QE":
        # Monthly → Quarterly: take last observation in each quarter
        return df.resample("QE").last()
    raise ValueError(f"Unsupported resampling: {native} -> {target}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def merge_macro_frames(*frames: MacroFrame, target_freq: str) -> pd.DataFrame:
    """Merge two or more MacroFrame objects into a single panel DataFrame.

    Parameters
    ----------
    *frames : MacroFrame
        Two or more MacroFrame objects to merge.  Each should be
        stationarity-transformed before merging (call ``.transform()`` first).
    target_freq : str
        Target observation frequency.  Accepted values:

        * ``"ME"`` or ``"monthly"`` — monthly basis.  Use this when FRED-MD
          is the primary dataset.  Quarterly frames are upsampled via ffill.
        * ``"QE"`` or ``"quarterly"`` — quarterly basis.  Use this when
          FRED-QD is the primary dataset.  Monthly frames are downsampled
          to quarter-end (last value in the quarter).

    Returns
    -------
    pd.DataFrame
        Merged panel with DatetimeIndex at *target_freq*, covering the
        intersection of all frames' date ranges (inner join).

    Warns
    -----
    UserWarning
        * When a frame with a different native frequency is resampled.
        * When a column name conflict is resolved by keeping the
          target-frequency frame's version.

    Examples
    --------
    >>> import macrocast as mc
    >>> md = mc.load_fred_md().transform()
    >>> qd = mc.load_fred_qd().transform()
    >>> panel = mc.merge_macro_frames(md, qd, target_freq="ME")

    >>> sd = mc.load_fred_sd(states=["CA", "TX"], variables=["UR"])
    >>> panel = mc.merge_macro_frames(md, sd, target_freq="ME")
    """
    if len(frames) < 1:
        raise ValueError("At least one MacroFrame is required.")

    target = _normalise_freq(target_freq)

    # ------------------------------------------------------------------
    # Step 1: resample each frame to target frequency
    # ------------------------------------------------------------------
    resampled: list[tuple[str, str, pd.DataFrame]] = []

    for frame in frames:
        native = _frame_native_freq(frame)
        df = frame.data.copy()

        if native != target:
            dataset = frame.metadata.dataset
            if native == "QE" and target == "ME":
                direction = "upsampled to monthly via forward-fill"
            else:
                direction = "downsampled to quarterly (last value per quarter)"

            warnings.warn(
                f"{dataset} has {frame.metadata.frequency} frequency but "
                f"target_freq='{target_freq}' was requested. "
                f"It will be {direction}. "
                "Verify that this resampling is appropriate for your analysis.",
                UserWarning,
                stacklevel=2,
            )
            df = _resample_to_target(df, native, target)

        resampled.append((frame.metadata.dataset, native, df))

    # ------------------------------------------------------------------
    # Step 2: resolve column conflicts
    # Primary frames (native == target) take precedence over resampled ones.
    # Among primaries, first frame wins.
    # ------------------------------------------------------------------
    # Sort so primary-freq frames come first
    ordered = sorted(resampled, key=lambda x: (0 if x[1] == target else 1))

    seen: dict[str, str] = {}   # column name -> dataset that owns it
    clean_frames: list[pd.DataFrame] = []

    for dataset, _native, df in ordered:
        conflicts = [c for c in df.columns if c in seen]
        if conflicts:
            owners = [seen[c] for c in conflicts]
            warnings.warn(
                f"Column conflict: {conflicts} from '{dataset}' already present "
                f"(from {owners}). Keeping the version from the primary dataset. "
                f"Dropping duplicate columns from '{dataset}'.",
                UserWarning,
                stacklevel=2,
            )
            df = df.drop(columns=conflicts)

        for col in df.columns:
            seen[col] = dataset
        clean_frames.append(df)

    # ------------------------------------------------------------------
    # Step 3: inner join on date index
    # ------------------------------------------------------------------
    merged = pd.concat(clean_frames, axis=1, join="inner")
    merged = merged.sort_index()

    return merged
