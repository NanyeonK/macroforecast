"""Data panel + targets for the GCLS (2022, JAE) replication.

Goulet Coulombe, Leroux, Stevanovic, Surprenant (2022), "How is Machine Learning
Useful for Macroeconomic Forecasting?", JAE 37(5), 920-964.

This module loads the official ZBW/JAE archive panel (``MainAnalysis/2018-01.csv``,
a genuine McCracken-Ng FRED-MD vintage), applies the FRED-MD transform codes with
the **CPI I(1) override (footnote 19: CPI treated as I(1), t-code 5, not I(2)/t-code
6)**, and exposes the 5 forecast targets as native ``TargetSpec`` objects.

Design source of truth: ``.dev-notes/phaseB_design_b4_gcls2022.md`` (S1b, S3) and
``.dev-notes/b4_scout_findings.md`` (verified data + API map).

Faithful-preprocessing note (mirrors the leak-free GCLS-2021 pipeline): the pipeline
is fed the RAW-level bundle + t-code metadata and applies the *official* transform
SPEC-LEVEL (refit per origin), so the stationary predictor panel is never built with
look-ahead. The 5 targets carry EXPLICIT (transform, policy) pairs (not derived from
the predictor t-code) that reproduce the paper's Eq.3/Eq.4 forecast objects.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import macroforecast as mf
from macroforecast.pipeline import TargetSpec

# --------------------------------------------------------------------------- #
# Archive paths
# --------------------------------------------------------------------------- #
# Data-only official archive (integrity-stamped MANIFEST.md, SHA256 verified).
ARCHIVE_ROOT = Path(
    "~/second_brain/00_wiki/raw/paper_code/jae_2910_glss_replication_20260602"
).expanduser()
ARCHIVE_ZIP = ARCHIVE_ROOT / "glss-files.zip"

# Extracted working copies (produced by the STAGE-1 extract step).
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "runs" / "gcls_b4_stage1" / "data"
MAIN_CSV = DATA_DIR / "2018-01.csv"

# The 4 interaction series (Table 3), aligned monthly. Staged now; not required for G1.
INTERACTION_FILES: dict[str, str] = {
    "MacroUncertainty": "MacroUncertaintyToCirculate.xlsx",
    "ANFCI": "National_Financial_Conditions_index.xls",
    "CSUSHPINSA": "CSUSHPINSA.xls",
    "UMCSENT": "UMCSENT.xls",
}

# --------------------------------------------------------------------------- #
# CPI I(1) override (GCLS footnote 19)
# --------------------------------------------------------------------------- #
# The 2018-01 FRED-MD vintage assigns CPIAUCSL t-code 6 (Delta^2 log = I(2)).
# GCLS fn.19 (following Medeiros et al.) treats the CPI as I(1): t-code 5 (Delta log).
# We override the headline CPI series that is (a) the INF forecast target and (b) a
# predictor in the panel. Other t-code-6 price sub-indices are retained at 6 in the
# predictor panel; extending I(1) to every price series is a G2 reconciliation item
# (does not change the INF target object, which is built I(1) below).
TCODE_CPI_OVERRIDE: dict[str, int] = {"CPIAUCSL": 5}

# --------------------------------------------------------------------------- #
# Targets (paper S4.1-4.2, Eq.3-4)
# --------------------------------------------------------------------------- #
# CRITICAL (mirrors the leak-free GCLS-2021 pipeline): the spec-level ``official``
# transform stationarises the WHOLE panel, so the target column (e.g. INDPRO) is
# already Delta-log by the time the target is built -- applying ``log_growth`` again
# would double-transform (empirically: log() masks the ~half of Delta-log values that
# are negative -> mostly-NaN target). We therefore build a dedicated one-period target
# object ``YOBJ__<col>`` from the RAW level, assign it t-code 1 (identity) so the
# official transform passes it through unchanged, and forecast it with
# ``transform="value"`` (+ the averaging POLICY). This reproduces the paper's Eq.3/Eq.4
# object regardless of the column's own predictor t-code (HOUST is t-code 4, CPI is
# t-code 6, etc.).
#
# column -> (alias, kind, policy):
#   kind "log_diff" -> Delta log Y  (I(1) log growth: INDPRO, INF/CPI, HOUST)
#   kind "diff"     -> Delta Y       (I(1) no-log change: UNRATE)
#   kind "level"    -> Y             (I(0) level: SPREAD/T10YFFM)
# policy "direct_average" -> (1/h) sum_{h'} object_{t+h'}  (Eq.4 average object)
# policy "direct"         -> object_{t+h}                   (Eq.3 h-ahead level)
YOBJ_PREFIX = "YOBJ__"
TARGET_TABLE: list[tuple[str, str, str, str]] = [
    # column,      alias,    kind,       policy
    ("INDPRO",   "INDPRO", "log_diff", "direct_average"),  # I(1) log   -> avg growth
    ("CPIAUCSL", "INF",    "log_diff", "direct_average"),  # I(1) log   -> avg growth (fn.19)
    ("HOUST",    "HOUST",  "log_diff", "direct_average"),  # I(1) log   -> avg growth
    ("UNRATE",   "UNRATE", "diff",     "direct_average"),  # I(1) no-log-> avg change
    ("T10YFFM",  "SPREAD", "level",    "direct"),          # I(0)       -> level
]


def yobj_column(col: str) -> str:
    return f"{YOBJ_PREFIX}{col}"


def gcls_targets() -> list[TargetSpec]:
    """The 5 GCLS-2022 forecast targets as native ``TargetSpec`` objects, each
    pointing at its raw-derived ``YOBJ__<col>`` object column (transform='value')."""
    return [
        TargetSpec(name=yobj_column(col), transform="value", policy=policy)
        for col, _alias, _kind, policy in TARGET_TABLE
    ]


def target_aliases() -> dict[str, str]:
    """YOBJ column -> paper alias (INF=CPIAUCSL, SPREAD=T10YFFM)."""
    return {yobj_column(col): alias for col, alias, _k, _p in TARGET_TABLE}


def _one_period_object(level, kind: str):
    """Stationary one-period forecast object from the RAW level (GCLS-2021 pattern)."""
    import numpy as np

    values = level.astype(float)
    if kind == "diff":
        return values.diff()
    if kind == "log_diff":
        positive = values.where(values > 0)
        return np.log(positive) - np.log(positive.shift(1))
    if kind == "level":
        return values
    raise ValueError(f"unknown target kind {kind!r}")


def augmented_bundle(csv: str | Path | None = None):
    """Raw-level bundle (CPI I(1) override) PLUS the 5 ``YOBJ__<col>`` target-object
    columns (t-code 1 identity). Returns ``(bundle, predictors)`` where ``predictors``
    is the list of the original FRED-MD series (the factor/shrinkage information set),
    excluding the YOBJ object columns so they can never leak as features."""
    import dataclasses
    import pandas as pd

    bundle = load_raw_bundle(csv)
    panel = bundle.panel.copy()
    codes = dict(bundle.metadata["transform_codes"])

    predictors = [c for c in bundle.panel.columns]  # the original FRED-MD series
    obj_frames = []
    for col, _alias, kind, _policy in TARGET_TABLE:
        obj = _one_period_object(bundle.panel[col], kind)
        obj.name = yobj_column(col)
        obj_frames.append(obj)
        codes[yobj_column(col)] = 1  # identity: official transform passes through
    new_panel = pd.concat([panel, pd.concat(obj_frames, axis=1)], axis=1)
    new_panel.attrs["macroforecast_transform_codes"] = dict(codes)

    meta = dict(bundle.metadata)
    meta["transform_codes"] = codes
    return dataclasses.replace(bundle, panel=new_panel, metadata=meta), predictors


# --------------------------------------------------------------------------- #
# Panel loading
# --------------------------------------------------------------------------- #
def load_raw_bundle(csv: str | Path | None = None) -> Any:
    """Load the raw-level FRED-MD bundle with the CPI I(1) override baked into its
    t-code metadata (both ``bundle.metadata['transform_codes']`` and the panel attrs,
    which the spec-level ``transform='official'`` preprocessing reads).

    Returns the ``DataBundle`` ready to feed ``pipeline_spec(data=bundle, ...)`` with
    ``preprocessing=preprocess_spec(transform='official', ...)``.
    """
    csv = Path(csv) if csv is not None else MAIN_CSV
    bundle = mf.load_fred_md(local_source=str(csv))
    codes = dict(bundle.metadata.get("transform_codes", {}) or {})
    for series, code in TCODE_CPI_OVERRIDE.items():
        if series in codes:
            codes[series] = code
    bundle.metadata["transform_codes"] = codes
    # Keep the panel attrs in lockstep (``_resolve_transform_codes`` reads metadata
    # first, but some code paths read the panel attrs).
    bundle.panel.attrs["macroforecast_transform_codes"] = dict(codes)
    return bundle


def transform_codes(csv: str | Path | None = None) -> dict[str, int]:
    """The FRED-MD t-code map WITH the CPI I(1) override applied."""
    bundle = load_raw_bundle(csv)
    return dict(bundle.metadata["transform_codes"])


def transformed_panel(csv: str | Path | None = None):
    """The stationary FRED-MD panel: official t-codes + CPI I(1) override applied.

    Burn-in rows (the first two, consumed by the max t-code-6 second difference) are
    dropped. Late-starting series retain NaN pending the pipeline's per-origin EM
    factor imputation (this is the raw materialisation for the STAGE-1 manifest, not
    the leak-free per-origin transform the pipeline runs).
    """
    from macroforecast.preprocessing import apply_tcode_transform

    bundle = load_raw_bundle(csv)
    codes = bundle.metadata["transform_codes"]
    tp = apply_tcode_transform(bundle.panel, codes)
    # Drop the 2-row differencing burn-in (t-code 6 = 2nd difference).
    tp = tp.iloc[2:]
    return tp


# --------------------------------------------------------------------------- #
# Interaction series (Table 3) -- staged, deferred for G1
# --------------------------------------------------------------------------- #
def load_interaction_series(data_dir: str | Path | None = None) -> dict[str, Any]:
    """Parse the 4 interaction series and align to month-start. Best-effort; used by
    Table 3 (G4), NOT required for G1. Returns {name: monthly pd.Series}."""
    import pandas as pd

    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    out: dict[str, Any] = {}
    for name, fname in INTERACTION_FILES.items():
        path = data_dir / fname
        if not path.exists():
            out[name] = None
            continue
        try:
            engine = "openpyxl" if path.suffix == ".xlsx" else "xlrd"
            raw = pd.read_excel(path, engine=engine)
        except Exception as exc:  # noqa: BLE001 -- staging only, never blocks G1
            out[name] = f"[parse-error: {type(exc).__name__}: {exc}]"
            continue
        out[name] = raw
    return out


# --------------------------------------------------------------------------- #
# STAGE-1 deliverable: panel.parquet + manifest.json
# --------------------------------------------------------------------------- #
def build_and_save_panel(out_dir: str | Path | None = None) -> dict[str, Any]:
    """Materialise ``panel.parquet`` (stationary, CPI I(1) override) + ``manifest.json``.

    Returns the manifest dict.
    """
    out_dir = Path(out_dir) if out_dir is not None else (
        Path(__file__).resolve().parent.parent.parent.parent / "runs" / "gcls_b4_stage1"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_raw_bundle()
    codes = dict(bundle.metadata["transform_codes"])
    raw_panel = bundle.panel
    tp = transformed_panel()

    parquet_path = out_dir / "panel.parquet"
    tp.to_parquet(parquet_path)

    manifest: dict[str, Any] = {
        "dataset": "FRED-MD (McCracken-Ng), official GCLS-2022 archive vintage 2018-01",
        "source_csv": str(MAIN_CSV),
        "archive_zip": str(ARCHIVE_ZIP),
        "n_series": int(raw_panel.shape[1]),
        "raw_date_range": [str(raw_panel.index[0].date()), str(raw_panel.index[-1].date())],
        "transformed_date_range": [str(tp.index[0].date()), str(tp.index[-1].date())],
        "transformed_shape": [int(tp.shape[0]), int(tp.shape[1])],
        "tcode_distribution": {str(k): int(v) for k, v in sorted(Counter(codes.values()).items())},
        "tcode_map": {k: int(v) for k, v in codes.items()},
        "cpi_i1_override": {
            "footnote": "fn.19 (CPI is I(1), not I(2)); following Medeiros et al.",
            "applied": TCODE_CPI_OVERRIDE,
            "note": (
                "CPIAUCSL t-code changed 6 -> 5 (Delta log). Applied to the headline "
                "CPI (INF target + predictor). Other t-code-6 price sub-indices kept "
                "at 6; extending I(1) to all price series is a G2 reconciliation item."
            ),
        },
        "series_count_reconciliation": {
            "vintage_series": int(raw_panel.shape[1]),
            "paper_nominal": 134,
            "note": (
                "The 2018-01 FRED-MD vintage carries 128 series; the paper's nominal "
                "'134 monthly indicators' is the standard FRED-MD nominal-vs-vintage "
                "gap (some nominal series are unavailable/dropped in this vintage). "
                "Non-blocking; retained-series parity vs the paper's variable table is "
                "a G2 check. Raw start 1959M01; the transformed panel begins 1959M03 "
                "(2-row Delta^2 burn-in) and the GCLS estimation window opens 1960M01."
            ),
        },
        "targets": [
            {
                "column": col, "alias": alias, "object_kind": kind,
                "object_column": yobj_column(col), "forecast_policy": policy,
                "target_transform": "value",
            }
            for col, alias, kind, policy in TARGET_TABLE
        ],
        "target_construction": (
            "YOBJ__<col> one-period object built from RAW level (log_diff/diff/level), "
            "t-code 1 (identity) so the official transform passes it through; forecast "
            "with transform='value' + policy (direct_average=Eq.4 avg object, "
            "direct=Eq.3 h-ahead level). Avoids the double-transform that stationarising "
            "the panel would otherwise inflict on the target."
        ),
        "interaction_series_staged": {
            name: (fname if (DATA_DIR / fname).exists() else "MISSING")
            for name, fname in INTERACTION_FILES.items()
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False))
    return manifest


if __name__ == "__main__":
    m = build_and_save_panel()
    print(json.dumps({k: v for k, v in m.items() if k != "tcode_map"}, indent=2))
