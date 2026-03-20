"""macrocast: Decomposing ML Forecast Gains in Macroeconomic Forecasting.

Layers
------
* Layer 1 (Data):     ``macrocast.data``        — FRED-MD/QD/SD loaders + MacroFrame
* Layer 2 (Pipeline): ``macrocast.pipeline``    — ForecastExperiment, models, features
* Layer 3 (Eval):     ``macrocast.evaluation``  — MSFE, PBSV, dual weights, MCS, DM

Quick start::

    from macrocast import load_fred_md, ForecastExperiment
"""

__version__ = "0.1.0"

from macrocast.data import (
    MacroFrame,
    MacroFrameMetadata,
    RealTimePanel,
    TransformCode,
    VariableMetadata,
    apply_tcode,
    apply_tcodes,
    classify_missing,
    handle_missing,
    list_available_vintages,
    load_fred_md,
    load_fred_qd,
    load_fred_sd,
    load_vintage_panel,
    merge_macro_frames,
)

__all__ = [
    "__version__",
    # Loaders
    "load_fred_md",
    "load_fred_qd",
    "load_fred_sd",
    # Core container
    "MacroFrame",
    "MacroFrameMetadata",
    "VariableMetadata",
    # Transforms
    "TransformCode",
    "apply_tcode",
    "apply_tcodes",
    # Missing
    "classify_missing",
    "handle_missing",
    # Vintages
    "list_available_vintages",
    "load_vintage_panel",
    "RealTimePanel",
    # Multi-dataset merge
    "merge_macro_frames",
    # Pipeline (Layer 2) — top-level convenience re-exports
    "ForecastExperiment",
    "ModelSpec",
    "FeatureSpec",
    "ResultSet",
    "ForecastRecord",
]

# Layer 2 convenience re-exports
from macrocast.pipeline import (  # noqa: E402
    FeatureSpec,
    ForecastExperiment,
    ForecastRecord,
    ModelSpec,
    ResultSet,
)
