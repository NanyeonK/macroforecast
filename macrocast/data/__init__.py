"""Data layer for macrocast.

Provides loaders for FRED-MD, FRED-QD, and FRED-SD datasets, along
with the MacroFrame container and transformation utilities.
"""

from macrocast.data.fred_md import load_fred_md
from macrocast.data.fred_qd import load_fred_qd
from macrocast.data.fred_sd import load_fred_sd
from macrocast.data.merge import merge_macro_frames
from macrocast.data.missing import classify_missing, handle_missing
from macrocast.data.schema import MacroFrame, MacroFrameMetadata, VariableMetadata
from macrocast.data.transforms import TransformCode, apply_tcode, apply_tcodes
from macrocast.data.vintages import (
    RealTimePanel,
    list_available_vintages,
    load_vintage_panel,
)

__all__ = [
    # Loaders
    "load_fred_md",
    "load_fred_qd",
    "load_fred_sd",
    # Multi-dataset merge
    "merge_macro_frames",
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
]
