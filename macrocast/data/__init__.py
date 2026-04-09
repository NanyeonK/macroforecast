"""macrocast.data — Raw data acquisition (FRED-MD, FRED-QD, FRED-SD)."""

from macrocast.data.fred_md import load_fred_md
from macrocast.data.fred_qd import load_fred_qd
from macrocast.data.fred_sd import load_fred_sd
from macrocast.data.merge import MergeResult, merge_macro_frames
from macrocast.data.registry import (
    get_data_task_defaults,
    get_dataset_defaults,
    get_target_defaults,
    load_data_task_registry,
    load_dataset_registry,
    load_target_registry,
    validate_data_task_registry,
    validate_dataset_registry,
    validate_target_registry,
)
from macrocast.data.schema import MacroFrame, MacroFrameMetadata, VariableMetadata
from macrocast.data.vintages import (
    RealTimePanel,
    list_available_vintages,
    load_vintage_panel,
)

__all__ = [
    "load_fred_md",
    "load_fred_qd",
    "load_fred_sd",
    "load_dataset_registry",
    "load_target_registry",
    "load_data_task_registry",
    "validate_dataset_registry",
    "validate_target_registry",
    "validate_data_task_registry",
    "get_dataset_defaults",
    "get_target_defaults",
    "get_data_task_defaults",
    "merge_macro_frames",
    "MergeResult",
    "MacroFrame",
    "MacroFrameMetadata",
    "VariableMetadata",
    "list_available_vintages",
    "load_vintage_panel",
    "RealTimePanel",
]
