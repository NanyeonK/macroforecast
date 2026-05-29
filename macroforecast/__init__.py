"""Pandas-first macro forecasting workflow tools."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.10.0a0"

_LAZY_EXPORTS = {
    "DEFAULT_RANDOM_SEED": ".meta",
    "configure": ".meta",
    "get_config": ".meta",
    "get_option": ".meta",
    "reset_config": ".meta",
    "use_config": ".meta",
    "DataBundle": ".data",
    "DataSpec": ".data",
    "as_panel": ".data",
    "attach_metadata": ".data",
    "metadata": ".data",
    "panel_info": ".data",
    "spec": ".data",
    "validate_panel": ".data",
    "combine": ".data",
    "list_vintages": ".data",
    "load_custom_csv": ".data",
    "load_custom_parquet": ".data",
    "load_fred_md": ".data",
    "load_fred_qd": ".data",
    "load_fred_sd": ".data",
    "load_fred_md_sd": ".data",
    "load_fred_qd_sd": ".data",
    "PreprocessedData": ".preprocessing",
    "preprocess": ".preprocessing",
    "reprocess": ".preprocessing",
    "DataSummaryReport": ".data_summary",
    "summarize_data": ".data_summary",
    "DataAnalysisReport": ".data_analysis",
    "analyze_data": ".data_analysis",
}

_LAZY_MODULES: tuple[str, ...] = (
    "meta",
    "data",
    "preprocessing",
    "data_summary",
    "data_analysis",
    "evaluation",
)

__all__ = sorted(set(_LAZY_EXPORTS) | set(_LAZY_MODULES))


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS) | set(_LAZY_MODULES))
