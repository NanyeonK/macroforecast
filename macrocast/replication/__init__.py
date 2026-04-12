"""Replication helpers retained as migration scaffolding.

Target architecture: paper studies should live as recipes/paths, not package-specific core modules.
Prefer `recipes/papers/*.yaml` plus recipe-aware compilation paths for new work.
"""

from macrocast.replication.clss2021 import CLSS2021, get_preset
from macrocast.replication.clss2021_runner import (
    load_clss2021_fixed_settings,
    load_clss2021_panel,
    relative_rmsfe_vs_ar,
    relative_rmsfe_vs_feature_baseline,
    run_clss2021_reduced_check,
    summarize_clss2021_run,
)

__all__ = [
    "CLSS2021",
    "get_preset",
    'load_clss2021_fixed_settings',
    'load_clss2021_panel',
    'relative_rmsfe_vs_ar',
    'relative_rmsfe_vs_feature_baseline',
    'run_clss2021_reduced_check',
    'summarize_clss2021_run',
]
