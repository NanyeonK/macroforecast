"""Global execution settings."""
from __future__ import annotations

from .config import (
    DEFAULT_RANDOM_SEED,
    MetaConfig,
    MetadataLevel,
    NJobs,
    OnError,
    StageDefaultScope,
    configure,
    get_config,
    get_option,
    resolve_n_jobs,
    reset_config,
    use_config,
)

__all__ = [
    "DEFAULT_RANDOM_SEED",
    "configure",
    "get_config",
    "get_option",
    "resolve_n_jobs",
    "MetaConfig",
    "MetadataLevel",
    "NJobs",
    "OnError",
    "StageDefaultScope",
    "reset_config",
    "use_config",
]
