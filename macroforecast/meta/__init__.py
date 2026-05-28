"""Global execution settings."""
from __future__ import annotations

from .config import MetaConfig, NJobs, OnError, configure, get_config, get_option, reset_config, use_config

__all__ = [
    "configure",
    "get_config",
    "get_option",
    "MetaConfig",
    "NJobs",
    "OnError",
    "reset_config",
    "use_config",
]
