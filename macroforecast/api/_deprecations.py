"""Deprecation shims for macroforecast public API parameter renames.

All deprecated-parameter detection is centralized here so that the
deprecation policy (DeprecationWarning, one-release window, v0.10.0
removal) is stated once.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable


_REMOVAL_VERSION = "v0.10.0"


def _warn(old: str, new: str) -> None:
    warnings.warn(
        f"{old}= is deprecated and will be removed in {_REMOVAL_VERSION}; "
        f"use {new}= instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def resolve_model(model: str | None, model_family: str | None, default: str) -> str:
    """Resolve the canonical ``model`` parameter, handling deprecated ``model_family``."""
    if model_family is not None:
        _warn("model_family", "model")
        if model is not None:
            raise ValueError(
                "Pass either model= or model_family= (deprecated), not both."
            )
        return model_family
    return model if model is not None else default


def resolve_models(
    models: Iterable[str] | None,
    model_families: Iterable[str] | None,
) -> Iterable[str] | None:
    """Resolve the canonical ``models`` parameter, handling deprecated ``model_families``."""
    if model_families is not None:
        _warn("model_families", "models")
        if models is not None:
            raise ValueError(
                "Pass either models= or model_families= (deprecated), not both."
            )
        return model_families
    return models


def resolve_benchmark_model(
    benchmark_model: str | None,
    benchmark_family: str | None,
    default: str,
) -> str:
    """Resolve the canonical ``benchmark_model`` parameter."""
    if benchmark_family is not None:
        _warn("benchmark_family", "benchmark_model")
        if benchmark_model is not None:
            raise ValueError(
                "Pass either benchmark_model= or benchmark_family= (deprecated), not both."
            )
        return benchmark_family
    return benchmark_model if benchmark_model is not None else default
