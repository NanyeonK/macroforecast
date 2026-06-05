"""Global execution settings for macroforecast."""
from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from typing import Any, Iterator, Literal, TypedDict, cast


OnError = Literal["raise", "continue"]
NJobs = int | Literal["auto"]
StageDefaultScope = Literal["full_panel", "origin_available", "fit_window"]
MetadataLevel = Literal["minimal", "standard", "full"]

DEFAULT_RANDOM_SEED: int = 42


class MetaConfig(TypedDict):
    random_seed: int | None
    n_jobs: NJobs
    on_error: OnError
    verbose: int
    default_preprocessing_scope: StageDefaultScope
    default_feature_scope: StageDefaultScope
    default_selection_scope: StageDefaultScope
    metadata_level: MetadataLevel


_DEFAULT_CONFIG: MetaConfig = {
    "random_seed": DEFAULT_RANDOM_SEED,
    "n_jobs": 1,
    "on_error": "raise",
    "verbose": 0,
    "default_preprocessing_scope": "origin_available",
    "default_feature_scope": "fit_window",
    "default_selection_scope": "fit_window",
    "metadata_level": "standard",
}

_CONFIG: dict[str, Any] = dict(_DEFAULT_CONFIG)
_LOCK = RLock()
_MISSING = object()


def configure(
    *,
    random_seed: int | None | object = _MISSING,
    n_jobs: NJobs | object = _MISSING,
    on_error: OnError | object = _MISSING,
    verbose: int | object = _MISSING,
    default_preprocessing_scope: StageDefaultScope | object = _MISSING,
    default_feature_scope: StageDefaultScope | object = _MISSING,
    default_selection_scope: StageDefaultScope | object = _MISSING,
    metadata_level: MetadataLevel | object = _MISSING,
) -> MetaConfig:
    """Update package-wide execution defaults and return the active config."""

    updates: dict[str, Any] = {}
    if random_seed is not _MISSING:
        updates["random_seed"] = _normalize_random_seed(random_seed)
    if n_jobs is not _MISSING:
        updates["n_jobs"] = _normalize_n_jobs(n_jobs)
    if on_error is not _MISSING:
        updates["on_error"] = _normalize_on_error(on_error)
    if verbose is not _MISSING:
        updates["verbose"] = _normalize_verbose(verbose)
    if default_preprocessing_scope is not _MISSING:
        updates["default_preprocessing_scope"] = _normalize_stage_default_scope(
            default_preprocessing_scope,
            name="default_preprocessing_scope",
        )
    if default_feature_scope is not _MISSING:
        updates["default_feature_scope"] = _normalize_stage_default_scope(
            default_feature_scope,
            name="default_feature_scope",
        )
    if default_selection_scope is not _MISSING:
        updates["default_selection_scope"] = _normalize_stage_default_scope(
            default_selection_scope,
            name="default_selection_scope",
        )
    if metadata_level is not _MISSING:
        updates["metadata_level"] = _normalize_metadata_level(metadata_level)

    with _LOCK:
        _CONFIG.update(updates)
        return cast(MetaConfig, dict(_CONFIG))


def get_config() -> MetaConfig:
    """Return a copy of the current package-wide execution defaults."""

    with _LOCK:
        return cast(MetaConfig, dict(_CONFIG))


def get_option(name: str) -> Any:
    """Return one setting from the current package-wide execution defaults."""

    with _LOCK:
        if name not in _CONFIG:
            valid = ", ".join(_DEFAULT_CONFIG)
            raise KeyError(f"unknown meta option {name!r}; expected one of: {valid}")
        return _CONFIG[name]


def resolve_n_jobs() -> int:
    """Return the configured worker count, resolving ``'auto'`` to the CPU count.

    This is the single resolution point so that ``meta.configure(n_jobs=...)``
    actually controls parallelism in callers that opt in (e.g. tree ensembles).
    """

    import os

    value = get_option("n_jobs")
    if value == "auto":
        return os.cpu_count() or 1
    return int(value)


def reset_config() -> MetaConfig:
    """Reset package-wide execution defaults to their initial values."""

    with _LOCK:
        _CONFIG.clear()
        _CONFIG.update(_DEFAULT_CONFIG)
        return cast(MetaConfig, dict(_CONFIG))


@contextmanager
def use_config(
    *,
    random_seed: int | None | object = _MISSING,
    n_jobs: NJobs | object = _MISSING,
    on_error: OnError | object = _MISSING,
    verbose: int | object = _MISSING,
    default_preprocessing_scope: StageDefaultScope | object = _MISSING,
    default_feature_scope: StageDefaultScope | object = _MISSING,
    default_selection_scope: StageDefaultScope | object = _MISSING,
    metadata_level: MetadataLevel | object = _MISSING,
) -> Iterator[MetaConfig]:
    """Temporarily update package-wide execution defaults inside a context."""

    with _LOCK:
        previous = cast(MetaConfig, dict(_CONFIG))
    try:
        active = configure(
            random_seed=random_seed,
            n_jobs=n_jobs,
            on_error=on_error,
            verbose=verbose,
            default_preprocessing_scope=default_preprocessing_scope,
            default_feature_scope=default_feature_scope,
            default_selection_scope=default_selection_scope,
            metadata_level=metadata_level,
        )
        yield active
    finally:
        with _LOCK:
            _CONFIG.clear()
            _CONFIG.update(previous)


def _normalize_random_seed(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("random_seed must be an int or None")
    if value < 0:
        raise ValueError("random_seed must be non-negative")
    return int(value)


def _normalize_n_jobs(value: object) -> NJobs:
    if value == "auto":
        return "auto"
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("n_jobs must be a positive int or 'auto'")
    if value < 1:
        raise ValueError("n_jobs must be a positive int or 'auto'")
    return int(value)


def _normalize_on_error(value: object) -> OnError:
    if value not in {"raise", "continue"}:
        raise ValueError("on_error must be 'raise' or 'continue'")
    return cast(OnError, value)


def _normalize_verbose(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("verbose must be a non-negative int")
    if value < 0:
        raise ValueError("verbose must be a non-negative int")
    return int(value)


def _normalize_stage_default_scope(value: object, *, name: str) -> StageDefaultScope:
    aliases = {
        "full": "full_panel",
        "full_panel": "full_panel",
        "global": "full_panel",
        "origin": "origin_available",
        "origin_available": "origin_available",
        "available": "origin_available",
        "fit": "fit_window",
        "fit_window": "fit_window",
        "train": "fit_window",
        "train_window": "fit_window",
    }
    key = str(value).lower().replace("-", "_")
    if key not in aliases:
        raise ValueError(f"{name} must be 'full_panel', 'origin_available', or 'fit_window'")
    return cast(StageDefaultScope, aliases[key])


def _normalize_metadata_level(value: object) -> MetadataLevel:
    key = str(value).lower().replace("-", "_")
    if key not in {"minimal", "standard", "full"}:
        raise ValueError("metadata_level must be 'minimal', 'standard', or 'full'")
    return cast(MetadataLevel, key)
