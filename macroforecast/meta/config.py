"""Global execution settings for macroforecast."""
from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from typing import Any, Iterator, Literal, TypedDict, cast

from macroforecast.api.defaults import DEFAULT_RANDOM_SEED


OnError = Literal["raise", "continue"]
NJobs = int | Literal["auto"]


class MetaConfig(TypedDict):
    random_seed: int | None
    n_jobs: NJobs
    on_error: OnError
    verbose: int


_DEFAULT_CONFIG: MetaConfig = {
    "random_seed": DEFAULT_RANDOM_SEED,
    "n_jobs": 1,
    "on_error": "raise",
    "verbose": 0,
}

_CONFIG: MetaConfig = dict(_DEFAULT_CONFIG)  # type: ignore[assignment]
_LOCK = RLock()
_MISSING = object()


def configure(
    *,
    random_seed: int | None | object = _MISSING,
    n_jobs: NJobs | object = _MISSING,
    on_error: OnError | object = _MISSING,
    verbose: int | object = _MISSING,
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

    with _LOCK:
        _CONFIG.update(cast(MetaConfig, updates))
        return dict(_CONFIG)  # type: ignore[return-value]


def get_config() -> MetaConfig:
    """Return a copy of the current package-wide execution defaults."""

    with _LOCK:
        return dict(_CONFIG)  # type: ignore[return-value]


def get_option(name: str) -> Any:
    """Return one setting from the current package-wide execution defaults."""

    with _LOCK:
        if name not in _CONFIG:
            valid = ", ".join(_DEFAULT_CONFIG)
            raise KeyError(f"unknown meta option {name!r}; expected one of: {valid}")
        return _CONFIG[name]  # type: ignore[literal-required]


def reset_config() -> MetaConfig:
    """Reset package-wide execution defaults to their initial values."""

    with _LOCK:
        _CONFIG.clear()
        _CONFIG.update(_DEFAULT_CONFIG)
        return dict(_CONFIG)  # type: ignore[return-value]


@contextmanager
def use_config(
    *,
    random_seed: int | None | object = _MISSING,
    n_jobs: NJobs | object = _MISSING,
    on_error: OnError | object = _MISSING,
    verbose: int | object = _MISSING,
) -> Iterator[MetaConfig]:
    """Temporarily update package-wide execution defaults inside a context."""

    with _LOCK:
        previous: MetaConfig = dict(_CONFIG)  # type: ignore[assignment]
    try:
        active = configure(
            random_seed=random_seed,
            n_jobs=n_jobs,
            on_error=on_error,
            verbose=verbose,
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
