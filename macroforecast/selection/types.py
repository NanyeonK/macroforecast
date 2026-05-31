from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from macroforecast.metrics import MetricLike

DistributionKind = Literal["float", "log_float", "int", "categorical"]


@dataclass(frozen=True)
class ParamDistribution:
    """Sampling rule for one hyperparameter."""

    kind: DistributionKind
    low: float | int | None = None
    high: float | int | None = None
    choices: tuple[Any, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready distribution description."""

        return {
            "kind": self.kind,
            "low": _json_ready(self.low),
            "high": _json_ready(self.high),
            "choices": _json_ready(self.choices),
        }

    def validate(self) -> None:
        """Validate distribution bounds before sampling or search execution."""

        if self.kind == "float":
            if self.low is None or self.high is None:
                raise ValueError("float distributions require low and high")
            if float(self.low) >= float(self.high):
                raise ValueError("float distributions require low < high")
            return
        if self.kind == "log_float":
            if self.low is None or self.high is None:
                raise ValueError("log_float distributions require low and high")
            if float(self.low) <= 0 or float(self.high) <= 0:
                raise ValueError("log_float distributions require positive bounds")
            if float(self.low) >= float(self.high):
                raise ValueError("log_float distributions require low < high")
            return
        if self.kind == "int":
            if self.low is None or self.high is None:
                raise ValueError("int distributions require low and high")
            low = float(self.low)
            high = float(self.high)
            if not low.is_integer() or not high.is_integer():
                raise ValueError("int distributions require integer bounds")
            if int(low) > int(high):
                raise ValueError("int distributions require low <= high")
            return
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError("categorical distributions require at least one choice")
            return
        raise ValueError(f"Unknown distribution kind: {self.kind!r}")

    def sample(self, rng: np.random.Generator) -> Any:
        self.validate()
        if self.kind == "float":
            assert self.low is not None and self.high is not None
            return float(rng.uniform(float(self.low), float(self.high)))
        if self.kind == "log_float":
            assert self.low is not None and self.high is not None
            return float(np.exp(rng.uniform(np.log(float(self.low)), np.log(float(self.high)))))
        if self.kind == "int":
            assert self.low is not None and self.high is not None
            return int(rng.integers(int(self.low), int(self.high) + 1))
        if self.kind == "categorical":
            return self.choices[int(rng.integers(0, len(self.choices)))]
        raise ValueError(f"Unknown distribution kind: {self.kind!r}")


@dataclass
class SearchSpec:
    """Parameter-search specification consumed by ``select_params``."""

    method: str
    param_grid: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    param_distributions: dict[str, ParamDistribution] = field(default_factory=dict)
    n_iter: int = 20
    random_state: int | None = None
    population_size: int = 12
    generations: int = 4
    mutation_rate: float = 0.2
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        """Return search-level metadata without trial results."""

        return _json_ready({
            "method": self.method,
            "n_iter": self.n_iter,
            "random_state": self.random_state,
            "population_size": self.population_size,
            "generations": self.generations,
            "mutation_rate": self.mutation_rate,
            "metadata": self.metadata,
        })

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready full search specification."""

        out = self.to_metadata()
        out["param_grid"] = _json_ready(self.param_grid)
        out["param_distributions"] = _json_ready(self.param_distributions)
        return out

    def to_json(
        self,
        path: str | Path | None = None,
        *,
        indent: int | None = 2,
    ) -> str:
        """Return JSON text, and optionally write it to ``path``."""

        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text


@dataclass(frozen=True)
class SearchTrial:
    """One evaluated parameter candidate."""

    trial: int
    params: dict[str, Any]
    score: float
    n_splits: int
    status: str = "ok"
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Return the row shape used in ``SearchResult.trials``."""

        return {
            "trial": self.trial,
            **self.params,
            "score": self.score,
            "n_splits": self.n_splits,
            "status": self.status,
            "error": self.error,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready trial dictionary."""

        return _json_ready(self.to_record())


class SearchError(RuntimeError):
    """Raised when parameter search cannot select any successful candidate."""

    def __init__(self, message: str, *, trials: pd.DataFrame | None = None) -> None:
        super().__init__(message)
        self.trials = pd.DataFrame() if trials is None else trials.copy()


@dataclass(frozen=True)
class SearchResult:
    """Result returned by ``select_params``."""

    best_params: dict[str, Any]
    best_score: float
    trials: pd.DataFrame
    metric: MetricLike
    method: str
    window: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Return a copy of the trial table."""

        return self.trials.copy()

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-ready result metadata without the trial table."""

        n_trials = int(len(self.trials))
        if "status" in self.trials:
            n_successful = int((self.trials["status"] == "ok").sum())
        else:
            n_successful = n_trials
        return _json_ready({
            "best_params": self.best_params,
            "best_score": self.best_score,
            "metric": self.metric,
            "method": self.method,
            "window": self.window,
            "n_trials": n_trials,
            "n_successful": n_successful,
            "n_failed": n_trials - n_successful,
            "metadata": self.metadata,
        })

    def to_dict(self, *, include_trials: bool = True) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""

        out = self.to_metadata()
        if include_trials:
            out["trials"] = _json_ready(self.trials.to_dict(orient="records"))
        return out

    def to_json(
        self,
        path: str | Path | None = None,
        *,
        include_trials: bool = True,
        indent: int | None = 2,
    ) -> str:
        """Return JSON text, and optionally write it to ``path``."""

        text = json.dumps(
            self.to_dict(include_trials=include_trials),
            indent=indent,
            ensure_ascii=False,
        )
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text


def _json_ready(value: Any) -> Any:
    if isinstance(value, ParamDistribution):
        return value.to_dict()
    if isinstance(value, SearchTrial):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, pd.Series):
        return _json_ready(value.to_dict())
    if isinstance(value, pd.DataFrame):
        return _json_ready(value.to_dict(orient="list"))
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NA:
        return None
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


__all__ = [
    "ParamDistribution",
    "SearchError",
    "SearchResult",
    "SearchSpec",
]
