from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import numpy as np


@dataclass(frozen=True)
class HPDistribution:
    type: Literal["float", "int", "categorical", "log_float"]
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] | None = None
    log: bool = False

    def sample(self, rng: np.random.RandomState):
        if self.type == "float":
            if self.low is None or self.high is None:
                raise ValueError("float distribution requires low and high")
            return float(rng.uniform(float(self.low), float(self.high)))
        if self.type == "log_float":
            if self.low is None or self.high is None:
                raise ValueError("log_float distribution requires low and high")
            return float(np.exp(rng.uniform(np.log(float(self.low)), np.log(float(self.high)))))
        if self.type == "int":
            if self.low is None or self.high is None:
                raise ValueError("int distribution requires low and high")
            return int(rng.randint(int(self.low), int(self.high) + 1))
        if self.type == "categorical":
            assert self.choices is not None
            return self.choices[rng.randint(len(self.choices))]
        raise ValueError(f"unknown distribution type {self.type!r}")


@dataclass(frozen=True)
class TuningSpec:
    search_algorithm: str
    tuning_objective: str
    tuning_budget: dict[str, Any]
    hp_space: dict[str, HPDistribution]
    validation_size_rule: str
    validation_size_config: dict[str, Any]
    validation_location: str
    embargo_gap: str
    embargo_gap_size: int
    seed: int | None


@dataclass(frozen=True)
class TuningTrial:
    trial_id: int
    hp_values: dict[str, Any]
    validation_score: float
    fit_time_seconds: float
    status: Literal["completed", "failed", "pruned"]


@dataclass(frozen=True)
class TuningResult:
    best_hp: dict[str, Any]
    best_score: float
    all_trials: tuple[TuningTrial, ...]
    search_algorithm: str
    total_trials: int
    total_time_seconds: float
    convergence_info: dict[str, Any]
