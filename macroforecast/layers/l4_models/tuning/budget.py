from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TuningBudget:
    max_trials: int | None = None
    max_time_seconds: float | None = None
    early_stop_trials: int | None = None
    min_improvement: float = 0.0
    _start_time: float = field(default_factory=time.time)
    _no_improvement_count: int = 0
    _best_score: float = float("inf")
    _trial_count: int = 0

    def exceeded(self, trials: list | None = None) -> bool:
        if self.max_trials is not None and self._trial_count >= self.max_trials:
            return True
        if self.max_time_seconds is not None and (time.time() - self._start_time) >= self.max_time_seconds:
            return True
        if self.early_stop_trials is not None and self._no_improvement_count >= self.early_stop_trials:
            return True
        return False

    def update(self, score: float):
        self._trial_count += 1
        if score < (self._best_score - self.min_improvement):
            self._best_score = score
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1
