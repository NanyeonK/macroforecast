from __future__ import annotations

import itertools
import time
from ..types import TuningResult, TuningTrial


def grid_search(eval_fn, hp_grid: dict[str, list], budget, search_algorithm: str = "grid_search") -> TuningResult:
    keys = list(hp_grid)
    combos = list(itertools.product(*[hp_grid[k] for k in keys]))
    trials = []
    start = time.time()
    for i, combo in enumerate(combos):
        if budget.exceeded(trials):
            break
        hp = dict(zip(keys, combo))
        t0 = time.time()
        score = eval_fn(hp)
        trials.append(TuningTrial(trial_id=i, hp_values=hp, validation_score=float(score), fit_time_seconds=time.time()-t0, status="completed"))
        budget.update(float(score))
    best = min(trials, key=lambda t: t.validation_score)
    return TuningResult(best_hp=best.hp_values, best_score=best.validation_score, all_trials=tuple(trials), search_algorithm=search_algorithm, total_trials=len(trials), total_time_seconds=time.time()-start, convergence_info={})
