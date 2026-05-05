from __future__ import annotations

import time
import numpy as np
from ..types import TuningResult, TuningTrial


def random_search(eval_fn, hp_space: dict, budget, random_state: int = 42, search_algorithm: str = "random_search") -> TuningResult:
    rng = np.random.RandomState(random_state)
    trials=[]
    start=time.time()
    i=0
    while not budget.exceeded(trials):
        hp={k: dist.sample(rng) for k,dist in hp_space.items()}
        t0=time.time()
        score=eval_fn(hp)
        trials.append(TuningTrial(trial_id=i, hp_values=hp, validation_score=float(score), fit_time_seconds=time.time()-t0, status="completed"))
        budget.update(float(score))
        i+=1
    best=min(trials, key=lambda t: t.validation_score)
    return TuningResult(best_hp=best.hp_values, best_score=best.validation_score, all_trials=tuple(trials), search_algorithm=search_algorithm, total_trials=len(trials), total_time_seconds=time.time()-start, convergence_info={})
