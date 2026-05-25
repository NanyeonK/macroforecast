from __future__ import annotations

import time
import optuna
from ..types import TuningResult, TuningTrial


def bayesian_optimization(eval_fn, hp_space: dict, budget, random_state: int = 42, search_algorithm: str = "bayesian_optimization") -> TuningResult:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial: optuna.Trial) -> float:
        hp={}
        for name,dist in hp_space.items():
            if dist.type == "log_float":
                hp[name]=trial.suggest_float(name, dist.low, dist.high, log=True)
            elif dist.type == "float":
                hp[name]=trial.suggest_float(name, dist.low, dist.high)
            elif dist.type == "int":
                hp[name]=trial.suggest_int(name, int(dist.low), int(dist.high))
            elif dist.type == "categorical":
                hp[name]=trial.suggest_categorical(name, list(dist.choices))
        return float(eval_fn(hp))
    sampler=optuna.samplers.TPESampler(seed=random_state)
    study=optuna.create_study(direction="minimize", sampler=sampler)
    start=time.time()
    study.optimize(objective, n_trials=budget.max_trials, timeout=budget.max_time_seconds)
    trials=[]
    for i,t in enumerate(study.trials):
        if t.value is None: continue
        trials.append(TuningTrial(trial_id=i, hp_values=dict(t.params), validation_score=float(t.value), fit_time_seconds=0.0, status="completed"))
    best=min(trials, key=lambda t: t.validation_score)
    return TuningResult(best_hp=best.hp_values, best_score=best.validation_score, all_trials=tuple(trials), search_algorithm=search_algorithm, total_trials=len(trials), total_time_seconds=time.time()-start, convergence_info={})
