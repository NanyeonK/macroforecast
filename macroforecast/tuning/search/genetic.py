from __future__ import annotations

import time
import numpy as np
from ..types import TuningResult, TuningTrial


def _tournament_select(population, fitness, tournament_size, n_select, rng):
    selected=[]
    for _ in range(n_select):
        idx=rng.choice(len(population), size=min(tournament_size, len(population)), replace=False)
        best=idx[np.argmin(fitness[idx])]
        selected.append(population[best].copy())
    return selected


def _crossover(parent1, parent2, hp_space, rng):
    child1, child2 = {}, {}
    for name, dist in hp_space.items():
        if dist.type == "categorical":
            if rng.random() < 0.5:
                child1[name], child2[name] = parent1[name], parent2[name]
            else:
                child1[name], child2[name] = parent2[name], parent1[name]
        else:
            p1, p2 = float(parent1[name]), float(parent2[name])
            d = abs(p1 - p2)
            low = max(min(p1, p2) - 0.5 * d, float(dist.low))
            high = min(max(p1, p2) + 0.5 * d, float(dist.high))
            c1 = rng.uniform(low, high)
            c2 = rng.uniform(low, high)
            if dist.type == "int":
                c1, c2 = int(round(c1)), int(round(c2))
            child1[name], child2[name] = c1, c2
    return child1, child2


def _mutate(individual, hp_space, rng, sigma_frac=0.1):
    name=list(hp_space.keys())[rng.randint(len(hp_space))]
    dist=hp_space[name]
    if dist.type == "categorical":
        individual[name] = dist.choices[rng.randint(len(dist.choices))]
    elif dist.type in {"float","log_float"}:
        sigma=(float(dist.high)-float(dist.low))*sigma_frac
        individual[name] = float(np.clip(float(individual[name]) + rng.normal(0, sigma), float(dist.low), float(dist.high)))
    elif dist.type == "int":
        delta=max(1, int((int(dist.high)-int(dist.low))*sigma_frac))
        individual[name] = int(np.clip(int(individual[name])+rng.randint(-delta, delta+1), int(dist.low), int(dist.high)))


def genetic_algorithm(eval_fn, hp_space: dict, budget, random_state: int = 42, population_size: int = 20, n_generations: int = 10, search_algorithm: str = "genetic_algorithm") -> TuningResult:
    rng=np.random.RandomState(random_state)
    if budget.max_trials is not None:
        population_size = min(population_size, max(2, int(budget.max_trials)))
    population=[{k: dist.sample(rng) for k,dist in hp_space.items()} for _ in range(population_size)]
    start=time.time()
    all_trials=[]
    fitness=[]
    for ind in population:
        if budget.exceeded(all_trials):
            break
        score=float(eval_fn(ind))
        fitness.append(score)
        all_trials.append(TuningTrial(trial_id=len(all_trials), hp_values=ind.copy(), validation_score=score, fit_time_seconds=0.0, status="completed"))
        budget.update(score)
    fitness=np.asarray(fitness, dtype=float)
    population=population[:len(fitness)]
    for gen in range(n_generations):
        if budget.exceeded(all_trials) or len(population) < 2:
            break
        parents=_tournament_select(population, fitness, 3, len(population), rng)
        offspring=[]
        for i in range(0, len(parents)-1, 2):
            c1,c2=_crossover(parents[i], parents[i+1], hp_space, rng)
            _mutate(c1, hp_space, rng)
            _mutate(c2, hp_space, rng)
            offspring.extend([c1,c2])
        new_population=[]
        new_fitness=[]
        for ind in offspring[:population_size]:
            if budget.exceeded(all_trials):
                break
            score=float(eval_fn(ind))
            new_population.append(ind)
            new_fitness.append(score)
            all_trials.append(TuningTrial(trial_id=len(all_trials), hp_values=ind.copy(), validation_score=score, fit_time_seconds=0.0, status="completed"))
            budget.update(score)
        if new_population:
            population=new_population
            fitness=np.asarray(new_fitness, dtype=float)
    best=min(all_trials, key=lambda t: t.validation_score)
    return TuningResult(best_hp=best.hp_values, best_score=best.validation_score, all_trials=tuple(all_trials), search_algorithm=search_algorithm, total_trials=len(all_trials), total_time_seconds=time.time()-start, convergence_info={})
