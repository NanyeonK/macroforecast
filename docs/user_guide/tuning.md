# Tuning engine

## Purpose

The tuning module provides hyperparameter optimization for all model families.
It sits between the recipe/compiler layer and the execution layer, providing search algorithms, temporal cross-validation, and budget enforcement.

## Architecture

```
macrocast/tuning/
  types.py          # TuningSpec, TuningResult, TuningTrial, HPDistribution
  engine.py         # run_tuning() orchestrator
  budget.py         # TuningBudget (max_trials, max_time, early_stop)
  hp_spaces.py      # per-model default HP distributions
  search/
    grid.py         # exhaustive grid search
    random.py       # random sampling from distributions
    bayesian.py     # TPE-based Bayesian optimization (optuna backend)
    genetic.py      # genetic algorithm (pure numpy)
  validation/
    splitter.py     # temporal CV splitters
    scorer.py       # validation scoring functions
```

## Search algorithms (4 operational)

- `grid_search` — exhaustive enumeration of discrete HP grid
- `random_search` — random sampling from HP distributions with budget enforcement
- `bayesian_optimization` — Tree-structured Parzen Estimator via optuna (optional dependency)
- `genetic_algorithm` — tournament selection, BLX-alpha crossover, Gaussian mutation (pure numpy)

## Validation splitters (4 operational)

- `LastBlockSplitter` — single held-out block at end of training window
- `RollingBlocksSplitter` — multiple rolling validation blocks
- `ExpandingValidationSplitter` — expanding walk-forward within training
- `BlockedKFoldSplitter` — blocked k-fold respecting temporal order

## Budget enforcement

- `max_trials` — maximum number of HP evaluations
- `max_time_seconds` — wall-clock time limit
- `early_stop_trials` — stop after N consecutive non-improving trials

## Tuning artifact

Every execution writes `tuning_result.json`:

```json
{
  "tuning_enabled": true,
  "model_family": "ridge",
  "search_algorithm": "bayesian_optimization",
  "best_hp": {"alpha": 0.0123},
  "best_score": 0.0045,
  "total_trials": 50,
  "total_time_seconds": 12.3
}
```

When tuning is not active, `tuning_enabled` is `false` and other fields are empty/zero.

## Dependencies

- `optuna` — required only for `bayesian_optimization` (import-guarded)
- All other algorithms use numpy only
