# `search_algorithm`

[Back to L4](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``search_algorithm`` on sub-layer ``L4_D_tuning`` (layer ``l4``).

## Sub-layer

**L4_D_tuning**

## Axis metadata

- Default: `'none'`
- Sweepable: True
- Status: operational

## Operational status summary

- Operational: 6 option(s)
- Future: 0 option(s)

## Options

### `none`  --  operational

No tuning; use the params block as-is.

Default. The recipe author has already chosen the hyperparameters.

Configures the ``search_algorithm`` axis on ``L4_D_tuning`` (layer ``l4``); the ``none`` value is materialised in the recipe's ``fixed_axes`` block under that sub-layer.

**When to use**

Default. Studies with hand-picked hyperparameters.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

_Last reviewed 2026-05-04 by macroforecast author._

### `cv_path`  --  operational

Regularisation path via RidgeCV / LassoCV.

Picks alpha from a grid via leave-one-out CV. Only applicable to ridge / lasso / elastic_net families.

**When to use**

Quick alpha selection; comparable to published cross-validated linear baselines.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

_Last reviewed 2026-05-04 by macroforecast author._

### `grid_search`  --  operational

Exhaustive grid over leaf_config.tuning_grid.

Sklearn ``GridSearchCV`` with ``TimeSeriesSplit`` cross-validation. Requires ``leaf_config.tuning_grid``.

**When to use**

Reproducible hyperparameter sweeps; comparison against published grid-tuned baselines.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

_Last reviewed 2026-05-04 by macroforecast author._

### `random_search`  --  operational

Random sampling of tuning_distributions.

Sklearn ``RandomizedSearchCV``. ``leaf_config.tuning_budget`` caps the iteration count.

**When to use**

Larger search spaces; black-box hyperparameter exploration.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

_Last reviewed 2026-05-04 by macroforecast author._

### `bayesian_optimization`  --  operational

Optuna TPE optimisation (optional dependency).

Requires ``pip install macroforecast[tuning]`` (optuna). Falls back to ``random_search`` when optuna isn't installed.

**When to use**

Expensive estimators where each fit costs many seconds; hyperparameter spaces with smooth landscapes.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

_Last reviewed 2026-05-04 by macroforecast author._

### `genetic_algorithm`  --  operational

Tournament-selection genetic algorithm.

Crossover-style evolution over hyperparameter dictionaries. ``leaf_config.genetic_algorithm_population`` and ``..._generations`` control budget.

**When to use**

Discrete / categorical hyperparameter spaces where TPE struggles.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

_Last reviewed 2026-05-04 by macroforecast author._
