# `genetic_algorithm_selection` -- Evolutionary feature subset search via genetic algorithm (Goldberg 1989).

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.genetic_algorithm_selection`.

## Function signature

```python
mf.functions.genetic_algorithm_selection(
    panel: pd.DataFrame,
    target: pd.Series,
    population_size: int,
    n_generations: int,
    crossover_prob: float,
    fitness_estimator: str,
    cv_folds: int,
    random_state: int,
    temporal_rule: str,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `target` | `pd.Series` | — | — | Supervisory signal aligned to the panel index. Must share at least one index value with panel; raises ValueError if the intersection is empty. |
| `population_size` | `int` | `30` | >= 4 | Number of candidate feature masks in each generation. Larger populations explore more of the subset space per generation. |
| `n_generations` | `int` | `50` | >= 1 | Number of evolutionary generations. More generations allow finer convergence at higher compute cost. |
| `crossover_prob` | `float` | `0.8` | in (0, 1] | Probability of applying uniform crossover to a parent pair. The complement probability reproduces a parent unchanged. |
| `fitness_estimator` | `str` | `'"ridge"'` | "ridge" | "lasso" | "ols" | Estimator used to evaluate each feature subset's CV accuracy. ``ridge`` is recommended for high-dim panels; ``ols`` for small subsets. |
| `cv_folds` | `int` | `3` | >= 2 | Number of time-series cross-validation folds per fitness evaluation. Higher values are more reliable but slow. |
| `random_state` | `int` | `0` | — | Random seed for population initialisation and genetic operators. |
| `temporal_rule` | `str` | `'"expanding_window_per_origin"'` | "expanding_window_per_origin" | "rolling_window_per_origin" | Controls when the genetic search is re-run per forecast origin. ``full_sample_once`` is hard-rejected. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Evolves a population of binary feature-inclusion masks to maximise cross-validated forecast accuracy. Algorithm:

1. Initialise ``population_size`` random binary masks (each bit indicates inclusion of one feature).
2. Evaluate each mask by fitting ``fitness_estimator`` on the sub-panel and computing ``cv_folds``-fold time-series CV MSE.
3. Select parents by tournament selection proportional to CV fitness.
4. Apply uniform crossover (rate ``crossover_prob``) and bit-flip mutation to produce the next generation.
5. Repeat for ``n_generations`` generations; return the feature subset of the fittest mask in the final population.

Genetic search is useful when the inclusion/exclusion objective is non-convex and greedy backward/forward procedures get stuck. ``temporal_rule`` governs when the GA is re-run per origin; ``full_sample_once`` is rejected.

**When to use**

Feature selection when the predictive objective is highly non-linear or when combinations of individually weak predictors form strong subsets inaccessible to greedy methods.

**When NOT to use**

Large panels (K > 200) or long time series where CV evaluation of many masks is computationally prohibitive -- prefer lasso_path_selection or boruta_selection.

## In recipe context

Set ``params.op = "genetic_algorithm_selection"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: genetic_algorithm_selection
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a pipeline of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Goldberg, D.E. (1989) Genetic Algorithms in Search, Optimization and Machine Learning. Addison-Wesley, Reading, MA.

## Related ops

See also: `feature_selection`, `boruta_selection`, `recursive_feature_elimination` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
