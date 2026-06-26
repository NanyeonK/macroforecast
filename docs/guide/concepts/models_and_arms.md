# Models and Arms

[Back to User Guide](../index.md)

An `Arm` is a target-agnostic configuration: preprocessing + features + a single
model. It is not a cell by itself; applied to a target and a horizon it forms one
cell (executed by one `run()` call), and in the evaluation it appears as exactly
one contender. An arm is one contender: one arm, one named entry in the
accuracy/DM/MCS table.

A `ModelSpec` is the description of a single model: its name, optional fixed
parameters, and optional parameter search space. Most model families are
accessible by name as a string (e.g. `"ar"`, `"random_forest"`, `"lasso"`).

The pipeline enforces that each arm contains exactly one model. Comparing models
means using multiple arms that are identical except for `model`. The helper
`pipeline.model_arms` builds one arm per model for a pure model comparison,
sharing a common preprocessing and feature spec.

## Key Callables

`pipeline.Arm` declares the full configuration for one contender.

`pipeline.model_arms` builds a list of arms differing only in their model,
sharing all other settings.

```python
import macroforecast as mf
from macroforecast.pipeline import Arm, model_arms

prep = mf.preprocessing.preprocess_spec(
    transform="official",
    outliers="iqr",
    impute="em_factor",
)
features = mf.feature_engineering.feature_spec(
    target="INDPRO",
    predictors="all",
    lags=None,
    feature_steps=[mf.feature_engineering.marx_step(name="MARX_X", max_lag=12)],
)

# Explicit arms: each declares its own model and optionally its own features.
arms = [
    Arm(
        name="AR",
        model="ar",
        is_benchmark=True,
    ),
    Arm(
        name="RF",
        model="random_forest",
        preprocessing=prep,
        features=features,
    ),
    Arm(
        name="EN",
        model="elastic_net",
        preprocessing=prep,
        features=features,
        nested_in_benchmark=True,  # EN nests the AR, Clark-West is licensed
    ),
]

# Shortcut for a pure model comparison (all arms share prep + features).
comparison_arms = model_arms(
    ["ridge", "lasso", "elastic_net"],
    preprocessing=prep,
    features=features,
    nested_in_benchmark=True,
)
```

## Executed walkthrough

Each arm exposes its name, model, and benchmark flag, and `model_arms` expands a
list of model names into one arm apiece:

```python
print([(a.name, a.model, a.is_benchmark) for a in arms])
print([a.name for a in comparison_arms])
```

```text
[('AR', 'ar', True), ('RF', 'random_forest', False), ('EN', 'elastic_net', False)]
['ridge', 'lasso', 'elastic_net']
```

The string names resolve against the built-in model registry. `list_model_specs`
returns the full catalogue:

```python
df = mf.list_model_specs()
print(len(df))
print(list(df["name"])[:10])
```

```text
76
['ols', 'ridge', 'nonneg_ridge', 'shrink_to_target_ridge', 'fused_difference_ridge', 'supervised_aggregation', 'component_aggregation', 'rank_aggregation', 'assemblage_regression', 'albacore_components']
```

## Reference

- [Pipeline reference page](../../reference/pipeline.md) — `Arm`, `model_arms`, `PipelineSpec`, `EvalSpec`, `CombinationContender`, and t-code to target mapping.
- [Models reference page](../../reference/models.md) — `ModelSpec`, `ModelFit`, and the list of built-in model families.
