# Custom Model

[Back to custom extensions](index.md)

This page is generated from the live callable signatures.

Use `custom_model` when the estimator is not built into `macroforecast`. It returns a `ModelSpec`, so it can be passed anywhere a built-in model spec is accepted.

Important current contract: there is no `metadata=` keyword. Use `description`, `parameters`, `default_params`, `search_spaces`, `input_kind`, `requires_extra`, `requires_scaling`, and `recommended_preprocessing` to describe the custom model.

## Callable Reference

### custom_model

Qualified name: `macroforecast.models.specs.custom_model`

#### Signature

```python
macroforecast.models.custom_model(name: str, fit_func: Callable[..., Any], *, family: str = "custom", default_params: Mapping[str, Any] | None = None, parameters: tuple[ModelParameter, ...] = (), search_spaces: SearchSpaces | None = None, default_search_method: str = "grid", default_preset: str = "standard", input_kind: InputKind = "supervised", backend: str = "custom", requires_extra: str | None = None, requires_scaling: bool = False, recommended_preprocessing: tuple[str, ...] = (), description: str | None = None) -> ModelSpec
```

#### Description

Build a user-owned ``ModelSpec`` without registering a package model.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |
| `fit_func` | positional or keyword | `Callable[..., Any]` | `required` |
| `family` | keyword only | `str` | `"custom"` |
| `default_params` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `parameters` | keyword only | `tuple[ModelParameter, ...]` | `()` |
| `search_spaces` | keyword only | `SearchSpaces \| None` | `None` |
| `default_search_method` | keyword only | `str` | `"grid"` |
| `default_preset` | keyword only | `str` | `"standard"` |
| `input_kind` | keyword only | `InputKind` | `"supervised"` |
| `backend` | keyword only | `str` | `"custom"` |
| `requires_extra` | keyword only | `str \| None` | `None` |
| `requires_scaling` | keyword only | `bool` | `False` |
| `recommended_preprocessing` | keyword only | `tuple[str, ...]` | `()` |
| `description` | keyword only | `str \| None` | `None` |

#### Returns

`ModelSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.models.custom_model(...)
```

### custom_model_ensemble

Qualified name: `macroforecast.model_ensemble.specs.custom_model_ensemble`

#### Signature

```python
macroforecast.model_ensemble.custom_model_ensemble(name: str, fit_func: Callable[..., Any], *, default_params: Mapping[str, Any] | None = None, parameters: tuple[ModelParameter, ...] = (), search_spaces: SearchSpaces | None = None, default_search_method: str = "grid", default_preset: str = "standard", backend: str = "custom", description: str | None = None) -> ModelSpec
```

#### Description

Build a user-owned fit-time model-ensemble spec.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |
| `fit_func` | positional or keyword | `Callable[..., Any]` | `required` |
| `default_params` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `parameters` | keyword only | `tuple[ModelParameter, ...]` | `()` |
| `search_spaces` | keyword only | `SearchSpaces \| None` | `None` |
| `default_search_method` | keyword only | `str` | `"grid"` |
| `default_preset` | keyword only | `str` | `"standard"` |
| `backend` | keyword only | `str` | `"custom"` |
| `description` | keyword only | `str \| None` | `None` |

#### Returns

`ModelSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_ensemble.custom_model_ensemble(...)
```

## Minimal Custom Model Example

```python
import numpy as np
import pandas as pd
import macroforecast as mf

class MeanFit:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value)

def mean_model(X: pd.DataFrame, y: pd.Series, *, offset: float = 0.0) -> MeanFit:
    return MeanFit(float(pd.Series(y).mean()) + offset)

model = mf.models.custom_model(
    "mean_model",
    mean_model,
    default_params={"offset": 0.0},
    search_spaces={"standard": {"offset": (-0.1, 0.0, 0.1)}},
    input_kind="supervised",
    requires_scaling=False,
    description="Mean benchmark with a tunable offset.",
)
```
