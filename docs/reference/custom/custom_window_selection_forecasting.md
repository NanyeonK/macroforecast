# Custom Window, Selection, And Forecasting

[Back to custom extensions](index.md)

This page is generated from the live callable signatures.

## Callable Reference

### custom_stage_policy

Qualified name: `macroforecast.window.policy.custom_stage_policy`

#### Signature

```python
macroforecast.window.custom_stage_policy(selector: Callable[..., Any], *, update: StageUpdate = "every_origin", apply_to: tuple[str, ...] | list[str] = ('fit', 'test'), metadata: Mapping[str, Any] | None = None) -> StagePolicy
```

#### Description

Create a stage policy whose sample labels are supplied by a callable.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `selector` | positional or keyword | `Callable[..., Any]` | `required` |
| `update` | keyword only | `StageUpdate` | `"every_origin"` |
| `apply_to` | keyword only | `tuple[str, ...] \| list[str]` | `("fit", "test")` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |

#### Returns

`StagePolicy`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.window.custom_stage_policy(...)
```

### custom_search

Qualified name: `macroforecast.model_selection.builders.custom_search`

#### Signature

```python
macroforecast.model_selection.custom_search(name: str, func: Callable[..., Any], *, param_grid: dict[str, Iterable[Any] | Any] | None = None, param_distributions: dict[str, ParamDistribution | Iterable[Any] | Any] | None = None, n_iter: int = 20, random_state: int | None = None, metadata: dict[str, Any] | None = None, **params: Any) -> SearchSpec
```

#### Description

Build a user-supplied parameter-search request.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |
| `func` | positional or keyword | `Callable[..., Any]` | `required` |
| `param_grid` | keyword only | `dict[str, Iterable[Any] \| Any] \| None` | `None` |
| `param_distributions` | keyword only | `dict[str, ParamDistribution \| Iterable[Any] \| Any] \| None` | `None` |
| `n_iter` | keyword only | `int` | `20` |
| `random_state` | keyword only | `int \| None` | `None` |
| `metadata` | keyword only | `dict[str, Any] \| None` | `None` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`SearchSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_selection.custom_search(...)
```

### custom_combination

Qualified name: `macroforecast.forecasting.combination.custom_combination`

#### Signature

```python
macroforecast.forecasting.custom_combination(name: str, func: Callable[..., Any], *, models: Sequence[str] | None = None, **params: Any) -> CombinationSpec
```

#### Description

Build a custom forecast-combination spec for ``forecasting.run``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |
| `func` | positional or keyword | `Callable[..., Any]` | `required` |
| `models` | keyword only | `Sequence[str] \| None` | `None` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`CombinationSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecasting.custom_combination(...)
```
