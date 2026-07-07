# Custom Features

[Back to custom extensions](index.md)

This page is generated from the live callable signatures.

## Callable Reference

### feature_spec

Qualified name: `macroforecast.feature_engineering.specs.feature_spec`

#### Signature

```python
macroforecast.feature_engineering.feature_spec(*, target: str | None = None, targets: Iterable[str] | None = None, horizon: int | None = None, horizons: Iterable[int] | int | None = None, predictors: "Literal['all'] | Iterable[str] | None" = None, lags: Iterable[int] | int | None = (0, 1), target_lags: Iterable[int] | int | None = None, rolling_windows: Iterable[int] | int | None = None, rolling_min_periods: int | None = None, add_time: bool = False, time_trend: bool = True, time_month: bool = False, time_quarter: bool = False, time_year: bool = False, pca_components: int | None = None, pca_columns: Iterable[str] | None = None, pca_scale: bool = True, pca_prefix: str = "pc", steps: Iterable[Mapping[str, Any]] | None = None, feature_steps: Iterable[Mapping[str, Any]] | None = None, include_original: bool = False, target_transform: TargetTransform = "level", target_mode: TargetMode = "direct", drop_missing: bool = True, metadata: Mapping[str, Any] | None = None) -> FeatureSpec
```

#### Description

Create a reusable feature-building specification.

Parameters define the target columns, horizons, predictor columns, simple
lag/rolling/PCA shortcuts, or an explicit ``feature_steps`` pipeline. The
returned spec is inert until a runner calls ``fit(...)`` or
``fit_transform(...)`` on a training panel, so stateful steps such as PCA,
sparse PCA, scaling, and feature selection are fitted inside the training
window rather than on the full sample.

``target``/``targets`` select the source series to forecast.
``horizon``/``horizons`` select direct forecast horizons. ``predictors`` may
be ``"all"``, an iterable of column names, ``None`` for metadata/default
resolution, or an empty iterable for target-only designs. ``lags`` and
``target_lags`` build simple lag matrices when no explicit step pipeline is
supplied. ``steps`` is an alias for ``feature_steps``.

Returns
FeatureSpec
    Frozen feature-builder configuration with ``fit``, ``fit_transform``,
    ``to_dict``, and ``to_metadata`` methods.

Example
>>> import macroforecast as mf
>>> features = mf.feature_engineering.feature_spec(
...     target="INDPRO",
...     predictors=["UNRATE", "CPIAUCSL"],
...     horizons=[1, 3],
...     lags=(0, 1, 2),
... )

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `target` | keyword only | `str \| None` | `None` |
| `targets` | keyword only | `Iterable[str] \| None` | `None` |
| `horizon` | keyword only | `int \| None` | `None` |
| `horizons` | keyword only | `Iterable[int] \| int \| None` | `None` |
| `predictors` | keyword only | `Literal['all'] \| Iterable[str] \| None` | `None` |
| `lags` | keyword only | `Iterable[int] \| int \| None` | `(0, 1)` |
| `target_lags` | keyword only | `Iterable[int] \| int \| None` | `None` |
| `rolling_windows` | keyword only | `Iterable[int] \| int \| None` | `None` |
| `rolling_min_periods` | keyword only | `int \| None` | `None` |
| `add_time` | keyword only | `bool` | `False` |
| `time_trend` | keyword only | `bool` | `True` |
| `time_month` | keyword only | `bool` | `False` |
| `time_quarter` | keyword only | `bool` | `False` |
| `time_year` | keyword only | `bool` | `False` |
| `pca_components` | keyword only | `int \| None` | `None` |
| `pca_columns` | keyword only | `Iterable[str] \| None` | `None` |
| `pca_scale` | keyword only | `bool` | `True` |
| `pca_prefix` | keyword only | `str` | `"pc"` |
| `steps` | keyword only | `Iterable[Mapping[str, Any]] \| None` | `None` |
| `feature_steps` | keyword only | `Iterable[Mapping[str, Any]] \| None` | `None` |
| `include_original` | keyword only | `bool` | `False` |
| `target_transform` | keyword only | `TargetTransform` | `"level"` |
| `target_mode` | keyword only | `TargetMode` | `"direct"` |
| `drop_missing` | keyword only | `bool` | `True` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |

#### Returns

`FeatureSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.feature_spec(...)
```

### custom_features

Qualified name: `macroforecast.feature_engineering.transforms.custom_features`

#### Signature

```python
macroforecast.feature_engineering.custom_features(data: FeatureInput, func: Callable[..., Any], *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, name: str | None = None, **params: Any) -> pd.DataFrame
```

#### Description

Apply a user supplied feature-engineering callable to a panel.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `func` | positional or keyword | `Callable[..., Any]` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `name` | keyword only | `str \| None` | `None` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.custom_features(...)
```

### custom_step

Qualified name: `macroforecast.feature_engineering.compose.custom_step`

#### Signature

```python
macroforecast.feature_engineering.custom_step(name: str, func: Callable[..., Any] | None = None, *, input: str = "panel", include: bool = True, columns: Iterable[str] | None = None, fit_func: Callable[..., Any] | None = None, transform_func: Callable[..., Any] | None = None, requires_target: bool = False, min_train_size: int | None = None, prefix: str | None = None, drop_missing: bool = False, **params: Any) -> dict[str, Any]
```

#### Description

Return a user-supplied feature step for ``feature_spec``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |
| `func` | positional or keyword | `Callable[..., Any] \| None` | `None` |
| `input` | keyword only | `str` | `"panel"` |
| `include` | keyword only | `bool` | `True` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `fit_func` | keyword only | `Callable[..., Any] \| None` | `None` |
| `transform_func` | keyword only | `Callable[..., Any] \| None` | `None` |
| `requires_target` | keyword only | `bool` | `False` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `prefix` | keyword only | `str \| None` | `None` |
| `drop_missing` | keyword only | `bool` | `False` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.custom_step(...)
```
