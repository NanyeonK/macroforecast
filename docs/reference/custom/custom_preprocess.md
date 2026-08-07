# Custom Preprocess

[Back to custom extensions](index.md)

This page is generated from the live callable signatures.

## Callable Reference

### preprocess_spec

Qualified name: `macroforecast.preprocessing.specs.preprocess_spec`

#### Signature

```python
macroforecast.preprocessing.preprocess_spec(**options: Any) -> PreprocessSpec
```

#### Description

Create a reusable preprocessing specification.

Keyword options are the same data-cleaning choices accepted by
``reprocess(...)``: frequency alignment, transform-code handling,
outlier policy, imputation policy, standardization, frame-edge handling,
and optional custom preprocessing steps. Stage timing and metadata are not
accepted here; they are supplied later through ``PreprocessSpec.fit(...)``
or by the forecasting/pipeline runner.

Custom preprocessing callables are safe for in-memory use. Disk-backed
preprocessing caches require a stable callable identity: use named
functions and set ``func.__mf_digest__`` whenever cached reuse should span
processes or runs. Anonymous lambdas without ``__mf_digest__`` are rejected
because they cannot be distinguished by a stable content identity.

With ``policy="fit_window"``, custom steps are re-executed on the apply
window at each origin. Those steps must be row-local/stateless; a custom
step that computes statistics from its whole input can read post-origin
rows and leak future information.

Returns
PreprocessSpec
    Frozen preprocessing configuration. Call ``fit(data)`` to get a
    ``FittedPreprocessor`` or ``fit_transform(data)`` to obtain a
    ``PreprocessedData`` object for the training panel.

Example
>>> import macroforecast as mf
>>> prep = mf.preprocessing.preprocess_spec(
...     transform="official",
...     outliers="iqr",
...     impute="em_factor",
...     standardize="zscore",
... )

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `options` | var keyword | `Any` | `required` |

#### Returns

`PreprocessSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.preprocess_spec(...)
```

### custom_preprocess

Qualified name: `macroforecast.preprocessing.preprocess.custom_preprocess`

#### Signature

```python
macroforecast.preprocessing.custom_preprocess(data: PreprocessInput, func: Callable[..., Any], *, metadata: Mapping[str, Any] | None = None, name: str | None = None, **params: Any) -> PreprocessedData
```

#### Description

Apply a user supplied preprocessing callable to a canonical panel.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `PreprocessInput` | `required` |
| `func` | positional or keyword | `Callable[..., Any]` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `name` | keyword only | `str \| None` | `None` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`PreprocessedData`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.custom_preprocess(...)
```

### custom_preprocess_step

Qualified name: `macroforecast.preprocessing.specs.custom_preprocess_step`

#### Signature

```python
macroforecast.preprocessing.custom_preprocess_step(name: str, func: Callable[..., Any] | None = None, *, fit_func: Callable[..., Any] | None = None, transform_func: Callable[..., Any] | None = None, row_local: bool = False, **params: Any) -> dict[str, Any]
```

#### Description

Return a custom preprocessing step for ``preprocess_spec(custom_steps=...)``.

There are two ways to write a step, and which one you need depends on
whether it aggregates.

**Row-local** -- each output row depends only on the same input row::

    custom_preprocess_step("log1p", log1p_step, row_local=True)

A log, a ratio between two columns, a sign: nothing that looks along the
index. These are safe under any policy, because re-running them on a longer
frame cannot change a row that was already there. Declare ``row_local=True``
to say so.

**Stateful** -- the step derives something from the sample first::

    custom_preprocess_step(
        "winsorize",
        fit_func=fit_bounds,        # (panel, metadata, **params) -> state
        transform_func=apply_bounds,  # (panel, state=..., metadata=..., **params) -> panel
    )

``fit_func`` sees only the estimation window and returns whatever state it
needs; ``transform_func`` receives that state and applies it. A quantile, a
mean, a fitted scaler, a selected column subset -- anything computed *from*
the data belongs in ``fit_func``, not recomputed inside ``transform_func``.

Why the split matters: under ``policy="fit_window"`` a step is applied to
each apply window, and that window contains the rows after the forecast
origin. A step that recomputes its own statistic there reads the future.
That is not hypothetical -- it changes forecasts, though whether it *reaches*
the forecast depends on the model. An affine leak (centering, rescaling) is
absorbed by an OLS intercept and leaves predictions untouched; a non-affine
one (clipping at a sample quantile) does not.

A bare ``func`` that declares neither is still accepted under
``policy="origin_available"``, where the sample is already restricted to
observable rows. Under ``fit_window`` it is refused: say ``row_local=True``
if it never aggregates, or split it into ``fit_func``/``transform_func`` if
it does.

For disk-backed preprocessing caches, set ``__mf_digest__`` on each callable
to a stable string and update it when behaviour changes. Without that opt-in
digest, the runner skips disk get/put for specs containing the callable and
recomputes instead of risking stale reuse.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |
| `func` | positional or keyword | `Callable[..., Any] \| None` | `None` |
| `fit_func` | keyword only | `Callable[..., Any] \| None` | `None` |
| `transform_func` | keyword only | `Callable[..., Any] \| None` | `None` |
| `row_local` | keyword only | `bool` | `False` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.custom_preprocess_step(...)
```
