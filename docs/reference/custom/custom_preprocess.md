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
macroforecast.preprocessing.custom_preprocess_step(name: str, func: Callable[..., Any], **params: Any) -> dict[str, Any]
```

#### Description

Return a custom preprocessing step for ``preprocess_spec(custom_steps=...)``.

For disk-backed preprocessing caches, set ``func.__mf_digest__`` to a stable
string and update it when the callable's behavior changes. Without that
opt-in digest, the runner skips disk get/put for specs containing the
callable and recomputes instead of risking stale reuse.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |
| `func` | positional or keyword | `Callable[..., Any]` | `required` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.custom_preprocess_step(...)
```
