# custom_preprocess

[Back to custom extensions](index.md)

Use custom preprocessing when a user cleaning operation belongs before feature
engineering. The callable should transform the canonical panel, not build
targets, lags, models, or forecast diagnostics.

## Direct Function

```python
mf.preprocessing.custom_preprocess(
    data,
    func,
    *,
    name=None,
    **params,
) -> mf.preprocessing.PreprocessedData
```

### Callable Signature

```python
func(panel: pandas.DataFrame, *, metadata: dict, **params)
```

### Accepted Return Types

| Return type | Meaning |
| --- | --- |
| `DataFrame` | Re-normalized as the new processed panel. |
| `DataBundle` | Uses the returned bundle panel and metadata. |
| `PreprocessedData` | Uses the returned preprocessing object directly. |
| `(DataFrame, metadata)` | Uses explicit panel and metadata pair. |

### Output

| Field | Meaning |
| --- | --- |
| `processed.panel` | Processed canonical panel. |
| `processed.metadata["custom_preprocess"]` | Custom step name, callable name, parameters, and inherited metadata. |
| `processed.steps` | Includes `{"step": "custom_preprocess", "name": ...}`. |

## Runner-Safe Step

```python
mf.preprocessing.custom_preprocess_step(
    name,
    func,
    **params,
) -> dict
```

Use this inside `preprocess_spec(custom_steps=[...])`:

```python
def add_spread(panel, *, metadata=None, scale=1.0):
    out = panel.copy()
    out["spread"] = (out["long_rate"] - out["short_rate"]) * scale
    return out

preprocessing = mf.preprocessing.preprocess_spec(
    transform="none",
    outliers="none",
    impute="none",
    standardize="none",
    custom_steps=[
        mf.preprocessing.custom_preprocess_step(
            "spread",
            add_spread,
            scale=100.0,
        ),
    ],
)
```

The runner applies this under its preprocessing policy. That means the same
callable can be used in full-sample, origin-available, fit-window, or
fixed-reference preprocessing depending on runner configuration.

## Boundary

| Put here | Put elsewhere |
| --- | --- |
| cleaning, deterministic variable recoding, adding raw derived series before feature creation | target creation, lags, rolling features, PCA, model fitting, forecast diagnostics |
