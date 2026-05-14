# Simple API quickstart

Run one default forecast with `mf.forecast(...)`.

```python
import macroforecast as mf

result = mf.forecast(
    "fred_md",
    target="INDPRO",
    start="1980-01",
    end="2019-12",
    horizons=[1, 3, 6],
)
```

The result is a `ForecastResult` facade over the runtime output:

```python
forecasts = result.forecasts
metrics = result.metrics
ranking = result.ranking
manifest = result.manifest
```

The high-level defaults are:

- `horizons=(1,)` if you do not pass `horizons`
- `model_family="ar_p"` if you do not pass `model_family`
- `random_seed=0` if you do not pass `random_seed`

The sample window is optional in the function signature, but most research runs should pass `start` and `end` explicitly so the study definition is reproducible.

To write artifacts to disk, pass `output_directory`:

```python
import macroforecast as mf

result = mf.forecast(
    "fred_md",
    target="INDPRO",
    start="1980-01",
    end="2019-12",
    horizons=[1],
    output_directory="outputs/indpro_default",
)

manifest_path = result.manifest_path
```
