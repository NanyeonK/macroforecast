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

- `horizons=(1,)` — single 1-step-ahead forecast
- `model_family="ar_p"` — autoregressive AR(p) on the target
- `random_seed=42`
- **Forecast strategy**: `direct` — fits a separate model for each horizon (`y_{t+h} ~ f(x_t)`)
- **Training window**: `expanding` — every refit uses all data from the sample start through the current origin
- **Refit policy**: `every_origin` — refits the model at every walk-forward origin
- **Benchmark minimum train size**: 5 observations (only affects the `zero_change` benchmark family)

All defaults live in `macroforecast.defaults`. To inspect or override them programmatically:

```python
from macroforecast.defaults import (
    DEFAULT_MODEL_FAMILY,
    DEFAULT_RANDOM_SEED,
    DEFAULT_HORIZONS,
    DEFAULT_FORECAST_STRATEGY,
    DEFAULT_TRAINING_START_RULE,
    DEFAULT_REFIT_POLICY,
)
```

To override the L4 forecast settings (strategy / window / refit) without writing
a full YAML recipe, use `.compare(axis_path, values)` on an `Experiment` — see
[Compare models and parameters](compare_models.md).

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
