# Read results

`mf.forecast(...)` and `Experiment.run(...)` return `ForecastResult`.

```python
import macroforecast as mf

result = mf.forecast(
    "fred_md",
    target="INDPRO",
    start="1980-01",
    end="2019-12",
    horizons=[1, 3],
    output_directory="outputs/indpro_default",
)
```

> **Date formats**: `start` / `end` accept ISO date strings: full `YYYY-MM-DD`, or partial `YYYY-MM` (normalized to first/last of month), or `YYYY` (normalized to year-start/year-end).

Use the table accessors for normal analysis:

```python
forecasts = result.forecasts
metrics = result.metrics
ranking = result.ranking
summary = result.mean(metric="mse")
```

Use cell accessors when you need runtime-level detail:

```python
cells = result.cells
succeeded = result.succeeded

first_cell = result.get(cells[0].cell_id)
```

Use artifact helpers when the run was executed with `output_directory`:

```python
manifest_path = result.manifest_path
manifest = result.read_json("manifest.json")
predictions_path = result.file_path("predictions.csv")
```

`result.manifest` is the underlying execution result object. `result.manifest_path` is only available when artifacts were written to disk.
