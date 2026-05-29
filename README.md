# macroforecast

`macroforecast` is being rebuilt as a pandas-first macro forecasting workflow
package. The current public surface is intentionally small:

- `macroforecast.meta`: package-wide defaults such as random seed and worker count.
- `macroforecast.data`: canonical date-indexed panels, metadata, FRED/custom loaders, and study data specs.
- `macroforecast.preprocessing`: direct pandas preprocessing callables.
- `macroforecast.data_summary`: one-panel summary tables.
- `macroforecast.data_analysis`: before/after preprocessing analysis.
- `macroforecast.evaluation`: reserved for the next callable evaluation pass.

The old YAML/runtime implementation is no longer part of the clean importable
package. A reference copy is preserved on the `legacy-runtime-reference` branch.

## Install

```bash
pip install -e ".[dev]"
```

Torch is not installed by default in this rebuild.

## Quick Use

```python
import macroforecast as mf

mf.configure(random_seed=42, n_jobs=1)

bundle = mf.data.load_custom_csv(
    "panel.csv",
    date="date",
    dataset="my_panel",
    frequency="monthly",
)

data_spec = mf.data.spec(
    bundle,
    target="INDPRO",
    horizons=[1, 3, 6],
    start="1990-01-01",
    end="2024-12-01",
)

processed = mf.preprocessing.reprocess(
    data_spec,
    transform="custom",
    transform_codes={"INDPRO": 5},
    outliers="iqr",
    impute="em_factor",
)

summary = mf.data_summary.summarize_data(processed.panel)
analysis = mf.data_analysis.analyze_data(bundle.panel, processed.panel)
```

## Data Shape

The standard panel is a `pandas.DataFrame` with:

- a `DatetimeIndex` named `date`
- one macro series per column
- numeric values in the cells
- dataset metadata stored separately and mirrored in `panel.attrs["macroforecast_metadata"]`

`macroforecast.data.load_*()` returns a `DataBundle(panel, metadata)`.
`macroforecast.data.spec(...)` attaches target, horizon, sample-window, and
predictor choices to that panel.

## Documentation

Function-level documentation lives under `docs/reference/`.

## License

MIT
