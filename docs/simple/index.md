# Simple Docs

This track is for researchers who want to run macroeconomic forecasting experiments without learning the internal recipe and registry system first.

The simple API is organized around `forecast` and `Experiment`.

```python
import macrocast as mc

result = mc.forecast(
    "fred_md",
    target="INDPRO",
    start="1980-01",
    end="2019-12",
    horizons=[1, 3, 6],
)
```

For comparisons and custom methods, use `Experiment`.

```python
exp = mc.Experiment(
    dataset="fred_md",
    target="INDPRO",
    start="1980-01",
    end="2019-12",
    horizons=[1, 3, 6],
)

result = exp.compare_models(["ar", "ridge"]).run()
```

The MVP contract is default-first: specify the forecasting question, then sweep only choices you care about. All resolved defaults are written to artifacts and manifests.

At Layer 0, Simple exposes only **Study Scope**. The shape of the call picks it:

- `forecast()` or `Experiment.run()` with one target and one method path -> `study_scope = one_target_one_method`
- `Experiment.compare_models([...])` with one target -> `study_scope = one_target_compare_methods`

The other Layer 0 execution-policy axes are filled from defaults:

- `failure_policy = fail_fast`
- `reproducibility_mode = seeded_reproducible`
- `compute_mode = serial`

Use Detail (code): Full when those policies should be reviewed or changed directly.

MVP public shapes:

- one default run
- one model-comparison run
- fixed custom model, custom preprocessor, target transformer, or FRED-SD inferred/empirical t-code policy inside those runs

The simple API intentionally exposes only direct comparison-cell and model-comparison routes. Advanced Layer 0 routes are present in the full grammar, but are not surfaced here until their public result contracts are fixed:

- preprocessing sweeps
- model x feature or model x preprocessing grids
- ablations
- replications
- benchmark suites
- multi-target wrappers

Those routes compile to explicit handoff contracts in the detailed docs instead of running through the simple `forecast()` path.

```{toctree}
:maxdepth: 1

quickstart
run_experiment
compare_models
sweep_only_what_you_care
fred_sd
add_custom_model
add_custom_preprocessor
read_results
```
