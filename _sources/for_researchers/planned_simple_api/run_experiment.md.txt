# Run An Experiment

> **API status note (current)**: this page uses the planned mf.forecast / mf.Experiment Python facade
> shape. Those are not yet exported from macroforecast.__all__. For working v0.6+ code, use
> macroforecast.run("recipe.yaml"), macroforecast.replicate("manifest.json"),
> the RecipeBuilder (macroforecast.scaffold.builder.RecipeBuilder), or
> python -m macroforecast scaffold. See [Simple Docs index](index.md) for the full status note.


`Experiment` is the public workspace for one forecasting question. It collects the few choices a researcher should specify, fills the rest from the default profile, then lowers to the internal recipe and execution engine.

```python
import macroforecast as mf

exp = mf.Experiment(
    dataset="fred_md",
    target="INDPRO",
    start="1980-01",
    end="2019-12",
    horizons=[1, 3, 6],
)

result = exp.run()
```

Required arguments:

- `dataset`: FRED source panel, one of `fred_md`, `fred_qd`, `fred_sd`, `fred_md+fred_sd`, or `fred_qd+fred_sd`
- `target`: target series name
- `start`: first sample date
- `end`: last sample date

Common optional arguments:

- `horizons`: forecast horizons, default `[1]`
- `frequency`: required only for `fred_sd` alone
- `custom_source_policy`: default `official_only`; use `custom_panel_only` or `official_plus_custom` for custom files
- `custom_source_path`: required when custom data is selected; parser/schema are inferred from the path extension and route frequency
- `vintage`: data vintage; if omitted, current/latest data is used
- `model_family`: default model, usually left as `ar`
- `primary_metric`: default `msfe`
- `random_seed`: default `42`

## Layer 0 In Simple

Simple exposes only the Layer 0 **Study Scope** decision. The run shape picks it:

- `exp.run()` with one target and one method path -> `one_target_one_method`
- `exp.compare_models([...]).run()` with one target -> `one_target_compare_methods`

The remaining Layer 0 policies are defaulted:

| Policy | Simple default | Effect |
|---|---|---|
| `failure_policy` | `fail_fast` | Stop on the first failed cell or variant. |
| `reproducibility_mode` | `seeded_reproducible` | Apply seed discipline; `random_seed` defaults to `42` and can be passed to `Experiment`. |
| `compute_mode` | `serial` | Execute sequentially. |

These resolved defaults are still written to the lowered recipe and manifest. To select non-default failure handling, reproducibility mode, or compute layout directly, use the Full YAML path in Detail (code).

Inspect the lowered recipe before running:

```python
recipe = exp.to_recipe_dict()
```

Run with a custom output root:

```python
result = exp.run(output_root="outputs/indpro_default")
```

The simple API is a facade. It does not bypass recipes, manifests, or provenance. Every default it fills is written to the manifest.

## Single Run Or Sweep

If no sweep axes are present, `run()` returns `ExperimentRunResult`.

```python
result = exp.run()
result.forecasts
result.metrics
result.manifest
```

If `.compare_models()` or `.sweep()` creates multiple variants, `run()` returns `ExperimentSweepResult`.

```python
sweep = exp.compare_models(["ridge", "lasso"]).run()
sweep.variants
sweep.compare("msfe")
```
