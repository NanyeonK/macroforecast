# macroforecast.meta

[Back to reference](index.md)

## Purpose

`macroforecast.meta` stores package-wide execution settings. These settings are
used when a run does not pass a more specific value through a direct function
argument. They do not choose data, preprocessing, features, models, evaluation
metrics, or output files.

The module is intentionally small. It owns defaults and temporary overrides;
direct function arguments and runner policies always take precedence over
global settings.

## Public Functions

| Function | Purpose |
| --- | --- |
| `configure` | Update one or more global execution settings. |
| `get_config` | Return the full active configuration. |
| `get_option` | Return one active configuration value. |
| `resolve_n_jobs` | Resolve the configured worker count (`'auto'` resolves to the CPU count). |
| `reset_config` | Restore package defaults. |
| `use_config` | Temporarily override settings inside a `with` block. |

## Public Values

| Symbol | Meaning |
| --- | --- |
| `DEFAULT_RANDOM_SEED` | Package default seed value, currently `42`. |
| `MetaConfig` | Dictionary-like output schema for active meta settings. |
| `NJobs` | Accepted worker-count type: positive integer or `"auto"`. |
| `OnError` | Stored failure-mode type: `"raise"` or `"continue"`. |
| `StageDefaultScope` | Runner stage-scope type: `"full_panel"`, `"origin_available"`, or `"fit_window"`. |
| `MetadataLevel` | Runner metadata detail type: `"minimal"`, `"standard"`, or `"full"`. |

## MetaConfig

`MetaConfig` is the output schema returned by `configure`, `get_config`,
`reset_config`, and yielded by `use_config`.

### Output Schema

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `random_seed` | <code>int &#124; None</code> | `42` | Seed used by stochastic functions when no run-specific seed is supplied. `None` leaves stochastic components unseeded. |
| `n_jobs` | <code>int &#124; "auto"</code> | `1` | Default worker count. `1` means serial execution; `"auto"` lets the package choose a bounded worker count. |
| `on_error` | failure mode | stop on error | Default cell failure behavior. See [Failure Mode Values](#failure-mode-values). |
| `verbose` | `int` | `0` | Default verbosity level for future logging surfaces. |
| `default_preprocessing_scope` | `"full_panel"`, `"origin_available"`, or `"fit_window"` | `"origin_available"` | Default `forecasting.run(..., preprocessing_policy=...)` scope when preprocessing is supplied and no explicit policy is passed. |
| `default_feature_scope` | `"full_panel"`, `"origin_available"`, or `"fit_window"` | `"fit_window"` | Default `feature_policy` scope. |
| `default_selection_scope` | `"full_panel"`, `"origin_available"`, or `"fit_window"` | `"fit_window"` | Default `model_selection_policy` scope. |
| `metadata_level` | `"minimal"`, `"standard"`, or `"full"` | `"standard"` | Runner metadata detail. `minimal` suppresses per-origin stage records; `standard` and `full` currently keep the same stage ledger. |

The default seed is owned by `macroforecast.meta.config.DEFAULT_RANDOM_SEED` and
is exported as `macroforecast.meta.DEFAULT_RANDOM_SEED`.

### Failure Mode Values

| User-facing mode | Stored value | Meaning |
| --- | --- | --- |
| Stop on error | `raise` | Stop immediately when a supported execution cell fails. |
| Continue where supported | `continue` | Record the failure and continue for call sites that support non-fatal errors. |

### Stage Scope Aliases

Stage-scope inputs are normalized before storage. Docs and metadata use the
canonical stored values.

| Accepted input | Stored value | Meaning |
| --- | --- | --- |
| `"full"`, `"full_panel"`, `"global"` | `"full_panel"` | Fit the stage on the full panel. |
| `"origin"`, `"origin_available"`, `"available"` | `"origin_available"` | Fit the stage on observations available at each forecast origin. |
| `"fit"`, `"fit_window"`, `"train"`, `"train_window"` | `"fit_window"` | Fit the stage on the model fit window. |

Example output:

```python
{
    "random_seed": 42,
    "n_jobs": 1,
    "on_error": "raise",
    "verbose": 0,
    "default_preprocessing_scope": "origin_available",
    "default_feature_scope": "fit_window",
    "default_selection_scope": "fit_window",
    "metadata_level": "standard",
}
```

## configure

Update package-wide execution settings and return the active configuration.

### Signature

```python
macroforecast.meta.configure(
    *,
    random_seed: int | None = ...,
    n_jobs: int | "auto" = ...,
    on_error: str = ...,
    verbose: int = ...,
    default_preprocessing_scope: "full_panel" | "origin_available" | "fit_window" = ...,
    default_feature_scope: "full_panel" | "origin_available" | "fit_window" = ...,
    default_selection_scope: "full_panel" | "origin_available" | "fit_window" = ...,
    metadata_level: "minimal" | "standard" | "full" = ...,
) -> MetaConfig
```

All inputs are keyword-only. Omitted inputs keep their current values.

### Input

| Name | Type | Default if omitted | Allowed Values | Meaning |
| --- | --- | --- | --- | --- |
| `random_seed` | <code>int &#124; None</code> | keep current value | non-negative integer or `None` | Sets the default seed for stochastic components. |
| `n_jobs` | <code>int &#124; "auto"</code> | keep current value | positive integer or `"auto"` | Sets the default worker count. |
| `on_error` | failure mode | keep current value | Stop on error (`raise`) or continue where supported (`continue`) | Sets default failure behavior. |
| `verbose` | `int` | keep current value | non-negative integer | Sets default verbosity. |
| `default_preprocessing_scope` | str | keep current value | `"full_panel"`, `"origin_available"`, `"fit_window"` | Sets the default preprocessing stage scope for `forecasting.run(...)`. |
| `default_feature_scope` | str | keep current value | `"full_panel"`, `"origin_available"`, `"fit_window"` | Sets the default feature-engineering stage scope. |
| `default_selection_scope` | str | keep current value | `"full_panel"`, `"origin_available"`, `"fit_window"` | Sets the default model-selection stage scope. |
| `metadata_level` | str | keep current value | `"minimal"`, `"standard"`, `"full"` | Sets the default runner metadata detail. |

Stage-scope inputs also accept the aliases listed in
[Stage Scope Aliases](#stage-scope-aliases). The returned `MetaConfig` always
stores the canonical value.

### Output

Returns `MetaConfig`, a copy of the full active configuration after the update.

| Output | Type | Meaning |
| --- | --- | --- |
| `config` | `MetaConfig` | The active package-wide execution settings. |

### Side Effects

`configure` changes global package state. Later calls that consult
`macroforecast.meta` will see the updated values.

### Validation

| Condition | Error |
| --- | --- |
| `random_seed` is negative | `ValueError` |
| `random_seed` is not `int` or `None` | `TypeError` |
| `n_jobs` is not a positive integer or `"auto"` | `TypeError` or `ValueError` |
| `on_error` is not a supported failure mode | `ValueError` |
| `verbose` is not a non-negative integer | `TypeError` or `ValueError` |
| stage default scope is not one of the allowed scopes | `ValueError` |
| `metadata_level` is not `"minimal"`, `"standard"`, or `"full"` | `ValueError` |

### Example

```python
import macroforecast as mf

config = mf.meta.configure(
    random_seed=7,
    n_jobs="auto",
    on_error="continue",
    default_feature_scope="origin_available",
    metadata_level="standard",
    verbose=1,
)

assert config["random_seed"] == 7
```

## get_config

Return the active package-wide execution settings.

### Signature

```python
macroforecast.meta.get_config() -> MetaConfig
```

### Input

No input.

### Output

Returns `MetaConfig`.

| Output | Type | Meaning |
| --- | --- | --- |
| `config` | `MetaConfig` | Copy of the active package-wide execution settings. |

The returned object is a copy. Mutating it does not change global package state.

### Example

```python
import macroforecast as mf

config = mf.meta.get_config()
print(config["n_jobs"])
```

## get_option

Return one active configuration value.

### Signature

```python
macroforecast.meta.get_option(name: str) -> object
```

### Input

| Name | Type | Allowed Values | Meaning |
| --- | --- | --- | --- |
| `name` | `str` | any `MetaConfig` key | Configuration key to read. |

### Output

Returns the value for `name`.

| `name` | Output Type |
| --- | --- |
| `"random_seed"` | <code>int &#124; None</code> |
| `"n_jobs"` | <code>int &#124; "auto"</code> |
| `"on_error"` | failure mode string |
| `"verbose"` | `int` |
| `"default_preprocessing_scope"` | `"full_panel"`, `"origin_available"`, or `"fit_window"` |
| `"default_feature_scope"` | `"full_panel"`, `"origin_available"`, or `"fit_window"` |
| `"default_selection_scope"` | `"full_panel"`, `"origin_available"`, or `"fit_window"` |
| `"metadata_level"` | `"minimal"`, `"standard"`, or `"full"` |

### Errors

| Condition | Error |
| --- | --- |
| `name` is not a known option | `KeyError` |

### Example

```python
import macroforecast as mf

seed = mf.meta.get_option("random_seed")
```

## reset_config

Restore package-wide execution settings to their defaults.

### Signature

```python
macroforecast.meta.reset_config() -> MetaConfig
```

### Input

No input.

### Output

Returns `MetaConfig` after reset.

| Key | Reset Value |
| --- | --- |
| `random_seed` | `42` |
| `n_jobs` | `1` |
| `on_error` | `"raise"` |
| `verbose` | `0` |
| `default_preprocessing_scope` | `"origin_available"` |
| `default_feature_scope` | `"fit_window"` |
| `default_selection_scope` | `"fit_window"` |
| `metadata_level` | `"standard"` |

### Side Effects

`reset_config` changes global package state.

### Example

```python
import macroforecast as mf

mf.meta.configure(random_seed=123, n_jobs=4)
config = mf.meta.reset_config()

assert config["random_seed"] == 42
assert config["n_jobs"] == 1
```

## use_config

Temporarily override package-wide execution settings inside a context manager.

### Signature

```python
macroforecast.meta.use_config(
    *,
    random_seed: int | None = ...,
    n_jobs: int | "auto" = ...,
    on_error: str = ...,
    verbose: int = ...,
    default_preprocessing_scope: "full_panel" | "origin_available" | "fit_window" = ...,
    default_feature_scope: "full_panel" | "origin_available" | "fit_window" = ...,
    default_selection_scope: "full_panel" | "origin_available" | "fit_window" = ...,
    metadata_level: "minimal" | "standard" | "full" = ...,
) -> Iterator[MetaConfig]
```

### Input

The inputs match `configure`.

| Name | Type | Default if omitted | Allowed Values | Meaning |
| --- | --- | --- | --- | --- |
| `random_seed` | <code>int &#124; None</code> | keep current value inside context | non-negative integer or `None` | Temporary default seed. |
| `n_jobs` | <code>int &#124; "auto"</code> | keep current value inside context | positive integer or `"auto"` | Temporary worker count. |
| `on_error` | failure mode | keep current value inside context | Stop on error (`raise`) or continue where supported (`continue`) | Temporary failure behavior. |
| `verbose` | `int` | keep current value inside context | non-negative integer | Temporary verbosity. |
| `default_preprocessing_scope` | str | keep current value inside context | `"full_panel"`, `"origin_available"`, `"fit_window"` | Temporary preprocessing default scope. |
| `default_feature_scope` | str | keep current value inside context | `"full_panel"`, `"origin_available"`, `"fit_window"` | Temporary feature default scope. |
| `default_selection_scope` | str | keep current value inside context | `"full_panel"`, `"origin_available"`, `"fit_window"` | Temporary model-selection default scope. |
| `metadata_level` | str | keep current value inside context | `"minimal"`, `"standard"`, `"full"` | Temporary metadata detail. |

Stage-scope inputs use the same alias normalization as `configure`.

### Output

Yields `MetaConfig`, the active configuration inside the context.

| Output | Type | Meaning |
| --- | --- | --- |
| `config` | `MetaConfig` | Active temporary settings inside the `with` block. |

### Side Effects

`use_config` changes global package state only for the duration of the `with`
block. The previous configuration is restored when the block exits, including
when an exception is raised.

### Example

```python
import macroforecast as mf

mf.meta.configure(random_seed=42)

with mf.meta.use_config(random_seed=7, n_jobs=2) as config:
    assert config["random_seed"] == 7
    # Run code here with the temporary settings.

assert mf.meta.get_option("random_seed") == 42
```

## Usage Behavior

Direct package functions can read `macroforecast.meta` when no more specific
value is provided by the caller.

| Setting | Use |
| --- | --- |
| `random_seed` | Used as the default run seed and propagated to stochastic estimators where supported. |
| `n_jobs` | Used as the default worker count for run-level and selected model-level parallel work. |
| `on_error` | Default failure handling. `raise` stops on failure; `continue` continues where supported. |
| `verbose` | Reserved as the package-wide verbosity setting. |
| `default_preprocessing_scope` | Used by `forecasting.run(...)` when `preprocessing` is supplied without `preprocessing_policy`. |
| `default_feature_scope` | Used by `forecasting.run(...)` when `feature_policy` is omitted. |
| `default_selection_scope` | Used by `forecasting.run(...)` when `model_selection_policy` is omitted. |
| `metadata_level` | Controls how much run-level metadata is recorded. `minimal` drops per-origin stage records; `standard` and `full` currently keep the same stage ledger. |

Forecast results record the active config under `metadata["run"]["config"]` so
a completed run can be audited against the settings in force at execution time.
