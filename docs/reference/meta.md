# macroforecast.meta

`macroforecast.meta` stores package-wide execution settings. These settings are
used when a run does not pass a more specific value through a direct function
argument. They do not choose data, preprocessing, features, models, evaluation
metrics, or output files.

The current public surface is:

| Function | Purpose |
| --- | --- |
| `configure` | Update one or more global execution settings. |
| `get_config` | Return the full active configuration. |
| `get_option` | Return one active configuration value. |
| `reset_config` | Restore package defaults. |
| `use_config` | Temporarily override settings inside a `with` block. |

YAML loading is intentionally outside this page. A future YAML wrapper can map a
file into the same `macroforecast.meta` functions.

## MetaConfig

`MetaConfig` is the output schema returned by `configure`, `get_config`,
`reset_config`, and yielded by `use_config`.

### Output Schema

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `random_seed` | `int | None` | `42` | Seed used by stochastic runtime components when no run-specific seed is supplied. `None` leaves stochastic components unseeded. |
| `n_jobs` | `int | "auto"` | `1` | Default worker count. `1` means serial execution; `"auto"` lets the runtime choose a bounded worker count. |
| `on_error` | `"raise" | "continue"` | `"raise"` | Default cell failure behavior. `"raise"` stops on failure; `"continue"` records the failure and continues where supported. |
| `verbose` | `int` | `0` | Default verbosity level for future runtime logging surfaces. |

Example output:

```python
{
    "random_seed": 42,
    "n_jobs": 1,
    "on_error": "raise",
    "verbose": 0,
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
    on_error: "raise" | "continue" = ...,
    verbose: int = ...,
) -> MetaConfig
```

All inputs are keyword-only. Omitted inputs keep their current values.

### Input

| Name | Type | Default if omitted | Allowed Values | Meaning |
| --- | --- | --- | --- | --- |
| `random_seed` | `int | None` | keep current value | non-negative integer or `None` | Sets the default seed for stochastic components. |
| `n_jobs` | `int | "auto"` | keep current value | positive integer or `"auto"` | Sets the default worker count. |
| `on_error` | `"raise" | "continue"` | keep current value | `"raise"`, `"continue"` | Sets default failure behavior. |
| `verbose` | `int` | keep current value | non-negative integer | Sets default verbosity. |

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
| `on_error` is not `"raise"` or `"continue"` | `ValueError` |
| `verbose` is not a non-negative integer | `TypeError` or `ValueError` |

### Example

```python
import macroforecast as mf

config = mf.meta.configure(
    random_seed=7,
    n_jobs="auto",
    on_error="continue",
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
| `name` | `str` | `"random_seed"`, `"n_jobs"`, `"on_error"`, `"verbose"` | Configuration key to read. |

### Output

Returns the value for `name`.

| `name` | Output Type |
| --- | --- |
| `"random_seed"` | `int | None` |
| `"n_jobs"` | `int | "auto"` |
| `"on_error"` | `"raise" | "continue"` |
| `"verbose"` | `int` |

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
    on_error: "raise" | "continue" = ...,
    verbose: int = ...,
) -> Iterator[MetaConfig]
```

### Input

The inputs match `configure`.

| Name | Type | Default if omitted | Allowed Values | Meaning |
| --- | --- | --- | --- | --- |
| `random_seed` | `int | None` | keep current value inside context | non-negative integer or `None` | Temporary default seed. |
| `n_jobs` | `int | "auto"` | keep current value inside context | positive integer or `"auto"` | Temporary worker count. |
| `on_error` | `"raise" | "continue"` | keep current value inside context | `"raise"`, `"continue"` | Temporary failure behavior. |
| `verbose` | `int` | keep current value inside context | non-negative integer | Temporary verbosity. |

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

## Runtime Behavior

The execution runtime reads `macroforecast.meta` when no more specific value is
provided by the caller.

| Setting | Runtime Use |
| --- | --- |
| `random_seed` | Used as the default run seed and propagated to stochastic estimators where supported. |
| `n_jobs` | Used as the default worker count for run-level and selected model-level parallel work. |
| `on_error` | Mapped to runtime failure handling: `"raise"` stops on failure, `"continue"` continues where supported. |
| `verbose` | Reserved as the package-wide verbosity setting. |

Run manifests record the active `meta_config` so a completed run can be audited
against the settings in force at execution time.
