# Meta

`macroforecast.meta` owns study-wide setup. It is the callable counterpart of
the YAML `0_meta` block and is also available as `macroforecast.l0(...)` for
backward-compatible authoring.

```python
import macroforecast as mf

setup = mf.meta.configure(
    failure_policy="fail_fast",
    reproducibility_policy="seeded_reproducible",
    compute_policy="serial",
)
```

This returns:

```python
{
    "0_meta": {
        "fixed_axes": {
            "failure_policy": "fail_fast",
            "reproducibility_policy": "seeded_reproducible",
            "compute_policy": "serial",
        },
        "leaf_config": {"random_seed": 42},
    }
}
```

## Axes

| Axis | Default | Choices |
| --- | --- | --- |
| `failure_policy` | `fail_fast` | `fail_fast`, `continue_on_failure` |
| `reproducibility_policy` | `seeded_reproducible` | `seeded_reproducible`, `exploratory` |
| `compute_policy` | `serial` | `serial`, `parallel` |

## Leaf Settings

| Setting | Default | Use |
| --- | --- | --- |
| `random_seed` | `42` when `seeded_reproducible` | deterministic replay |
| `parallel_unit` | `cells` when `compute_policy="parallel"` | parallel work split |
| `n_workers` | unset | worker count or `auto` |
| `gpu_deterministic` | unset | deterministic GPU mode flag |

`meta` does not choose data, preprocessing, features, models, or evaluation
metrics. It only controls study setup and execution policy.

For the generated low-level option page, see [L0 Study Setup](generated/l0/index.md).
