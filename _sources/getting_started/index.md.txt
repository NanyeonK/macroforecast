# Getting Started

Start from the canonical layer system, not from the legacy stage labels.

```text
L0 -> L1 -> L2 -> L3(DAG) -> L4(DAG) -> L5 -> L6 -> L7(DAG) -> L8
        |      |      |       |
       L1.5   L2.5   L3.5    L4.5 diagnostics
```

## Choose Your Path

| Path | Use when | Start with |
|---|---|---|
| Layer design | You want to see the layer/DAG architecture before writing YAML. | [Navigator Docs](../navigator/index.md) |
| Runnable core recipe | You want the current L1-L8 artifact path and output directory. | [Quickstart](quickstart.md) |
| Runtime support matrix | You need to know what is executed today versus schema-only. | [Runtime Support Matrix](runtime_support.md) |
| Planned simple API | You want to preview the upcoming high-level Python facade (`mc.forecast` / `mc.Experiment`). For v0.5.x use `macrocast.run` or Detail Docs. | [Simple Docs (planned)](../simple/index.md) |
| Contract detail | You need exact layer contracts, artifacts, or custom hooks. | [Detail Docs](../detail/index.md) |

## Minimal Runtime Call

```python
from macrocast.core import execute_minimal_forecast

result = execute_minimal_forecast(open("my_layer_recipe.yaml").read())
print(result.sink("l5_evaluation_v1").metrics_table)
```

## YAML Shape

List layers use `fixed_axes` and optional `leaf_config`. Graph layers use `nodes` and `sinks`.

```yaml
1_data:
  fixed_axes: {...}
2_preprocessing:
  fixed_axes: {...}
3_feature_engineering:
  nodes: [...]
  sinks: {...}
4_forecasting_model:
  nodes: [...]
  sinks: {...}
5_evaluation:
  fixed_axes: {...}
6_statistical_tests:
  enabled: true
  sub_layers: {...}
7_interpretation:
  nodes: [...]
  sinks: {...}
8_output:
  fixed_axes: {...}
```

Read [Quickstart](quickstart.md) for a complete runnable recipe.

```{toctree}
:hidden:
:maxdepth: 1

quickstart
first_study
runtime_support
understanding_output
../install
```
