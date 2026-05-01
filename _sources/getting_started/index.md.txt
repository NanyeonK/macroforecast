# 0. Getting Started

Install the package, then choose the execution surface that matches your recipe.

## Install

For active development, install from the repository:

```bash
git clone https://github.com/NanyeonK/macrocast.git
cd macrocast
pip install -e .
```

Verify the import:

```python
import macrocast
print("macrocast imported")
```

For optional model and interpretation backends, see [Installation](../install.md).

## Execution Paths

| Path | Use when | Start with |
|---|---|---|
| Core layer-contract runtime | You want the current L1-L8 artifact path, diagnostics, L6/L7 lightweight artifacts, and L8 output directory. | [Quickstart](quickstart.md) |
| Runtime support matrix | You need to know what is actually executed today versus schema-only. | [Runtime Support Matrix](runtime_support.md) |
| Simple code | You want the older high-level Python facade. | [Simple Docs](../simple/index.md) |
| Detail docs | You need exact layer contracts, artifact contracts, or custom hooks. | [Detail Docs](../detail/index.md) |

## Core Runtime In One Screen

```python
from macrocast.core import execute_minimal_forecast

result = execute_minimal_forecast(open("my_layer_recipe.yaml").read())
print(result.sink("l5_evaluation_v1").metrics_table)
```

Layer-contract recipes use top-level blocks such as:

```yaml
1_data:
  fixed_axes: {...}
2_preprocessing:
  fixed_axes: {...}
3_feature_engineering:
  nodes: [...]
4_forecasting_model:
  nodes: [...]
5_evaluation:
  fixed_axes: {...}
8_output:
  fixed_axes: {...}
```

Read [Quickstart](quickstart.md) for a complete runnable recipe.

## Legacy Paths

The repository still contains legacy compiler and simple-facade docs. Those paths remain useful for existing recipes, but they do not describe the full L0-L8 layer-contract runtime. When in doubt, check [Runtime Support Matrix](runtime_support.md).

```{toctree}
:hidden:
:maxdepth: 1

quickstart
first_study
runtime_support
understanding_output
stages_reference
../install
```
