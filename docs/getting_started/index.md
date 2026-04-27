# 0. Getting Started

This page is intentionally small. Install the package, then choose one of three usage paths.

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

## Three usage paths

| Path | Use when | Start with |
|---|---|---|
| Navigator path | You want to see valid choices, disabled branches, and YAML before running anything. | [Open Navigator App](../navigator_app/index.html) |
| Simple code | You want a quick forecast, model comparison, or small sweep from Python. | [Simple Docs](../simple/index.md) |
| Detail code / YAML | You need exact layer control, custom methods, replication, or auditable contracts. | [Detail Docs](../detail/index.md) |

## Path 1: Navigator to YAML to CLI

1. Open the [Navigator App](../navigator_app/index.html).
2. Choose each layer in order.
3. Inspect disabled branches and compatibility messages.
4. Export the YAML recipe.
5. Run the recipe:

```bash
macrocast-navigate resolve recipe.yaml
macrocast-navigate run recipe.yaml --output-root results/my-run
```

## Path 2: Simple Python code

```python
import macrocast as mc

result = mc.forecast(
    dataset="fred_md",
    target="INDPRO",
    start="1980-01",
    end="2019-12",
    horizons=[1, 3, 6],
)
```

For comparisons:

```python
exp = mc.Experiment(
    dataset="fred_md",
    target="INDPRO",
    start="1980-01",
    end="2019-12",
    horizons=[1, 3, 6],
)

result = exp.compare_models(["ar", "ridge", "lasso"]).run()
```

## Path 3: Detail code / YAML

Use the full layer grammar when you need exact decisions:

```yaml
path:
  0_meta:
    fixed_axes:
      research_design: single_path_benchmark
  1_data_task:
    fixed_axes:
      dataset: fred_md
      target_structure: single_target_point_forecast
  3_training:
    fixed_axes:
      model_family: ridge
```

The full contract is documented layer by layer in [Detail Docs](../detail/index.md).

```{toctree}
:hidden:
:maxdepth: 1

quickstart
first_study
understanding_output
stages_reference
../install
```
