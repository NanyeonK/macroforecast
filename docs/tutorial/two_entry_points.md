# Two entry points — standalone models vs recipe pipeline

This page compares the two entry points at a conceptual level. For a hands-on
narrative, start with {doc}`01_first_forecast`.

macroforecast exposes two ways to run forecasting models:

| | Standalone models | Recipe pipeline |
|---|---|---|
| Interface | `from macroforecast.models import X` | YAML file |
| Scope | Single model, direct Python call | Full L0–L8 pipeline |
| Bit-exact replay | Re-run the script | `mf.recipes.run(manifest.json)` |
| Output | Python return value | Artifact directory + manifest |
| Best for | Notebooks, scripting, quick comparisons | Reproducible papers, sweep studies |

---

## Pick your path

### Use standalone models when

- You are working in a **Jupyter notebook or one-off script** and want Python
  return values rather than file artifacts.
- You already have **your own data and pipeline** and want to fit a model or
  compute a metric without writing YAML.
- You want to **mix and match** macroforecast model classes with scikit-learn or
  pandas code directly.

```python
from macroforecast.models import LinearAR
import pandas as pd

model = LinearAR(p=2)
model.fit(X_train, y_train)   # no YAML, no output directory
preds = model.predict(X_test)
print(preds)
```

See {doc}`01_first_forecast` for a complete worked example.

---

### Use the recipe pipeline when

- You want a **self-contained, reproducible study** that a colleague can
  replicate from a single file.
- You need **sweep experiments**, trying multiple model families,
  regularisation strengths, or evaluation windows with a single recipe.
- You are using **FRED-MD / FRED-QD / FRED-SD** datasets and want
  the built-in vintage management, tcode transforms, and McCracken-Ng
  group definitions.
- You want the full **L6 test battery** (DM / GW / CW / MCS / SPA / StepM /
  residual battery / density tests) run automatically after evaluation.

```python
import macroforecast.recipes as mf_recipes

result = mf_recipes.run("recipe.yaml", output_directory="out/")
# mf_recipes.run is also callable as mf.run -- both are valid aliases
print(result.cells[0].sink_hashes)
replication = mf_recipes.replicate("out/manifest.json")
assert replication.sink_hashes_match   # bit-exact
```

---

## Transition path

If you start with standalone models and later need reproducibility or
systematic sweeps, graduate to the recipe pipeline. The model families are
the same underneath. `LinearAR(p=2)` in standalone mode uses the same
implementation as `model: ar_p` in a recipe. Adding a YAML recipe around
your study gains provenance, sweep expansion, and the full L6 test battery
without changing the underlying models.

---

## Decision guide

```text
Working in a notebook or one-off script?
    YES  ->  Standalone  ->  from macroforecast.models import X
    NO
     |
     v
Need bit-exact replication across runs?
    YES  ->  Recipe pipeline  ->  mf.recipes.run("recipe.yaml")
    NO
     |
     v
Need FRED-MD/QD/SD dataset integration?
    YES  ->  Recipe pipeline  ->  data: {fixed_axes: {dataset: fred_md}}
    NO
     |
     v
Need full test battery (DM / CW / MCS)?
    YES  ->  Recipe pipeline  ->  6_statistical_tests: {enabled: true, ...}
    NO   ->  Standalone  ->  individual test callables in mf.functions
```
