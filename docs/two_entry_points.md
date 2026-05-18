# Two entry points — recipe DSL vs standalone callables

macroforecast exposes two ways to use its operations:

| | Recipe DSL | Standalone callables |
|---|---|---|
| Interface | YAML file | `mf.functions.<name>(...)` |
| Scope | Full 12-layer DAG | Single operation |
| Bit-exact replay | `mf.replicate(manifest.json)` | Re-run the script |
| Output | Artifact directory + manifest | Python return value |
| Best for | Reproducible papers, sweep studies | Notebooks, scripting, partial pipelines |

---

## Pick your path

### Use the recipe DSL when

- You want a **self-contained, reproducible study** that a colleague can
  replicate from a single file: `mf.replicate("out/manifest.json")` verifies
  every artifact bit-exactly.
- You need **sweep experiments** — trying multiple model families,
  regularisation strengths, or evaluation windows with a single recipe.
- You are using **FRED-MD / FRED-QD / FRED-SD** datasets and want
  the built-in vintage management, tcode transforms, and McCracken-Ng
  group definitions.
- You want the full **L6 test battery** (DM / GW / CW / MCS / SPA / StepM /
  residual battery / density tests) run automatically after evaluation.

```python
import macroforecast as mf

result = mf.run("recipe.yaml", output_directory="out/")
print(result.cells[0].sink_hashes)            # per-cell sink hashes
replication = mf.replicate("out/manifest.json")
assert replication.sink_hashes_match           # bit-exact
```

See [User guide](user_guide.md) and [Recipe API](recipe_api/index.md).

---

### Use standalone callables when

- You already have **your own data and pipeline** and just want one
  operation — fit a ridge, compute a Theil U1, run a DM test.
- You are working in a **Jupyter notebook** or script and want Python
  return values rather than file artifacts.
- You want to **mix-and-match** macroforecast ops with scikit-learn or
  pandas code without writing YAML.

```python
import macroforecast as mf
import numpy as np

# L4: fit ridge
rng = np.random.RandomState(42)
X = rng.randn(100, 5)
y = X @ np.array([1, 2, 3, 4, 5]) + 0.5 * rng.randn(100)

result = mf.functions.ridge_fit(X, y, alpha=1.0)
print(result.coef_)

# L5: Theil U1
u1 = mf.functions.theil_u1(y, result.predict(X))
print(f"Theil U1 = {u1:.4f}")
```

See [Standalone functions](standalone_functions/index.md).

---

## Transition path

If you start with standalone callables and later need reproducibility or
sweeps, you can graduate to the recipe DSL. The operations are the same
underneath — `mf.functions.ridge_fit(X, y, alpha=1.0)` uses the same
adapter as `family: ridge` + `alpha: 1.0` in a recipe.

---

## Decision guide

```text
Need bit-exact replication or sweep study?
    YES  →  Recipe DSL  →  mf.run("recipe.yaml")
    NO
     |
     ↓
Need FRED-MD/QD/SD dataset integration?
    YES  →  Recipe DSL  →  1_data: {fixed_axes: {dataset: fred_md}}
    NO
     |
     ↓
Working in a notebook or scripting context?
    YES  →  Standalone  →  mf.functions.<name>(...)
    NO
     |
     ↓
Need full L6 test battery or L7 interpretation pipeline?
    YES  →  Recipe DSL  →  6_statistical_tests: {enabled: true, ...}
    NO   →  Standalone  →  individual test / importance callables
```

---

## Related

- [Getting started](getting_started.md)
- [User guide](user_guide.md)
- [Standalone functions overview](standalone_functions/index.md)
- [Recipe API](recipe_api/index.md)
