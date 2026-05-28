# How_to + Tutorial Import Audit — 2026-05-27

**Scope**: all `from macroforecast` / `import macroforecast` import statements in  
`docs/how_to/` (excluding `replications/`), `docs/tutorial/` (excluding `replications/`),  
and `docs/getting_started.md`.

**Method**: each multi-line import statement was extracted (balanced-parenthesis  
collection) from every `python` code fence and executed via  
`python3 -c "<import>"` from the repo root.  
Execution environment: `python3` in the installed editable package at  
`/home/nanyeon99/project/macroforecast`.

---

**Total imports audited**: 41  
**PASS**: 40  
**FAIL before fix**: 1  
**FAIL after fix**: 0  

---

## Results table

| File | Line | Import / usage | Result |
|------|------|---------------|--------|
| `docs/how_to/add_custom_dataset.md` | 44 | `import macroforecast as mf` | PASS |
| `docs/how_to/add_custom_model.md` | 12 | `import macroforecast as mf` | PASS |
| `docs/how_to/advanced_recipes.md` | 61 | `import macroforecast as mf` | PASS |
| `docs/how_to/advanced_recipes.md` | 131 | `from macroforecast.features.selection import Boruta` | PASS |
| `docs/how_to/bayesian_var_minnesota.md` | 16 | `from macroforecast.models import BVAR, BVARMinnesota` | PASS |
| `docs/how_to/chow_lin_disaggregation.md` | 16 | `from macroforecast.features.transforms import chow_lin_disaggregate` | PASS |
| `docs/how_to/compare_midas_variants.md` | 16 | `from macroforecast.models import (MidasAlmon, MidasBeta, MidasStep, UnrestrictedMidas,)` | PASS |
| `docs/how_to/feature_selection_boruta.md` | 15 | `from macroforecast.features.selection import Boruta` | PASS |
| `docs/how_to/forecast_volatility_realized_garch.md` | 15 | `from macroforecast.models import RealizedGARCH` | PASS |
| `docs/how_to/irf_pesaran_shin_girf.md` | 16 | `from macroforecast.models import VAR` | PASS |
| `docs/how_to/irf_pesaran_shin_girf.md` | 17 | `from macroforecast.interpretation import GIRF` | PASS |
| `docs/how_to/partial_layer_execution.md` | 34 | `from macroforecast.core import (materialize_l1, materialize_l2, materialize_l3_minimal, materialize_l4_minimal, materialize_l5_minimal, execute_l1_l2, execute_minimal_forecast, execute_node,)` | PASS |
| `docs/how_to/partial_layer_execution.md` | 70 | `import macroforecast as mf` | PASS |
| `docs/how_to/partial_layer_execution.md` | 135 | `from macroforecast.core import execute_l1_l2, execute_minimal_forecast` | PASS |
| `docs/how_to/partial_layer_execution.md` | 274 | `from macroforecast.core import materialize_l1, materialize_l2` | PASS |
| `docs/how_to/replicate_a_study.md` | 33 | `import macroforecast as mf` | PASS |
| `docs/how_to/replicate_a_study.md` | 59 | `import macroforecast` | PASS |
| `docs/how_to/reproducibility_policy.md` | 17 | `import macroforecast` | PASS |
| `docs/how_to/simple_api/compare_models.md` | 6 | `import macroforecast as mf` | PASS |
| `docs/how_to/simple_api/fred_sd.md` | 8 | `import macroforecast as mf` | PASS |
| `docs/how_to/simple_index.md` | 8 | `import macroforecast as mf` | PASS |
| `docs/how_to/simple_api/quickstart.md` | 6 | `import macroforecast as mf` | PASS |
| `docs/how_to/simple_api/quickstart.md` | 41 | `from macroforecast.defaults import (..., DEFAULT_FORECAST_POLICY, ...)` | **FAIL → FIXED** |
| `docs/how_to/simple_api/read_results.md` | 6 | `import macroforecast as mf` | PASS |
| `docs/how_to/simple_api/run_experiment.md` | 6 | `import macroforecast as mf` | PASS |
| `docs/how_to/sweep_over_models.md` | 14 | `import macroforecast as mf` | PASS |
| `docs/how_to/troubleshooting.md` | 116 | `import macroforecast as mf` | PASS |
| `docs/how_to/troubleshooting.md` | 186 | `from macroforecast.core import materialize_l1, materialize_l2, materialize_l3_minimal` | PASS |
| `docs/how_to/troubleshooting.md` | 187 | `from macroforecast.core.yaml import parse_recipe_yaml` | PASS |
| `docs/how_to/use_extension_points.md` | 25 | `import macroforecast as mf` | PASS |
| `docs/tutorial/00_install.md` | 41 | `import macroforecast` | PASS |
| `docs/tutorial/01_first_forecast.md` | 13 | `import macroforecast` | PASS |
| `docs/tutorial/01_first_forecast.md` | 54 | `from macroforecast.models import LinearAR` | PASS |
| `docs/tutorial/01_first_forecast.md` | 121 | `import macroforecast.recipes as mf_recipes` | PASS |
| `docs/tutorial/02_full_study.md` | 60 | `from macroforecast.models import LinearAR` | PASS |
| `docs/tutorial/02_full_study.md` | 96 | `from macroforecast.models import PrincipalComponentRegression, FactorAugmentedAR` | PASS |
| `docs/tutorial/02_full_study.md` | 146 | `import macroforecast.recipes as mf_recipes` | PASS |
| `docs/tutorial/03_custom_model.md` | 170 | `import macroforecast.custom as mf_custom` | PASS |
| `docs/tutorial/04_custom_preprocessor.md` | 86 | `import macroforecast as mf` | PASS |
| `docs/tutorial/two_entry_points.md` | 30 | `from macroforecast.models import LinearAR` | PASS |
| `docs/tutorial/two_entry_points.md` | 56 | `import macroforecast.recipes as mf_recipes` | PASS |

---

## Fix applied

**File**: `docs/how_to/simple_api/quickstart.md`, line 45  
**Before**: `DEFAULT_FORECAST_POLICY`  
**After**: `DEFAULT_FORECAST_STRATEGY`  
**Reason**: `DEFAULT_FORECAST_POLICY` was never exported from `macroforecast.defaults`.  
The correct constant name is `DEFAULT_FORECAST_STRATEGY`  
(confirmed via `python3 -c "import macroforecast.defaults as d; print([x for x in dir(d) if 'FORECAST' in x])"` → `['DEFAULT_FORECAST_STRATEGY']`).
