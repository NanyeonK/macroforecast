# Recipe Defaults

Cycle 10 source of truth for user-facing defaults is `macroforecast.defaults`.

Public defaults:

| Constant | Value |
| --- | --- |
| `DEFAULT_RANDOM_SEED` | `42` |
| `DEFAULT_MODEL_FAMILY` | `"ar_p"` |
| `DEFAULT_HORIZONS` | `(1,)` |
| `DEFAULT_FORECAST_STRATEGY` | `"direct"` |
| `DEFAULT_TRAINING_START_RULE` | `"expanding"` |
| `DEFAULT_REFIT_POLICY` | `"every_origin"` |

Docs should not teach `random_seed: 0`, `model_family: "ar"`, or legacy default-profile grammar.

`build_default_recipe_dict()` (Cycle 11b1 fix) emits a recipe using the current top-level layer keys and the constants above.
