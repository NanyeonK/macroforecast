# Terminology

Date: 2026-04-22

This project uses `target` as the canonical package term for the variable being
forecast.

## Target Versus Y

Use `target` in public package language:

- recipe keys and manifest fields;
- registry axis names and descriptions;
- docs and examples;
- artifact column descriptions;
- extension protocol prose.

Use `y` only when it is clearly local mathematical notation or a short internal
array variable next to `X`, such as `model.fit(X, y)` style estimator code.
When docs discuss papers that write `Y_t` or `y_{t+h}`, introduce it as paper
notation and then return to `target`.

| Context | Preferred term |
|---|---|
| Public API | `target`, `targets`, `target_transformer` |
| Registry | `horizon_target_construction`, `target_transform`, `target_missing_policy` |
| Manifest/docs | target series, target scale, target level |
| Local numerical code | `X_train`, `y_train` is acceptable |
| Paper formulas | `Y_t` or `y_{t+h}` only after explicitly marking it as notation |

## Current Compatibility Names

Some older artifact columns and values still contain `y`:

- `y_true`, `y_pred`, `y_true_level`, `y_pred_level`, `y_pred_model_scale`;
- legacy `horizon_target_construction=future_target_level_t_plus_h`.

Keep these accepted for backward compatibility. New docs and generated recipes
should prefer canonical target-language values such as
`future_target_level_t_plus_h`. Future artifact migrations should add target
aliases before removing any `y_*` columns.
