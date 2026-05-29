# macroforecast.evaluation

[Back to reference](index.md)

`macroforecast.evaluation` is intentionally minimal in the current clean
package. The legacy recipe evaluation schema was removed from the importable
surface and is preserved only on the `legacy-runtime-reference` branch.

## Current Status

| Item | Value |
| --- | --- |
| Import path | `macroforecast.evaluation` |
| Callable functions | none yet |
| Input contract | to be defined after model/result objects are rebuilt |
| Output contract | to be defined after model/result objects are rebuilt |

## Planned Boundary

Evaluation should later accept direct pandas/result objects, not a YAML runtime
artifact. Expected function families:

- point forecast metrics such as MSE, RMSE, MAE, and R2 OOS
- benchmark-relative metrics
- horizon and target aggregation
- ranking tables
- compact metadata attached to evaluation results
