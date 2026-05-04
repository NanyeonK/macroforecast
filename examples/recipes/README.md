# Recipe gallery

50+ reference recipes for macrocast. Run any with:

```python
import macrocast
result = macrocast.run("examples/recipes/<name>.yaml", output_directory="out/")
```

## Highlights

| File | What it shows |
|------|---------------|
| `l4_minimal_ridge.yaml` | Smallest end-to-end example. Custom panel + ridge. |
| `l4_random_forest.yaml` | RandomForest with seed propagation. |
| `l4_quantile_regression_forest.yaml` | **v0.3** — Meinshausen QRF with quantile bands. |
| `l4_bagging.yaml` | **v0.3** — bootstrap-aggregated ridge. |
| `l4_ensemble_ridge_xgb_vs_ar1.yaml` | Horse race with explicit benchmark. |
| `l4_mrf_placeholder.yaml` | Coulombe (2024) GTVP MRF demo. |
| `l4_regime_separate_fit.yaml` | Per-regime model fitting. |
| `l3_cascade_pca_on_marx.yaml` | L3 DAG: MARX → PCA → ... |
| `l1_with_regime.yaml` | L1.G external NBER regime activation. |
| `horse-race-model.yaml` | Multi-cell sweep across families. |

## Layer-specific minimal / full examples

For each layer ``l{N}`` we ship ``l{N}_minimal.yaml`` (smallest config that
parses + runs) and ``l{N}_full.yaml`` (every axis populated).

## Replication scripts

See ``examples/replication/`` for paper-replication scripts (e.g.
``coulombe_2024_mrf_fred_md.py``).
