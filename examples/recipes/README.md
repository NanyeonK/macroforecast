# Recipe gallery

이 폴더의 yaml은 그대로 `mf.run('recipe.yaml')` 실행 가능하다.
Partial-layer 예제 (syntax snippets, not end-to-end runnable)는
`docs/recipe-snippets/` 참조.

Run any recipe with:

```python
import macroforecast
result = macroforecast.run("examples/recipes/<name>.yaml", output_directory="out/")
```

## Runnable examples

| File | What it shows |
|------|---------------|
| `l4_minimal_ridge.yaml` | Smallest end-to-end example. Custom panel + ridge. |
| `l4_quantile_regression_forest.yaml` | **v0.3** — Meinshausen QRF with quantile bands. |
| `l4_bagging.yaml` | **v0.3** — bootstrap-aggregated ridge. |
| `l4_ensemble_ridge_xgb_vs_ar1.yaml` | Horse race with explicit benchmark. |
| `l6_standard.yaml` | AR-p benchmark vs. ridge with DM-HLN and MCS blocks. |
| `l6_full_replication.yaml` | Broader L6 statistical-test coverage. |

## In-progress (PR7b scope — not yet runnable)

The following recipes fail `mf.run()` due to stale `fixed_axes` syntax on L3
sections (Pattern 2 from `docs/_audit/recipe-sweep-2026-05-26.md`). They will
be migrated to the nodes/sinks DAG pattern in PR7b.

- `l1_estimated_markov_switching.yaml`
- `l1_minimal.yaml`
- `l1_with_regime.yaml`
- `l2_fred_sd_alignment.yaml`
- `l2_minimal.yaml`
- `goulet_coulombe_2021_replication.yaml` (stale axis name `custom_source_policy`)

## Replication scripts

See ``examples/replication/`` for paper-replication scripts (e.g.
``coulombe_2024_mrf_fred_md.py``).
