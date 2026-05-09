# Paper-replication recipes (v0.9 Phase 2)

Each YAML in this directory replicates one of the 16 macro-forecasting
papers tracked by the v0.9 paper-coverage pass. Recipes are written as
**recipe patterns** over the existing atomic primitive vocabulary, not
as paper-named monolithic family options. This makes the algorithmic
decomposition visible at the recipe level and means new papers reuse
existing primitives rather than inflating the registry.

## Status legend

* **operational** -- runs end-to-end on the current `main`. Smoke-tested
  in `tests/test_examples_smoke.py` (curated subset).
* **pre-promotion** -- recipe is canonical; one or more atomic primitives
  it depends on are still `future`-status. Validator hard-rejects the
  recipe at run-time until the primitive lands. Paper, decomposition,
  and missing primitives are noted in the recipe's top comment.

## Index

| # | Paper | Recipe | Status | Decomposition |
|---|---|---|---|---|
| 1 | Scaled PCA (Huang/Jiang/Li/Tong/Zhou 2022) | `scaled_pca.yaml` | TBD | `scaled_pca` op (already operational) |
| 2 | The Macroeconomy as a Random Forest (Coulombe 2024) | `macroeconomic_random_forest.yaml` | TBD | `macroeconomic_random_forest` family |
| 3 | To Bag is to Prune -- PRF baseline (Coulombe 2024) | `perfectly_random_forest.yaml` | **operational** | `extra_trees(max_features=1)` |
| 3b | To Bag is to Prune -- Booging (Coulombe 2024) | `booging.yaml` | pre-promotion | `bagging(strategy=sequential_residual)` (future) |
| 3c | To Bag is to Prune -- MARSquake (Coulombe 2024) | `marsquake.yaml` | pre-promotion | `bagging(base_family=mars)` (mars future) |
| 4 | Adaptive Moving Average (Coulombe & Klieber 2025) | `adaptive_ma_rf.yaml` | pre-promotion | `adaptive_ma_rf` L3 op (future) |
| 5 | TVP as Ridge / 2SRR (Coulombe 2025) | `two_step_ridge.yaml` | pre-promotion | chained `ridge` + `ridge(prior=random_walk)` (future) |
| 6 | Hemisphere Neural Networks (Coulombe et al. 2025 JAE) | `hemisphere_nn.yaml` | pre-promotion | `mlp(architecture=hemisphere, loss=volatility_emphasis)` (future) |
| 7 | OLS as Attention (Coulombe 2026) | `ols_attention_demo.yaml` | TBD | conceptual paper -- demo with `ols` + `transformer` |
| 8 | Anatomy of OOS Forecasting (Borup et al. 2022) | `anatomy_oos.yaml` | pre-promotion | `oshapley_vi` + `pbsv` L7 ops (future, anatomy package) |
| 9 | Dual Interpretation of ML Forecasts (Coulombe et al. 2024) | `dual_interpretation.yaml` | pre-promotion | `dual_decomposition` L7 op (future); op output carries inline HHI / short / turnover / leverage |
| 10 | Maximally Forward-Looking Core Inflation (Coulombe et al. 2024) | `maximally_forward_looking.yaml` | pre-promotion | `asymmetric_trim` L2 + `ridge(coefficient_constraint=nonneg)` (future) |
| 11 | Sparse Macro Factors (Zhou) | `sparse_macro_factors.yaml` | TBD | `sparse_pca` + `var` |
| 12 | Macroeconomic Data Transformations Matter (Coulombe 2021) | `macroeconomic_data_transformations.yaml` | TBD | `ma_increasing_order` (MARX) + `pca` (rotation) + `cumulative_average` (path-avg) |
| 13 | How is ML Useful for Macro Forecasting (Coulombe et al. 2022 JAE) | `ml_useful_macro.yaml` | TBD | sweep machinery over `family` × `regularization` × `cv` × `loss` |
| 14 | Slow-Growing Trees (Coulombe 2024) | `slow_growing_trees.yaml` | pre-promotion | `decision_tree(split_shrinkage=η)` (future) |
| 15 | Deprecated — no paper anchor; helper cut 2026-05-08 | — | — | — |
| 16 | Arctic Amplification VAR / VARCTIC (Coulombe & Goebel 2021) | `arctic_var.yaml` | TBD | `var` + `historical_decomposition` + `orthogonalised_irf` + `fevd` |

## Calling them programmatically

The Python helper module `macroforecast.recipes.paper_methods` exposes
each replication as a function that returns a recipe dict ready for
`macroforecast.run`:

```python
import macroforecast as mf
from macroforecast.recipes.paper_methods import perfectly_random_forest

result = mf.run(perfectly_random_forest(target="y", horizon=1))
print(result.cells[0].runtime_result.artifacts["l4_forecasts_v1"].forecasts)
```

Helper docstrings mirror the YAML's top comment so the algorithmic
decomposition is visible from either entry point.
