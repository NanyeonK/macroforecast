# `ridge` -- Ridge regression (L2-regularised OLS).

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.ridge_fit`.

## Function signature

```python
mf.functions.ridge_fit(
    vol_model: str enum {"ewma", "garch11"} | None,
    random_state: int | None,
    *,
    alpha: float = 1.0,
    prior: str enum {"none", "random_walk", "shrink_to_target", "fused_difference"} = 'none',
    coefficient_constraint: str enum {"none", "nonneg"} = 'none',
)
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `alpha` | `float` | `1.0` | >=0 | L2 regularisation strength. Larger values shrink coefficients more aggressively toward zero. |
| `prior` | `str enum {"none", "random_walk", "shrink_to_target", "fused_difference"}` | `'none'` | — | Coefficient prior. ``none`` = standard ridge. ``random_walk`` = Goulet Coulombe (2025 IJF) TVP-as-ridge two-step estimator. ``shrink_to_target`` = Albacore_comps Variant A (simplex non-neg + target penalty). ``fused_difference`` = Albacore_ranks Variant B (fused-difference penalty). |
| `coefficient_constraint` | `str enum {"none", "nonneg"}` | `'none'` | — | Sign / cone constraint. ``nonneg`` enforces β >= 0 via augmented NNLS (Assemblage Regression, Coulombe et al. 2024). Ignored when ``prior`` is ``shrink_to_target`` or ``fused_difference`` (those priors handle non-negativity internally). |
| `vol_model` | `str enum {"ewma", "garch11"} | None` | — | only used when prior='random_walk' | Volatility model for step-2 Omega_eps reconstruction in the random-walk estimator. ``ewma`` = RiskMetrics lambda=0.94 (no extra deps). ``garch11`` = GARCH(1,1) via the ``arch`` package; auto-falls back to EWMA if unavailable. |
| `random_state` | `int | None` | — | — | Random seed for stochastic sub-steps (currently unused in the standard ridge path; reserved for future Monte Carlo extensions). |

## Behavior

Closed-form ridge: ``β = (X'X + αI)⁻¹ X'y``. Shrinks coefficients toward zero proportional to the regularisation strength α (``params.alpha``).

Default α = 1.0. The ``cv_path`` search algorithm uses ``RidgeCV`` to pick α from a grid via leave-one-out CV; the ``grid_search`` / ``random_search`` algorithms can sweep over leaf_config.tuning_grid['alpha'].

**When to use**

High-dimensional macro panels with collinear predictors; standard benchmark.

**v0.9 sub-axes** (default values preserve standard ridge):
* ``params.prior`` -- prior on the coefficients. ``none`` (default) keeps standard ridge.
  - ``random_walk`` (operational v0.9.1) -- Goulet Coulombe (2025 IJF) 'Time-Varying Parameters as Ridge Regressions' two-step closed-form estimator with a random-walk kernel on coefficient deviations. Yields per-time β path via the cumulative-sum reparametrisation β_k = C_RW · θ_k. Helper ``_TwoStageRandomWalkRidge``.
  - ``shrink_to_target`` (operational v0.9.1) -- Maximally Forward-Looking Core Inflation Albacore_comps Variant A (Goulet Coulombe / Klieber / Barrette / Goebel 2024). ``arg min ‖y − Xw‖² + α‖w − w_target‖²`` s.t. ``w ≥ 0``, ``w'1 = 1``. Solved via scipy SLSQP. Limit cases: α=0 → unconstrained / NNLS; α→∞ → returns w_target. Helper ``_ShrinkToTargetRidge``. Sub-axis params: ``prior_target`` (default uniform 1/K), ``prior_simplex`` (default True).
  - ``fused_difference`` (operational v0.9.1) -- Maximally FL Albacore_ranks Variant B. ``arg min ‖y − Xw‖² + α‖Dw‖²`` s.t. ``w ≥ 0``, ``mean(y) = mean(Xw)``, where D is the first-difference operator. Pairs with the L3 ``asymmetric_trim`` op (B-6 v0.8.9) for rank-space transformation. Limit cases: α=0 → standard OLS / NNLS; α→∞ → uniform weights (level set by mean equality). Helper ``_FusedDifferenceRidge``. Sub-axis params: ``prior_diff_order`` (default 1), ``prior_mean_equality`` (default True).
* ``params.coefficient_constraint`` -- sign / cone constraints. ``none`` (default) is unconstrained; ``nonneg`` (operational v0.8.9) implements the assemblage non-negative ridge.
* ``params.vol_model`` (random_walk only) -- volatility model for the step-2 Ω_ε reconstruction. ``ewma`` (default; RiskMetrics λ=0.94; no extra deps) or ``garch11`` (requires ``arch>=5.0``; auto-falls-back to EWMA when missing).

## In recipe context

Set ``params.family = "ridge"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  family: ridge
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Hoerl & Kennard (1970) 'Ridge regression: biased estimation for nonorthogonal problems', Technometrics 12(1).
* Goulet Coulombe (2025) 'Time-Varying Parameters as Ridge Regressions', International Journal of Forecasting 41:982-1002. doi:10.1016/j.ijforecast.2024.08.006.
* Goulet Coulombe / Klieber / Barrette / Goebel (2024) 'Maximally Forward-Looking Core Inflation' -- Albacore_comps (shrink_to_target Variant A) and Albacore_ranks (fused_difference Variant B).

## Related ops

See also: `lasso`, `elastic_net`, `lasso_path` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
