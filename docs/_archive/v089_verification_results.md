# v0.8.9 verification results

Honesty-pass audit results. Per-item entries appended in execution
order. Plan: ``v089_verification_plan.md``.

## V1.1 ridge.coefficient_constraint=nonneg (B-1)

**Status**: ✅ Pass

**Evidence**:

* Augmented-system equivalence proven algebraically:
  ``‖[y;0] − [X; √α·I]β‖² = ‖y − Xβ‖² + α‖β‖²``. So NNLS on the
  augmented system minimises the ridge objective under β ≥ 0.
* Direct ``scipy.optimize.nnls`` on the augmented system returns
  identical coefficients to ``_NonNegRidge`` (max abs diff = 0.0).
* On non-binding-constraint synthetic data, recovered coefficients
  match ``sklearn.linear_model.Ridge`` to 0.01 (only difference is
  intercept handling: ``_NonNegRidge`` centres y; ``Ridge`` includes
  intercept in the L2 penalty by default).
* On binding constraint (one negative truth), recovered β has the
  negative coefficient clipped to 0 and the others positive.
* Edge cases: empty input returns zero coefficients without crash;
  NaN in X handled via ``fillna(0)``.

**Discrepancies**: none.

**Action**: none.


## V1.2 dual_decomposition (linear) (B-3)

**Status**: ✅ Pass

**Evidence**:

* Representer identity ``ŷ = W @ y`` verified to machine precision
  (~1e-15 max abs diff) on a 50×3 synthetic ridge problem with
  ``fit_intercept=False``.
* Closed-form weight matrix ``W = X(X'X + αI)⁻¹X'`` matches the
  package's computation exactly (max abs diff = 0 on the synthetic
  problem; pinv-based path).
* Frame layout matches the L7 importance contract: index = training
  row labels, columns = ``[mean_weight, abs_mean_weight, max_abs_weight]``,
  with full ``(n_test × n_train)`` weight matrix attached as
  ``attrs['dual_weights']``.
* Portfolio metrics inline at ``attrs['portfolio_metrics']``: columns
  ``[hhi, short, turnover, leverage]``; non-negativity invariants
  hold; ``turnover[0] = 0`` by construction (no previous row).
* Non-linear families (RF) correctly raise ``NotImplementedError``
  pointing at v0.9.x Phase 2.
* Empty input handled without crash.

**Discrepancies**: 

* The implementation defaults to *in-sample* dual weights
  (X_test = X_train). For walk-forward use cases users must call
  with the test subset explicitly. This is documented in the helper
  docstring; not a runtime correctness issue.
* Linear-only support is intentional (paper applies dual decomposition
  to non-linear models too). Documented in CHANGELOG; non-linear
  extension scheduled for v0.9.x Phase 2.

**Action**: none for v0.8.9 (linear scope is honest).


## V1.3 bagging.strategy=block (B-4)

**Status**: ✅ Pass

**Evidence**:

* Block invariant (consecutive integers mod n) verified across 20
  trials × 6 blocks = 120 blocks; all blocks consecutive.
* ``block_length`` parameter respected for L ∈ {3, 7, 12}: each block
  has length L and consecutive structure.
* Uniform coverage verified over 5000 bootstrap draws on n=100:
  mean count per index = 2008, std = 32 (1.6%); each index sampled
  approximately uniformly.
* Variance reduction on AR(1) ``φ=0.7`` data: block and standard
  bagging both produce comparable per-bag variance; both run to
  completion on the wrapped ridge base learner.
* Wrapper-level ``_draw_indices`` for unknown ``strategy`` falls
  back to i.i.d. bootstrap; the explicit future-gate for
  ``sequential_residual`` is in the dispatch layer (already covered
  by ``test_v09_sub_axis_future_gates_raise_at_runtime``).

**Discrepancies**: Implementation is the *circular* moving-block
bootstrap (block wraps at n via modulo). The classical Künsch
non-circular variant samples block starts only from
``{0, ..., n - block_length}``. The circular variant is a
standard accepted variant; behaviour difference is minor at the
sample boundaries. Documented for clarity but not a correctness
issue.

**Action**: add a one-line note to the OPTION_DOCS prose for
``bagging.strategy`` clarifying "circular variant". ✅ done.


## V1.4 blocked_oob_reality_check (B-5)

**Status**: ✅ Pass

**Evidence**:

* Type-I rate against equal-skill benchmark (n_sims=200, n=80,
  α=0.05, n_bootstraps=300): empirical rejection rate = 0.060,
  within 2× nominal α (the 2× tolerance reflects finite-sample
  bootstrap noise; 0.060 vs 0.050 is one bootstrap-standard-error
  away, which is acceptable).
* Power against clear-winner candidate (cand has 40% of benchmark's
  loss): power = 1.0 over 50 simulations.
* Edge cases: n < 2 returns empty DataFrame without crash; missing
  benchmark name raises ValueError with informative message.
* Single-side test on noisy mid-skill data runs to completion with
  reasonable p-value (mid-tail).

**Discrepancies**: 

* The recentring in the implementation uses
  ``recentred = boot_means - d_bar`` then
  ``p = mean(recentred >= d_bar)``. This is the *one-sided
  bootstrap p-value with mean-recentring* variant, which is
  equivalent (after algebraic simplification) to
  ``p = mean(boot_means >= 2 * d_bar)`` — a slightly less standard
  formulation than White's original. The empirical type-I and
  power results confirm the test is correctly sized in practice.
* For a fully canonical White (2000) reality check, recentring uses
  ``(d_bar - max(0, d_bar))`` to make the null distribution match
  the least-favourable configuration. Our implementation is the
  simpler "centre on observed mean" variant; for the multi-model
  case (which we do not yet support) the two diverge. Single-model
  (one candidate vs benchmark) case is correct as implemented.

**Action**: when v0.9.x adds multi-candidate joint-test support,
re-verify with the canonical White least-favourable recentring.


## V1.5 asymmetric_trim (B-6)

**Status**: ✅ Pass

**Evidence**:

* Per-row sort identity: 50 rows × 7 cols all match ``np.sort(arr,
  axis=1)`` exactly.
* Idempotent on already-sorted input.
* Per-row monotone: ``rank_1 <= rank_2 <= ... <= rank_K`` for all rows.
* Centred 3-month MA verified by hand-computed expected values on a
  small example: actual matches expected exactly.
* Edge cases: empty input returns empty DataFrame; K=1 returns
  identity (single-column input is already its own sort).

**Discrepancies**: none. Trivially correct (np.sort + rolling.mean
with min_periods=1).

**Action**: none.


## V2.5 sparse_pca runtime vs Zhou paper

**Status**: ⚠️ **Partial** — `sparse_pca` op is operational and uses
sklearn's standard ``SparsePCA`` (Zou-Hastie-Tibshirani 2006 via
Mairal et al. 2009 dictionary learning), but **Zhou's "Sparse
Macro-Finance Factors" paper uses a different sparse PCA variant**:
Sparse Component Analysis (SCA) per Chen and Rohe (2023).

**Evidence (from review at ``second_brain/wiki/papers/reviews/guofu_zhou_sparse_macro_factors-477d3fda.md``)**:

* Zhou paper objective: ``min ‖X − ZDΘ'‖_F subject to Z ∈ S(T,J),
  Θ ∈ S(M,J), ‖Θ‖₁ ≤ ζ`` (Chen-Rohe 2023 SCA).
* sklearn ``SparsePCA``: L1-penalised regression formulation
  (Zou-Hastie-Tibshirani 2006); does NOT match the Chen-Rohe SCA
  objective.
* The two variants produce **different sparse loadings** — they
  are not equivalent reformulations.
* Additionally, Zhou uses Supervised PCA (Giglio/Xiu/Zhang 2025) for
  risk premium estimation; this is also not in the package.

**Discrepancies**:

1. ``sparse_pca`` op accurate as a *generic* sparse PCA primitive
   (Zou-Hastie-Tibshirani 2006) but does NOT replicate Zhou's
   specific construction (Chen-Rohe 2023 SCA).
2. The 614-test-asset risk premium estimation in Zhou requires
   Supervised PCA (Giglio et al. 2025); this is a separate primitive
   not in the package.

**Action**:

* CHANGELOG entry under v0.8.9 sparse_macro_factors helper: clarify
  that the recipe uses the sklearn (Zou-Hastie-Tibshirani) variant,
  not the Chen-Rohe SCA used in Zhou's paper.
* Add to v0.9.x roadmap (``v09_paper_coverage_plan.md``):
  - new L3 op ``sparse_pca_chen_rohe`` for SCA (Chen-Rohe 2023)
  - new L3 op ``supervised_pca`` for Giglio-Xiu-Zhang 2025 SPCA
* Update ``sparse_macro_factors`` paper helper docstring to note
  it produces a related-but-not-identical decomposition. ✅ done.


## V2.2 macroeconomic_random_forest runtime vs Coulombe (2024)

**Status**: ❌ **FAIL → fixed in v0.8.9 honesty pass via re-anchor to mrf-web**

**Pre-fix discrepancy**:

The previous in-house ``_MRFWrapper`` implemented **only one of three**
paper-defining pieces of Goulet Coulombe (2024) "The Macroeconomy as a
Random Forest" (arXiv:2006.12724):

| Paper piece | Pre-fix ``_MRFWrapper`` |
|---|---|
| (a) per-leaf local linear regression of ``y`` on the state vector | ✅ present (sklearn ``RandomForestRegressor`` augmented with a normalised time-trend column, then per-(tree, leaf) ``LinearRegression``) |
| (b) random walk kernel / Olympic podium for time-varying β smoothing | ❌ missing (no time-variation regularisation; only data-driven splits on the time column) |
| (c) Block Bayesian Bootstrap (Taddy 2015 ext.) for forecast ensembles + intervals | ❌ missing |

The "GTVP" claim in the docstring was overclaimed — the wrapper was a
random forest with leaf-local linear regressions, not the paper's
random-walk-regularised time-varying parameter forest.

This was operational-but-overclaimed since v0.2 #187. The v0.8.9
verification audit detected it on the V2.2 line of
``v089_verification_plan.md``.

**Fix landed in v0.8.9 (re-anchor to mrf-web)**:

A new helper, ``_MRFExternalWrapper``, replaces the in-house
``_MRFWrapper``. It delegates the full algorithm to Ryan Lucas's
reference implementation, **vendored under
``macroforecast/_vendor/macro_random_forest/``** with four surgical
numpy 2.x / pandas 2.x compatibility patches (full delta:
``_vendor/macro_random_forest/PATCHES.md``). **No algorithmic
changes**. Upstream URL:
<https://github.com/RyanLucas3/MacroRandomForest>; original MIT licence
preserved alongside the vendored source. See ``THIRD_PARTY_NOTICES.md``
at the repository root for the attribution table.

Wrapper contract:

* ``fit(X, y)`` caches the training panel.
* ``predict(X_new)`` builds the ``[train ; test]`` data frame mrf-web
  expects, sets ``oos_pos`` to the test row positions, runs
  ``MacroRandomForest._ensemble_loop()`` with the full paper procedure,
  caches ``output['betas']`` (T × K+1 GTVP series) and
  ``output['pred_ensemble']`` (B × n_oos bootstrap forecasts), and
  returns the ``output['pred']`` slice.
* L7 ``mrf_gtvp`` now reads ``_cached_betas`` directly (column 0 =
  intercept, columns 1..K = per-time-step coefficients on the state
  vector). ``np.nanmean`` for importance because OOS rows are
  intentionally NaN under mrf-web's in-sample-only Bayesian bootstrap.

No new install extra: the family works out of the box once
``pip install macroforecast`` is done.

**Verification (this V2.2 entry)**:

* End-to-end smoke test on synthetic AR(1) panel (n=80, 3 features,
  B=10, train=60 / test=20): wrapper returns finite forecasts; betas
  cache populated at shape (80, 4); pred_ensemble at shape (10, 20).
* Known-answer test
  (``test_v22_mrf_external_wrapper_matches_vendored_reference``):
  ``_MRFExternalWrapper`` produces forecast values bit-identical to a
  direct ``MacroRandomForest._ensemble_loop()`` call with matching seed
  and parameters (rtol 1e-10).
* L7 mrf_gtvp regression
  (``test_mrf_gtvp_produces_per_row_coefficient_path``): coefficient
  path length equals (n_train + n_test); per-feature importance ≥ 0.
* Determinism + parameter sensitivity: matching ``random_state``
  reproduces forecasts to machine precision; distinct seeds produce
  distinct forecasts. (``test_mrf_seed_makes_predictions_deterministic``,
  ``test_mrf_distinct_seeds_produce_distinct_forecasts``.)

**Discrepancies (post-fix)**:

* Sklearn-style ``fit``/``predict`` separation does not match mrf-web's
  fit-and-forecast pipeline. Each ``predict`` call therefore re-runs
  the full ``_ensemble_loop``. Walk-forward studies pay this cost N
  times (one per origin). Acceptable for v0.8.9; future optimisation
  (collapse fit + multi-origin predict into one ``_ensemble_loop``)
  tracked for v0.9.x.
* Old hyperparameters ``n_estimators`` / ``max_depth`` no longer drive
  forest size: ``n_estimators`` becomes an alias for mrf-web's ``B``
  (bootstrap iterations); ``max_depth`` is silently ignored because
  the paper algorithm uses RW + ridge regularisation, not tree depth.

**Action**: ✅ runtime re-anchor landed. ✅ vendored under
``macroforecast/_vendor/macro_random_forest/`` with LICENSE and
PATCHES.md preserved. ✅ ``THIRD_PARTY_NOTICES.md`` at repo root.
✅ known-answer + L7 + determinism tests added. ✅ OPTION_DOCS entry
updated with the citation requirement (Goulet Coulombe 2024 +
Lucas 2022). ✅ CHANGELOG entry under v0.8.9 honesty-pass section.
✅ encyclopedia regen.


## V2.3 var + IRF + FEVD + historical_decomposition vs Coulombe & Göbel (2021)

**Status**: ⚠️ 2/4 ✅ Pass + 2/4 fixed in v0.8.9 honesty pass

Paper: Goulet Coulombe, P. and Göbel, M. (2021) "Arctic Amplification of
Anthropogenic Forcing: A Vector Autoregressive Analysis", *Journal of
Climate* 34, doi:10.1175/JCLI-D-20-0324.1. VARCTIC method = Bayesian
VAR with Minnesota prior, **Cholesky** identification (CO₂ ordered
first → atmospheric → temperature → ice), lag length **manually
specified** (P=12 for VARCTIC 8, P=3 for VARCTIC 18).

### V2.3 (a) — `var` lag selection

**Status**: ✅ Pass.

Runtime ``_VARWrapper.fit`` (``runtime.py``) accepts ``params['n_lag']``
as a user-controlled lag length and calls
``statsmodels.tsa.api.VAR(data).fit(p)`` with that value. **No
automatic selection** (no AIC/BIC/HQIC), which **matches** the paper's
explicit P=12 / P=3 choice -- the authors set lag length by domain
knowledge, not a formal selection criterion. A future ``lag_selection``
sub-axis (information-criterion search) is tracked as a v0.9.x quality-
of-life addition; not blocking for v0.8.9.

### V2.3 (b) — IRF identification: `generalized_irf` was misnamed

**Status**: ❌ FAIL → fixed in v0.8.9 honesty pass via op rename.

**Pre-fix discrepancy**:

The op ``generalized_irf`` (operational since v0.2 #189) was named after
Pesaran-Shin (1998) generalized impulse-response analysis, which is
**order-invariant**: each shock is constructed as the multivariate-
normal projection of all residuals onto the j-th canonical direction,
yielding an IRF table that does not depend on variable ordering.

The runtime, however, returned **Cholesky orthogonalised IRFs**
(``statsmodels.tsa.vector_ar.irf.IRAnalysis.orth_irfs``) -- order-
**dependent** by construction. The two methods are distinct algorithms;
calling Cholesky output "generalized" is a name misuse that misleads
users picking ops by paper reference.

**Fix landed in v0.8.9**:

* New op ``orthogonalised_irf`` registered in ``DEFAULT_FIGURE_MAPPING``
  and OPTION_DOCS (``Sims 1980`` reference). Routes to the same
  ``statsmodels orth_irfs`` runtime path as before -- numerical output
  identical, only the op label changed.
* ``generalized_irf`` removed from ``DEFAULT_FIGURE_MAPPING`` and added
  to ``FUTURE_OPS`` with a docstring pointing to the Pesaran-Shin
  paper. The L7 runtime dispatcher raises ``NotImplementedError`` with
  a citation-aware message redirecting users to ``orthogonalised_irf``.
* Validator (``layers/l7.py``) accepts both ``orthogonalised_irf`` and
  ``generalized_irf`` as VAR-only ops; the future-gate fires at
  execute time.
* Encyclopedia regenerated; existing replication recipes (``arctic_var.yaml``,
  ``recipes/paper_methods.varctic_arctic_amplification`` docstring) updated
  to use ``orthogonalised_irf``.

**Verification (this V2.3 (b) entry)**:

* Schema regression
  (``test_v23_generalized_irf_is_future_gated``): ``orthogonalised_irf``
  in ``OPERATIONAL_OPS``; ``generalized_irf`` in ``FUTURE_OPS`` and
  raises ``NotImplementedError`` with message containing the
  ``orthogonalised_irf`` redirect.
* Cholesky-factor invariant
  (``test_v23_orthogonalised_irf_returns_cholesky_response``): the
  horizon-0 orthogonalised IRF matrix equals
  ``np.linalg.cholesky(Σᵤ)`` to ``atol=1e-10``, confirming the runtime
  delegates to statsmodels' standard Cholesky decomposition.

**Action**: ✅ rename + future-gate landed. ✅ encyclopedia regen.
✅ tests pinning the rename. CHANGELOG entry under v0.8.9 honesty-pass
section.

### V2.3 (c) — FEVD normalization

**Status**: ✅ Pass.

Runtime ``_var_impulse_frame`` (op_name="fevd") delegates to
``statsmodels.tsa.vector_ar.fevd.FEVD.decomp``, which normalises the
forecast-error variance contribution of each (Cholesky-identified)
structural shock to sum to 1 across shocks at each horizon. This is
the standard FEVD construction and matches the paper's setup
(VARCTIC uses Cholesky identification → Cholesky FEVD).

The op output averages the per-horizon shares across the requested
horizons (``decomp[:, target_index, :].mean(axis=0)``), which the
docstring in ``_var_impulse_frame`` makes explicit. The paper's
figures present per-horizon shares; users wanting per-horizon detail
can call the op with ``n_periods=1`` and walk the loop, but the
default returns a horizon-averaged summary appropriate for L7
importance ranking.

### V2.3 (d) — `historical_decomposition` was a proxy

**Status**: ❌ FAIL → fixed in v0.8.9 honesty pass with a real
Burbidge-Harrison HD implementation.

**Pre-fix discrepancy**:

The op claimed to be a Burbidge-Harrison (1985) historical
decomposition, but the runtime computed:

```python
response = (np.abs(orth_irfs[:, target, :]).sum(axis=0)) * std(resid[:, j])
```

This is a per-shock importance score that combines the L1 mass of the
IRF (a *time-invariant* quantity) with the residual standard deviation
(a *cross-sectional* statistic). It has **no phase alignment** with the
realised target path -- the same number falls out regardless of when
the shocks actually occurred. The Burbidge-Harrison procedure expresses
the *path* of each variable as the sum of contributions from each
orthogonalised shock at each historical date; our proxy collapses time
out and replaces it with an unrelated scalar.

**Fix landed in v0.8.9**:

The runtime now implements canonical HD:

1. Cholesky-decompose the residual covariance: ``P P' = Σᵤ``.
2. Recover structural shocks: ``e*_t = P⁻¹ u_t``.
3. Convolve with the orthogonalised IRF coefficients: for the target
   variable ``i`` and shock ``j``,
   ``hd[t, i, j] = Σ_{s=0..t}  orth_irfs[s, i, j] · e*_{t-s, j}``.
4. Per-shock importance: ``importance[j] = Σ_t |hd[t, target, j]|``.

The reduction to a *summed-absolute* importance keeps the L7 frame
schema unchanged (one row per shock); a future v0.9.x extension can
surface the full ``hd[t, i, j]`` time path as an additional artefact
for stacked-bar visualisation.

**Verification (this V2.3 (d) entry)**:

* Reconstruction lower bound
  (``test_v23_historical_decomposition_reconstructs_target_path``):
  total per-shock importance (sum of ``|hd[t, target, j]|`` across
  ``t`` and ``j``) is between 0.5× and 5× the L1 fluctuation of the
  realised target around its sample mean. The previous proxy could
  miss this band wildly because it was decoupled from the realised
  residual sequence.
* Path dependence (same test): re-fitting on a different residual
  realisation (different seed, same DGP parameters) produces a
  distinct importance vector. The proxy was constant in residual
  realisation up to the residual-std factor, so its per-shock
  importance was nearly identical across draws -- the new HD is
  correctly seed-dependent.

**Discrepancies (post-fix)**:

* The L7 frame schema returns a per-shock summary, not the full
  ``(T_resid, K)`` HD time series. Downstream renderers that wanted
  the time path (e.g. stacked-bar over time) will need a v0.9.x
  artefact extension to surface the full ``hd_target`` array.
* HD is sensitive to the variable ordering (Cholesky identification);
  this is the same caveat as for ``orthogonalised_irf`` and is
  documented in both OPTION_DOCS entries.

**Action**: ✅ runtime rewritten with canonical Burbidge-Harrison
algorithm. ✅ known-answer tests pinning reconstruction bound + path
dependence. CHANGELOG entry under v0.8.9 honesty-pass section.


## V2.4 dfm_mixed_mariano_murasawa runtime vs Mariano-Murasawa (2003 / 2010)

**Status**: ❌ FAIL → fixed in v0.8.9 honesty pass with two patches.

Plan reference: V2.4 audits the ``dfm_mixed_mariano_murasawa`` family
operational since v0.25 #245. The expected procedure is the
Mariano-Murasawa (2003) monthly-state aggregator dynamic factor model
fitted by Kalman filter + EM, with the Mariano-Murasawa (2010) Eq. 4
AR(1) idiosyncratic-error spec as the canonical default.

### Pre-fix behaviour

The runtime ``_DFMMixedFrequency`` documented two paths:

1. **Mixed-frequency**: when ``mixed_frequency=True`` or
   ``column_frequencies`` is provided, route to
   ``statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ``.
2. **Single-frequency** (fallback): route to
   ``statsmodels.tsa.statespace.dynamic_factor.DynamicFactor``.

Both delegate to statsmodels state-space estimators (Kalman + MLE) --
not PCA approximations -- so the *intent* matched the paper. The
problem was that the **mixed-frequency path had never run successfully**:

* **Bug 1 -- ``k_endog_monthly`` argument conflict**. The runtime
  passed ``endog_quarterly=...`` together with
  ``k_endog_monthly=len(monthly)``. ``DynamicFactorMQ.__init__``
  rejects this combination with ``ValueError``: when
  ``endog_quarterly`` is supplied, ``k_endog_monthly`` is inferred from
  the primary endog shape and *cannot* be specified explicitly. Every
  invocation raised on construction, the silent ``try/except`` caught
  the exception, and the ``for ... else`` clause routed the user into
  the single-frequency fallback **without warning**.

* **Bug 2 -- Quarterly index shape mismatch**. ``DynamicFactorMQ``
  requires the quarterly endog to carry a quarterly-frequency
  ``DateTimeIndex`` (``freqstr`` starting with 'Q'). Users naturally
  supply quarterly variables NaN-padded at non-quarter-end months on a
  monthly index (e.g. FRED-QD inside a FRED-MD panel); this shape was
  rejected by statsmodels and the runtime had no normalisation step to
  convert it.

Combined effect: every recipe declaring ``mixed_frequency=True`` (the
M-M 2003 path) silently degraded to the single-frequency
``DynamicFactor`` (which is a generic DFM, not the Mariano-Murasawa
mixed-state model). The ``operational`` claim from v0.25 #245 was
overclaimed for the mixed-frequency contract.

### Fix landed in v0.8.9

Two surgical patches in ``_DFMMixedFrequency._fit_mixed_frequency``:

* **Argument fix**. When ``endog_quarterly`` is non-None, drop
  ``k_endog_monthly`` from the kwargs (statsmodels infers it). When
  there are no quarterly variables, keep ``k_endog_monthly`` as before.

* **Quarterly index normalisation**. Drop all-NaN rows in the
  quarterly block, then reindex to a quarterly ``DateTimeIndex`` with
  ``freq='QE'`` (pandas 3.0 spelling; statsmodels' frequency check
  inspects only ``freqstr[0]``, which is ``'Q'`` for both ``'Q'`` and
  ``'QE'``). The conversion only fires when the user-supplied index is
  a ``DatetimeIndex``, so non-monthly inputs are passed through.

Two diagnostic surfaces added to make a future silent-fallback
regression detectable:

* ``_mq_failure_reason``: ``str | None`` populated when MQ requested
  but did not run (statsmodels missing / insufficient data / no
  monthly anchor / fit raised). Replaces the previous "exception
  swallowed, no trace" behaviour.
* ``_idiosyncratic_ar1``: ``True`` when the M-M (2010) Eq. 4 spec was
  active, ``False`` when the runtime fell back to ``i.i.d.``
  idiosyncratic errors, ``None`` when MQ did not run.

### Verification (this V2.4 entry)

* **Pure-monthly MQ runs with M-M 2010 Eq. 4 spec**
  (``test_v24_dfm_mq_pure_monthly_uses_mariano_murasawa_2010_ar1``):
  ``mixed_frequency=True`` with all-monthly columns drives
  ``_mode == "mixed_frequency_mq"``, ``_idiosyncratic_ar1 is True``,
  ``_mq_failure_reason is None``. ``_results`` exposes ``llf`` +
  ``filter_results`` (state-space invariants).
* **Mixed M+Q with NaN-padded quarterly column**
  (``test_v24_dfm_mq_mixed_m_q_handles_quarterly_nan_padded_input``):
  user supplies ``q1`` NaN-padded on monthly index; runtime drops the
  NaN rows, reindexes to quarterly DateTimeIndex, runs DFMQ;
  forecasts finite. Pre-fix this path silently degraded to single-
  frequency.
* **Single-frequency uses state-space DFM**
  (``test_v24_dfm_single_frequency_falls_back_to_state_space_dfm``):
  ``mixed_frequency=False`` routes to ``DynamicFactor`` (Kalman +
  MLE), ``_mode == "single_frequency"``, ``llf`` and ``filter_results``
  present.
* **Diagnostic surfaces honestly**
  (``test_v24_dfm_mq_failure_surfaces_in_diagnostic_attribute``):
  declaring all variables quarterly (no monthly anchor) makes MQ
  refuse; the runtime sets ``_mq_failure_reason`` to ``"no monthly
  variables declared"`` and falls back. The user can now distinguish
  intended single-frequency from a silently-failed MQ.

### Discrepancies (post-fix)

* The ``maxiter=20`` cap on ``DynamicFactorMQ.fit`` is short relative
  to statsmodels' default of 50; small-sample MQ fits may emit
  ``ConvergenceWarning`` while still producing valid forecasts. Tracked
  for v0.9.x as an exposed param.
* The single-frequency path does not implement the Mariano-Murasawa
  monthly aggregator (it would not apply -- the aggregator equation
  is the *raison d'être* of the mixed-frequency model). When users
  request mixed-frequency but every column is declared monthly, the
  runtime correctly routes to ``DynamicFactorMQ`` (pure-monthly MQ);
  this is documented behaviour.

### Action

* ✅ Argument-conflict patch landed in
  ``runtime.py:_DFMMixedFrequency._fit_mixed_frequency``.
* ✅ Quarterly index normalisation patch landed in the same method.
* ✅ Diagnostic attributes ``_mq_failure_reason`` /
  ``_idiosyncratic_ar1`` exposed.
* ✅ Four known-answer tests in
  ``tests/core/test_v09_paper_coverage.py``.
* CHANGELOG entry under v0.8.9 honesty-pass section.


## V2.1 scaled_pca runtime vs Huang/Zhou (2022) authors' MATLAB

**Status**: ❌ **FAIL → fixed in v0.8.9 honesty pass**

**Pre-fix discrepancy**:

The previous ``_pca_factors(variant='scaled_pca')`` implemented a
**row-wise ``|target|`` weighting** of observations, not the paper's
column-wise predictive-slope β scaling. This is a fundamentally
different algorithm.

* Paper's algorithm (from authors' MATLAB ``sPCAest.m``):
  1. Standardise X column-wise (z-score, ddof=1)
  2. For each column j, β_j = univariate OLS slope of target on Xs[:, j]
  3. Scale each column j of Xs by β_j (signed)
  4. PCA on the scaled matrix
* Our previous implementation:
  1. Centre X (NOT standardise)
  2. Compute row-wise weights from |target − mean(target)| / max
  3. Multiply each row by (1 + |target|/max)
  4. sklearn PCA

Synthetic test confirmed: factor correlation between previous
implementation and authors' algorithm was 0.977 (not 1.0 → different
algorithms). On the same data, authors' algorithm tracked the true
latent factor at 0.988 vs our previous at 0.963.

**Fix landed in v0.8.9**:

New helper ``_scaled_pca_huang_zhou`` implements the four-step paper
algorithm exactly. Verified against authors' MATLAB:

* Per-column β agrees to machine precision (max abs diff ~ 1e-16)
* Resulting first factor correlates 1.000 with authors' algorithm
* Test ``test_v21_scaled_pca_matches_huang_zhou_2022_authors_matlab``
  pins this in the regression suite.

**Discrepancies (post-fix)**:

* Authors' ``pc_T`` uses ``F'F/T = I`` factor normalisation; we use
  sklearn ``PCA`` default (factors scaled by singular values). Up to
  sign and per-column scalar, the factor *directions* are identical;
  downstream regressions absorb the scaling difference into the
  coefficients. Documented in the helper docstring; non-issue for
  prediction.

**Action**: ✅ runtime fix landed. ✅ known-answer test added.
✅ helper docstring documents normalisation difference. CHANGELOG
entry needed under v0.8.9 honesty-pass section.

