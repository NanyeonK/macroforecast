# `op`

[Back to L7](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``op`` on sub-layer ``L7_A_importance_dag_body`` (layer ``l7``).

## Sub-layer

**L7_A_importance_dag_body**

## Axis metadata

- Default: `'permutation_importance'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 34 option(s)
- Future: 6 option(s)

## Options

### `accumulated_local_effect`  --  operational

Apley & Zhu (2020) accumulated local effects -- PDP alternative robust to correlation.

For feature ``j``, computes the cumulative local change ``Σ_{k≤K} E_{X_{-j} | x_j ∈ bin_k}[∂f/∂x_j]·Δx_j``. The binning + conditioning eliminates the 'extrapolation into low-density regions' bias of plain PDPs.

**When to use**

Correlated feature panels (FRED-MD / -QD) where PDPs are misleading.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Apley & Zhu (2020) 'Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models', JRSS Series B 82(4): 1059-1086.

**Related options**: [`partial_dependence`](#partial-dependence)

_Last reviewed 2026-05-05 by macroforecast author._

### `attention_weights`  --  operational

OLS-as-attention closed-form attention matrix (Goulet Coulombe 2026).

Goulet Coulombe (2026) 'OLS as an Attention Mechanism' Eq. 3 closed form: ``Ω = X_test · (X'_train · X_train)⁻¹ · X'_train``. The ``(n_test, n_train)`` matrix encodes how strongly each test point attends to each training point under an OLS / ridge fit, identical to the representer expansion of the dual ridge solution. Output table carries one row per training observation (per-test-point weight aggregates) plus the full attention matrix and representer-identity diagnostics inline via ``frame.attrs``.

Promoted from ``future`` to ``operational`` in Phase B-10 (paper-10 replication). Compatible with linear-family L4 models (``ols`` / ``ridge`` / ``lasso`` / ``elastic_net`` / ``bayesian_ridge`` / ``huber``).

**When to use**

Linear-family attribution as a kernel-attention map; pedagogical / replication of paper-10 Coulombe (2026).

**When NOT to use**

Non-linear models (the closed form requires a linear estimator).

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Goulet Coulombe (2026) 'OLS as an Attention Mechanism', working paper -- Eq. 3 closed-form attention matrix.

**Related options**: [`dual_decomposition`](#dual-decomposition), [`model_native_linear_coef`](#model-native-linear-coef), [`shap_linear`](#shap-linear)

_Last reviewed 2026-05-05 by macroforecast author._

### `bootstrap_jackknife`  --  operational

Bootstrap / jackknife confidence bands around any importance score.

Wraps another importance op and re-runs it on ``B`` stationary-bootstrap (Politis-White 2004) or jackknife resamples. Emits ``(score_mean, score_p2.5, score_p97.5)`` per feature; pair with the ``boxplot`` figure type.

**When to use**

Reporting confidence-banded importance rankings.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Politis & White (2004) 'Automatic Block-Length Selection for the Dependent Bootstrap', Econometric Reviews 23(1): 53-70.

**Related options**: [`rolling_recompute`](#rolling-recompute)

_Last reviewed 2026-05-05 by macroforecast author._

### `boruta_selection`  --  future

_(no schema description for `boruta_selection`)_

> TBD: option doc not yet authored for this value. The encyclopedia falls back to the bare schema description above. PRs adding a full ``OptionDoc`` entry under ``macroforecast/scaffold/option_docs/l7.py`` are welcome.

### `bvar_pip`  --  operational

Posterior inclusion probabilities for BVAR / Bayesian linear models.

For each predictor ``j``, returns ``P(β_j ≠ 0 | data)`` -- the posterior probability that the variable enters the model with non-zero effect. Compatible with ``bvar_minnesota`` / ``bvar_normal_inverse_wishart`` / ``bayesian_ridge``.

**When to use**

Bayesian model selection; comparing variable importance under posterior uncertainty.

**When NOT to use**

Frequentist models -- use ``lasso_inclusion_frequency`` for an analogous stability score.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Koop & Korobilis (2010) 'Bayesian Multivariate Time Series Methods for Empirical Macroeconomics', Foundations and Trends in Econometrics 3(4): 267-358.

**Related options**: [`lasso_inclusion_frequency`](#lasso-inclusion-frequency)

_Last reviewed 2026-05-05 by macroforecast author._

### `cumulative_r2_contribution`  --  operational

Cumulative R² gain from adding features one at a time (forward-selection ranking).

Re-fits the L4 estimator with features added in descending order of marginal contribution; each step records the cumulative OOS-R² achieved. Pair with the ``lineplot`` figure type to visualise the marginal information value of each predictor.

**When to use**

Quantifying how many predictors the model actually needs to reach a target R².

**When NOT to use**

Highly correlated features -- the order is sensitive to entry rules.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Stock & Watson (2012) 'Generalized Shrinkage Methods for Forecasting using Many Predictors', JBES 30(4): 481-493.

**Related options**: [`lasso_inclusion_frequency`](#lasso-inclusion-frequency), [`lofo`](#lofo)

_Last reviewed 2026-05-05 by macroforecast author._

### `deep_lift`  --  operational

DeepLIFT (Shrikumar 2017) -- difference-from-reference attribution.

Decomposes the difference ``f(x) - f(x')`` into per-feature contributions using rescaled-difference / reveal-cancel rules for non-linear activations. Faster than integrated gradients but with less rigorous axiomatic backing.

**When to use**

NN attribution where integrated-gradients runtime is too high.

**When NOT to use**

When the completeness / sensitivity axioms matter -- prefer integrated gradients.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Shrikumar, Greenside & Kundaje (2017) 'Learning Important Features Through Propagating Activation Differences', ICML.

**Related options**: [`integrated_gradients`](#integrated-gradients), [`gradient_shap`](#gradient-shap), [`saliency_map`](#saliency-map), [`shap_deep`](#shap-deep)

_Last reviewed 2026-05-05 by macroforecast author._

### `dual_decomposition`  --  operational

Forecast-as-weighted-training-targets via the representer theorem (Coulombe et al. 2024); equivalently a restricted attention module (Goulet Coulombe 2026).

Goulet Coulombe / Goebel / Klieber (2024) 'Dual Interpretation of ML Forecasts'. Surfaces each prediction as a weighted combination of historical training targets; weights recovered through the representer theorem applied to the fitted model. Atomic L7 primitive: SHAP-family ops decompose by feature contribution, this op decomposes by training-row contribution, the natively interpretable view for small-sample temporally-ordered macro panels.

**Linear families** (operational v0.8.9): ridge / OLS / lasso via closed-form ``w(xₜ) = X(X'X + αI)⁻¹xₜ``.

**Tree-bagging ensembles** (operational v0.9.1 dev-stage v0.9.0B-5): RandomForestRegressor / ExtraTreesRegressor via the leaf-co-occurrence kernel ``wⱼ(xₜ) = (1/B) Σ_b 1[j ∈ B_b] · 1[leaf_b(xₜ) == leaf_b(xⱼ)] / leaf_size_b(xⱼ)`` where ``B_b`` is tree b's bootstrap subset (sklearn ``estimators_samples_``). Reproduces ``forest.predict`` to machine precision (~4e-16). Helper ``_rf_leaf_cooccurrence_weights`` in core/runtime.py.

Output frame layout: rows = training row labels, columns = ``mean_weight``, ``abs_mean_weight``, ``max_abs_weight``. Full ``(n_test × n_train)`` weight matrix attached as ``frame.attrs['dual_weights']``; ``frame.attrs['method']`` carries ``'linear_closed_form'`` or ``'rf_leaf_cooccurrence_kernel'`` for downstream renderers.

**Inline portfolio diagnostics.** The output artifact also carries the four portfolio metrics from the same paper (HHI = ``Σwⱼ²``, short = ``Σ max(0,-wⱼ)``, turnover = ``‖wₜ - wₜ₋₁‖₁``, leverage = ``‖w‖₁``) at ``frame.attrs['portfolio_metrics']``. These are trivial numpy reductions on the primary dual weights and do not warrant their own L7 op (decomposition discipline).

**OLS-as-attention equivalence.** Goulet Coulombe (2026) 'Ordinary Least Squares as an Attention Mechanism' (SSRN 5200864) shows that the same dual representation ``ŷ_test = F_test F_train' y_train`` (eq. 7) coincides with a *restricted attention module*: queries ``Q = X_test W``, keys ``K = X_train W`` with ``W = U Λ^{-½}``, values ``V = y``, and the softmax replaced by the identity (eqs. 17-19). The training-row weights ``ωⱼᵢ = ⟨Fⱼ, Fᵢ⟩`` surfaced by this op are exactly the (restricted) attention weights of that paper. Same compute, different vocabulary -- no separate runtime needed.

Boosted-tree (gradient_boosting / xgboost / lightgbm) and NN extensions are deferred: residual-bagging and learned non-linear models do not admit a clean sum-of-training-targets dual representation.

**When to use**

Decomposing macro forecasts into training-target contributions; explaining ML predictions to econometric audiences; bridging classical OLS to transformer-attention literature; per-prediction provenance for tree ensembles.

**When NOT to use**

Boosted-tree / NN families (gradient_boosting, xgboost, lightgbm, mlp, lstm, etc.) -- raises NotImplementedError; the residual-bagging structure does not factor into a sum-of-training-targets representation.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Goulet Coulombe, Goebel & Klieber (2024) 'Dual Interpretation of Machine Learning Forecasts', arXiv:2412.13076.
* Goulet Coulombe (2026) 'Ordinary Least Squares as an Attention Mechanism', SSRN 5200864 -- shows OLS predictions ŷ_test = F_test F_train' y_train (eq. 7) coincide with a restricted attention module (eqs. 17-19, identity activation, tied W_Q W_K' = (X_train' X_train)^{-1}). The dual_decomposition op already implements the same compute via the closed-form ridge representer; no separate runtime needed.

**Related options**: [`permutation_importance`](#permutation-importance), [`shap_kernel`](#shap-kernel)

_Last reviewed 2026-05-05 by macroforecast author._

### `fevd`  --  operational

Forecast error variance decomposition (Sims 1980).

For a fitted VAR (``var`` / ``factor_augmented_var`` / ``bvar_*``), decomposes the h-step-ahead forecast error variance into shares attributable to each orthogonalised shock. Default Cholesky orthogonalisation; ordering is set by the column order of the VAR. statsmodels ``fevd`` backend.

**When to use**

Standard VAR analysis; interpreting how shocks propagate across variables.

**When NOT to use**

Non-VAR models -- use ``permutation_importance`` instead.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Sims (1980) 'Macroeconomics and Reality', Econometrica 48(1): 1-48.

**Related options**: [`historical_decomposition`](#historical-decomposition), [`generalized_irf`](#generalized-irf), [`forecast_decomposition`](#forecast-decomposition)

_Last reviewed 2026-05-05 by macroforecast author._

### `forecast_decomposition`  --  operational

Decompose a single forecast into per-feature contributions.

For a single (cell, target, horizon) forecast, returns a table ``(feature → contribution)`` summing to ``forecast - benchmark``. Linear models: ``β_j x_j``. Trees: Tree SHAP. NN: gradient SHAP. Universal entry point unified across families -- delegates to the appropriate family-specific op.

**When to use**

Reporting feature contributions for a specific forecast (e.g. 'why is the model bullish on Q3 GDP').

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'

**Related options**: [`shap_tree`](#shap-tree), [`shap_linear`](#shap-linear), [`shap_deep`](#shap-deep)

_Last reviewed 2026-05-05 by macroforecast author._

### `friedman_h_interaction`  --  operational

Friedman & Popescu (2008) H-statistic for two-way feature interactions.

For feature pair ``(j, k)``, computes ``H²_{jk} = Σ[PD_{jk}(x_j, x_k) - PD_j(x_j) - PD_k(x_k)]² / Σ PD²_{jk}``. ``H² ∈ [0, 1]``; the share of the joint partial-dependence variance attributable to non-additive structure.

**When to use**

Identifying which feature pairs the model treats non-additively.

**When NOT to use**

Wide panels -- the M² PDP grid grows expensive.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Friedman & Popescu (2008) 'Predictive Learning via Rule Ensembles', Annals of Applied Statistics 2(3): 916-954.

**Related options**: [`shap_interaction`](#shap-interaction), [`partial_dependence`](#partial-dependence)

_Last reviewed 2026-05-05 by macroforecast author._

### `generalized_irf`  --  future

Pesaran-Shin (1998) generalized impulse-response function (future, v0.9.x).

Order-invariant IRF where each shock is constructed as the multivariate-normal projection of all residuals onto the j-th canonical direction. Distinct from Cholesky orthogonalised IRFs (which use a recursive lower-triangular rotation). **Future** -- the runtime currently raises NotImplementedError. For the Cholesky variant operational since v0.2, use ``orthogonalised_irf``.

**When to use**

VAR analysis where the variable ordering has no theoretical motivation -- order-invariance is the desired property.

**When NOT to use**

When a recursive identification IS theoretically motivated -- use ``orthogonalised_irf`` instead.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Pesaran & Shin (1998) 'Generalized impulse response analysis in linear multivariate models', Economics Letters 58(1): 17-29.

**Related options**: [`fevd`](#fevd), [`historical_decomposition`](#historical-decomposition), [`orthogonalised_irf`](#orthogonalised-irf)

_Last reviewed 2026-05-05 by macroforecast author._

### `gradient_shap`  --  operational

Gradient SHAP -- expectation-of-gradient SHAP approximation (Lundberg-Lee 2017).

Approximates SHAP values via expected gradients at random interpolations between input and a baseline distribution. Captum-backed; requires the ``macroforecast[deep]`` extra.

**When to use**

Differentiable models (NN families) where exact SHAP is too expensive.

**When NOT to use**

Non-NN models.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Lundberg & Lee (2017) 'A Unified Approach to Interpreting Model Predictions', NeurIPS 30: 4765-4774.

**Related options**: [`shap_deep`](#shap-deep), [`integrated_gradients`](#integrated-gradients), [`saliency_map`](#saliency-map), [`deep_lift`](#deep-lift)

_Last reviewed 2026-05-05 by macroforecast author._

### `group_aggregate`  --  operational

Aggregate per-feature importance into pre-defined block sums (FRED-SD blocks, theme blocks).

Sums (or means) per-feature importance scores over groups defined by a user-supplied or built-in mapping table. v0.25 ships 8 built-in blocks: 8-group FRED-MD + 14-group FRED-QD + 50-state FRED-SD grids.

Required input for the FRED-SD ``us_state_choropleth`` figure.

**When to use**

FRED-MD / -QD / -SD analyses where per-series importance should roll up to thematic / geographic blocks.

**When NOT to use**

Custom panels lacking a meaningful grouping.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', JBES 34(4): 574-589.

**Related options**: [`lineage_attribution`](#lineage-attribution), [`transformation_attribution`](#transformation-attribution)

_Last reviewed 2026-05-05 by macroforecast author._

### `historical_decomposition`  --  operational

Historical decomposition (Burbidge-Harrison 1985) of the realised series into structural shocks.

Reconstructs each variable's realised path as the convolution of orthogonalised IRF coefficients (Cholesky-rotated structural form) with the time series of structural shocks recovered from the reduced-form residuals. Returns the per-shock cumulative absolute contribution to the target variable's realised fluctuations; the row labels match the VAR variable ordering.

**When to use**

Telling the historical narrative -- which shocks drove specific recessions / expansions.

**When NOT to use**

Non-VAR models.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Burbidge & Harrison (1985) 'A historical decomposition of the great depression to determine the role of money', JME 16(1): 45-54.

**Related options**: [`fevd`](#fevd), [`orthogonalised_irf`](#orthogonalised-irf)

_Last reviewed 2026-05-05 by macroforecast author._

### `integrated_gradients`  --  operational

Integrated gradients (Sundararajan 2017) -- path-integral attribution.

Computes ``(x_j - x'_j) · ∫₀¹ ∂f(x' + α(x - x')) / ∂x_j dα`` for a baseline ``x'`` (default zero). Satisfies the completeness axiom (sum of attributions equals ``f(x) - f(x')``). Captum-backed.

**When to use**

Axiomatically-grounded NN attribution (Sundararajan completeness + sensitivity properties).

**When NOT to use**

Non-NN models; pathological models where integration along the linear path is misleading.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Sundararajan, Taly & Yan (2017) 'Axiomatic Attribution for Deep Networks', ICML.

**Related options**: [`gradient_shap`](#gradient-shap), [`saliency_map`](#saliency-map), [`deep_lift`](#deep-lift)

_Last reviewed 2026-05-05 by macroforecast author._

### `lasso_inclusion_frequency`  --  operational

Bootstrap inclusion frequency for Lasso-selected features (Bach 2008).

For each feature ``j``, computes the share of ``B`` Lasso fits (on bootstrap or rolling-window resamples) for which ``β̂_j ≠ 0``. Returns a stability score in ``[0, 1]``. v0.25 supports ``sampling = bootstrap | rolling | both`` (via leaf_config).

**When to use**

Feature-selection stability audit for Lasso / Lasso-Path / Elastic Net.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Bach (2008) 'Bolasso: model consistent Lasso estimation through the bootstrap', ICML.
* Meinshausen & Bühlmann (2010) 'Stability selection', JRSS Series B 72(4): 417-473.

**Related options**: [`model_native_linear_coef`](#model-native-linear-coef), [`bootstrap_jackknife`](#bootstrap-jackknife)

_Last reviewed 2026-05-05 by macroforecast author._

### `lasso_path_selection`  --  future

_(no schema description for `lasso_path_selection`)_

> TBD: option doc not yet authored for this value. The encyclopedia falls back to the bare schema description above. PRs adding a full ``OptionDoc`` entry under ``macroforecast/scaffold/option_docs/l7.py`` are welcome.

### `lineage_attribution`  --  operational

Trace importance back through L3 feature lineage to the L1 raw source.

For each L3 feature, walks the L3.metadata ``column_lineage`` graph to identify the chain of transforms that produced it; attributes the L7 importance score back to the L1 raw column at the head of the lineage chain.

Solves the 'PCA factors are most important; what does that mean in terms of original variables?' problem.

**When to use**

Pipelines with PCA / factor / dimensionality-reduction stages where downstream importance must be traced back to raw inputs.

**When NOT to use**

Pipelines with only direct-input features (no L3 transforms).

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'

**Related options**: [`group_aggregate`](#group-aggregate), [`transformation_attribution`](#transformation-attribution)

_Last reviewed 2026-05-05 by macroforecast author._

### `lofo`  --  operational

Leave-one-feature-out (LOFO) refit importance.

For each predictor ``j``, refits the L4 estimator on the panel with column ``j`` removed and reports the OOS-loss delta. More expensive than permutation importance (one extra fit per feature) but free from the permutation-and-correlation interaction.

Compatible with every L4 family; runtime scales as ``n_features × cost_per_fit``.

**When to use**

Small / medium feature panels (< 100) where N-extra fits are affordable.

**When NOT to use**

Wide panels (n_features > 200) -- prohibitive runtime.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Lemaître, Aridas & Nogueira (2018) 'imbalanced-learn', JMLR 18(17): 1-5 -- LOFO popularised; pre-dating refit-importance traditions in econometrics.

**Related options**: [`permutation_importance`](#permutation-importance)

_Last reviewed 2026-05-05 by macroforecast author._

### `lstm_hidden_state`  --  future

_(no schema description for `lstm_hidden_state`)_

> TBD: option doc not yet authored for this value. The encyclopedia falls back to the bare schema description above. PRs adding a full ``OptionDoc`` entry under ``macroforecast/scaffold/option_docs/l7.py`` are welcome.

### `model_native_linear_coef`  --  operational

Standardised regression coefficients from a fitted linear model.

Returns ``β̂_j`` for each predictor as the importance score; with ``standardize=True`` (default) the predictors are pre-scaled so coefficients are directly comparable. Compatible with every linear-family L4 model (``ols / ridge / lasso / elastic_net / lasso_path / bayesian_ridge / huber / glmboost``).

Cheapest meaningful importance score; the natural sanity-check to run before the more expensive permutation / SHAP families.

**When to use**

Linear-model baselines; quick interpretation when a tree / NN model is overkill.

**When NOT to use**

Non-linear models -- coefficients no longer summarise marginal effects.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Greene (2018) 'Econometric Analysis', 8th ed., Pearson, Chapter 4.

**Related options**: [`model_native_tree_importance`](#model-native-tree-importance), [`lasso_inclusion_frequency`](#lasso-inclusion-frequency)

_Last reviewed 2026-05-05 by macroforecast author._

### `model_native_tree_importance`  --  operational

Mean-decrease-impurity importance from a fitted tree ensemble.

Returns sklearn's ``feature_importances_`` for the fitted estimator -- the average reduction in node impurity attributable to each feature, weighted by node sample count. Available for every tree-family L4 model (``decision_tree`` / ``random_forest`` / ``extra_trees`` / ``gradient_boosting`` / ``xgboost`` / ``lightgbm`` / ``catboost``).

Cheap and built-in; biases toward high-cardinality features. For unbiased tree importance, prefer ``permutation_importance`` or ``permutation_importance_strobl``.

**When to use**

Quick first-pass tree importance; pair with permutation importance for bias-correction.

**When NOT to use**

High-cardinality continuous features dominate -- known MDI bias (Strobl et al. 2007).

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Breiman (2001) 'Random Forests', Machine Learning 45(1): 5-32.
* Strobl, Boulesteix, Zeileis & Hothorn (2007) 'Bias in random forest variable importance measures', BMC Bioinformatics 8: 25.

**Related options**: [`permutation_importance`](#permutation-importance), [`permutation_importance_strobl`](#permutation-importance-strobl)

_Last reviewed 2026-05-05 by macroforecast author._

### `mrf_gtvp`  --  operational

Macroeconomic Random Forest GTVP -- per-leaf time-varying coefficients (Coulombe 2024).

Compatible only with the ``macroeconomic_random_forest`` L4 family. For each leaf ``ℓ`` and predictor ``j``, returns the leaf-local linear coefficient ``β̂_{j, ℓ}``; the full output is an ``(n_leaves × n_features)`` GTVP (Generalised Time-Varying Parameter) panel.

**When to use**

Coulombe (2024) MRF interpretation; spotting non-linearity captured by the leaf partition.

**When NOT to use**

Non-MRF models.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Coulombe (2024) 'The Macroeconomic Random Forest', Journal of Applied Econometrics 39(7): 1190-1209.

**Related options**: [`rolling_recompute`](#rolling-recompute), [`model_native_tree_importance`](#model-native-tree-importance)

_Last reviewed 2026-05-05 by macroforecast author._

### `orthogonalised_irf`  --  operational

Cholesky-orthogonalised impulse-response function (Sims 1980).

Standard structural-VAR IRF: residual covariance Σᵤ is Cholesky-decomposed P P' = Σᵤ; the structural shocks P⁻¹ u_t are orthogonalised by construction. ``orth_irfs[s, i, j]`` is the response of variable ``i`` at horizon ``s`` to a unit structural shock to variable ``j`` at time 0. **Order-dependent**: the variable ordering in the recipe determines the recursive causal scheme imposed.

**When to use**

VAR analysis with a theoretically motivated recursive identification (e.g. monetary policy ordered last; supply ordered first).

**When NOT to use**

When the variable ordering is arbitrary -- file a v0.9.x request for ``generalized_irf`` (Pesaran-Shin 1998 order-invariant variant, currently future-gated).

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Sims (1980) 'Macroeconomics and Reality', Econometrica 48(1): 1-48.

**Related options**: [`fevd`](#fevd), [`historical_decomposition`](#historical-decomposition)

_Last reviewed 2026-05-05 by macroforecast author._

### `oshapley_vi`  --  operational

Out-of-sample SHAP-style variable importance (Borup et al. 2022) [schema; runtime via anatomy package].

Borup, Goulet Coulombe, Montes-Rojas, Schutte & Veiga (2022) 'Anatomy of Out-of-Sample Forecasting Accuracy'. Recomputes Shapley-style feature contributions on the *out-of-sample* loss rather than in-sample fit, addressing the distribution-shift mismatch where in-sample SHAP misranks features that matter for OOS accuracy.

Atomic primitive -- existing in-sample ``shap_*`` ops do not compose into oShapley-VI. Runtime delegates to the Borup et al. ``anatomy`` Python package as an optional dep (``pip install macroforecast[anatomy]``). Schema-only in v0.9.0; operational promotion lands once the anatomy integration is wired.

**When to use**

OOS-aware variable importance for macro forecast audits; replicating Borup et al. (2022).

**When NOT to use**

Pre-promotion. Without the anatomy extra installed.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Borup, Goulet Coulombe, Montes-Rojas, Schutte & Veiga (2022) 'Anatomy of Out-of-Sample Forecasting Accuracy', SSRN 4278745.

**Related options**: [`shap_kernel`](#shap-kernel), [`shap_tree`](#shap-tree), [`permutation_importance`](#permutation-importance), [`pbsv`](#pbsv)

_Last reviewed 2026-05-05 by macroforecast author._

### `partial_dependence`  --  operational

Friedman (2001) partial dependence plot.

For feature ``j``, computes ``E_{X_{-j}}[f(x_j, X_{-j})]`` over a grid of ``x_j`` values. Visualises the marginal effect of ``x_j`` on the prediction averaged over the joint distribution of remaining features. sklearn ``partial_dependence`` backend.

**When to use**

Visualising marginal feature effects; first-pass non-linearity audit.

**When NOT to use**

Highly correlated features -- PDP averages over impossible regions of feature space. Use ``accumulated_local_effect`` instead.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Friedman (2001) 'Greedy Function Approximation: A Gradient Boosting Machine', Annals of Statistics 29(5): 1189-1232.

**Related options**: [`accumulated_local_effect`](#accumulated-local-effect), [`friedman_h_interaction`](#friedman-h-interaction)

_Last reviewed 2026-05-05 by macroforecast author._

### `pbsv`  --  operational

Performance-Based Shapley Value (Borup et al. 2022) [schema; runtime via anatomy package].

OOS accuracy decomposition: Shapley-attributes the forecast performance improvement over a benchmark to each feature coalition's contribution. Differs from ``oshapley_vi`` in decomposing the *accuracy gain* rather than the OOS loss; they are companion ops covering the two faces of OOS Shapley.

Runtime delegates to ``anatomy`` package. Schema-only in v0.9.0.

**When to use**

Decomposing OOS forecast skill by feature; benchmark-relative interpretation studies.

**When NOT to use**

Pre-promotion. Without the anatomy extra installed.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Borup, Goulet Coulombe, Montes-Rojas, Schutte & Veiga (2022) 'Anatomy of Out-of-Sample Forecasting Accuracy', SSRN 4278745.

**Related options**: [`oshapley_vi`](#oshapley-vi), [`permutation_importance`](#permutation-importance)

_Last reviewed 2026-05-05 by macroforecast author._

### `permutation_importance`  --  operational

Breiman-Fisher-Rudin (2019) model-agnostic permutation importance.

For each predictor ``j``, computes the increase in OOS loss when ``x_j`` is randomly permuted. The score is ``L(y, f(X_perm_j)) - L(y, f(X))`` averaged over ``n_repeats`` (default 10). Model-agnostic: works for every L4 family.

Bias-free alternative to ``model_native_tree_importance``; the gold-standard fallback for any model that does not expose a native importance attribute.

**When to use**

Default importance score for non-linear models; comparing across model families.

**When NOT to use**

Highly correlated predictors -- permutation breaks the dependence and inflates importance. Use ``permutation_importance_strobl`` instead.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Fisher, Rudin & Dominici (2019) 'All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously', JMLR 20(177): 1-81.
* Breiman (2001) 'Random Forests', Machine Learning 45(1): 5-32.

**Related options**: [`permutation_importance_strobl`](#permutation-importance-strobl), [`lofo`](#lofo), [`model_native_tree_importance`](#model-native-tree-importance)

_Last reviewed 2026-05-05 by macroforecast author._

### `permutation_importance_strobl`  --  operational

Strobl (2008) conditional permutation importance.

Permutes ``x_j`` only within bins defined by the joint distribution of correlated predictors, eliminating the extrapolation bias of plain permutation importance for correlated features. v0.3 implementation uses tree-partition bins (Strobl et al. 2008 §4).

**When to use**

Highly correlated macro panels (FRED-MD / -QD with redundant aggregates).

**When NOT to use**

When predictor correlations are negligible -- the cheaper plain permutation importance suffices.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Strobl, Boulesteix, Kneib, Augustin & Zeileis (2008) 'Conditional variable importance for random forests', BMC Bioinformatics 9: 307.

**Related options**: [`permutation_importance`](#permutation-importance)

_Last reviewed 2026-05-05 by macroforecast author._

### `recursive_feature_elimination`  --  future

_(no schema description for `recursive_feature_elimination`)_

> TBD: option doc not yet authored for this value. The encyclopedia falls back to the bare schema description above. PRs adding a full ``OptionDoc`` entry under ``macroforecast/scaffold/option_docs/l7.py`` are welcome.

### `rolling_recompute`  --  operational

Re-compute any importance score on a rolling-window basis.

Applies an inner importance op (e.g. ``permutation_importance``) on each of K rolling-window subsamples; emits a ``(K × n_features)`` matrix tracking how importance evolves over time. Pair with the ``heatmap`` or ``lineplot`` figure type.

**When to use**

Detecting time-varying feature importance; structural-stability audits.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'

**Related options**: [`bootstrap_jackknife`](#bootstrap-jackknife), [`mrf_gtvp`](#mrf-gtvp)

_Last reviewed 2026-05-05 by macroforecast author._

### `saliency_map`  --  operational

Saliency map (Simonyan 2014) -- absolute gradient at the input.

Returns ``|∂f / ∂x_j|`` evaluated at the input. The earliest and simplest gradient-based attribution; useful as a baseline but susceptible to gradient-saturation issues that integrated gradients address.

**When to use**

Quick NN attribution baseline; sanity-check vs more elaborate methods.

**When NOT to use**

Production attribution -- prefer integrated gradients or SHAP.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Simonyan, Vedaldi & Zisserman (2014) 'Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps', ICLR Workshops.

**Related options**: [`integrated_gradients`](#integrated-gradients), [`gradient_shap`](#gradient-shap), [`deep_lift`](#deep-lift)

_Last reviewed 2026-05-05 by macroforecast author._

### `shap_deep`  --  operational

Deep SHAP -- DeepLIFT-based SHAP for neural networks.

DeepLIFT (Shrikumar 2017) interpreted as Shapley-value approximation. Compatible with the ``mlp`` / ``lstm`` / ``gru`` / ``transformer`` L4 families when the ``macroforecast[deep]`` extra is installed (captum backend).

**When to use**

Neural-network forecasters (LSTM / GRU / Transformer / MLP).

**When NOT to use**

Non-NN models -- use ``shap_tree`` / ``shap_linear`` / ``shap_kernel`` instead.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Lundberg & Lee (2017) 'A Unified Approach to Interpreting Model Predictions', NeurIPS 30: 4765-4774.
* Shrikumar, Greenside & Kundaje (2017) 'Learning Important Features Through Propagating Activation Differences', ICML.

**Related options**: [`shap_tree`](#shap-tree), [`shap_kernel`](#shap-kernel), [`deep_lift`](#deep-lift), [`gradient_shap`](#gradient-shap), [`integrated_gradients`](#integrated-gradients)

_Last reviewed 2026-05-05 by macroforecast author._

### `shap_interaction`  --  operational

SHAP interaction values -- pairwise feature-interaction Shapley.

Lundberg-Erion-Lee (2020) extension that decomposes each SHAP value into a main-effect term plus pairwise interaction terms. Available for tree ensembles via the same polynomial-time algorithm as ``shap_tree``.

Output is an ``(n × M × M)`` tensor; pair with the ``heatmap`` figure type for visualisation.

**When to use**

Identifying which feature pairs drive the model's non-additive structure.

**When NOT to use**

Wide feature panels -- the ``M²`` storage cost grows quickly.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Lundberg & Lee (2017) 'A Unified Approach to Interpreting Model Predictions', NeurIPS 30: 4765-4774.
* Lundberg, Erion & Lee (2020) 'From local explanations to global understanding with explainable AI for trees', Nature Machine Intelligence 2: 56-67.

**Related options**: [`shap_tree`](#shap-tree), [`friedman_h_interaction`](#friedman-h-interaction)

_Last reviewed 2026-05-05 by macroforecast author._

### `shap_kernel`  --  operational

Kernel SHAP -- model-agnostic Shapley value approximation.

Lundberg-Lee (2017) weighted-LIME estimator that approximates Shapley values for any model via local linear regression on perturbed inputs. Slow (O(2^M) coalitions sampled) but universally applicable.

**When to use**

Non-tree, non-linear, non-deep models (SVM, kNN, custom callables).

**When NOT to use**

Trees (use ``shap_tree``) or linear models (use ``shap_linear``) -- both are dramatically faster.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Lundberg & Lee (2017) 'A Unified Approach to Interpreting Model Predictions', NeurIPS 30: 4765-4774.

**Related options**: [`shap_tree`](#shap-tree), [`shap_linear`](#shap-linear)

_Last reviewed 2026-05-05 by macroforecast author._

### `shap_linear`  --  operational

Linear SHAP -- closed-form Shapley values for linear models.

For a fitted linear model ``f(x) = β'x + b``, the SHAP value for feature ``j`` reduces to ``β_j (x_j - E[x_j])``. Uses the training-sample mean as the reference. Available for every linear L4 family.

**When to use**

Linear models when the SHAP per-row decomposition is needed (otherwise ``model_native_linear_coef`` suffices).

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Lundberg & Lee (2017) 'A Unified Approach to Interpreting Model Predictions', NeurIPS 30: 4765-4774.

**Related options**: [`model_native_linear_coef`](#model-native-linear-coef), [`shap_tree`](#shap-tree), [`shap_kernel`](#shap-kernel)

_Last reviewed 2026-05-05 by macroforecast author._

### `shap_tree`  --  operational

Tree SHAP -- exact polynomial-time Shapley values for tree ensembles.

Lundberg-Erion-Lee (2020) algorithm computing exact Shapley values in ``O(T·L·D²)`` time (T trees, L leaves, D depth) instead of ``O(2^M)`` brute-force. Available for ``random_forest`` / ``extra_trees`` / ``gradient_boosting`` / ``xgboost`` / ``lightgbm`` / ``catboost``.

Returns per-prediction SHAP values; the ``output_table_format`` L7.B axis controls whether the result is the global mean-``|SHAP|`` ranking or the per-row decomposition.

**When to use**

Default importance op for tree ensembles; exact and fast.

**When NOT to use**

Non-tree models -- use ``shap_kernel`` or ``shap_linear`` instead.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Lundberg & Lee (2017) 'A Unified Approach to Interpreting Model Predictions', NeurIPS 30: 4765-4774.
* Lundberg, Erion & Lee (2020) 'From local explanations to global understanding with explainable AI for trees', Nature Machine Intelligence 2: 56-67.

**Related options**: [`shap_kernel`](#shap-kernel), [`shap_linear`](#shap-linear), [`shap_interaction`](#shap-interaction), [`shap_deep`](#shap-deep)

_Last reviewed 2026-05-05 by macroforecast author._

### `stability_selection`  --  future

_(no schema description for `stability_selection`)_

> TBD: option doc not yet authored for this value. The encyclopedia falls back to the bare schema description above. PRs adding a full ``OptionDoc`` entry under ``macroforecast/scaffold/option_docs/l7.py`` are welcome.

### `transformation_attribution`  --  operational

Shapley over pipelines -- decompose forecast skill across alternative L3 transforms.

Multi-cell sweep aggregator: given multiple pipelines that differ in their L3 transform choices, computes the Shapley share of each transform's contribution to the metric improvement. v0.25 uses the Castro-Gómez-Tejada (2009) permutation-Shapley sampler when ``n_pipelines > 8``.

**When to use**

Interpreting horse-race sweeps -- which L3 transform delivers the win?

**When NOT to use**

Single-pipeline studies; sweeps with fewer than 3 alternative pipelines.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Castro, Gómez & Tejada (2009) 'Polynomial calculation of the Shapley value based on sampling', Computers & Operations Research 36(5): 1726-1730.

**Related options**: [`lineage_attribution`](#lineage-attribution), [`group_aggregate`](#group-aggregate)

_Last reviewed 2026-05-05 by macroforecast author._
