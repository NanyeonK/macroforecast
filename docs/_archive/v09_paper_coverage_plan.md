# v0.9.1 paper-coverage promotion plan (single published cut)

This document is the promotion roadmap for the algorithmic implementations
v0.8.9 registered as `future`. **All work is bundled into the single
published cut v0.9.1**; the 5 internal dev-stage labels v0.9.0A through
v0.9.0E below are working tags for staged commits during the v0.9.1
development cycle. None of v0.9.0A-E ship to PyPI separately.

**Dev stage map**:

| Stage | Scope | Effort |
|---|---|---|
| **v0.9.0A** | Errata patch (E1-E4) + paper-10 doc cross-ref + encyclopedia regen | 1 d |
| **v0.9.0B** | Tier 2 batch: 2SRR / Booging / SGT / Maximally FL G1+G2 / dual non-linear | 12-15 d |
| **v0.9.0C** | Tier 3 + Sparse Macro G3: AlbaMA / HNN / sparse_pca_chen_rohe / supervised_pca | 10-12 d |
| **v0.9.0D** | anatomy adapter Path B (degraded — uses final-window fit) | 2-3 d |
| **v0.9.0E** | anatomy adapter Path A (faithful) + L4PerOriginModelsArtifact | 2 d |
| **v0.9.1** | Published cut bundling A-E | (cut + tag + PyPI publish) |

The plan is grouped into Tier-based milestones reflecting implementation
effort and risk:

The grouping reflects implementation effort and risk:

* **Tier 1** -- closed-form / scipy-based; promotion is a thin wrapper
  with a known answer for testing. Lands first.
* **Tier 2** -- custom algorithm in pure NumPy; needs paper sections
  for the algorithmic detail but no learned components.
* **Tier 3** -- learned algorithms requiring deep PDF read and
  potentially custom architectures (HNN). Heaviest.
* **Tier 4** -- external library integration (`anatomy` for oShapley-
  VI / PBSV). Light implementation but depends on adapter design.

Phase 0 (PDF deep-read of the 8 paper algorithm sections) precedes any
algorithmic implementation. PDFs are already in second_brain at
``wiki/raw/papers/`` (verified 15/16 in the v0.8.9 audit). The
``graphify`` skill is the recommended tool for pseudocode extraction.

## Phase 0 -- PDF deep-read (selectable)

| Item | PDF (in second_brain) |
|---|---|
| Booging | `arXiv:2008.07063` |
| AlbaMA | `arXiv:2501.13222` §3 |
| 2SRR | `10.1016/j.ijforecast.2024.08.006` §3 |
| HNN | `Philippe_Goulet_Coulombe_JAE_version.pdf` |
| Borup oShapley-VI / PBSV | `10.2139ssrn.4278745.pdf` (cross-check with `anatomy` package source) |
| Dual Interpretation | `arXiv:2412.13076` representer formulas |
| Maximally FL Core | `Maximally Forward-Looking Core Inflation.pdf` |
| Slow-Growing Trees | `10.1007978-3-031-43601-7_4.pdf` |

Pseudocode + algorithm sketches land in
``docs/replications/<paper>_algorithm_notes.md`` ahead of code work.

## Tier 1 closed-form / scipy (already shipped in v0.8.9)

| # | Item | Algorithm sketch | Files | Est |
|---|---|---|---|---|
| 1 | `ridge.coefficient_constraint=nonneg` | `scipy.optimize.nnls` on augmented system `[X; sqrt(α)·I] β = [y; 0]`, β ≥ 0 | `core/runtime.py` ridge dispatch | 2 h |
| 2 | `dual_decomposition` (linear families) | Ridge: wₜ = xₜ' (X'X+αI)⁻¹ X'. OLS / Lasso analogues. Output artifact also carries inline portfolio diagnostics (HHI / short / turnover / leverage) from the same paper -- 4 trivial numpy reductions on the dual weights, no separate L7 op | `core/runtime.py` L7 dispatch | 4 h |
| 4 | `bagging.strategy=block` | `_BaggingWrapper` modification: replace i.i.d. bootstrap with consecutive `block_length`-row blocks | `core/runtime.py` `_BaggingWrapper` | 3 h |
| 5 | `blocked_oob_reality_check` (L5) | Block-bootstrap on OOS residuals; reject H0 when median MSE_diff < 0 at α=0.05 | `core/runtime.py` L5 dispatch | 4 h |
| 6 | `asymmetric_trim` (L2) | Per-predictor target-correlation-weighted lower / upper quantile cutoffs; trim observations beyond | `core/runtime.py` L2 path | 4 h |

**Phase 1 quality bar.** Closed-form, so each promotion has a
known-answer test:

* `nonneg`: matches OLS-on-augmented-system when no constraint is binding.
* `dual_decomposition`: ridge prediction matches Σⱼ wⱼ · yⱼ at every
  forecast date. Inline portfolio diagnostics: HHI on uniform weights
  = 1/n; turnover with identical wₜ and wₜ₋₁ = 0.
* `bagging.strategy=block`: variance reduction at least matches plain
  bagging on serially-correlated synthetic data.
* `blocked_oob`: rejects at α=0.05 against trivial-improvement
  benchmark; type-I rate ~0.05 against equal-skill benchmark.
* `asymmetric_trim`: idempotent (re-applying does not change output).

## Tier 2 custom NumPy (dev-stage v0.9.0B, ~12-15 d)

| # | Item | Sketch + risk | Est |
|---|---|---|---|
| 7 | `decision_tree.split_shrinkage` (SGT) | **In-fit** soft-weighted tree per Goulet Coulombe (2024) Algorithm 1: at each split step, observations not satisfying the rule receive weight `(1 − η)` rather than 0; leaf weight propagation `ω_i^l ← ω_i^l · (1 − η · I(rule))`; Herfindahl `H_l ≡ Σ ω_i²` stopping criterion. **sklearn extension is insufficient** (the splits themselves change under soft weights); requires a custom soft-weighted tree implementation (~250-350 LOC). The post-fit leaf-multiplication sketch this entry previously carried was **incorrect** — corrected on the 2026-05-07 audit. | 5-7 d |
| 8 | `bagging.strategy=booging` (Booging) | **Outer B = 100 bags of (intentionally over-fitted) inner Stochastic Gradient Boosting** + Data Augmentation (`X̃ = X + N(0, σ_k/3)`) + per-bag column dropping (20%). Sampling without replacement, fraction 2/3 or 3/4 (paper Appendix A.2). The inner SGB is *not* tuned by CV on `S` — it is over-fit on purpose, and outer bagging plays the pruning role. The pre-2026-05-07 sketch ("outer K rounds: bag a base learner on residuals") **did not match the paper's algorithm**; the schema option name should be `booging` (alias `sequential_residual` retained for back-compat). | 1.5-2 d |
| 9 | `dual_decomposition` (kernel / RF) | Non-linear case: leaf-co-occurrence kernel for RF; explicit kernel for kernel methods. Linear case in Phase 1. | 1-2 d |
| 10 | `ridge.prior=random_walk` (2SRR) | Paper's **closed-form** two-step generalised ridge (Coulombe 2025 IJF Eq. 11): step 1 ridge with RW reparametrisation `Z = WC` (`C_RW` = lower-triangular ones), step 2 generalised ridge with `Ω_θ = diag(σ²_u) ⊗ I_T` and `Ω_ε = diag(σ²_ε)` recovered from step-1 residuals via GARCH(1,1). Two `T × T` matrix inverts, no iteration. **Risk register entry RESOLVED on 2026-05-07 audit** — paper Eq. 11 is unambiguously closed-form. | 2-2.5 d |
| 10a | `ridge.prior=shrink_to_target` (Maximally FL Albacore_comps Variant A) | Goulet Coulombe et al. Eq. 1: `arg min ‖y − Xw‖² + λ‖w − w_target‖²` s.t. `w ≥ 0, w'1 = 1`. New constrained QP slot composing with existing `coefficient_constraint=nonneg`. Implementation via scipy SLSQP (small K) or `cvxpy[clarabel]` (production). When `w_target = 0` and `simplex=False`, reduces to existing `_NonNegRidge` (B-1). **Added 2026-05-07** — was absent from this plan. | 1.5-2 d |
| 10b | `ridge.prior=fused_difference` (Maximally FL Albacore_ranks Variant B) | Goulet Coulombe et al. Eq. 2: `arg min ‖y − Ow‖² + λ‖Dw‖²` s.t. `w ≥ 0`, `mean(y) = mean(Ow)`, where `O = sort(X)` is the rank-space data and `D` is the first-difference operator. λ → ∞ collapses to uniform `1/K`; λ = 0 is sparse OLS. Composes with `asymmetric_trim` (B-6 v0.8.9) for the rank-space transformation and existing `coefficient_constraint=nonneg`. **Added 2026-05-07**. | 1.5-2 d |

## Tier 3 learned algorithms (dev-stage v0.9.0C, ~10-12 d)

| # | Item | Why heavy | Dependency | Est |
|---|---|---|---|---|
| 11 | `adaptive_ma_rf` (AlbaMA) | Novel algorithm: RF determines per-observation window length. Phase 0 produces pseudocode; implementation builds on sklearn RF primitives. | sklearn RF | 3-4 d |
| 12 | `mlp.architecture=hemisphere` (HNN) | Dual-head torch architecture (mean / variance) + emphasis loss tying the heads. Extends the existing `_TorchSequenceModel`. Smoke test on synthetic; full reproduction needs FRED-MD. | `[deep]` extra (already exists) | 3-5 d |
| 13 | `mlp.loss=volatility_emphasis` | HNN constraint loss; lands together with the architecture in #12. | (with #12) | (included) |

**Phase 3 quality bar.** Smoke tests on synthetic data + paper's toy
example reproduction where the PDF includes one. Real-data
reproduction (e.g. HNN on FRED-MD) is a v0.9.4 documentation milestone,
not a runtime gate.

## Tier 4 anatomy adapter (dev-stages v0.9.0D Path B + v0.9.0E Path A; ~2-3 d each)

| # | Item | Sketch | Est |
|---|---|---|---|
| 14 | `oshapley_vi` + `pbsv` | L7 dispatch via single `Anatomy.explain(transformer=...)` call (anatomy 0.1.6 has **no** dedicated `oshapley_vi` / `pbsv` methods — corrected on 2026-05-07 audit). For `oshapley_vi`: default identity transformer → local explanations → mean of `\|values\|` across OOS instances per Borup Eq. 16. For `pbsv`: squared-error transformer → global scalar explanation per Eq. 24. **Path B (v0.9.3)**: use the final-window fitted model for every period; degraded relative to Eq. 11 semantics; surfaces `status="degraded"` warning. **Path A (v0.9.4)**: requires new `L4PerOriginModelsArtifact` to persist per-origin fitted estimators. | 2-3 d (B) + 2 d (A) |

**Adapter sketch (Path B; corrected on 2026-05-07 audit):**

```python
def _l7_anatomy_op(op, model, l3_features, l4_models, l4_train, params):
    try:
        from anatomy import (
            Anatomy, AnatomyModel, AnatomyModelProvider, AnatomyModelOutputTransformer,
        )
    except ImportError as exc:
        raise NotImplementedError(
            "[anatomy] extra missing -- pip install macroforecast[anatomy]"
        ) from exc

    feature_cols = list(model.feature_names)
    y_name = l3_features.y_final.name
    origins = l4_train.forecast_origins
    X_full, y_full = l3_features.X_final.data, l3_features.y_final.data

    def _provider_fn(key):
        origin = origins[key.period]
        win_lo, win_hi = l4_train.training_window_per_origin[(model.model_id, origin)]
        train = X_full.loc[win_lo:win_hi, feature_cols].copy()
        train[y_name] = y_full.loc[win_lo:win_hi]
        test = X_full.loc[[origin], feature_cols].copy()
        test[y_name] = y_full.loc[[origin]]
        # Path B: final-window fit for every period -> status="degraded".
        # Path A (v0.9.4): swap in l4_per_origin_models[(model_id, origin)].
        fitted = l4_models.artifacts[key.model_name].fitted_object
        return AnatomyModelProvider.PeriodValue(
            train=train, test=test,
            model=AnatomyModel(pred_fn=lambda xs: np.asarray(fitted.predict(xs)).ravel()),
        )

    provider = AnatomyModelProvider(
        n_periods=len(origins), n_features=len(feature_cols),
        model_names=[model.model_id], y_name=y_name, provider_fn=_provider_fn,
    )
    anat = Anatomy(provider=provider, n_iterations=int(params.get("n_iterations", 500))) \
            .precompute(n_jobs=int(params.get("n_jobs", 1)))

    if op == "oshapley_vi":
        df = anat.explain()                          # local; mean abs over OOS rows
        per_feat = df.drop(columns="base_contribution").abs().mean(axis=0)
    elif op == "pbsv":
        def _se(y_hat, y):
            return float(np.mean((y - y_hat) ** 2))
        df = anat.explain(transformer=AnatomyModelOutputTransformer(transform=_se))
        per_feat = df.drop(columns="base_contribution").iloc[0]
    else:
        raise ValueError(op)

    return pd.DataFrame({
        "feature": feature_cols,
        "importance": per_feat.reindex(feature_cols).abs().values,
        "coefficient": per_feat.reindex(feature_cols).values,  # signed PBSV / mean Shapley
        "status": "degraded" if op == "pbsv" else "operational",  # path-B downgrade
    })
```

**Path A (v0.9.4)**: introduce `L4PerOriginModelsArtifact` in `core/types.py`
keyed by `(model_id, origin)`; opt-in storage in the L4 walk-forward loop
when the recipe declares `oshapley_vi` or `pbsv`. The adapter switches to
`l4_per_origin.fitted_for(model_id, origin)` and clears the degraded
warning.

## Tier-extra V2.5 follow-through Sparse Macro Factors (bundled into dev-stage v0.9.0C; ~3.75 d)

Added on 2026-05-07 audit. Two new L3 ops alongside the existing `sparse_pca`
(which keeps its sklearn `SparsePCA` semantics).

| # | Item | Sketch | Est |
|---|---|---|---|
| 15 | `sparse_pca_chen_rohe` | Chen-Rohe (2023) Sparse Component Analysis with non-diagonal `D`: `min ‖X − ZDΘ'‖_F` s.t. `Z ∈ S(T,J), Θ ∈ S(M,J), ‖Θ‖_1 ≤ ζ`. Implemented via the equivalent bilinear convex-hull form `max ‖Z'XΘ‖_F` over `H(T,J) × H(M,J)` (Zhou-Rapach 2025 Eq. 4) — alternating SVD-projection of Z + L1-budget projection of Θ. Pure NumPy. Used as the macro-side stage of Rapach & Zhou (2025). | 1.5 d |
| 16 | `supervised_pca` | Giglio-Xiu-Zhang (2025): for each target column, screen panel by univariate correlation, retain top `⌊q·N⌋`, run PCA on the screened sub-panel. Refines Giglio-Xiu (2021) three-pass for weak-factor robustness. Distinct from `partial_least_squares` (NIPALS) and `scaled_pca` (column β). Used as the asset-side stage of Rapach-Zhou (2025) for risk-premium estimation. | 0.5 d |

## Total budget (revised 2026-05-07)

| Dev stage | Scope | Promotions | Calendar |
|---|---|---|---|
| Phase 0 (PDF readthrough) | already done in v0.8.9 audit | 0 | done |
| (Tier 1, shipped in v0.8.9) | nonneg ridge / dual / block bagging / blocked_oob / asymmetric_trim | 6 | done |
| **v0.9.0A** | Errata + paper-10 doc cross-ref | 1 doc patch | 1 d |
| **v0.9.0B** | Tier 2 batch (Booging / SGT / 2SRR / G1 / G2 / dual non-linear) | 6 | 12-15 d |
| **v0.9.0C** | Tier 3 + Tier-extra (AlbaMA / HNN / SCA / SPCA) | 4 | 10-12 d |
| **v0.9.0D** | anatomy Path B | 2 (oshapley_vi, pbsv degraded) | 2-3 d |
| **v0.9.0E** | anatomy Path A | (path B → A) + `L4PerOriginModelsArtifact` | 2 d |
| **v0.9.1** | Single published cut bundling A-E | 13 promotions | (cut/tag/publish) |
| **Total dev work** | | **13** new promotions | **~28-34 person-days full-time** |

Dev stages B / C are independent and can be parallelised. Stage D must
precede E (E requires `L4PerOriginModelsArtifact`). Stage A is a 1-day
standalone patch and unblocks nothing else, but lands first to keep the
roadmap doc honest before any code work begins.

## Risk register (revised 2026-05-07)

* **2SRR closed-form** (Phase 2 #10): ✅ **RESOLVED** on 2026-05-07
  audit. Paper Eq. 11 is unambiguously closed-form (two `T × T` solves,
  no iteration). The "fall back to iterative ridge" mitigation is
  unnecessary.
* **SGT in-fit vs post-fit** (Phase 2 #7): ❌ **CONFIRMED in-fit** on
  2026-05-07 audit. Paper Algorithm 1 explicitly applies the soft
  weight `(1 − η)` during weight propagation at every split step.
  sklearn `DecisionTreeRegressor` cannot reproduce SGT — a custom
  soft-weighted tree implementation is required (~250-350 LOC; effort
  bumped from 1-2 d to **5-7 d**).
* **HNN reproduction fidelity** (Phase 3 #12): synthetic-data smoke
  passes do not guarantee real-data parity with the JAE 2025 paper.
  Real-data reproduction is documented as a v0.9.4 milestone.
  **Additionally on 2026-05-07 audit**: ν proxy from a plain-NN OOB
  MSE roughly doubles training cost per cell — cache the ν estimate
  in the manifest so sweeps over `(architecture, loss)` don't recompute.
* **AlbaMA window selector** (Phase 3 #11): if RF variance is poorly
  controlled, output may not match the paper's smoothness; tuning
  budget is ~1 d on top of the implementation. **Additionally on
  2026-05-07 audit**: paper is silent on per-t refit vs fit-once-with-
  leaf-restriction in one-sided mode. Safe interpretation = per-t refit.
* **anatomy API stability**: `anatomy 0.1.6` is the current PyPI
  release; the adapter depends on `Anatomy.explain(...)` and
  `AnatomyModelProvider` / `PeriodKey` / `PeriodValue` nested-class
  layout. **2026-05-07 audit**: anatomy 0.1.6 has *no* dedicated
  `oshapley_vi` / `pbsv` methods — both metrics are derived from
  `Anatomy.explain(transformer=...)`. Vendor lock to `>=0.1.6,<0.2`.
  CI parity test pins the API surface so a 0.1.7 break is caught.
* **anatomy per-origin refit** (Phase 4 #14): ❌ **NEW risk surfaced
  on 2026-05-07 audit**. Borup paper Eq. 11 requires the per-origin
  fitted model `f̂(·; W_i, h)`. macroforecast's `L4ModelArtifactsArtifact`
  keeps a single fitted_object per `model_id`. Path B (v0.9.3) uses
  the final-window fit and surfaces `status="degraded"`. Path A
  (v0.9.4) introduces `L4PerOriginModelsArtifact` for the faithful
  semantics — opt-in to control the memory cost.
* **Booging schema name** (Phase 2 #8): ❌ **CONFIRMED misnamed** on
  2026-05-07 audit. Paper's algorithm is "outer bagging of inner SGB
  + DA", *not* "bag-on-residuals". Schema option renamed to `booging`
  with `sequential_residual` retained as alias for back-compat.
* **Maximally FL solver** (Phase 2 #10a + #10b): scipy SLSQP for the
  constrained QP is robust but slow on K > 100. Future v0.9.x
  optimisation: switch to `cvxpy[clarabel]` or `osqp` when K is large.
* **Sparse Macro Factors solver** (Phase 5 #15): pure-NumPy alternating
  bilinear maximisation may converge slowly on M > 200. Numba-JIT the
  inner SVD-projection if needed.

## Honesty pass discipline

Every promotion that lands operationally must satisfy the quality bar
for its Tier (closed-form known-answer test for Tier 1; toy-example
reproduction for Tiers 2/3; adapter contract test for Tier 4). Items
that ship as minimum-viable proxies get demoted in the next honesty
pass (PR #163 / v0.2 / v0.25 pattern). The Tier-grouped milestone
structure exists to prevent that pattern from recurring.
