# v0.8.9 verification plan

Honesty-pass audit of paper-method support claims **before** the v0.8.9
cut. Each verification item reads the package's runtime implementation
of a claimed-operational method and compares it against the published
procedure in the paper. Items split into V1 (just-promoted in v0.8.9
B-1..B-6) and V2 (claimed-operational since earlier versions).

The audit is **gating**: any item marked ❌ ("does not match published
procedure") triggers either (a) demotion to ``future`` with a clear
honesty-pass entry, or (b) inline fix before cut. ⚠️ ("partial match")
items get a CHANGELOG note about scope and a v0.9.x follow-up issue.

## V1 -- v0.8.9 Phase 1 promotions (just landed)

| # | Item | Verification approach | Expected outcome |
|---|---|---|---|
| V1.1 | `ridge.coefficient_constraint=nonneg` (B-1) | Augmented-system math: `min ‖y - Xβ‖² + α‖β‖²` ≡ `min ‖[y;0] - [X; √α·I]β‖²`, so NNLS on the augmented system = constrained ridge. Verify NaN handling, intercept centring | ✅ closed-form correct |
| V1.2 | `dual_decomposition` linear (B-3) | Representer identity: ridge `ŷ = X(X'X+αI)⁻¹X'y = W y`. Compute W on synthetic data, verify W @ y matches `Ridge.predict(X)` (centring excluded) | ✅ identity holds in ridge case |
| V1.3 | `bagging.strategy=block` (B-4) | Künsch (1989) moving-block invariants: (a) each block is `block_length` consecutive integers (mod n), (b) variance reduction at least matches plain bagging on AR(1) data | ✅ both invariants pass |
| V1.4 | `blocked_oob_reality_check` (B-5) | White (2000) + block extension: (a) type-I error ~α=0.05 against equal-skill benchmark over 100 sims, (b) power = 1.0 against clear-winner over 100 sims, (c) recentring d_bar - boot_d_bar correctness | ✅ size + power |
| V1.5 | `asymmetric_trim` (B-6) | Per-row sort: trivially correct (np.sort axis=1). Smoothing: centred rolling mean. Verify against Maximally FL §3 description ("3-month MA centred") | ✅ matches |

## V2 -- claimed-operational since earlier releases

| # | Item | Paper / source | What to verify |
|---|---|---|---|
| V2.1 | `scaled_pca` op | Huang/Jiang/Li/Tong/Zhou (2022) | Read `_scaled_pca` in runtime; verify (a) predictive slope scale per predictor j (`sₘ_j = Σₜ yₜ₊₁·xₜⱼ / Σₜ xₜⱼ²`), (b) PCA on scaled matrix (X · diag(s)). The two-step procedure must match paper's eq (1)-(2). |
| V2.2 | `macroeconomic_random_forest` family (v0.2 #187) | Coulombe (2024) SSRN 3633110 | Read `_MRFWrapper`; verify (a) per-leaf local linear regression on (X, y) at each leaf, (b) random walk kernel / Olympic podium for time-variation regularization, (c) Block Bayesian Bootstrap (Taddy 2015 ext.) for confidence intervals. CLAUDE.md notes v0.1 was a plain RandomForest; v0.2 #187 should have promoted it -- **verify the v0.2 implementation actually has all three pieces, not just per-leaf linear**. |
| V2.3 | `var` + `historical_decomposition` + `generalized_irf` + `fevd` | Coulombe & Goebel (2021) JCLI-D-20-0324 | Verify (a) VAR(p) lag selection scheme matches paper, (b) IRF identification (Cholesky vs Pesaran-Shin generalized), (c) FEVD normalization. Per-paper schema check. |
| V2.4 | `dfm_mixed_mariano_murasawa` family (v0.25 #245) | Mariano & Murasawa (2003) | Read `_DFMMixedFrequency`; verify Kalman state-space EM (not PCA approximation). Sea Ice DFM paper PDF not in second_brain so we verify against the generic Mariano-Murasawa procedure. |
| V2.5 | `sparse_pca` op | Zhou (et al.) Sparse Macro Factors | Read `sparse_pca` runtime; verify it matches sklearn `SparsePCA` (which is Zou-Hastie-Tibshirani 2006) -- this is the standard construction. If Zhou's paper uses a different sparsity scheme (e.g. group sparsity), document the discrepancy. |

## Pass criteria

For each V item:

* ✅ **Pass** — runtime matches the published procedure end-to-end.
* ⚠️ **Partial** — core idea correct but at least one piece is approximated
  or not implemented faithfully. **Action**: CHANGELOG note documenting
  the scope; v0.9.x follow-up issue.
* ❌ **Fail** — runtime does not match the published procedure.
  **Action**: demote to ``future`` with honesty-pass entry; cannot cut
  v0.8.9 with this status remaining.

Already known from paper-method audit (independent of code reading):

* ⚠️ #10 Maximally FL Core Inflation: B-1 `nonneg_ridge` is generic;
  Albacore_comps requires `λ‖w − w_headline‖²` shrinkage-to-headline +
  sum-to-1; Albacore_ranks requires `λ‖Dw‖²` fused-ridge. Both deferred
  to v0.9.x as new sub-axis values (`ridge.prior=fused_difference` and
  `ridge.prior=shrink_to_target`). CHANGELOG B-1 entry will note this
  primitive vs full-Albacore distinction.

* ❌ #8 Anatomy: MAS (Model Accordance Score) and MAS hypothesis testing
  not in v0.9.x roadmap. Action: extend v09 plan doc Phase 4 (anatomy
  adapter) to include MAS as a third op alongside oshapley_vi / pbsv.

## Execution order

V1.1 → V1.2 → V1.3 → V1.4 → V1.5 (just-promoted, fastest), then  
V2.5 → V2.3 → V2.1 → V2.4 → V2.2 (existing ops, ascending complexity).

V2.2 (MRF) is the riskiest -- the v0.2 #187 promotion claim is the
strongest in the codebase, but the CLAUDE.md note suggests v0.1 was
demoted as a plain RandomForest, and the v0.2 promotion may not have
implemented all three pieces (per-leaf linear + RW kernel + block
Bayesian bootstrap). If V2.2 fails, MRF gets demoted with a #2
honesty-pass entry.

## Output artefact

Per-item verification result lands in
``docs/architecture/v089_verification_results.md`` (created by V-Plan
on the first execution and appended to as items complete). Format per
item:

```
### V1.1 ridge.coefficient_constraint=nonneg
**Status**: ✅ Pass
**Evidence**: ...read of _NonNegRidge code...
**Discrepancies**: none
**Action**: none
```
