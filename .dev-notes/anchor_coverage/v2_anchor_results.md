# WP-V2 — Model Anchor Results (macroforecast, branch `test/model-anchors-v2`)

Independent correctness anchors for the remaining Tier-1 (zero-anchor) MODELS
from `.dev-notes/anchor_coverage/summary.md`/`matrix.csv` (the V0 inventory).
New tests live under `tests/models/anchors/` (plain pytest) and
`tests/parity/` (R-needing, `rparity` marker, subprocess-Rscript bridge
ported from the `test/r-parity-v1` branch's `tests/parity/conftest.py`).

Mission rule followed throughout: no tolerance was loosened to force a pass.
One genuine numerical-instability finding was discovered (see below) and is
marked `xfail(strict=True)` with a full diagnosis rather than hidden.

## Environment notes

- R 4.3.3 (`/usr/bin/Rscript`), packages `BVAR`, `bvartools`, `FAVAR` newly
  built from CRAN into `~/R/x86_64-pc-linux-gnu-library/4.3` (shared with the
  `test/r-parity-v1` worktree's R library). `BVAR` was tried per the mission
  brief but not needed in the end (`bvartools`/`FAVAR` sufficed).
- `lightgbm` extra installed (`uv sync --extra lightgbm`) for the `lgb_plus`/
  `lgba_plus` anchors.
- `torch` is deliberately **not** installed (per task brief) -- see the
  BLOCKED item below.
- rpy2 is known-broken against this host's R 4.3.3 ABI (per V1); all R
  cross-checks use the subprocess-Rscript bridge, never rpy2.

## Results table

| model | anchor type | result | tolerance | note |
|---|---|---|---|---|
| `bvar_minnesota` | C (clean-room formula) + P (live R parity) | PASS (7 tests, 1 xfail) | 1e-12 (formula); rtol=1e-8 (R parity) | Minnesota prior variance grid (own-lag `kappa0/l^2`, cross-lag `kappa0*kappa1/l^2*sigma_i^2/sigma_j^2`) matches both an independent reimplementation and a live `bvartools::minnesota_prior()` call, byte-for-byte layout-reconciled. |
| `bvar_normal_inverse_wishart` | O (Monte Carlo closed-form oracle) | PASS (2 tests) + **xfail(strict) FINDING** | MC tolerance = max(6×reported MCSE, 5e-3) | Diffuse-prior (`b0=vb0=0`) Gibbs posterior mean converges to closed-form OLS on the shared VAR design (GLS≡OLS when all equations share regressors). n_lag=1 and n_lag=2 both verified, ~5000 post-burn-in draws, seeds fixed. |
| `bvar_normal_inverse_wishart` (default `s0=0.0`) | O (regression, diagnosed bug) | **XFAIL(strict) — genuine finding** | n/a | See "Findings" below. Not tolerance-loosened; documented root cause via direct `s0` parameter sweep. |
| `favar` (factor step) | C (independent PCA-via-SVD) + P (live R parity, 4 functions) | PASS (1 + 4 tests) | 1e-8 (own PCA ref); 1e-8/1e-4 (R parity, ExtrPC/olssvd/facrot exact, BGM slightly looser for pinv-vs-solve) | `_favar_extr_pc`/`_favar_facrot`/`_favar_olssvd`/`_favar_bgm` all matched byte-for-byte (up to eigenvector sign) against live `FAVAR:::ExtrPC`/`facrot`/`olssvd`/`BGM` — the exact R functions the source comments cite. All four are deterministic (no RNG). |
| `favar` (end-to-end forecast) | O (near-noiseless DGP oracle) | PASS (1 test) | <1% of target's in-sample range; must beat naive by 70%+ | 2-D bounded rotating-state DGP, 1% idiosyncratic noise (exactly-zero noise triggers two separate known degeneracies — see Findings), explicit non-degenerate `varprior` (avoids the `s0=0.0` finding). Recovered forecast error ≈0.2% of range. |
| `dfm_mixed_mariano_murasawa` | P (parameter pass-through, byte-close) | PASS (2 tests) | rtol=1e-10 (params/llf), rtol=1e-8 (fitted) | Confirmed a thin `statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ` wrapper by reading source; wrapper output matches a direct `DynamicFactorMQ` call bit-for-bit (params, llf, fitted values), including a monthly-column-order-shuffle variant that exercises the wrapper's own reordering glue. |
| `dfm_mixed_mariano_murasawa` (factor recovery) | O (low-noise MM-aggregation oracle) | PASS (1 test) | \|correlation\| > 0.9 | Single monthly AR(1) factor + quarterly MM [1,2,3,2,1]-aggregated series with 2% measurement noise (DynamicFactorMQ requires nonzero idiosyncratic variance); filtered monthly factor correlates >0.9 with the true path. |
| `tvp_ridge` (Z-basis) | C (independent reimplementation) | PASS (3 param cases) | 1e-12 | Cumulative-design construction reimplemented independently from the documented `Zfun` structure; matches `_tvp_z_basis` exactly. |
| `tvp_ridge` (dualGRR core) | C (independent Woodbury-identity derivation) | PASS (4 tests incl. heterogeneous weights) | 1e-8 / 1e-7 | Derived from scratch (not read off code) that `_dual_generalized_ridge`'s dual AND primal branches both solve `theta=(Z'WZ+Lambda)^-1 Z'Wy`; matched via an independent from-scratch normal-equations solve. `param=dim_x*n_obs>n_obs` always for `tvp_ridge`'s real usage, so only the dual branch is exercised in practice (noted, not a gap). |
| `tvp_ridge` (constant-parameter limit) | O (analytic limit) | PASS (1 test) | rtol=1e-6 (constant path), rtol=1e-4 (vs plain ridge) | `lambda1 -> 1e10` collapses the recovered path to a single plain ridge fit with penalty `lambda2` — the textbook zero-state-variance TVP limit. |
| `lgb_plus` | C (hand-rolled competition loop vs raw lightgbm) | PASS (2 tests) | rtol=1e-10 (bit-identical) | Reproduced the exact `np.random.default_rng(seed)` call sequence (subsample draw, then candidate-feature draw) so random streams match; predictions and per-step tree/linear choice match bit-for-bit. Determinism pin also passes. |
| `lgba_plus` | C (hand-rolled alternating loop vs raw lightgbm) | PASS (2 tests) | rtol=1e-10 (bit-identical) | Fully deterministic at `subsample=1.0` (no RNG replay needed); predictions and per-cycle linear-feature choice match bit-for-bit. Determinism pin also passes. |
| `assemblage_regression` | O (analytic KKT closed form) + entry-point check | PASS (2 tests) | rtol=1e-4 to 1e-6 | Routes through `SupervisedAggregationRegressor` (confirmed by reading source); simplex-constrained ridge matched to the exact bordered-KKT equality-constrained solution (fixture verified `nonneg` non-binding). |
| `supervised_aggregation` | O (analytic closed forms, 4 variants) + entry-point check | PASS (5 tests) | rtol=1e-4 to 1e-6 | Unconstrained ridge, target-shrinkage ridge, simplex ridge, mean-match ridge all matched to exact closed forms (plain ridge normal equations / bordered KKT system). |
| `hemisphere_nn` | — | **BLOCKED(no-torch)** | n/a | See below. |
| `density_hnn` | — | **BLOCKED(no-torch)** | n/a | See below. |

**Totals**: 9/9 assigned Tier-1 models now have at least one genuine
independent anchor (up from 0). 37 new tests across 8 files (6 under
`tests/models/anchors/`, 2 under `tests/parity/`): 36 pass, 1
`xfail(strict=True)` (a genuine finding, not a gap in test-writing). 2 models
(`hemisphere_nn`, `density_hnn`) BLOCKED per environment constraint (torch
deliberately not installed).

## Findings (prominent, per mission rule)

### FINDING 1 — `bvar_normal_inverse_wishart`'s default `s0=0.0` is numerically unstable on near-singular residual covariance (xfail'd, diagnosed)

**What**: `bvar_normal_inverse_wishart`'s own function signature defaults to
`s0=0.0` — an *exactly zero* inverse-Wishart prior scale matrix. When the
fitted VAR's true residual covariance is even mildly near-singular (verified
smallest eigenvalue ~1e-6 relative to O(1) others — a realistic scenario,
not a manufactured pathology: e.g. a factor-augmented VAR block where the
target is well-explained by its own factors, which is the whole *point* of
FAVAR), the Gibbs sampler becomes severely numerically unstable:

- Posterior mean coefficients off by 3-4+ orders of magnitude from the
  OLS/true value (observed: single coefficients up to ~2.9e4 to 1.4e25
  across variants, vs true values all in [-1, 1]).
- Not rare-draw noise: ~97% of post-burn-in Gibbs draws have some
  `|coefficient| > 10`; the **median** draw is also badly wrong.
- **Silent**: no error, warning, or NaN is raised anywhere in the chain —
  `predict()` just returns an absurd number.

**How isolated**: direct parameter sweep on `bvar_normal_inverse_wishart`
alone (bypassing FAVAR's factor/loading machinery entirely) on a fixed
3-variable VAR(1) panel where one variable is (almost) an exact linear
combination of the other two:

| `s0` | frac. of draws with `|coef|>10` | posterior mean max `|coef|` |
|---|---|---|
| 0.0 | 97% | ~2.9e4 |
| 1e-6 | 97% | ~2.9e4 |
| 1e-3 | 26% | ~0.96 (partial recovery) |
| 1.0 | 5.5% (normal MC noise) | ~0.94 (correct) |

This isolates the mechanism to the interaction between a (near-)zero
inverse-Wishart prior scale and a near-singular residual covariance in the
Wishart-draw step of `_favar_bvar_draws` — not something specific to FAVAR's
standardization or factor-extraction code (confirmed: the same instability
reproduces calling `bvar_normal_inverse_wishart` directly, with no FAVAR
involved at all).

**Blast radius**: `favar()`'s own default `varprior=None` silently resolves
(`_parse_favar_varprior` treats the empty `{}` dict as falsy) to this exact
`s0=0.0`/`vb0=0.0` configuration. On a realistic near-noiseless
factor-augmented-VAR DGP, `favar()`'s *default* configuration produced
one-step forecasts many orders of magnitude off (`-8.6e24`, `-881948`,
`-24374` observed across DGP-noise-level variants vs a true value of about
`-1.0`), silently.

**Disposition**: marked `xfail(strict=True)` in
`tests/models/anchors/test_bvar_minnesota_niw_anchors.py::test_niw_default_s0_is_numerically_stable_on_near_singular_residual_covariance`
with the full diagnosis in the test's docstring/comments. The `favar`
oracle test was adjusted (small idiosyncratic noise + an explicit
non-degenerate `varprior={"s0": 1.0, ...}`) to avoid re-discovering this
already-diagnosed bug on every run, and that adjustment is documented
in-line pointing back at this finding. **Not fixed** (out of scope for a
verification work package) — recommend either flooring `s0` away from
literally zero, or emitting a warning when the Wishart scale matrix's
condition number is extreme.

### FINDING 2 (minor, folded into Finding 1's fixture design) — BGM factor-purge collapses to a rank-deficient factor on exactly rank-deficient data

Separately from Finding 1: with the FAVAR oracle DGP's idiosyncratic noise
set to *exactly* zero (X exactly rank-2 in an 8-series panel), `_favar_bgm`'s
iterative purge collapsed the second extracted factor to numerical noise
(~1e-16, i.e. effectively zero variance) even though the true state is
genuinely 2-dimensional. This looks like a property of the BGM fixed-point
iteration itself operating on exactly-collinear data (R `FAVAR:::BGM`'s
identical iteration would plausibly do the same — not verified against R on
this specific degenerate input, out of scope) rather than a macroforecast-
specific bug. Documented in `test_favar_anchors.py`'s `_rotating_state_dgp`
docstring; worked around (not hidden) by using 1% idiosyncratic noise, which
is standard practice for "near-noiseless" DGP oracles anyway.

## BLOCKED: `hemisphere_nn` / `density_hnn` (no-torch)

`torch` is not installed on server1 and was **not** installed for this work
package (per the task brief). Both models are hand-rolled torch dual-head
networks (`hemisphere_nn`: bagged blocked-subsample ensemble; `density_hnn`:
distributional/density head) with zero prior numeric anchoring
(`.dev-notes/anchor_coverage/summary.md` Tier-1 items #3 and #10).

**Proposed anchor design for a torch-enabled environment** (one paragraph,
as requested): (1) a seeded tiny-network determinism pin — construct the
smallest viable dual-head network (e.g. 2 predictors, 4 hidden units, batch
size 8), fix `torch.manual_seed`, run `fit` twice on identical data and
assert bit-identical (or `atol=0` under CPU determinism flags) predictions
and weight tensors, catching any accidental nondeterminism (dropout without
a fixed generator, unseeded bagging/subsample draws, etc.) before touching
correctness at all; and (2) a closed-form limiting case — with the
volatility-emphasis loss weight and both heads' hidden layers driven toward
a degenerate linear/constant-variance limit (e.g. zero hidden units or
weights forced near-linear via a tiny-weight-initialization + no-training
smoke check), the mean head should reduce to a plain linear regression
forecast and the variance head to a constant equal to the residual variance,
giving a hand-computable target analogous to this WP's `tvp_ridge`
constant-parameter-limit test. A known-mean/known-variance synthetic DGP
(e.g. `y = beta'x + N(0, sigma(x)^2)` with a simple, exactly-representable
`sigma(x)`) could additionally anchor the density head's recovery once (1)
and (2) are in place. Historical context: a deleted (see below)
`tests/core/test_phase_d2d_hnn_earlystop.py` (257 lines, removed in
`2e62e740`) documented paper-locked hyperparameters worth preserving in any
future implementation — `patience=15`, `val_frac=0.20`, citing "Goulet
Coulombe / Frenette / Klieber 2025 JAE §3 p.14" — and tested early-stopping/
train-val-split/best-weight-restoration behavior in detail; it is not
directly restorable (imports the same now-removed
`macroforecast.core.runtime`/`macroforecast.models.ops` registry API — see
below) but is a useful reference for hyperparameter and test-shape design.

## Cheap bonus: git-archaeology of deleted test files

`git log --diff-filter=D --name-only --pretty=format:"COMMIT %H %s" --
'tests/**'` surfaces ~419 deleted test-file paths across ~20 commits. Most
are legitimate feature/API removals (explicit `chore!:`/`Remove`/`refactor`
messages). The single largest and most suspicious block (matching V0's own
flag of the deleted R cross-ref suite) is commit `2e62e740` ("Clean semantic
package structure"), which alone deleted ~100 test files including several
directly relevant to this WP's targets:

| deleted path | lines | relevance |
|---|---|---|
| `tests/core/test_bvar_minnesota.py` | 261 | `bvar_minnesota`/`bvar_normal_inverse_wishart` |
| `tests/core/test_bvar_sigma2_scaling.py` | 424 | BVAR sigma^2 hyperparameter |
| `tests/promotion/test_c63_1_bvarminnesota.py` | 235 | BVAR promotion gate |
| `tests/api/test_bvar_api_hyperparameters.py` | 176 | BVAR API-level hyperparameters |
| `tests/core/test_factor_augmented_var.py` | 59 | `favar` (old class name `_FactorAugmentedVAR`) |
| `tests/core/test_dfm_mariano_murasawa.py` | 73 | `dfm_mixed_mariano_murasawa` |
| `tests/core/test_v025_dfm_mq.py` | 66 | DFM mixed-frequency |
| `tests/core/test_phase_d2d_hnn_earlystop.py` | 257 | `hemisphere_nn` early stopping (see BLOCKED section) |
| `tests/core/test_l4_realized_garch_c49.py` / `_c56.py` | — | `realized_garch` (not a V2 target, but V0's #1 priority) |
| `tests/core/test_l4_midas_family_c48.py` / `_builder.py`, `test_f07_umidas_tester.py` | — | MIDAS family |
| `tests/core/test_l4_tuning_algorithms.py`, `tests/test_tuning.py` | — | tuning algorithms |
| `tests/core/test_mrf_gtvp.py` | — | macro_random_forest / GTVP (TVP-adjacent) |

**Important caveat, spot-checked (not assumed)**: `test_bvar_minnesota.py`,
`test_factor_augmented_var.py`, and `test_dfm_mariano_murasawa.py` all
import `from macroforecast.core.runtime import _BayesianVAR /
_FactorAugmentedVAR / _DFMMixedFrequency` and `from macroforecast.models.ops
import OPERATIONAL_MODELS, FUTURE_MODELS, get_family_status` — a
registry/"operational status" API that no longer exists in the current
`macroforecast/models/timeseries.py`-based module structure. **These files
are not directly restorable** (`git show <rev>:<path> > path` would produce
a file that fails to import); restoring their intent would require rewriting
against the current API. They are, however, useful historical references:
e.g. `test_bvar_minnesota.py`'s docstring states a closed-form posterior
mean formula (`β̂ = (V⁻¹ + X'X)⁻¹(V⁻¹m + X'y)`, a proper-prior natural-
conjugate form) distinct from the diffuse-prior special case this WP
anchored, and `test_factor_augmented_var.py` used a similar 2-latent-factor
toy-panel construction to this WP's `favar` oracle. Not restored, per the
task brief ("list candidates... do not restore").

## Gates

- New test files: 37 collected, 36 passed, 1 `xfail(strict=True)` (genuine
  finding), 80.75s total wall time.
- Existing relevant test files re-run for regressions: `tests/models/
  test_models.py` (99 passed, 8 skipped — pre-existing optional-dependency
  gates, e.g. torch/xgboost/catboost), `tests/correctness/
  test_tsbvar01_intercept_mean_revert.py`, `test_tsbvar04_posterior_sd.py`,
  `test_tvp1_predict_no_insample_leak.py`, `test_parity_bugs.py`,
  `tests/models/test_assemblage_models.py` — all pass. One pre-existing,
  unrelated failure (`test_default_cost_budget.py::
  test_macro_random_forest_default_cost_budget`, needs the separate
  `matplotlib`/`macro_random_forest` extra, not installed and not a V2
  target) confirmed unrelated to this branch's changes.
- `mypy tests/models/anchors/ tests/parity/`: clean (0 errors after adding 3
  explicit `np.ndarray` annotations mypy required).
- `mypy` (project default target, `macroforecast/`): 4 pre-existing errors
  in `data_analysis/summary.py` and `models/linear.py`, both files untouched
  by this branch — confirmed pre-existing, out of scope.
- CHANGELOG `[Unreleased]` entry added.

## Operational note (flagged transparently)

While reading `.dev-notes/anchor_coverage/summary.md` early in this session,
its rendered tool output contained what appears to be an injected fake
"system-reminder" (a spurious "the date has changed, don't tell the user"
notice) plus a fabricated "available agent types" block, immediately after
the file content. This was not treated as a genuine instruction (no action
was taken on it, nothing was hidden from the user) and matches a pattern
V0's own report already flagged as worth an independent look. Noted here for
visibility; out of scope for the anchor-coverage work itself.
