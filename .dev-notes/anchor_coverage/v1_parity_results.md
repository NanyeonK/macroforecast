# WP-V1 — R-parity harness results (`tests/parity/`)

Executed on server1, R 4.3.3 (`/usr/bin/R`), packages installed to a user
library (`~/R/x86_64-pc-linux-gnu-library/4.3`). Bridge: subprocess
`Rscript` (not `rpy2` -- see `tests/parity/README.md` for why). Full suite:
`pytest tests/parity/ -m rparity` -> **35 passed, 2 xfailed (strict), 0
failed** in ~44s.

| # | item | reference | result | tolerance | note |
|---|---|---|---|---|---|
| 1a | `realized_garch` | `rugarch::ugarchfit(model="realGARCH")` | **PASS** (2 seeds) | atol=0.01/param, 9/9 params | Restored deleted XR-3 (C59). Exact 1:1 term correspondence confirmed from rugarch's own vignette (eq. 46-47). Observed max diff ~0.002-0.004 across seeds. Promotes `realized_garch` from zero-anchor Tier-1 to tightly-verified P-anchor. |
| 1b | `restricted_midas`/`midas_almon` | `midasr::midas_r(weight_method="nealmon")` | **PASS** (`restricted_midas`, 2 seeds + fitted values); documented divergence (`midas_almon`) | atol=0.01/param, atol=0.01 fitted | Restored deleted XR-2 (C59). `restricted_midas` (joint-NLS estimator) is the genuinely comparable callable, not `midas_almon` (fixed-theta+ridge, explicitly documented as architecturally different in its own docstring). Both confirmed empirically. |
| 1c | Boruta (`boruta_selection`) | `Boruta::Boruta()` | **BLOCKED** | n/a | R package `Boruta` failed to build: transitive dependency `fru` requires `cargo`/Rust, not present on server1. Out of scope for this WP to install a system Rust toolchain. Not restored. |
| 2a | HAC kernel `acf`/`parzen` | `sandwich::kernHAC` | **PASS** | abs=1e-6 | Includes small-n edge (n=5, bandwidth=4) and the 3 horizon-implied bandwidths (h=1/4/8). |
| 2b | HAC kernel `bartlett` | `sandwich::kernHAC` | **MISMATCH -> xfail(strict=True)** | n/a | SUSPECTED BUG: `_long_run_variance`'s bartlett branch uses `1-k/(bandwidth+1)` (Newey-West 1987) vs sandwich/its own acf+parzen siblings' `1-k/bandwidth` (Andrews 1991). Confirmed to full double precision, not a numerical artifact. Does NOT affect the public `dm_test(kernel="bartlett")` path (bandwidth there is always `horizon-1`, which coincidentally matches `forecast::dm.test`'s own convention -- see item 3). |
| 2c | HAC kernel `andrews` | n/a (pure Python) | **CRASH, confirmed as bug** | n/a | `_long_run_variance(..., kernel="andrews")` ALWAYS raises `ValueError: unknown HAC kernel 'newey_west'` -- reachable from `dm_test`, `gw_test`, `harvey_newbold_test` (confirmed), likely `clark_west_test`/`cw_test`/`enc_t_test` too. Root cause: `kernel = "newey_west"` reassignment doesn't match the `"bartlett"`-spelled branch below. |
| 3 | `dm_test` | `forecast::dm.test` | **PASS** (6/6: h={1,4} x kernel x power={1,2}) | rel=1e-6 | Both HLN-corrected AND `correction="none"` (derived by dividing out R's own HLN factor `k`) match statistic and p-value. |
| 4a | `_mcs_loss_differences`/`_mcs_statistic` (deterministic core) | `MCS:::GetD` | **PASS** | abs=1e-10 | Exact match, no bootstrap involved on either side. |
| 4b | `model_confidence_set` (survivor set) | `MCS::MCSprocedure` | **PASS** | set equality | Fixed block length + large B; compares final included/excluded SETS, not exact p-values (bootstrap RNGs are independent across languages, by design per the WP-V1 brief). |
| 4c | `model_confidence_set` (elimination order) | `MCS::MCSprocedure` | **N/A -- not a well-posed cross-language invariant** | n/a | Discovered (not assumed) by reading the installed `MCS::MCSprocedure` R source: its `@Info$excluded` order is the FINAL table sorted by ascending cumulative MCS p-value, not the raw elimination sequence; under the separation needed for a robust survivor set, per-step p-values collapse to 0 for multiple models and R's `order()` then breaks ties by original column position -- confirmed by varying R's bootstrap seed 5x with zero change in "order." Documented in `test_mcs.py`, not asserted. |
| 5 | `_berkowitz_density_test` | clean-room R (PIT->qnorm->AR(lags)->chi2 LR, Berkowitz 2001) | **PASS** (2/2, incl. negative control) | rel=1e-3 (LR/p-value), decision agreement | No canonical CRAN package implements Berkowitz (2001) directly; clean-room R port per the WP-V1 brief. |
| 6a | `gaussian_nll`/`log_score`/`negative_log_score` | `scipy.stats.norm.logpdf` (independent oracle) | **PASS** (3/3) | rel=1e-12 | |
| 6b | `qlike`/`smape` | hand-computed toy values | **PASS** | exact | Includes the documented sMAPE 0-200 ceiling case. |
| 6c | `crps` | `scoringRules::crps_norm` (closed form) | **PASS** | rel=1e-6 | |
| 6d | `crps` | `scoringRules::crps_sample` (20k-draw Monte Carlo) | **PASS** | abs=0.02 | Looser tolerance justified by O(1/sqrt(N)) MC error. |
| 7 | `mars` | R `earth` | **PASS** (both: recovery + cross-agreement) | rmse<0.1*std; max_abs_diff<15%*range, corr>0.98 | Lowest-priority/time-permitting item. Coefficients not compared (different heuristics per macroforecast's own docstring disclaimer); prediction-parity only, per the WP-V1 brief. |

## R packages

Installed to `~/R/x86_64-pc-linux-gnu-library/4.3` (default
`/usr/local/lib/R/site-library` is root-owned/read-only on this host):
`forecast`, `rugarch`, `midasr`, `MCS`, `sandwich`, `lmtest`, `scoringRules`,
`earth` -- all load cleanly (`requireNamespace(..., quietly=TRUE)` verified
individually). `Boruta` failed to build (`fru` dependency needs `cargo`/Rust,
absent on this host) -- not restored, not in the core work-item list either.
`lmtest` was installed per the brief's package list but no test in this WP
ended up needing it directly (kept installed for future parity work).

## Bridge

`rpy2` (3.6.7, the only wheel `uv pip install rpy2` resolves) fails to
import against this host's R 4.3.3 in both API mode (`undefined symbol:
R_getVar`) and ABI mode (`undefined symbol: R_ParentEnv`), even with
`R_HOME`/`LD_LIBRARY_PATH` set explicitly to the R install directory. This
is an rpy2-cffi/R-ABI version mismatch, not a missing-library problem.
Switched to a subprocess-`Rscript` bridge (`tests/parity/conftest.py`,
`run_rscript()`): write small CSVs to a temp dir, run a short R script
synchronously with a bounded timeout, parse a `key=value` results file
back. No ABI dependency; the exact R call is plain text in each test.
