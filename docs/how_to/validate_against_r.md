# Validate Against R Reference Implementations

This guide explains how to run the optional R cross-reference test suite, which
validates key macroforecast estimators against independently maintained R packages.
Use this when you want an independent sanity check beyond the Python unit tests.

## When to use

R cross-reference validation is useful in two situations. First, when you have
promoted a new estimator to `operational` status and want to confirm that its
outputs are consistent with a reference implementation in R. Second, when you
observe unexpected results from a macroforecast estimator and want to rule out
algorithmic divergence from the published method.

The cross-reference tests in `tests/core/test_r_crossref_c59.py` cover three
estimators: `boruta_selection` (L3, vs `Boruta::Boruta()`), `midas_almon` (L4,
vs `midasr`), and `realized_garch` (L4, vs `rugarch`). Small numerical
differences are expected and accepted; the tests check that results agree within
a functional tolerance rather than bit-for-bit identity.

## Install requirements

You need Python, R, and the three R packages installed. On the Python side:

```bash
pip install "macroforecast[validation]"
```

This installs `rpy2>=3.5`, the Python-to-R bridge. On the R side, start an R
session and install the three packages:

```r
install.packages("Boruta")
install.packages("midasr")
install.packages("rugarch")
```

You also need a working R installation reachable from your PATH. Verify with:

```bash
Rscript --version
python3 -c "import rpy2; print(rpy2.__version__)"
```

Both commands should return without error.

## Run the cross-reference tests

```bash
pytest tests/core/test_r_crossref_c59.py -v
```

Expected output with rpy2 and R packages installed:

```
tests/core/test_r_crossref_c59.py::test_xr1_boruta_vs_r_boruta PASSED
tests/core/test_r_crossref_c59.py::test_xr2_midas_almon_vs_r_midasr PASSED
tests/core/test_r_crossref_c59.py::test_xr3_hhs_vs_r_rugarch PASSED
```

Expected output without rpy2 (the default environment for most users):

```
SKIPPED  tests/core/test_r_crossref_c59.py - rpy2 not installed
```

The module-level `pytest.importorskip("rpy2")` fires immediately when rpy2 is
absent, marking the entire file as skipped. This is the expected outcome in CI
environments that do not install R. The main test suite (pytest without `-m`)
still passes.

## Accepted tolerances

The three tests check agreement within these bounds:

| Test | Metric | Threshold |
| --- | --- | --- |
| Boruta vs `Boruta::Boruta()` | Jaccard similarity of selected feature sets | >= 0.70 |
| MIDAS Almon vs `midasr` | Relative prediction error (y_hat) | <= 0.10 |
| Realized GARCH vs `rugarch` | Per-parameter absolute difference | per-param (see test) |

Differences within these thresholds arise from legitimate algorithmic variation.
For example, `midas_almon` uses Nelder-Mead NLS with polynomial Almon weights,
while `midasr` uses nealmon (normalized exponential Almon polynomial); these are
functional-form cousins, not identical implementations. Similarly, the Boruta
shadow permutation scheme and RNG seeding differ between Python and R.

## Caveats

Only install the `[validation]` extra on machines that have R and the required R
packages. Importing `rpy2` without a working R installation raises `ImportError`
at import time. The extra is intentionally excluded from the `ci` aggregate extra
(`macroforecast[ci]`) because the CI matrix does not include an R runtime.

If a cross-reference test fails (not skips), first check whether the R package
version matches the one used when the test was written, then inspect the per-test
tolerance table in `tests/core/test_r_crossref_c59.py` to understand whether the
divergence is within expected variation.
