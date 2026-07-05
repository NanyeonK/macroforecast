"""Live R cross-check of the Minnesota prior formula vs ``bvartools::minnesota_prior``.

WP-V2. ``bvar_minnesota``'s docstring explicitly claims a
"FAVAR::BVAR-aligned" / bvartools-``minnesota_prior``-aligned prior, but per
``.dev-notes/anchor_coverage/matrix.csv`` this had "zero executable R
(FAVAR::BVAR/bvartools) comparison anywhere" before this test. ``bvartools``
installs cleanly from CRAN on this host (unlike ``FAVAR``'s heavier Gibbs
machinery, which is anchored separately in ``tests/models/anchors/test_favar_anchors.py``
via a DGP oracle instead), so this is a genuine executable parity check, not
just a deterministic re-derivation.

Uses the subprocess-Rscript bridge from ``tests/parity/conftest.py`` (ported
from the WP-V1 ``test/r-parity-v1`` branch's ``tests/parity/`` pattern --
rpy2 is known-broken against this host's R 4.3.3 ABI).
"""
from __future__ import annotations

import numpy as np
import pytest

from macroforecast.models.timeseries import _favar_minnesota_prior
from tests.parity.conftest import require_r, run_rscript

pytestmark = pytest.mark.rparity


def _panel(n_obs: int = 200, seed: int = 99) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phi = np.array([[0.5, 0.1, -0.05], [0.05, 0.4, 0.1], [0.0, -0.1, 0.6]])
    values: np.ndarray = np.zeros((n_obs, 3), dtype=float)
    for t in range(1, n_obs):
        values[t] = phi @ values[t - 1] + rng.normal(scale=[1.0, 1.5, 0.7])
    return values


@pytest.mark.parametrize("kappa0,kappa1,n_lag", [(2.0, 0.5, 1), (3.0, 0.4, 2)])
def test_minnesota_prior_matches_bvartools_precision(kappa0, kappa1, n_lag):
    require_r("bvartools")
    values = _panel()
    k = values.shape[1]
    _, precision = _favar_minnesota_prior(values, n_lag, kappa0, kappa1)
    # bvartools::minnesota_prior builds V[l, (r-1)*k+j] -- an (k eq-rows,
    # n_reg cols) grid indexed exactly like our own `variance` array (row =
    # equation, col = (lag-1)*k + regressor) -- then does
    # `v_i <- diag(1 / c(V))`. R's `matrix(V)` (single-argument form)
    # flattens column-major, exactly like numpy's `flatten(order="F")`.
    # Confirmed by reading the bvartools::minnesota_prior source directly
    # (`print(minnesota_prior)` in R). So no index permutation is needed;
    # the two diagonals are directly comparable position-for-position.
    ours_bvartools_layout = np.diagonal(precision).copy()

    flat_values = ",".join(f"{v:.17g}" for v in values.flatten(order="C"))
    script = f"""
    suppressMessages(library(bvartools))
    values <- matrix(c({flat_values}), ncol = {k}, byrow = TRUE)
    colnames(values) <- c("v1", "v2", "v3")
    Y <- ts(values, start = c(2000, 1), frequency = 12)
    data <- gen_var(Y, p = {n_lag}, deterministic = "none")
    prior <- minnesota_prior(data, kappa0 = {kappa0}, kappa1 = {kappa1}, kappa3 = 5, sigma = "AR")
    emit("precision_diag", diag(prior$v_i))
    """
    result = run_rscript(script)
    r_precision_diag = np.array([float(v) for v in result["precision_diag"].split(",")])

    np.testing.assert_allclose(ours_bvartools_layout, r_precision_diag, rtol=1e-8, atol=1e-10)
