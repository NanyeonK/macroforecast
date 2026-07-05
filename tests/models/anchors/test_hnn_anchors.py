"""Independent correctness anchors for ``hemisphere_nn`` / ``density_hnn`` (WP-A4).

Both models were Tier-1 zero-anchor AND explicitly ``BLOCKED(no-torch)`` per
``.dev-notes/anchor_coverage/v2_anchor_results.md``'s WP-V2 note: torch was
deliberately not installed for that work package, so ``hemisphere_nn``
(``_TorchHemisphereNNRegressor`` / ``_TorchHemisphereNet``) and
``density_hnn`` (``_TorchDensityHNNRegressor`` / ``_TorchDensityHNNNet``,
citing Goulet Coulombe, Frenette & Klieber 2025 JAE / Aionx ``DensityHNN``)
had never been checked against any independent numeric target. WP-A4
installs CPU-only torch and closes that gap, following the WP-V2 note's own
proposed anchor design (its BLOCKED section) with four anchor types:

1. Seeded determinism pin (``test_*_determinism_pin``): identical
   ``random_state`` -> bit-identical ``fit``+``predict`` output AND
   bagged-member ``state_dict()`` tensors across two independent fits; a
   different seed must change the result (catches silent RNG leaks, e.g. an
   unseeded dropout mask or subsample draw). Empirically verified
   bit-identical (``atol=0``) on this host's default torch CPU thread
   settings (48 logical CPUs, torch default 24 intra-op threads) across
   repeated runs -- no tolerance was loosened to force a pass. Both
   wrappers seed both RNG streams they use: ``torch.manual_seed`` (network
   init + dropout) and a ``numpy.random.default_rng`` instance (blocked
   subsample / block-bootstrap draws) -- both keyed off the same
   ``random_state`` -- so this is a genuine, not accidental, determinism
   guarantee for a CPU-only run.

2. Closed-form architecture limit (``test_*_zero_weight_architecture_limit``):
   neither ``_TorchHemisphereNet`` nor ``_TorchDensityHNNNet`` exposes a
   literal "zero hidden units"/"no nonlinearity" configuration through its
   public constructor kwargs -- ``_torch_dense_stack`` always inserts a
   ``ReLU`` after every ``Linear``, and ``lc``/``lm``/``lv``/
   ``common_layers``/``mean_layers``/``volatility_layers`` are all clipped
   to >= 1, ``neurons`` to >= 2 (read directly from
   ``macroforecast/models/neural.py``, not assumed). The true
   hand-computable degenerate limit this architecture admits is: zero every
   ``Linear`` layer's weight AND bias except the final head layer's bias.
   Every hidden layer then maps any input to the zero vector
   (``ReLU(0 @ W + 0) = 0``, propagated through every subsequent zeroed
   layer), so the head's final ``Linear`` collapses to a pure additive
   constant independent of ``X`` -- exactly the "unconditional
   mean/variance" limit this WP's sibling ``tvp_ridge``
   constant-parameter-limit anchor (``test_tvp_ridge_anchors.py``) checks
   for a different model class. Verified directly against hand-chosen
   target constants to float32 precision; no training involved (pure
   forward-pass architecture check, isolating the network wiring from any
   optimizer behavior).

   ``density_hnn``'s volatility head has an additional, architecture-level
   closed form worth anchoring on its own: ``_TorchDensityHNNNet.__call__``
   computes ``sigma = volatility_emphasis * raw_sigma / raw_sigma.mean()``
   (citing Aionx ``DensityHNN.base_architecture``). Under the zero-weight
   limit, every row's ``raw_sigma`` is IDENTICAL (a constant, since the head
   ignores ``X`` entirely), so ``raw_sigma.mean() == raw_sigma`` row-wise
   and ``sigma`` collapses to EXACTLY ``volatility_emphasis``, regardless of
   the raw pre-normalization bias chosen -- verified empirically with two
   different raw-bias values, both giving identical ``sigma``, confirming
   the normalization does what the source comment claims.

3. DGP recovery smoke (``test_*_recovers_low_noise_linear_dgp``): a
   low-noise linear DGP (``y = beta'x + small noise``) that both model
   classes can represent. Out-of-sample RMSE against the TRUE signal (not
   the noisy ``y``) must beat a naive constant (train-mean) forecast by a
   loose-but-meaningful margin. Band rationale: both models actually
   achieve an 84-89% RMSE reduction vs. naive on this DGP (verified); the
   assertion threshold (RMSE <= 50% of naive RMSE, i.e. >= 50% improvement)
   leaves a wide safety margin against environment-driven float noise while
   still being a meaningful, non-trivial recovery bar (a broken/untrained
   network would not clear it).

4. ``density_hnn``-only density-output contract
   (``test_density_hnn_density_output_contract_on_low_noise_dgp``): finite
   mu/sigma^2, sigma^2 > 0 for every prediction (hard, always-true
   contract), plus a loose MEDIAN-based calibration sanity check anchored
   on the model's OWN realized out-of-sample squared error -- not the
   unobservable true DGP noise floor, since the model only ever sees
   residuals from its own, necessarily imperfect, mean predictions, so its
   variance head is calibrated to *total* predictive uncertainty, not
   irreducible noise (verified: median predicted variance is consistently
   within a single order of magnitude of the realized out-of-sample MSE
   across configurations). See the Finding below for why this test uses the
   median, not a per-row bound.

## Finding (reported per mission rule, not hidden, NOT xfail'd)

``density_hnn``'s per-row predicted variance has a very wide right tail even
on a genuinely near-noiseless DGP (verified in a throwaway script: true
noise variance 1e-6, median predicted variance ~0.03-0.3 across several
``n_estimators``/``prior_estimators``/``block_size`` configurations, but
individual out-of-sample test rows occasionally reach 20-50x the median).
Isolated by ablation (``rescale_volatility=False`` still shows the same
tail -- if anything a *larger* one: median 0.80 vs. 0.26, max 54.5 vs.
21.1), so this is NOT an artifact of the OOB log-linear volatility-rescaling
step (``_fit_aionx_volatility_rescaling``) as initially suspected; it
reproduces at the base bagged-ensemble level
(``_TorchDensityHNNRegressor._predict_scaled_matrices``'s per-member
``sigma`` disagreement for certain rows, averaged across members but not
smoothed away by ``np.mean(sigma_matrix, axis=1)``). This looks like a
genuine (if unflattering) property of small-``n_estimators`` bagged
density-ensembles disagreeing sharply on individual out-of-typical-range
rows, not a macroforecast-specific coding bug -- consistent with the
architecture-limit anchor above showing the forward pass itself is exactly
correct. Not fixed here (out of scope for a verification work package); the
DGP-recovery/density-contract tests below use a ROBUST (median, not
per-row) calibration statistic specifically so this tail behavior does not
make the tests flaky, while still asserting the hard per-row contract
(finite, positive) unconditionally.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

import macroforecast as mf  # noqa: E402
from macroforecast.models.neural import (  # noqa: E402
    _TorchDensityHNNNet,
    _TorchHemisphereNet,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _zero_all_but_final_bias(seq: Any, final_bias: float) -> None:
    """Zero every ``Linear`` layer's weight+bias inside a torch container,
    except the LAST ``Linear``'s bias, which is set to ``final_bias``.

    With every weight zeroed, ``Linear(x) = bias`` regardless of ``x``; with
    every hidden bias also zeroed, ``ReLU(0) = 0`` propagates a zero vector
    through every layer up to the final one, so the whole stack collapses to
    the constant ``final_bias`` independent of the input. This is the
    "zero-width hemisphere" degenerate limit referenced in the module
    docstring -- read from the architecture, not assumed.
    """
    linears = [module for module in seq.modules() if isinstance(module, torch.nn.Linear)]
    assert linears, "expected at least one nn.Linear layer to zero"
    for linear in linears:
        torch.nn.init.zeros_(linear.weight)
        torch.nn.init.zeros_(linear.bias)
    linears[-1].bias.data.fill_(final_bias)


def _inverse_softplus(value: float) -> float:
    """Inverse of ``softplus(x) = log(1 + exp(x))``, used to hand-pick a
    bias that makes ``softplus(bias) + 1e-6`` equal a target variance."""
    return float(np.log(np.expm1(value)))


def _assert_state_dicts_equal(a: dict[str, Any], b: dict[str, Any]) -> None:
    for key, value in a.items():
        other = b[key]
        if isinstance(value, dict):
            _assert_state_dicts_equal(value, other)
        else:
            assert torch.equal(value, other), f"tensor mismatch at {key!r}"


def _make_linear_dgp(
    T: int = 180, K: int = 3, noise_sd: float = 0.05, seed: int = 123
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Low-noise linear DGP: y = beta'x + small noise. Representable exactly
    by a linear model, so both HNN classes (which include a linear model as
    a special case of their function class) should recover it well above a
    naive constant-forecast baseline."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((T, K)), columns=[f"x{i}" for i in range(K)]
    )
    beta = np.array([1.2, -0.8, 0.5])[:K]
    signal = X.to_numpy(dtype=float) @ beta
    y = pd.Series(signal + noise_sd * rng.standard_normal(T), name="y")
    return X, y, signal


def _xy(K: int = 4, seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((80, K)), columns=[f"x{i}" for i in range(K)]
    )
    beta = rng.standard_normal(K)
    y = pd.Series(
        X.to_numpy(dtype=float) @ beta * 0.5 + 0.05 * rng.standard_normal(80),
        name="y",
    )
    return X, y


# ---------------------------------------------------------------------------
# Anchor 1: seeded determinism pin.
# ---------------------------------------------------------------------------


def test_hemisphere_nn_determinism_pin() -> None:
    X, y = _xy()

    def _fit(seed: int):
        return mf.models.hemisphere_nn(
            X, y, neurons=8, max_epochs=15, n_estimators=3, patience=5,
            random_state=seed, device="cpu",
        )

    fit_a = _fit(42)
    fit_b = _fit(42)
    fit_c = _fit(99)

    pred_a = fit_a.predict(X.iloc[-5:]).to_numpy(dtype=float)
    pred_b = fit_b.predict(X.iloc[-5:]).to_numpy(dtype=float)
    pred_c = fit_c.predict(X.iloc[-5:]).to_numpy(dtype=float)
    var_a = fit_a.predict_variance(X.iloc[-5:])
    var_b = fit_b.predict_variance(X.iloc[-5:])

    np.testing.assert_array_equal(pred_a, pred_b)
    np.testing.assert_array_equal(var_a, var_b)
    _assert_state_dicts_equal(
        fit_a.estimator.models_[0].state_dict(), fit_b.estimator.models_[0].state_dict()
    )
    assert not np.allclose(pred_a, pred_c, atol=1e-9), (
        "different random_state produced identical predictions -- possible "
        "silent RNG leak (seed not actually threaded through fit)"
    )


def test_density_hnn_determinism_pin() -> None:
    X, y = _xy()

    def _fit(seed: int):
        return mf.models.density_hnn(
            X, y, neurons=8, max_epochs=15, n_estimators=3, prior_estimators=3,
            block_size=8, patience=5, random_state=seed, device="cpu",
        )

    fit_a = _fit(42)
    fit_b = _fit(42)
    fit_c = _fit(99)

    pred_a = fit_a.predict(X.iloc[-5:]).to_numpy(dtype=float)
    pred_b = fit_b.predict(X.iloc[-5:]).to_numpy(dtype=float)
    pred_c = fit_c.predict(X.iloc[-5:]).to_numpy(dtype=float)
    var_a = fit_a.predict_variance(X.iloc[-5:])
    var_b = fit_b.predict_variance(X.iloc[-5:])

    np.testing.assert_array_equal(pred_a, pred_b)
    np.testing.assert_array_equal(var_a, var_b)
    _assert_state_dicts_equal(
        fit_a.estimator.models_[0].state_dict(), fit_b.estimator.models_[0].state_dict()
    )
    assert not np.allclose(pred_a, pred_c, atol=1e-9), (
        "different random_state produced identical predictions -- possible "
        "silent RNG leak (seed not actually threaded through fit)"
    )


# ---------------------------------------------------------------------------
# Anchor 2: closed-form zero-weight architecture limit.
# ---------------------------------------------------------------------------


def test_hemisphere_nn_zero_weight_architecture_limit() -> None:
    torch.manual_seed(0)
    n_features = 5
    net = _TorchHemisphereNet(
        torch=torch, n_features=n_features, lc=2, lm=2, lv=2, neurons=8, dropout=0.0
    )
    net.eval()

    target_mean = 3.7
    target_variance = 2.0
    bias_variance = _inverse_softplus(target_variance - 1e-6)

    _zero_all_but_final_bias(net.core, 0.0)
    _zero_all_but_final_bias(net.mean_head, target_mean)
    _zero_all_but_final_bias(net.variance_head, bias_variance)

    x = torch.tensor(
        np.random.default_rng(1).standard_normal((10, n_features)), dtype=torch.float32
    )
    mean, variance = net(x)

    np.testing.assert_allclose(
        mean.detach().numpy(), np.full(10, target_mean), rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(
        variance.detach().numpy(), np.full(10, target_variance), rtol=0, atol=1e-5
    )
    # Every row must be IDENTICAL (input-independent), not merely close to
    # the target on average -- this is the defining property of the limit.
    assert np.unique(np.round(mean.detach().numpy(), 6)).size == 1
    assert np.unique(np.round(variance.detach().numpy(), 6)).size == 1


def test_density_hnn_zero_weight_architecture_limit() -> None:
    torch.manual_seed(0)
    n_features = 5
    volatility_emphasis = 0.65
    net = _TorchDensityHNNNet(
        torch=torch,
        n_features=n_features,
        common_layers=2,
        mean_layers=2,
        volatility_layers=2,
        neurons=8,
        dropout=0.0,
        volatility_emphasis=volatility_emphasis,
    )
    net.eval()

    target_mean = -1.2
    _zero_all_but_final_bias(net.core, 0.0)
    _zero_all_but_final_bias(net.mean_head, target_mean)
    _zero_all_but_final_bias(net.volatility_head, 0.3)

    x = torch.tensor(
        np.random.default_rng(1).standard_normal((10, n_features)), dtype=torch.float32
    )
    mean, sigma = net(x)

    np.testing.assert_allclose(
        mean.detach().numpy(), np.full(10, target_mean), rtol=0, atol=1e-5
    )
    # sigma = volatility_emphasis * raw_sigma / raw_sigma.mean() collapses to
    # EXACTLY volatility_emphasis when raw_sigma is constant across rows --
    # independent of the raw pre-normalization bias chosen (0.3 here).
    np.testing.assert_allclose(
        sigma.detach().numpy(), np.full(10, volatility_emphasis), rtol=0, atol=1e-5
    )

    # Confirm the raw-bias-independence claim directly: a different raw bias
    # must still produce the identical normalized sigma.
    _zero_all_but_final_bias(net.volatility_head, 5.0)
    _, sigma_alt = net(x)
    np.testing.assert_allclose(
        sigma_alt.detach().numpy(), np.full(10, volatility_emphasis), rtol=0, atol=1e-5
    )


# ---------------------------------------------------------------------------
# Anchor 3: low-noise linear DGP recovery smoke test.
# ---------------------------------------------------------------------------


def test_hemisphere_nn_recovers_low_noise_linear_dgp() -> None:
    X, y, signal = _make_linear_dgp()
    n_train = 140
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test = X.iloc[n_train:]
    signal_test = signal[n_train:]
    naive_rmse = float(np.sqrt(np.mean((y_train.mean() - signal_test) ** 2)))

    fit = mf.models.hemisphere_nn(
        X_train, y_train, neurons=16, max_epochs=60, n_estimators=8,
        patience=10, learning_rate=0.01, random_state=0, device="cpu",
    )
    pred = fit.predict(X_test).to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((pred - signal_test) ** 2)))

    assert np.isfinite(pred).all()
    # Loose band: empirically ~84% improvement over naive; require >= 50%
    # (rmse <= 0.5 * naive_rmse) to leave a wide safety margin while still
    # ruling out a broken/untrained network (which would not beat naive).
    assert rmse <= 0.5 * naive_rmse, (
        f"hemisphere_nn rmse={rmse:.4f} did not beat 50% of naive rmse="
        f"{naive_rmse:.4f} on a representable low-noise linear DGP"
    )


def test_density_hnn_recovers_low_noise_linear_dgp() -> None:
    X, y, signal = _make_linear_dgp()
    n_train = 140
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test = X.iloc[n_train:]
    signal_test = signal[n_train:]
    naive_rmse = float(np.sqrt(np.mean((y_train.mean() - signal_test) ** 2)))

    fit = mf.models.density_hnn(
        X_train, y_train, neurons=16, max_epochs=60, n_estimators=15,
        prior_estimators=15, block_size=14, patience=10, learning_rate=0.01,
        random_state=0, device="cpu",
    )
    pred = fit.predict(X_test).to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((pred - signal_test) ** 2)))

    assert np.isfinite(pred).all()
    assert rmse <= 0.5 * naive_rmse, (
        f"density_hnn rmse={rmse:.4f} did not beat 50% of naive rmse="
        f"{naive_rmse:.4f} on a representable low-noise linear DGP"
    )


# ---------------------------------------------------------------------------
# Anchor 4: density_hnn density-output contract.
# ---------------------------------------------------------------------------


def test_density_hnn_density_output_contract_on_low_noise_dgp() -> None:
    X, y, signal = _make_linear_dgp()
    n_train = 140
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test = X.iloc[n_train:]
    signal_test = signal[n_train:]

    fit = mf.models.density_hnn(
        X_train, y_train, neurons=16, max_epochs=60, n_estimators=15,
        prior_estimators=15, block_size=14, patience=10, learning_rate=0.01,
        random_state=0, device="cpu",
    )
    pred = fit.predict(X_test).to_numpy(dtype=float)
    variance = fit.predict_variance(X_test)
    mse = float(np.mean((pred - signal_test) ** 2))

    # Hard contract: always true regardless of calibration quality.
    assert np.isfinite(variance).all(), "density_hnn produced non-finite variance"
    assert (variance > 0).all(), "density_hnn produced non-positive variance"

    # Loose, ROBUST (median, not per-row -- see module docstring Finding)
    # calibration sanity check: the model's own variance head should be
    # within an order of magnitude of its own realized out-of-sample MSE,
    # not off by many orders of magnitude in either direction (e.g. a
    # constant near-zero variance head, or a completely uncorrelated huge
    # constant, would fail this).
    median_variance = float(np.median(variance))
    assert 0.05 * mse <= median_variance <= 50.0 * mse, (
        f"density_hnn median predicted variance {median_variance:.4g} is not "
        f"within a loose [0.05x, 50x] band of the realized out-of-sample "
        f"MSE {mse:.4g}"
    )
