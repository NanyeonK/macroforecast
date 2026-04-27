from __future__ import annotations

import numpy as np
import pytest

from macrocast.execution.models.midasr import (
    almonp_weights,
    fit_midasr_restricted_direct,
    fit_midasr_nealmon_direct,
    genexp_weights,
    harstep_weights,
    nbeta_weights,
    nealmon_weights,
)


def test_nealmon_weights_match_midasr_flat_normalization() -> None:
    weights = nealmon_weights([2.0, 0.0, 0.0], 4)

    assert weights == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert float(weights.sum()) == pytest.approx(2.0)


def test_almonp_weights_match_midasr_raw_polynomial() -> None:
    weights = almonp_weights([1.0, 2.0], 3)

    assert weights == pytest.approx([3.0, 5.0, 7.0])


def test_nbeta_weights_match_midasr_normalized_beta() -> None:
    weights = nbeta_weights([2.0, 2.0, 3.0], 5)

    assert weights[0] == pytest.approx(0.0, abs=1e-12)
    assert weights[1:4] == pytest.approx([0.9, 0.8, 0.3])
    assert weights[4] == pytest.approx(0.0, abs=1e-12)
    assert float(weights.sum()) == pytest.approx(2.0)


def test_genexp_weights_match_midasr_generalized_exponential() -> None:
    weights = genexp_weights([1.0, 2.0, 3.0, 4.0], 3)

    assert weights == pytest.approx(
        [
            1.0,
            1.02 * np.exp(0.0304),
            1.04 * np.exp(0.0616),
        ]
    )


def test_harstep_weights_match_midasr_har_rv_step() -> None:
    weights = harstep_weights([1.0, 2.0, 4.0], 20)

    assert weights[0] == pytest.approx(1.6)
    assert weights[1:5] == pytest.approx([0.6, 0.6, 0.6, 0.6])
    assert weights[5:] == pytest.approx([0.2] * 15)
    assert float(weights.sum()) == pytest.approx(7.0)


def test_fit_midasr_nealmon_direct_returns_finite_prediction() -> None:
    lag_tensor = np.asarray(
        [
            [[1.0, 0.5, 0.25], [0.1, 0.2, 0.3]],
            [[1.1, 0.6, 0.30], [0.2, 0.1, 0.2]],
            [[1.2, 0.7, 0.35], [0.3, 0.2, 0.1]],
            [[1.3, 0.8, 0.40], [0.4, 0.3, 0.2]],
        ],
        dtype=float,
    )
    y_train = np.asarray([1.0, 1.1, 1.2, 1.3], dtype=float)
    pred_lag_tensor = np.asarray([[1.4, 0.9, 0.45], [0.5, 0.4, 0.3]], dtype=float)

    fit = fit_midasr_nealmon_direct(
        lag_tensor,
        y_train,
        pred_lag_tensor,
        degree=2,
        max_nfev=250,
    )

    assert fit.success is True
    assert np.isfinite(fit.prediction)
    assert len(fit.weights_by_term) == 2
    assert len(fit.weights_by_term[0]) == 3


@pytest.mark.parametrize(
    ("weight_family", "degree", "lag_count", "true_params"),
    [
        ("nbeta", 0, 5, [0.8, 1.4, 2.2]),
        ("genexp", 0, 5, [0.2, 0.1, -1.0, 0.5]),
        ("harstep", 0, 20, [0.3, 0.2, 0.1]),
    ],
)
def test_fit_midasr_restricted_direct_supports_registered_weight_families(
    weight_family: str,
    degree: int,
    lag_count: int,
    true_params: list[float],
) -> None:
    rng = np.random.default_rng(42)
    lag_tensor = rng.normal(size=(40, 1, lag_count))
    pred_lag_tensor = rng.normal(size=(1, lag_count))
    true_weights = {
        "nbeta": nbeta_weights,
        "genexp": genexp_weights,
        "harstep": harstep_weights,
    }[weight_family](true_params, lag_count)
    y_train = 0.25 + lag_tensor[:, 0, :] @ true_weights

    fit = fit_midasr_restricted_direct(
        lag_tensor,
        y_train,
        pred_lag_tensor,
        weight_family=weight_family,
        degree=degree,
        max_nfev=1000,
    )

    assert fit.success is True
    assert np.isfinite(fit.prediction)
    assert len(fit.weights_by_term) == 1
    assert len(fit.weights_by_term[0]) == lag_count


def test_fit_midasr_restricted_direct_supports_almonp() -> None:
    lag_tensor = np.asarray(
        [
            [[1.0, 0.5, 0.25], [0.1, 0.2, 0.3]],
            [[1.1, 0.6, 0.30], [0.2, 0.1, 0.2]],
            [[1.2, 0.7, 0.35], [0.3, 0.2, 0.1]],
            [[1.3, 0.8, 0.40], [0.4, 0.3, 0.2]],
        ],
        dtype=float,
    )
    y_train = np.asarray([1.0, 1.1, 1.2, 1.3], dtype=float)
    pred_lag_tensor = np.asarray([[1.4, 0.9, 0.45], [0.5, 0.4, 0.3]], dtype=float)

    fit = fit_midasr_restricted_direct(
        lag_tensor,
        y_train,
        pred_lag_tensor,
        weight_family="almonp",
        degree=2,
        max_nfev=250,
    )

    assert fit.success is True
    assert np.isfinite(fit.prediction)
    assert len(fit.weights_by_term) == 2
    assert len(fit.weights_by_term[0]) == 3
