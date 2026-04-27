from __future__ import annotations

import numpy as np
import pytest

from macrocast.execution.models.midasr import (
    almonp_weights,
    fit_midasr_nealmon_direct,
    nealmon_weights,
)


def test_nealmon_weights_match_midasr_flat_normalization() -> None:
    weights = nealmon_weights([2.0, 0.0, 0.0], 4)

    assert weights == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert float(weights.sum()) == pytest.approx(2.0)


def test_almonp_weights_match_midasr_raw_polynomial() -> None:
    weights = almonp_weights([1.0, 2.0], 3)

    assert weights == pytest.approx([3.0, 5.0, 7.0])


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

