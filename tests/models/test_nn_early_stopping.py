"""`nn` must be able to stop early on a held-out tail.

The training loop ran a fixed epoch budget over the whole fit window with no held-out
data and no stopping rule, so the only defences against overfitting were `dropout` and
`weight_decay`. Early stopping is the standard protocol for this setting and could not
be expressed at all.

The split is by time rather than at random: these are ordered observations, and a
shuffled split would let the network validate against its own future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf

ARCH = dict(
    hidden_layer_sizes=(16, 8, 4, 2),
    activation="relu",
    optimizer="adam",
    max_epochs=100,
    random_state=0,
)


@pytest.fixture()
def overfit_data():
    """Weak signal, many features -- the setting where a fixed epoch budget overfits."""
    rng = np.random.default_rng(0)
    n, p = 300, 40
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    y = pd.Series(0.3 * X["x0"] - 0.2 * X["x1"] + rng.normal(0.0, 1.0, n), name="y")
    return X.iloc[:220], y.iloc[:220], X.iloc[220:], y.iloc[220:]


def test_defaults_are_unchanged(overfit_data) -> None:
    pytest.importorskip("torch")
    Xtr, ytr, Xte, _ = overfit_data
    fit = mf.models.nn(Xtr, ytr, **ARCH)

    assert fit.estimator.validation_fraction == 0.0
    assert fit.estimator.early_stopping_patience is None
    # the whole epoch budget is used when nothing is held out
    assert len(fit.estimator.training_history_["loss"]) == ARCH["max_epochs"]
    assert "val_loss" not in fit.estimator.training_history_

    again = mf.models.nn(Xtr, ytr, **ARCH)
    np.testing.assert_allclose(
        np.asarray(fit.predict(Xte), dtype=float),
        np.asarray(again.predict(Xte), dtype=float),
    )


def test_early_stopping_stops_and_restores_the_best_epoch(overfit_data) -> None:
    pytest.importorskip("torch")
    Xtr, ytr, _, _ = overfit_data
    fit = mf.models.nn(
        Xtr, ytr, validation_fraction=0.2, early_stopping_patience=5, **ARCH
    )
    est = fit.estimator

    assert est.stopped_early_ is True
    ran = len(est.training_history_["loss"])
    assert ran < ARCH["max_epochs"]
    assert len(est.training_history_["val_loss"]) == ran
    # the retained parameters are the best-scoring epoch, not the last one
    assert est.best_epoch_ is not None and est.best_epoch_ <= ran


def test_early_stopping_improves_out_of_sample_where_the_budget_overfits(
    overfit_data,
) -> None:
    pytest.importorskip("torch")
    Xtr, ytr, Xte, yte = overfit_data
    plain = mf.models.nn(Xtr, ytr, **ARCH)
    stopped = mf.models.nn(
        Xtr, ytr, validation_fraction=0.2, early_stopping_patience=5, **ARCH
    )
    truth = yte.to_numpy(dtype=float)
    mse_plain = float(np.mean((np.asarray(plain.predict(Xte), float) - truth) ** 2))
    mse_stopped = float(np.mean((np.asarray(stopped.predict(Xte), float) - truth) ** 2))

    assert mse_stopped < mse_plain


def test_patience_without_a_validation_split_is_refused(overfit_data) -> None:
    pytest.importorskip("torch")
    Xtr, ytr, _, _ = overfit_data
    with pytest.raises(ValueError, match="validation_fraction"):
        mf.models.nn(Xtr, ytr, early_stopping_patience=5, validation_fraction=0.0, **ARCH)


def test_validation_fraction_is_bounded(overfit_data) -> None:
    pytest.importorskip("torch")
    Xtr, ytr, _, _ = overfit_data
    with pytest.raises(ValueError, match="validation_fraction"):
        mf.models.nn(Xtr, ytr, validation_fraction=1.0, **ARCH)
