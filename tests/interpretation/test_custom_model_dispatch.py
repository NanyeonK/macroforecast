"""Built-in interpretation methods, applied to an INJECTED custom model (#446 risk 1).

`#446` flags that `interpretation` dispatches on model class, so what happens to a
model registered through `mf.models.custom_model` is unverified. `custom_interpretation`
(the user supplying the *method*) is covered elsewhere; this is the other direction --
the user supplying the *model* and asking a built-in method to explain it.

Dispatch here is duck-typing, not isinstance: `linear_coefficients` reads `coef_`,
`tree_importance` reads `feature_importances_`. So a custom estimator that exposes the
right attribute is in scope for those methods whether or not anyone intended it, and
the model-agnostic methods (`permutation_importance`, `partial_dependence`,
`lofo_importance`) are in scope for every custom model because they only ever call
`predict`.

Three things must hold, and the third is the one worth writing a test for:

1. a custom model whose coefficients are known gets those coefficients back;
2. the model-agnostic methods work on a custom model with nothing but `predict`;
3. a custom model that cannot support a method **fails loudly**. The failure mode this
   file exists to catch is a method that returns a plausible table for a model it
   cannot actually explain -- a wrong exhibit is worse than a missing one, because
   nobody checks a number that appeared.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf

N = 120
BETA = {"a": 2.0, "b": -1.5, "c": 0.0}


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({k: rng.normal(size=N) for k in BETA})
    y = pd.Series(
        sum(BETA[k] * X[k] for k in BETA) + rng.normal(size=N) * 0.01, name="y"
    )
    return X, y


class _LinearLike:
    """A custom estimator that exposes the linear surface by duck-typing."""

    def __init__(self, coef: np.ndarray, names: list[str]) -> None:
        self.coef_ = coef
        self.feature_names_in_ = np.asarray(names, dtype=object)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_


class _OpaqueModel:
    """A custom estimator with nothing but ``predict``."""

    def __init__(self, coef: np.ndarray) -> None:
        self._coef = coef

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self._coef


def _fit_linear_like(X: pd.DataFrame, y: pd.Series, **_: object) -> _LinearLike:
    coef, *_rest = np.linalg.lstsq(np.asarray(X, dtype=float), np.asarray(y, dtype=float), rcond=None)
    return _LinearLike(coef, list(X.columns))


def _fit_opaque(X: pd.DataFrame, y: pd.Series, **_: object) -> _OpaqueModel:
    coef, *_rest = np.linalg.lstsq(np.asarray(X, dtype=float), np.asarray(y, dtype=float), rcond=None)
    return _OpaqueModel(coef)


def test_a_custom_model_registers_and_fits(data) -> None:
    """The registration path itself, so a later failure is not blamed on dispatch."""
    X, y = data
    spec = mf.models.custom_model("dispatch_linear_like", _fit_linear_like)
    assert spec.name == "dispatch_linear_like"
    fit = spec.fit_func(X, y)
    np.testing.assert_allclose(
        fit.coef_, [BETA[c] for c in X.columns], rtol=0, atol=5e-3
    )


def test_linear_coefficients_reads_a_custom_models_coefficients(data) -> None:
    """Duck-typed dispatch must return the CUSTOM model's numbers, not a default."""
    X, y = data
    fit = _fit_linear_like(X, y)
    table = mf.interpretation.linear_coefficients(fit)

    assert set(table["feature"]) == set(X.columns), (
        f"feature names do not come from the custom model: {list(table['feature'])}"
    )
    got = dict(zip(table["feature"], table["coefficient"], strict=True))
    for name, beta in BETA.items():
        assert abs(got[name] - beta) < 5e-3, (
            f"{name}: interpretation reported {got[name]:.4f}, the model has {beta}"
        )


def test_a_model_without_coefficients_is_refused_not_guessed(data) -> None:
    """The assertion this file exists for: no plausible table for an opaque model."""
    X, y = data
    opaque = _fit_opaque(X, y)
    # The message has to name the missing surface, or a user cannot act on it.
    with pytest.raises(ValueError, match="coef_"):
        mf.interpretation.linear_coefficients(opaque)


def test_tree_importance_refuses_a_model_that_is_not_a_tree(data) -> None:
    X, y = data
    fit = _fit_linear_like(X, y)
    with pytest.raises(ValueError, match="feature_importances_"):
        mf.interpretation.tree_importance(fit)


def test_permutation_importance_works_on_a_predict_only_model(data) -> None:
    """Model-agnostic methods only need ``predict``, so a custom model is in scope.

    The oracle is the DGP: feature ``c`` has a zero coefficient, so permuting it must
    not raise the loss, while ``a`` (|beta| = 2.0) must matter more than ``b`` (1.5).
    """
    X, y = data
    opaque = _fit_opaque(X, y)
    table = mf.interpretation.permutation_importance(
        opaque, X, y, n_repeats=8, random_state=0
    )

    imp = dict(zip(table["feature"], table["importance"], strict=True))
    assert imp["c"] < imp["b"] < imp["a"], (
        "permutation importance does not recover the known ordering "
        f"|beta_a|=2.0 > |beta_b|=1.5 > |beta_c|=0: {imp}"
    )
    assert imp["c"] < 0.05 * imp["a"], (
        f"an unused feature was given real importance: c={imp['c']:.4g}, a={imp['a']:.4g}"
    )


def test_partial_dependence_of_a_custom_model_is_the_models_own_slope(data) -> None:
    """For a linear predict, PD in feature j must be a line of slope beta_j.

    Written out independently of the implementation, so a shared bug cannot make both
    sides agree.
    """
    X, y = data
    opaque = _fit_opaque(X, y)
    table = mf.interpretation.partial_dependence(opaque, X, features="a", grid_size=12)

    xs = np.asarray(table["value"], dtype=float)
    ys = np.asarray(table["prediction"], dtype=float)
    slope = np.polyfit(xs, ys, 1)[0]
    assert abs(slope - BETA["a"]) < 5e-2, (
        f"partial dependence slope {slope:.4f} does not match the model's beta_a={BETA['a']}"
    )


def test_lofo_on_a_custom_model_needs_its_fit_func(data) -> None:
    """LOFO refits, so a custom model must hand it the callable that builds one.

    Without ``fit_func`` there is no way to rebuild a custom estimator, and the
    honest outcomes are a clear error or a correct result -- not a table built from
    some other model.
    """
    X, y = data
    table = mf.interpretation.lofo_importance(
        _fit_opaque(X, y), X, y, fit_func=lambda Xi, yi: _fit_opaque(Xi, yi)
    )
    imp = dict(zip(table["feature"], table["importance"], strict=True))
    assert imp["c"] < imp["a"], (
        f"dropping an unused feature hurt more than dropping the strongest one: {imp}"
    )
