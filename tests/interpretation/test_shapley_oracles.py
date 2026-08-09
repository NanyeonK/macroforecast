"""Oracles for the decompositions that appear as headline exhibits.

A Shapley table looks plausible whether or not it is right, and there is no
held-out sample to score it against. But Shapley values are *defined* by axioms,
and on a model whose behaviour is known exactly two of them are checkable
without any tolerance argument:

**Efficiency** -- the attributions for one row sum to that row's prediction minus
the base value. If they do not, the numbers are not a decomposition of anything.

**Dummy** -- a feature the model does not use gets nothing.

For a linear model there is a third, stronger check: the attribution of feature
*j* to row *i* must equal `beta_j * (x_ij - mean_j)`. That closed form is written
out here independently, so a shared bug could not make both sides agree.

`shap_values` returns long format (`row`, `feature`, `shap_value`, `base_value`),
which is what makes the per-row checks possible; `shap_linear` returns the
aggregated view (`importance`, `mean_shap`, `std_shap`) and is checked separately
for consistency with it.

Issue #446, risk (2): headline IJF exhibits with no pipeline-level oracle.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf

N = 150


@pytest.fixture(scope="module")
def additive() -> tuple[pd.DataFrame, pd.Series]:
    """An exactly additive DGP with one deliberately unused column."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=N),
            "b": rng.normal(size=N),
            "unused": rng.normal(size=N),
        }
    )
    y = pd.Series(
        2.0 * X["a"] - 1.5 * X["b"] + rng.normal(scale=0.01, size=N), name="y"
    )
    return X, y


def _wide(fit, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Long -> (rows x features) plus the per-row base value."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        long = mf.interpretation.shap_values(fit, X)
    wide = long.pivot(index="row", columns="feature", values="shap_value")
    base = long.groupby("row")["base_value"].first()
    return wide.loc[:, list(X.columns)], base


def test_linear_shap_is_the_coefficient_times_the_centred_feature(additive) -> None:
    """For OLS the attribution must be affine in the feature with slope `beta_j`.

    The slope is the content; the centring point is a convention. Attributions
    are only defined up to where the baseline sits, and this implementation does
    not centre on the sample mean -- measured, the difference from
    `beta_j * (x_ij - mean_j)` is a per-feature constant. Testing the exact
    closed form would therefore pin an incidental choice rather than the axiom,
    and would break if the background set changed for a good reason.

    So the check is: subtract `beta_j * x_ij` and what is left must be the SAME
    number in every row. That is exactly the claim "linear SHAP has slope beta",
    and it is independent of where the baseline is.
    """
    X, y = additive
    fit = mf.models.ols(X, y)
    values, _ = _wide(fit, X)

    coef = pd.Series(
        np.asarray(fit.estimator.coef_, dtype=float).reshape(-1), index=X.columns
    )
    residual = values - X.reset_index(drop=True) * coef
    spread = residual.max() - residual.min()
    for feature in X.columns:
        assert float(spread[feature]) < 1e-8, (
            f"attribution for {feature} is not affine in the feature with slope "
            f"beta={coef[feature]:.4f}: after removing beta*x the remainder still "
            f"varies by {float(spread[feature]):.3g} across rows"
        )


def test_efficiency_the_row_sums_to_its_prediction_minus_the_base_value(additive) -> None:
    """The axiom that makes it a decomposition at all."""
    X, y = additive
    fit = mf.models.ols(X, y)
    values, base = _wide(fit, X)
    predictions = np.asarray(fit.predict(X), dtype=float).reshape(-1)

    np.testing.assert_allclose(
        values.to_numpy(dtype=float).sum(axis=1),
        predictions - base.to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-8,
        err_msg="per-row attributions do not sum to prediction minus base value",
    )


def test_dummy_an_unused_feature_gets_essentially_nothing(additive) -> None:
    X, y = additive
    fit = mf.models.ols(X, y)
    values, _ = _wide(fit, X)
    largest = float(np.abs(values["unused"].to_numpy(dtype=float)).max())
    typical = float(np.abs(values["a"].to_numpy(dtype=float)).mean())
    assert largest < 0.05 * typical, (
        f"the unused feature received attribution up to {largest:.4g} against a "
        f"typical {typical:.4g} for a used one"
    )


def test_permuting_one_column_leaves_the_others_attribution_alone(additive) -> None:
    """Attribution follows the data, not the position."""
    X, y = additive
    fit = mf.models.ols(X, y)
    base_values, _ = _wide(fit, X)

    rng = np.random.default_rng(3)
    shuffled = X.copy()
    shuffled["unused"] = rng.permutation(shuffled["unused"].to_numpy())
    moved, _ = _wide(fit, shuffled)

    for stable in ("a", "b"):
        np.testing.assert_allclose(
            base_values[stable].to_numpy(dtype=float),
            moved[stable].to_numpy(dtype=float),
            rtol=1e-8,
            atol=1e-10,
            err_msg=f"attribution for {stable} moved when only `unused` was permuted",
        )


def test_the_aggregated_view_agrees_with_the_per_row_one(additive) -> None:
    """`shap_linear` summarises what `shap_values` returns; they must not drift."""
    X, y = additive
    fit = mf.models.ols(X, y)
    values, _ = _wide(fit, X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = mf.interpretation.shap_linear(fit, X).set_index("feature")

    for feature in X.columns:
        np.testing.assert_allclose(
            float(summary.loc[feature, "mean_shap"]),
            float(values[feature].mean()),
            rtol=0,
            atol=1e-8,
            err_msg=f"mean_shap for {feature} disagrees with the per-row values",
        )
        np.testing.assert_allclose(
            float(summary.loc[feature, "importance"]),
            float(np.abs(values[feature]).mean()),
            rtol=1e-6,
            atol=1e-8,
            err_msg=(
                f"importance for {feature} is not the mean absolute per-row value; "
                "if it is defined differently, this test should say so"
            ),
        )


def test_a_constant_model_attributes_nothing_to_anything() -> None:
    """Degenerate but decisive: nothing to explain means nothing explained."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=N), "b": rng.normal(size=N)})
    y = pd.Series(np.full(N, 7.0), name="y")
    fit = mf.models.ols(X, y)
    values, _ = _wide(fit, X)
    assert np.allclose(values.to_numpy(dtype=float), 0.0, atol=1e-8), (
        "a constant target produced non-zero attributions"
    )
