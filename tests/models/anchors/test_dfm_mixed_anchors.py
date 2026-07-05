"""Independent correctness anchors for ``dfm_mixed_mariano_murasawa``.

WP-V2. Per ``.dev-notes/anchor_coverage/matrix.csv``, this model's
"Mariano-Murasawa" alignment claim was "asserted only as metadata substring,
no executable R (dfms/nowcasting) comparison". Reading
``macroforecast/models/timeseries.py`` (``_MixedFrequencyDFM``,
``dfm_mixed_mariano_murasawa``, ``_prepare_mixed_frequency_panel``) confirms
this is a genuine backend WRAPPER around
``statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ`` -- there is
no hand-rolled Kalman/EM code here to anchor against a paper formula. The
correctness risk is entirely in macroforecast's own glue: column reordering
(monthly-first, quarterly-last), frequency inference, and the parameters
threaded through to ``DynamicFactorMQ``. So the two anchors below target
that glue directly, not statsmodels' own (separately, upstream-tested)
EM/Kalman machinery.

1. ``test_parameter_passthrough_matches_direct_statsmodels_call``: builds a
   monthly+quarterly panel already in the exact shape
   ``_prepare_mixed_frequency_panel`` would produce (so the glue is a no-op
   for this fixture), fits it through ``dfm_mixed_mariano_murasawa``, and
   separately calls ``DynamicFactorMQ`` directly with identical arguments on
   the identical frame. Parameters, log-likelihood, and fitted values must
   match to numerical noise floor (they should be bit-for-bit reproducible
   given both paths construct and fit the exact same statsmodels object with
   the same solver settings).

2. ``test_mixed_frequency_low_noise_oracle_recovers_factor_path``: a single
   monthly latent factor plus a quarterly variable built by the documented
   Mariano-Murasawa [1, 2, 3, 2, 1] temporal-aggregation weights on the
   factor's monthly increments, with small measurement noise (DynamicFactorMQ
   requires a proper, non-degenerate idiosyncratic variance; exactly zero
   noise is degenerate for an EM/Kalman fit, so "low-noise" rather than
   literally noiseless -- documented at the fixture). The filtered monthly
   factor must recover the true factor path up to the standard PCA/DFM sign
   and scale indeterminacy (checked via correlation, not raw value
   equality).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


# ---------------------------------------------------------------------------
# Anchor 1: parameter pass-through vs a direct statsmodels DynamicFactorMQ call.
# ---------------------------------------------------------------------------


def _mixed_frequency_panel(n_months: int = 60, seed: int = 0):
    """Monthly panel with 2 monthly series + 1 quarterly series, already in
    ``_prepare_mixed_frequency_panel``'s exact output shape: a monthly
    DatetimeIndex (month-end), monthly columns first, quarterly columns
    last, and the quarterly column NaN except at quarter-end months.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-31", periods=n_months, freq="ME")
    factor = np.zeros(n_months)
    for t in range(1, n_months):
        factor[t] = 0.6 * factor[t - 1] + rng.normal(scale=1.0)
    m1 = 1.0 * factor + rng.normal(scale=0.3, size=n_months)
    m2 = -0.7 * factor + rng.normal(scale=0.3, size=n_months)
    quarterly_growth = 0.9 * factor + rng.normal(scale=0.3, size=n_months)

    frame = pd.DataFrame({"m1": m1, "m2": m2, "q1": quarterly_growth}, index=idx)
    frame.index.name = "date"
    is_quarter_end = idx.month.isin([3, 6, 9, 12])
    frame.loc[~is_quarter_end, "q1"] = np.nan
    return frame


@pytest.mark.reference
def test_parameter_passthrough_matches_direct_statsmodels_call():
    from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

    frame = _mixed_frequency_panel()

    fit = mf.models.dfm_mixed_mariano_murasawa(
        frame,
        target="q1",
        monthly_columns=["m1", "m2"],
        quarterly_columns=["q1"],
        n_factors=1,
        factor_order=1,
        idiosyncratic_ar1=True,
        standardize=True,
        maxiter=200,
        tolerance=1e-7,
    )
    wrapped_params = fit.estimator.params_
    wrapped_llf = fit.estimator.llf_
    assert wrapped_params is not None and wrapped_llf is not None

    direct_model = DynamicFactorMQ(
        frame,
        k_endog_monthly=2,
        factors=1,
        factor_orders=1,
        idiosyncratic_ar1=True,
        standardize=True,
    )
    direct_result = direct_model.fit(maxiter=200, tolerance=1e-7, disp=False)

    pd.testing.assert_series_equal(
        wrapped_params.sort_index(), direct_result.params.sort_index(),
        check_exact=False, rtol=1e-10, atol=1e-12,
    )
    assert wrapped_llf == pytest.approx(float(direct_result.llf), rel=1e-10)

    wrapped_fitted = fit.estimator.fitted_target_
    direct_fitted = direct_result.fittedvalues["q1"]
    # DynamicFactorMQ's own `fittedvalues` comes back on a month-START index
    # (its internal PeriodIndex round-trip) while our wrapper's input frame
    # is month-END; that's a cosmetic label difference, not a correctness
    # one, so compare aligned-by-position values, not labels.
    assert len(wrapped_fitted) == len(direct_fitted)
    np.testing.assert_allclose(
        wrapped_fitted.to_numpy(), direct_fitted.to_numpy(), rtol=1e-8, atol=1e-10
    )


@pytest.mark.reference
def test_parameter_passthrough_respects_monthly_columns_order_override():
    """Same anchor with monthly_columns given in a NON-default order, to
    confirm the wrapper's reordering-to-monthly-first-quarterly-last glue
    (the actual risk surface per the module docstring) doesn't silently
    permute values relative to a direct call built the same way.
    """
    from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

    frame = _mixed_frequency_panel(seed=1)
    # Feed columns to the public API in a shuffled order; the wrapper must
    # still produce monthly-first/quarterly-last for statsmodels.
    shuffled = frame[["q1", "m2", "m1"]]

    fit = mf.models.dfm_mixed_mariano_murasawa(
        shuffled,
        target="q1",
        monthly_columns=["m2", "m1"],
        quarterly_columns=["q1"],
        n_factors=1,
        factor_order=1,
        maxiter=200,
        tolerance=1e-7,
    )
    ordered = frame[["m2", "m1", "q1"]]
    direct_model = DynamicFactorMQ(
        ordered, k_endog_monthly=2, factors=1, factor_orders=1,
        idiosyncratic_ar1=True, standardize=True,
    )
    direct_result = direct_model.fit(maxiter=200, tolerance=1e-7, disp=False)

    pd.testing.assert_series_equal(
        fit.estimator.params_.sort_index(), direct_result.params.sort_index(),
        check_exact=False, rtol=1e-10, atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Anchor 2: low-noise Mariano-Murasawa mixed-frequency factor-recovery oracle.
# ---------------------------------------------------------------------------


@pytest.mark.reference
@pytest.mark.slow
def test_mixed_frequency_low_noise_oracle_recovers_factor_path():
    rng = np.random.default_rng(2026)
    n_months = 120
    idx = pd.date_range("2000-01-31", periods=n_months, freq="ME")

    true_factor = np.zeros(n_months)
    for t in range(1, n_months):
        true_factor[t] = 0.7 * true_factor[t - 1] + rng.normal(scale=1.0)
    noise_scale = 0.02
    m1 = true_factor + rng.normal(scale=noise_scale, size=n_months)
    m2 = -1.3 * true_factor + rng.normal(scale=noise_scale, size=n_months)

    # Mariano-Murasawa [1, 2, 3, 2, 1] temporal aggregation of the monthly
    # factor's own level onto quarter-end months (the documented MM weight
    # pattern for a flow/growth-rate quarterly series built from monthly
    # increments), plus small measurement noise. DynamicFactorMQ requires a
    # proper (nonzero) idiosyncratic variance to fit at all; this is
    # deliberately "low-noise", not literally zero.
    weights = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    quarterly_agg = np.full(n_months, np.nan)
    for t in range(4, n_months):
        if idx[t].month in (3, 6, 9, 12):
            quarterly_agg[t] = np.dot(weights, true_factor[t - 4 : t + 1]) + rng.normal(
                scale=noise_scale
            )

    frame = pd.DataFrame({"m1": m1, "m2": m2, "q1": quarterly_agg}, index=idx)
    frame.index.name = "date"

    fit = mf.models.dfm_mixed_mariano_murasawa(
        frame,
        target="q1",
        monthly_columns=["m1", "m2"],
        quarterly_columns=["q1"],
        n_factors=1,
        factor_order=1,
        maxiter=300,
        tolerance=1e-7,
    )
    filtered = fit.estimator.factors_filtered_
    assert filtered is not None and filtered.shape[1] == 1
    recovered = filtered.iloc[:, 0].to_numpy()

    # Sign/scale of the recovered factor are not identified (standard PCA/DFM
    # indeterminacy); compare via correlation, not raw values.
    valid = ~np.isnan(recovered)
    correlation = np.corrcoef(recovered[valid], true_factor[valid])[0, 1]
    assert abs(correlation) > 0.9, correlation
