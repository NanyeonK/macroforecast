"""Density/interval forecasting wired through the managed pipeline (Phase 1).

Reality check (see CHANGELOG [Unreleased] and docs/reference/pipeline.md,
"Density and interval forecasting"): ``variance_prediction``/
``quantile_predictions`` emission onto the forecast table already existed
before this change (``forecasting/policies/base.py::_variance_series``/
``_quantile_frame``, called from the direct policy), and
``macroforecast.metrics.evaluate_forecasts``/``macroforecast.tests`` already
had the full registry-driven density-metric and calibration-test machinery.
What did NOT exist was ``pipeline/evaluate.py::evaluate()`` ever calling any
of it. These tests cover what this lane actually built:

- ``density_table``/``calibration_table``, EvalSpec-gated, wrapping the
  pre-existing ``evaluate_forecasts``/``density_interval_tests``/
  ``interval_coverage_test`` machinery.
- byte-identity of the pre-existing keys (accuracy/significance/mcs) on a
  variance-emitting fixture under a DEFAULT EvalSpec, against a golden
  fixture captured from the base commit (never regenerated from the code
  under test -- see the two ``_golden/density_defaults_*.parquet`` files and
  the pre-existing ``_golden/evalspec_defaults_*.parquet`` point-only case).
- the pipeline-level (not just ``forecasting.run()``-level) emission
  contract: a variance-emitting arm alongside a point-only arm in the SAME
  ``run_pipeline`` call.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, evaluate, pipeline_spec, run_arms, run_pipeline

_GOLDEN_DIR = Path(__file__).parent / "_golden"


def _bundle(n=96):
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, n)
    frame = pd.DataFrame({"y": 1.0 + 2.0 * x + rng.standard_normal(n) * 0.05, "x1": x}, index=idx)
    return mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})


def _window():
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=36),
        val=mf.window.val_last_block(size=12),
        test=mf.window.test_origins(horizon=1, step=3),
    )


class _GoldenVarianceFit:
    """Deterministic fit exposing ``predict_variance`` (same shape as the
    ``x_variance_model`` fixture ``tests/forecasting/test_forecasting.py``
    already anchors the emission contract with)."""

    def __init__(self, level: float):
        self._level = level

    def predict(self, X):
        return np.full(len(X), self._level)

    def predict_variance(self, X):
        return np.full(len(X), 1.0)


def _golden_variance_model(X, y):
    return _GoldenVarianceFit(float(np.asarray(y, dtype=float).mean()))


def _feats():
    return mf.feature_engineering.feature_spec(
        target="y", predictors=["x1"], lags=1, target_lags=(0, 1)
    )


def _spec_point_only(**over):
    """Identical recipe to ``tests/pipeline/test_evalspec_threading.py``'s
    ``_spec()`` -- also what produced ``evalspec_defaults_master.parquet``."""
    kw = dict(
        data=_bundle(), targets=["y"], horizons=[1], window=_window(),
        arms=[Arm("AR", model="ar", features=_feats()),
              Arm("OLS", model="ols", features=_feats(), nested_in_benchmark=True),
              Arm("RIDGE", model="ridge", features=_feats(), nested_in_benchmark=True)],
        evaluation=EvalSpec(benchmark="AR"), save_models=False,
    )
    kw.update(over)
    return pipeline_spec(**kw)


def _spec_with_variance_arm(**over):
    """Same recipe plus a fourth, variance-emitting arm -- also what produced
    ``density_defaults_master.parquet`` (generated from the base commit, see
    that fixture's generation note in CHANGELOG [Unreleased])."""
    variance_spec = mf.models.ModelSpec(
        name="var_model", family="test", fit_func=_golden_variance_model,
    )
    kw = dict(
        data=_bundle(), targets=["y"], horizons=[1], window=_window(),
        arms=[Arm("AR", model="ar", features=_feats()),
              Arm("OLS", model="ols", features=_feats(), nested_in_benchmark=True),
              Arm("RIDGE", model="ridge", features=_feats(), nested_in_benchmark=True),
              Arm("VAR_MODEL", model=variance_spec, features=_feats())],
        evaluation=EvalSpec(benchmark="AR"), save_models=False,
    )
    kw.update(over)
    return pipeline_spec(**kw)


def _point_only_golden_master() -> pd.DataFrame:
    return pd.read_parquet(_GOLDEN_DIR / "evalspec_defaults_master.parquet")


def _variance_golden_master() -> pd.DataFrame:
    return pd.read_parquet(_GOLDEN_DIR / "density_defaults_master.parquet")


# --------------------------------------------------------------------------- #
# 1. byte-identity of forecasts/accuracy/significance/mcs, default EvalSpec,
#    on BOTH a point-only fixture and a variance-emitting-model fixture.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("table", ["accuracy", "significance", "mcs"])
def test_default_evalspec_byte_identical_point_only_golden(table):
    """Point-only fixture: unaffected by this lane's changes at all."""
    spec = _spec_point_only()
    master = _point_only_golden_master()
    got = evaluate(master, spec)[table].reset_index(drop=True)
    golden = pd.read_parquet(_GOLDEN_DIR / f"evalspec_defaults_{table}.parquet").reset_index(drop=True)
    pd.testing.assert_frame_equal(got, golden, atol=1e-12)


@pytest.mark.parametrize("table", ["accuracy", "significance", "mcs"])
def test_default_evalspec_byte_identical_variance_emitting_golden(table):
    """Variance-emitting fixture, DEFAULT EvalSpec (no density metric
    requested): accuracy/significance/mcs must be byte-identical to the base
    commit's own evaluate() on the SAME master frame, even though that frame
    now carries a real (non-null) variance_prediction column.
    """
    spec = _spec_with_variance_arm()
    master = _variance_golden_master()
    assert master["variance_prediction"].notna().any()
    got = evaluate(master, spec)[table].reset_index(drop=True)
    golden = pd.read_parquet(_GOLDEN_DIR / f"density_defaults_{table}.parquet").reset_index(drop=True)
    pd.testing.assert_frame_equal(got, golden, atol=1e-12)


def test_default_evalspec_density_and_calibration_empty_regardless_of_columns():
    """Neither fixture requests a density metric/calibration test by default,
    so density/calibration are empty for BOTH -- point-only and
    variance-emitting alike.
    """
    for spec, master in [
        (_spec_point_only(), _point_only_golden_master()),
        (_spec_with_variance_arm(), _variance_golden_master()),
    ]:
        res = evaluate(master, spec)
        assert res["density"].empty
        assert res["calibration"].empty


# --------------------------------------------------------------------------- #
# 2. density_table: requested density metrics compute, gracefully degrade for
#    contenders without the column, and hand-computed CRPS matches.
# --------------------------------------------------------------------------- #

def test_density_table_computes_gaussian_bundle_for_variance_emitting_contender():
    spec = _spec_with_variance_arm(
        evaluation=EvalSpec(benchmark="AR", metrics=("rmse", "crps"))
    )
    master = _variance_golden_master()
    res = evaluate(master, spec)
    density = res["density"]
    assert not density.empty
    # requesting "crps" also gets the bundled "gaussian_nll" (evaluate_forecasts's
    # own pre-existing, documented bundling behavior -- see density_table's docstring).
    assert {"crps", "gaussian_nll"} <= set(density.columns)

    var_row = density.loc[density["contender"] == "VAR_MODEL"].iloc[0]
    assert np.isfinite(var_row["crps"]) and var_row["crps"] > 0.0
    assert np.isfinite(var_row["gaussian_nll"])

    # AR/OLS/RIDGE never emit a variance -> gracefully NaN, no crash.
    for contender in ("AR", "OLS", "RIDGE"):
        rows = density.loc[density["contender"] == contender]
        if not rows.empty:
            assert rows["crps"].isna().all()


def test_hand_computed_gaussian_crps_matches_density_table():
    """CRPS for a Gaussian predictive density has a closed form (Gneiting &
    Raftery 2007); hand-compute it independently of ``macroforecast.metrics``
    and compare to what ``density_table`` reports.
    """
    spec = _spec_with_variance_arm(
        evaluation=EvalSpec(benchmark="AR", metrics=("rmse", "crps"))
    )
    master = _variance_golden_master()
    sub = (
        master.loc[master["contender"] == "VAR_MODEL", ["actual", "prediction", "variance_prediction"]]
        .dropna()
    )
    assert len(sub) > 0
    sigma = np.sqrt(sub["variance_prediction"].to_numpy(dtype=float))
    z = (sub["actual"].to_numpy(dtype=float) - sub["prediction"].to_numpy(dtype=float)) / sigma
    hand_crps = float(
        np.mean(sigma * (z * (2.0 * sps.norm.cdf(z) - 1.0) + 2.0 * sps.norm.pdf(z) - 1.0 / np.sqrt(np.pi)))
    )

    res = evaluate(master, spec)
    got = res["density"].loc[res["density"]["contender"] == "VAR_MODEL", "crps"].iloc[0]
    assert np.isclose(got, hand_crps, atol=1e-10)

    # cross-check against macroforecast.metrics.crps directly too.
    from macroforecast.metrics import crps as crps_fn

    assert np.isclose(
        crps_fn(sub["actual"], sub["prediction"], sub["variance_prediction"]), hand_crps
    )


def test_density_metric_without_variance_column_raises_actionable_error():
    """Requesting a density metric on a forecast frame with NO
    variance_prediction column at all raises the actionable ValueError
    evaluate_forecasts already raises -- not a silent NaN/empty result.
    """
    spec = _spec_with_variance_arm(
        evaluation=EvalSpec(benchmark="AR", metrics=("rmse", "crps"))
    )
    master = _point_only_golden_master()  # no variance_prediction column at all
    with pytest.raises(ValueError, match="variance_prediction"):
        evaluate(master, spec)


def test_density_metric_not_requested_is_a_true_no_op():
    """density_table must not even inspect master's columns unless a density
    metric is actually requested -- a point-only master with no
    variance_prediction column and a default EvalSpec must not raise.
    """
    spec = _spec_point_only()
    master = _point_only_golden_master()
    res = evaluate(master, spec)
    assert res["density"].empty


# --------------------------------------------------------------------------- #
# 3. calibration_table: constructed intervals (coverage) and known-calibrated
#    vs known-miscalibrated PITs (berkowitz / pit_autocorr).
# --------------------------------------------------------------------------- #

class _EvalSpecDouble:
    """Duck-typed spec carrying only ``evaluation`` -- the same "spec double"
    convention ``pipeline/evaluate.py`` already documents (see
    ``_eval_metrics``/``_eval_tests``'s module comment): calibration_table
    only reads ``spec.evaluation``, so a synthetic frame test does not need a
    full ``PipelineSpec``/window/data machinery.
    """

    def __init__(self, evaluation):
        self.evaluation = evaluation


def _calibration_master(actual, prediction, variance) -> pd.DataFrame:
    n = len(actual)
    return pd.DataFrame({
        "target": ["y"] * n, "horizon": [1] * n, "contender": ["M"] * n,
        "actual": actual, "prediction": prediction, "variance_prediction": variance,
    })


def test_berkowitz_and_pit_autocorr_distinguish_calibrated_from_miscalibrated():
    rng = np.random.default_rng(1234)
    n = 300
    prediction = np.zeros(n)
    variance = np.ones(n)

    # Well-calibrated: actual literally drawn from N(prediction, variance), so
    # the Gaussian PIT is Uniform(0,1) with no serial dependence by construction.
    calibrated_actual = prediction + np.sqrt(variance) * rng.standard_normal(n)

    # Miscalibrated: true variance is 9x the REPORTED variance, and residuals
    # are strongly autocorrelated (AR(1), coefficient 0.85) -- both the
    # variance mismatch and the serial dependence should be caught.
    eps = rng.standard_normal(n)
    resid = np.zeros(n)
    for t in range(1, n):
        resid[t] = 0.85 * resid[t - 1] + eps[t]
    resid = resid / resid.std()
    miscalibrated_actual = prediction + 3.0 * np.sqrt(variance) * resid

    spec = _EvalSpecDouble(EvalSpec(benchmark="M", tests=("berkowitz", "pit_autocorr")))

    from macroforecast.pipeline.evaluate import calibration_table

    calibrated = calibration_table(_calibration_master(calibrated_actual, prediction, variance), spec)
    miscalibrated = calibration_table(_calibration_master(miscalibrated_actual, prediction, variance), spec)

    assert set(calibrated["test"]) == {"berkowitz", "pit_autocorr"}
    assert set(miscalibrated["test"]) == {"berkowitz", "pit_autocorr"}

    cal_berkowitz = calibrated.loc[calibrated["test"] == "berkowitz"].iloc[0]
    mis_berkowitz = miscalibrated.loc[miscalibrated["test"] == "berkowitz"].iloc[0]
    assert cal_berkowitz["reject"] is False or cal_berkowitz["reject"] == False  # noqa: E712
    assert mis_berkowitz["reject"] is True or mis_berkowitz["reject"] == True  # noqa: E712

    cal_pit_autocorr = calibrated.loc[calibrated["test"] == "pit_autocorr"].iloc[0]
    mis_pit_autocorr = miscalibrated.loc[miscalibrated["test"] == "pit_autocorr"].iloc[0]
    assert not bool(cal_pit_autocorr["reject"])
    assert bool(mis_pit_autocorr["reject"])


def test_coverage_on_constructed_intervals():
    """Constructed [lower, upper] intervals with a KNOWN empirical coverage.

    The 0.05/0.95 quantile pair implies a nominal 90% interval, so the Kupiec
    POF test (which is TWO-sided: over-coverage rejects just like
    under-coverage) must NOT reject when the empirical miss rate is exactly
    the nominal 10%, and MUST reject at a 50% miss rate.
    """
    n = 200
    actual = np.zeros(n)

    def _frame(miss_fraction: float) -> pd.DataFrame:
        n_miss = int(round(n * miss_fraction))
        lower = np.full(n, -1.0)
        upper = np.full(n, 1.0)
        # deterministic, evenly-spaced misses so the interval bounds themselves
        # stay simple constants (only which rows fall outside varies).
        miss_idx = np.linspace(0, n - 1, n_miss, dtype=int) if n_miss else np.array([], dtype=int)
        actual_vals = actual.copy()
        actual_vals[miss_idx] = 5.0  # well outside [-1, 1]
        quantile_predictions = [{"0.05": lo, "0.95": hi} for lo, hi in zip(lower, upper)]
        return pd.DataFrame({
            "target": ["y"] * n, "horizon": [1] * n, "contender": ["M"] * n,
            "actual": actual_vals, "quantile_predictions": quantile_predictions,
        })

    spec = _EvalSpecDouble(EvalSpec(benchmark="M", tests=("coverage",)))
    from macroforecast.pipeline.evaluate import calibration_table

    good = calibration_table(_frame(miss_fraction=0.10), spec)  # exactly nominal
    bad = calibration_table(_frame(miss_fraction=0.5), spec)

    good_row = good.loc[good["test"] == "coverage"].iloc[0]
    bad_row = bad.loc[bad["test"] == "coverage"].iloc[0]

    assert np.isclose(good_row["coverage_rate"], 0.9)
    assert np.isclose(bad_row["coverage_rate"], 0.5)
    assert not bool(good_row["reject"])
    assert bool(bad_row["reject"])


def test_calibration_test_without_required_column_raises_actionable_error():
    from macroforecast.pipeline.evaluate import calibration_table

    master = _point_only_golden_master()  # no variance_prediction / quantile_predictions
    spec = _EvalSpecDouble(EvalSpec(benchmark="AR", tests=("dm", "berkowitz")))
    with pytest.raises(ValueError, match="variance_prediction"):
        calibration_table(master, spec)

    spec_cov = _EvalSpecDouble(EvalSpec(benchmark="AR", tests=("coverage",)))
    with pytest.raises(ValueError, match="quantile_predictions"):
        calibration_table(master, spec_cov)


def test_calibration_not_requested_is_a_true_no_op():
    spec = _EvalSpecDouble(EvalSpec(benchmark="AR", tests=("dm", "cw", "mcs")))
    from macroforecast.pipeline.evaluate import calibration_table

    master = _point_only_golden_master()
    assert calibration_table(master, spec).empty


# --------------------------------------------------------------------------- #
# 4. pipeline-level (not just forecasting.run()-level) emission: a
#    variance-emitting arm alongside a point-only arm in ONE run_pipeline call.
# --------------------------------------------------------------------------- #

def test_run_pipeline_emission_variance_present_for_one_arm_absent_for_others():
    spec = _spec_with_variance_arm()
    report = run_pipeline(spec)
    forecasts = report.forecasts

    var_rows = forecasts.loc[forecasts["contender"] == "VAR_MODEL", "variance_prediction"]
    assert not var_rows.empty
    assert var_rows.notna().all()
    assert (var_rows.to_numpy(dtype=float) > 0.0).all()

    for contender in ("AR", "OLS", "RIDGE"):
        rows = forecasts.loc[forecasts["contender"] == contender, "variance_prediction"]
        assert not rows.empty
        assert rows.isna().all()

    # report.density/report.calibration exist (not None) and are empty under
    # the default EvalSpec this spec uses.
    assert isinstance(report.density, pd.DataFrame)
    assert isinstance(report.calibration, pd.DataFrame)
    assert report.density.empty
    assert report.calibration.empty


def test_run_pipeline_with_requested_density_metric_populates_report_density():
    spec = _spec_with_variance_arm(evaluation=EvalSpec(benchmark="AR", metrics=("rmse", "crps")))
    report = run_pipeline(spec)
    assert not report.density.empty
    assert "crps" in report.density.columns
    var_row = report.density.loc[report.density["contender"] == "VAR_MODEL"].iloc[0]
    assert np.isfinite(var_row["crps"])


def test_requested_density_metric_does_not_perturb_point_tables():
    """metrics=("rmse","relative_mse","r2_oos","crps") must produce the SAME
    accuracy/significance/mcs as the plain default -- the density metric is
    routed to report.density, never through the point accuracy path.
    """
    master = _variance_golden_master()
    default = evaluate(master, _spec_with_variance_arm())
    with_density = evaluate(master, _spec_with_variance_arm(
        evaluation=EvalSpec(
            benchmark="AR", metrics=("rmse", "relative_mse", "r2_oos", "crps")
        )
    ))
    for table in ("accuracy", "significance", "mcs"):
        pd.testing.assert_frame_equal(
            default[table].reset_index(drop=True),
            with_density[table].reset_index(drop=True),
            atol=1e-12,
        )
    assert default["density"].empty
    assert not with_density["density"].empty


def test_run_pipeline_quantile_metrics_and_coverage_end_to_end():
    """A natively quantile-emitting model (quantile_regression_forest) through
    the FULL managed pipeline: requesting pinball_loss lands the per-level
    pinball columns plus the symmetric-pair coverage/interval bundle in
    report.density, and the 'coverage' calibration test scores the QRF's
    empirical interval coverage while skipping the quantile-less AR arm.
    """
    spec = pipeline_spec(
        data=_bundle(), targets=["y"], horizons=[1], window=mf.window.spec(
            estimation=mf.window.estimation_expanding(min_size=36),
            val=mf.window.val_last_block(size=12),
            test=mf.window.test_origins(horizon=1, step=6),
        ),
        arms=[Arm("AR", model="ar", features=_feats()),
              Arm("QRF", model="quantile_regression_forest", features=_feats(),
                  params={"n_estimators": 10, "random_state": 0,
                          "quantile_levels": (0.05, 0.5, 0.95)},
                  model_selection=None)],
        evaluation=EvalSpec(benchmark="AR", metrics=("rmse", "pinball_loss"),
                            tests=("dm", "coverage")),
        save_models=False,
    )
    report = run_pipeline(spec)

    density = report.density
    assert not density.empty
    qrf = density.loc[density["contender"] == "QRF"].iloc[0]
    for column in ("pinball_loss_q0_05", "pinball_loss_q0_5", "pinball_loss_q0_95",
                   "coverage_q0_05_q0_95", "interval_score_q0_05_q0_95"):
        assert column in density.columns
        assert np.isfinite(qrf[column])
    ar = density.loc[density["contender"] == "AR"].iloc[0]
    assert pd.isna(ar["pinball_loss_q0_5"])  # AR emits no quantiles -> NaN

    calibration = report.calibration
    assert set(calibration["contender"]) == {"QRF"}  # AR skipped, not raised on
    row = calibration.loc[calibration["test"] == "coverage"].iloc[0]
    assert 0.0 <= row["coverage_rate"] <= 1.0
    assert row["n_obs"] > 0


# --------------------------------------------------------------------------- #
# 5. rescore() end-to-end: a checkpointed variance-emitting run rescored from
#    disk alone reproduces the live run's density table.
# --------------------------------------------------------------------------- #

def test_rescore_reproduces_density_table_from_checkpoint(tmp_path):
    from macroforecast.pipeline import rescore

    def _spec_ckpt(ckpt_dir):
        variance_spec = mf.models.ModelSpec(
            name="var_model", family="test", fit_func=_golden_variance_model,
        )
        return pipeline_spec(
            data=_bundle(), targets=["y"], horizons=[1], window=_window(),
            arms=[Arm("AR", model="ar", features=_feats()),
                  Arm("VAR_MODEL", model=variance_spec, features=_feats())],
            evaluation=EvalSpec(benchmark="AR", metrics=("rmse", "crps")),
            checkpoint_dir=str(ckpt_dir), save_models=False,
        )

    ckpt_dir = tmp_path / "ckpt"
    live = run_pipeline(_spec_ckpt(ckpt_dir))
    assert not live.density.empty

    rescored = rescore(ckpt_dir, _spec_ckpt(ckpt_dir))
    assert not rescored.density.empty

    key = ["target", "horizon", "contender"]
    live_density = live.density.sort_values(key).reset_index(drop=True)
    re_density = rescored.density.sort_values(key).reset_index(drop=True)
    # same contenders, same crps values -- the checkpoint carried the variance.
    assert list(live_density["contender"]) == list(re_density["contender"])
    live_crps = live_density.set_index("contender")["crps"]
    re_crps = re_density.set_index("contender")["crps"]
    assert np.isfinite(live_crps["VAR_MODEL"])
    assert np.isclose(re_crps["VAR_MODEL"], live_crps["VAR_MODEL"], atol=1e-12)
    assert pd.isna(live_crps["AR"]) and pd.isna(re_crps["AR"])
