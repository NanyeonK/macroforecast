"""EvalSpec.metrics / EvalSpec.tests / EvalSpec.loss are actually consumed.

Before this change, ``pipeline/evaluate.py`` read only ``spec.evaluation.benchmark``
/ ``cw_for_nested`` / ``mcs_alpha`` -- ``EvalSpec.metrics`` and ``EvalSpec.tests``
were declared fields that were silently ignored, so a user passing
``EvalSpec(metrics=("mae",), tests=("dm",))`` got the hard-coded rmse/relative_mse/
r2_oos + dm/cw/mcs outputs regardless. These tests pin:

- the golden byte-identity of the DEFAULT EvalSpec's ``evaluate()`` output against
  a fixture captured from the pre-threading code at the base commit
  (``master.parquet`` is the pinned master forecast frame, subset to the plain
  columns ``evaluate()`` reads -- the full frame carries struct columns parquet
  cannot hold; ``accuracy``/``significance``/``mcs``.parquet are the BASE
  COMMIT's own ``evaluate()`` output, verified identical whether computed on the
  full or the subset master -- see ``_golden/evalspec_defaults_*.parquet``;
  the files sit directly in ``_golden/`` because the repo ``.gitignore``
  un-ignores exactly ``tests/**/_golden/*.parquet``);
- that a custom metric (string-registry or callable) actually appears as its own
  accuracy column with the hand-computed value, and that requesting fewer metrics
  actually removes the defaults (the historical bug);
- that ``EvalSpec.tests`` actually gates which significance tests run, and that an
  unsupported name raises at ``pipeline_spec`` build time;
- that ``EvalSpec.loss`` threads into the DM loss differential (matched against a
  hand-computed ``mf.tests.dm_test`` call) and the MCS loss matrix, while Clark-West
  -- invalid under non-quadratic loss -- is skipped with a ``UserWarning`` instead
  of silently computed against the wrong loss;
- that ``rescore()`` honors a custom metric automatically, since it just calls
  ``evaluate()``.

To regenerate the golden fixture after an INTENTIONAL change to the DEFAULT
accuracy/significance/mcs formulas: build the fixture spec below, ``run_arms`` it,
subset the master frame to the columns in ``master.parquet`` and write it, then
run the CURRENT-DEFAULT-BEHAVIOR commit's own ``evaluate(master, spec)`` on that
pinned master and write its accuracy/significance/mcs -- never regenerate the
"before" side from the code under test, that would defeat the point of the test.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import (
    Arm, EvalSpec, TargetSpec, evaluate, pipeline_spec, rescore, run_pipeline,
)

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


def _spec(**over):
    """Same fixture ``test_evaluate_stage2.py`` uses -- also what generated the
    golden fixture's pinned master frame, so keep this identical to that recipe.
    """
    feats = mf.feature_engineering.feature_spec(target="y", predictors=["x1"], lags=1, target_lags=(0, 1))
    kw = dict(
        data=_bundle(), targets=["y"], horizons=[1], window=_window(),
        arms=[Arm("AR", model="ar", features=feats),
              Arm("OLS", model="ols", features=feats, nested_in_benchmark=True),
              Arm("RIDGE", model="ridge", features=feats, nested_in_benchmark=True)],
        evaluation=EvalSpec(benchmark="AR"), save_models=False,
    )
    kw.update(over)
    return pipeline_spec(**kw)


def _golden_master() -> pd.DataFrame:
    return pd.read_parquet(_GOLDEN_DIR / "evalspec_defaults_master.parquet")


# --------------------------------------------------------------------------- #
# 1. byte-identity of the default (unaffected) path
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("table", ["accuracy", "significance", "mcs"])
def test_default_evalspec_is_byte_identical_to_pre_threading_golden(table):
    """The default ``EvalSpec()`` must produce EXACTLY what the pre-threading
    code produced on the same master frame -- rmse/relative_mse/r2_oos and
    dm/cw/mcs, unchanged formulas, unchanged column order.
    """
    spec = _spec()
    master = _golden_master()
    got = evaluate(master, spec)[table].reset_index(drop=True)
    golden = pd.read_parquet(
        _GOLDEN_DIR / f"evalspec_defaults_{table}.parquet"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, golden, atol=1e-12)


# --------------------------------------------------------------------------- #
# 2. EvalSpec.metrics is consumed
# --------------------------------------------------------------------------- #

def test_metrics_reduced_to_one_name_drops_the_other_defaults():
    """The historical bug: ``metrics=("mae",)`` used to silently still produce
    rmse/relative_mse/r2_oos. Now the user gets exactly what they asked for.
    """
    from macroforecast.metrics import mae as mae_fn

    spec = _spec(evaluation=EvalSpec(benchmark="AR", metrics=("mae",)))
    master = _golden_master()
    acc = evaluate(master, spec)["accuracy"]

    assert list(acc.columns) == [
        "target", "horizon", "contender", "mae", "n_common", "is_benchmark", "benchmark_present",
    ]
    for contender in ["AR", "OLS", "RIDGE"]:
        sub = master[master["contender"] == contender]
        expected = mae_fn(sub["actual"].to_numpy(dtype=float), sub["prediction"].to_numpy(dtype=float))
        got = acc.loc[acc["contender"] == contender, "mae"].iloc[0]
        assert np.isclose(got, expected)


def test_custom_callable_metric_appears_as_its_own_column():
    def double_absolute_error(y_true, y_pred):
        return float(np.mean(2.0 * np.abs(np.asarray(y_true) - np.asarray(y_pred))))

    spec = _spec(evaluation=EvalSpec(benchmark="AR", metrics=("rmse", double_absolute_error)))
    master = _golden_master()
    acc = evaluate(master, spec)["accuracy"]

    assert list(acc.columns) == [
        "target", "horizon", "contender", "rmse", "double_absolute_error",
        "n_common", "is_benchmark", "benchmark_present",
    ]
    for contender in ["AR", "OLS", "RIDGE"]:
        sub = master[master["contender"] == contender]
        expected = double_absolute_error(sub["actual"].to_numpy(), sub["prediction"].to_numpy())
        got = acc.loc[acc["contender"] == contender, "double_absolute_error"].iloc[0]
        assert np.isclose(got, expected)


def test_unknown_metric_name_raises_via_metric_registry():
    spec = _spec(evaluation=EvalSpec(benchmark="AR", metrics=("not_a_real_metric",)))
    master = _golden_master()
    with pytest.raises(ValueError, match="Unknown metric"):
        evaluate(master, spec)


# --------------------------------------------------------------------------- #
# 3. EvalSpec.tests is consumed
# --------------------------------------------------------------------------- #

def test_tests_dm_only_excludes_cw_and_mcs():
    spec = _spec(evaluation=EvalSpec(benchmark="AR", tests=("dm",)))
    master = _golden_master()
    res = evaluate(master, spec)
    sig = res["significance"]
    assert {"dm_stat", "dm_p"} <= set(sig.columns)
    assert "cw_p" not in sig.columns
    assert res["mcs"].empty


def test_tests_empty_gives_accuracy_only():
    spec = _spec(evaluation=EvalSpec(benchmark="AR", tests=()))
    master = _golden_master()
    res = evaluate(master, spec)
    assert not res["accuracy"].empty
    assert res["significance"].empty
    assert res["mcs"].empty


def test_unknown_test_name_raises_at_spec_build_time():
    with pytest.raises(ValueError, match="unsupported"):
        _spec(evaluation=EvalSpec(benchmark="AR", tests=("dm", "not_a_real_test")))


def test_test_options_must_reference_requested_tests():
    with pytest.raises(ValueError, match="test_options"):
        _spec(
            evaluation=EvalSpec(
                benchmark="AR",
                tests=("dm",),
                test_options={"mcs": {"n_boot": 25}},
            )
        )


def test_test_options_validate_underlying_callable_kwargs():
    spec = _spec(
        evaluation=EvalSpec(
            benchmark="AR",
            tests=("dm",),
            test_options={"dm": {"kernel": "bartlett", "correction": "hln"}},
        )
    )
    assert spec.evaluation.test_options["dm"]["kernel"] == "bartlett"

    with pytest.raises(ValueError, match="unsupported option"):
        _spec(
            evaluation=EvalSpec(
                benchmark="AR",
                tests=("dm",),
                test_options={"dm": {"not_a_dm_kwarg": True}},
            )
        )


# --------------------------------------------------------------------------- #
# 4. EvalSpec.loss threads into DM/MCS; CW is skipped (with warning) under it
# --------------------------------------------------------------------------- #

def _abs_loss(y_true, y_pred):
    return np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float))


def test_custom_loss_matches_hand_computed_dm_and_skips_cw_with_warning():
    from macroforecast.tests import dm_test

    spec = _spec(evaluation=EvalSpec(benchmark="AR", tests=("dm", "cw", "mcs"), loss=_abs_loss))
    master = _golden_master()

    with pytest.warns(UserWarning, match="Clark-West"):
        res = evaluate(master, spec)

    sig = res["significance"]
    assert {"dm_stat", "dm_p"} <= set(sig.columns)
    assert "cw_p" not in sig.columns  # skipped globally, not silently computed
    # MCS is loss-agnostic and keeps running under the custom loss.
    assert not res["mcs"].empty

    bench = master[master["contender"] == "AR"].set_index("origin")[["prediction", "actual"]]
    for contender in ["OLS", "RIDGE"]:
        c = master[master["contender"] == contender].set_index("origin")[["prediction"]]
        common = bench.index.intersection(c.index)
        y = bench.loc[common, "actual"].to_numpy(dtype=float)
        fb = bench.loc[common, "prediction"].to_numpy(dtype=float)
        fc = c.loc[common, "prediction"].to_numpy(dtype=float)
        loss_b = _abs_loss(y, fb)
        loss_c = _abs_loss(y, fc)
        expected = dm_test(loss_c, loss_b, horizon=1, input_type="loss")
        row = sig[sig["contender"] == contender].iloc[0]
        assert np.isclose(row["dm_stat"], expected.statistic)
        assert np.isclose(row["dm_p"], expected.p_value)


def test_no_custom_loss_no_warning_and_cw_present(recwarn):
    """Sanity counterpart: without a custom loss, CW runs as before, no warning."""
    spec = _spec(evaluation=EvalSpec(benchmark="AR", tests=("dm", "cw", "mcs")))
    master = _golden_master()
    res = evaluate(master, spec)
    cw_warnings = [w for w in recwarn.list if "Clark-West" in str(w.message)]
    assert not cw_warnings
    assert "cw_p" in res["significance"].columns


# --------------------------------------------------------------------------- #
# 5. rescore() honors a custom metric automatically (it just calls evaluate())
# --------------------------------------------------------------------------- #

def _mape_like(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_pred - y_true) / y_true)))


def _rescore_toy(checkpoint_dir):
    idx = pd.date_range("1990-01-01", periods=140, freq="MS")
    rng = np.random.default_rng(5)
    cols = {f"S{i}": rng.normal(size=140) for i in range(5)}
    cols["Y"] = np.cumsum(rng.normal(size=140)) + 50.0  # keep away from 0 for the ratio metric
    panel = pd.DataFrame(cols, index=idx)
    panel.index.name = "date"
    bundle = mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})
    win = mf.window.from_cutoffs(
        test_start="2000-01-01", test_end="2000-12-01", mode="expanding",
        val_method="last_block", retrain_every=1,
    )
    feats = mf.feature_engineering.feature_spec(target="Y", predictors="all", lags=range(1, 3))
    return pipeline_spec(
        data=bundle, targets=[TargetSpec(name="Y")], horizons=[1, 3], window=win,
        arms=[Arm(name="RIDGE", model="ridge", features=feats, is_benchmark=True),
              Arm(name="LASSO", model="lasso", features=feats)],
        evaluation=EvalSpec(benchmark="RIDGE", metrics=("rmse", _mape_like)),
        checkpoint_dir=(str(checkpoint_dir) if checkpoint_dir is not None else None),
        save_models=False,
    )


def test_rescore_roundtrip_honors_custom_metric(tmp_path):
    ckpt = tmp_path / "ckpt"
    live = run_pipeline(_rescore_toy(ckpt))
    assert "_mape_like" in live.accuracy.columns

    rescored = rescore(ckpt, _rescore_toy(ckpt))
    assert "_mape_like" in rescored.accuracy.columns

    key = ["target", "horizon", "contender"]
    live_acc = live.accuracy.sort_values(key).reset_index(drop=True)
    re_acc = rescored.accuracy.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(live_acc, re_acc, atol=1e-12)
