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
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
import macroforecast.pipeline.spec as spec_mod
from macroforecast.pipeline import (
    Arm, EvalSpec, SubsampleWindow, TargetSpec, evaluate, pipeline_spec, rescore, run_pipeline,
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


def _multi_horizon_master() -> pd.DataFrame:
    rows = []
    for origin in range(12):
        for horizon in (1, 2):
            actual = 10.0 + 0.2 * origin + 0.1 * horizon
            rows.extend(
                [
                    {
                        "target": "y", "horizon": horizon, "origin": origin,
                        "date": pd.Timestamp("2000-01-31") + pd.offsets.MonthEnd(origin),
                        "contender": "AR", "prediction": actual + 0.50 + 0.02 * horizon,
                        "actual": actual,
                    },
                    {
                        "target": "y", "horizon": horizon, "origin": origin,
                        "date": pd.Timestamp("2000-01-31") + pd.offsets.MonthEnd(origin),
                        "contender": "OLS", "prediction": actual + 0.15 + 0.01 * ((origin + horizon) % 3),
                        "actual": actual,
                    },
                    {
                        "target": "y", "horizon": horizon, "origin": origin,
                        "date": pd.Timestamp("2000-01-31") + pd.offsets.MonthEnd(origin),
                        "contender": "RIDGE",
                        "prediction": actual + (0.18 if horizon == 1 else 0.46) + 0.01 * (origin % 2),
                        "actual": actual,
                    },
                ]
            )
    return pd.DataFrame(rows)


def _dated_master(n: int = 60) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2018-01-31", periods=n, freq="ME")
    for origin, date in enumerate(dates):
        actual = 1.0 + 0.03 * origin + 0.2 * np.sin(origin / 4.0)
        rows.extend(
            [
                {
                    "target": "y", "horizon": 1, "origin": origin, "date": date,
                    "contender": "AR", "prediction": actual + 0.35, "actual": actual,
                },
                {
                    "target": "y", "horizon": 1, "origin": origin, "date": date,
                    "contender": "OLS", "prediction": actual + 0.10, "actual": actual,
                },
            ]
        )
    return pd.DataFrame(rows)


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


def test_mad_metric_appears_in_accuracy_table_with_hand_oracle():
    from macroforecast.metrics import mad as mad_fn

    spec = _spec(evaluation=EvalSpec(benchmark="AR", metrics=("mad",)))
    master = _golden_master()
    acc = evaluate(master, spec)["accuracy"]

    assert list(acc.columns) == [
        "target", "horizon", "contender", "mad", "n_common", "is_benchmark", "benchmark_present",
    ]
    for contender in ["AR", "OLS", "RIDGE"]:
        sub = master[master["contender"] == contender]
        expected = mad_fn(sub["actual"].to_numpy(dtype=float), sub["prediction"].to_numpy(dtype=float))
        got = acc.loc[acc["contender"] == contender, "mad"].iloc[0]
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


def test_unknown_metric_name_raises_at_spec_build_via_metric_registry():
    with pytest.raises(ValueError, match="Unknown metric"):
        _spec(evaluation=EvalSpec(benchmark="AR", metrics=("not_a_real_metric",)))


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


def test_arch_backed_tests_require_arch_extra(monkeypatch):
    monkeypatch.setattr(spec_mod, "_arch_available", lambda: False)

    with pytest.raises(ImportError) as excinfo:
        _spec(evaluation=EvalSpec(benchmark="AR", tests=("dm", "spa")))

    message = str(excinfo.value)
    assert "spa" in message
    assert 'pip install "macroforecast[arch]"' in message


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


def test_wired_pairwise_tests_emit_long_significance_rows():
    spec = _spec(
        evaluation=EvalSpec(
            benchmark="AR",
            tests=("gw", "enc_new", "enc_t", "pt", "hm", "ag", "gr"),
        )
    )
    sig = evaluate(_golden_master(), spec)["significance"]

    assert {"target", "horizon", "contender", "test", "statistic", "p_value", "reject", "n_obs"} <= set(sig.columns)
    rows = sig.dropna(subset=["test"])
    assert set(rows["test"]) == {"gw", "enc_new", "enc_t", "pt", "hm", "ag", "gr"}
    for name in {"gw", "enc_new", "enc_t", "pt", "hm", "ag", "gr"}:
        assert set(rows.loc[rows["test"] == name, "contender"]) == {"OLS", "RIDGE"}


def test_pairwise_test_options_reach_underlying_callable(monkeypatch):
    spec = _spec(
        evaluation=EvalSpec(
            benchmark="AR",
            tests=("gw",),
            test_options={"gw": {"small_sample": False, "alpha": 0.2}},
        )
    )
    calls = []

    def spy(loss_a, loss_b, *, horizon=1, instruments=None, alpha=0.05, small_sample=True):
        calls.append(
            {
                "horizon": horizon,
                "alpha": alpha,
                "small_sample": small_sample,
                "n": len(loss_a),
            }
        )
        return mf.tests.TestResult(
            statistic=1.0,
            p_value=0.5,
            decision=False,
            alternative="conditional_equal_predictive_ability",
            n_obs=len(loss_a),
        )

    monkeypatch.setattr(mf.tests, "giacomini_white_test", spy)

    sig = evaluate(_golden_master(), spec)["significance"]

    assert len(calls) == 2
    assert {call["small_sample"] for call in calls} == {False}
    assert {call["alpha"] for call in calls} == {0.2}
    assert set(sig["test"]) == {"gw"}
    assert set(sig["statistic"]) == {1.0}


def test_custom_loss_skips_nested_quadratic_pairwise_tests_with_warning():
    spec = _spec(
        evaluation=EvalSpec(
            benchmark="AR",
            tests=("gw", "enc_new", "enc_t"),
            loss=_abs_loss,
        )
    )
    with pytest.warns(UserWarning, match="ENC-NEW"):
        sig = evaluate(_golden_master(), spec)["significance"]

    assert set(sig["test"]) == {"gw"}


def test_set_comparison_tests_land_in_mcs_report():
    pytest.importorskip("arch")

    spec = _spec(
        evaluation=EvalSpec(
            benchmark="AR",
            tests=("spa", "rc", "stepm"),
            test_options={
                "spa": {"n_boot": 25, "block_length": 2},
                "rc": {"n_boot": 25, "block_length": 2},
                "stepm": {"n_boot": 25, "block_length": 2},
            },
        )
    )
    res = evaluate(_golden_master(), spec)

    assert res["significance"].empty
    mcs = res["mcs"]
    assert {"target", "horizon", "contender", "test", "superior", "reject", "n_obs"} <= set(mcs.columns)
    assert set(mcs["test"]) == {"spa", "rc", "stepm"}
    for name in {"spa", "rc", "stepm"}:
        assert set(mcs.loc[mcs["test"] == name, "contender"]) == {"OLS", "RIDGE"}


def test_set_comparison_test_options_reach_underlying_callable(monkeypatch):
    pytest.importorskip("arch")

    spec = _spec(
        evaluation=EvalSpec(
            benchmark="AR",
            tests=("spa",),
            test_options={"spa": {"n_boot": 17, "block_length": 2, "random_state": 123}},
        )
    )
    calls = []

    def spy(loss_panel, *, benchmark, loss="squared_error", alpha=0.05, n_boot=1000,
            block_length="auto", bootstrap_method="stationary_bootstrap",
            p_value_type="consistent", studentize=True, nested=False,
            random_state=0, target="target", horizon="horizon", origin="origin",
            model="model_id"):
        calls.append(
            {
                "benchmark": benchmark,
                "loss": loss,
                "n_boot": n_boot,
                "block_length": block_length,
                "random_state": random_state,
                "rows": len(loss_panel),
            }
        )
        return {
            "records": [
                {
                    "target": "y",
                    "horizon": 1,
                    "benchmark": benchmark,
                    "superior_models": ["OLS"],
                    "decision": True,
                    "p_value": 0.01,
                    "n_obs": 9,
                    "n_models": 2,
                    "mean_loss_difference": {"OLS": 0.25, "RIDGE": -0.10},
                    "status": "computed",
                }
            ]
        }

    monkeypatch.setattr(mf.tests, "superior_predictive_ability_test", spy)

    mcs = evaluate(_golden_master(), spec)["mcs"]

    assert calls == [
        {
            "benchmark": "AR",
            "loss": "squared_error",
            "n_boot": 17,
            "block_length": 2,
            "random_state": 123,
            "rows": len(_golden_master()),
        }
    ]
    assert set(mcs["test"]) == {"spa"}
    assert bool(mcs.loc[mcs["contender"] == "OLS", "superior"].iloc[0]) is True
    assert mcs.loc[mcs["contender"] == "RIDGE", "mean_loss_difference"].iloc[0] == -0.10


def test_joint_multi_horizon_tests_emit_joint_significance_rows_and_do_not_affect_paper_table():
    spec = _spec(
        horizons=[1, 2],
        evaluation=EvalSpec(
            benchmark="AR",
            tests=("dm", "uspa", "aspa"),
            test_options={
                "uspa": {"n_boot": 99, "block_length": 3, "random_state": 7},
                "aspa": {"n_boot": 99, "block_length": 3, "random_state": 7},
            },
        ),
    )
    res = evaluate(_multi_horizon_master(), spec)
    sig = res["significance"]
    joint = sig.loc[sig["horizon"] == "joint"]

    assert set(joint["test"]) == {"uspa", "aspa"}
    for name in {"uspa", "aspa"}:
        assert set(joint.loc[joint["test"] == name, "contender"]) == {"OLS", "RIDGE"}
    assert set(joint["n_horizons"]) == {2}

    table = mf.reporting.paper_accuracy_table(SimpleNamespace(**res))
    assert "hjoint" not in table.data.columns
    assert {"h1", "h2"} <= set(table.data.columns)


def test_joint_multi_horizon_tests_require_multiple_horizons():
    with pytest.raises(ValueError, match="multi-horizon"):
        _spec(evaluation=EvalSpec(benchmark="AR", tests=("uspa",)))


def test_end_to_end_report_with_dm_gw_gr_and_spa_is_coherent():
    pytest.importorskip("arch")

    spec = _spec(
        evaluation=EvalSpec(
            benchmark="AR",
            tests=("dm", "gw", "gr", "spa"),
            test_options={"spa": {"n_boot": 25, "block_length": 2}},
        )
    )
    res = evaluate(_golden_master(), spec)

    assert not res["accuracy"].empty
    assert {"dm_stat", "dm_p"} <= set(res["significance"].columns)
    long = res["significance"].dropna(subset=["test"])
    assert set(long["test"]) == {"gw", "gr"}
    assert set(res["mcs"]["test"]) == {"spa"}
    assert set(res["mcs"]["contender"]) == {"OLS", "RIDGE"}


def test_paper_accuracy_table_ignores_long_form_significance_rows():
    spec = _spec(evaluation=EvalSpec(benchmark="AR", tests=("dm", "pt", "gw")))
    res = evaluate(_golden_master(), spec)

    table = mf.reporting.paper_accuracy_table(SimpleNamespace(**res))

    assert list(table.data.columns) == ["Model", "h1"]
    assert set(table.data["Model"]) == {"AR (benchmark)", "OLS", "RIDGE"}


def test_gr_defaults_horizon_hac_lag_and_respects_override(monkeypatch):
    calls = []

    def spy(loss_a, loss_b, *, method="giacomini_rossi", window_ratio=0.5,
            dmv_fullsample=True, lag_truncate=0, alpha=0.05):
        calls.append({"lag_truncate": lag_truncate, "method": method, "n": len(loss_a)})
        return {
            "statistic": 1.0,
            "decision": False,
            "n_obs": len(loss_a),
            "critical_value": 2.0,
            "window_size": 5,
            "lag_truncate": lag_truncate,
        }

    monkeypatch.setattr(mf.tests, "conditional_predictive_ability_test", spy)
    master = _dated_master(n=40).assign(horizon=6)

    default_spec = _spec(
        horizons=[6],
        evaluation=EvalSpec(benchmark="AR", tests=("gr",)),
    )
    default_sig = evaluate(master, default_spec)["significance"]
    assert {call["lag_truncate"] for call in calls} == {5}
    assert set(default_sig["lag_truncate"]) == {5}
    assert set(default_sig["lag_truncate_source"]) == {"default_min_h_minus_1_5"}

    calls.clear()
    override_spec = _spec(
        horizons=[6],
        evaluation=EvalSpec(
            benchmark="AR",
            tests=("gr",),
            test_options={"gr": {"lag_truncate": 2}},
        ),
    )
    override_sig = evaluate(master, override_spec)["significance"]
    assert {call["lag_truncate"] for call in calls} == {2}
    assert set(override_sig["lag_truncate_source"]) == {"user"}


def test_dm_kwargs_merge_into_test_options_with_test_options_precedence():
    spec = _spec(
        evaluation=EvalSpec(
            benchmark="AR",
            tests=("dm",),
            dm_kwargs={"kernel": "parzen", "alpha": 0.2},
            test_options={"dm": {"kernel": "acf"}},
        )
    )

    assert spec.evaluation.test_options["dm"] == {"kernel": "acf", "alpha": 0.2}


def test_dead_evalspec_fields_raise_at_spec_build_time():
    with pytest.raises(ValueError, match="evaluation.by is not implemented"):
        _spec(evaluation=EvalSpec(benchmark="AR", by=("target",)))
    with pytest.raises(ValueError, match="evaluation.primary_axis is not implemented"):
        _spec(evaluation=EvalSpec(benchmark="AR", primary_axis="model"))
    with pytest.raises(ValueError, match="evaluation.multiple_testing is not implemented"):
        _spec(evaluation=EvalSpec(benchmark="AR", multiple_testing="holm"))


def test_mcs_method_is_not_echoed_as_applied_provenance():
    from macroforecast.pipeline.run import _spec_echo

    echo = _spec_echo(_spec(evaluation=EvalSpec(benchmark="AR", mcs_method="stationary")))

    assert "mcs_method" not in echo["evaluation"]


def test_encompassing_rows_without_reference_values_are_inconclusive():
    spec = _spec(evaluation=EvalSpec(benchmark="AR", tests=("enc_new", "enc_t")))
    sig = evaluate(_golden_master(), spec)["significance"]

    assert set(sig["test"]) == {"enc_new", "enc_t"}
    assert set(sig["status"]) == {"inconclusive"}
    assert sig["reject"].isna().all()


def test_directional_constant_contender_forecast_emits_degenerate_rows():
    master = _dated_master(n=40)
    master.loc[master["contender"] == "OLS", "prediction"] = 1.0
    spec = _spec(evaluation=EvalSpec(benchmark="AR", tests=("pt", "hm", "ag")))

    sig = evaluate(master, spec)["significance"]

    assert set(sig["test"]) == {"pt", "hm", "ag"}
    assert set(sig["status"]) == {"degenerate"}
    assert set(sig["reason"]) == {"constant_forecast"}
    assert sig["reject"].isna().all()


def test_mincer_zarnowitz_pipeline_row_appears_with_horizon_default_hac():
    spec = _spec(evaluation=EvalSpec(benchmark="AR", tests=("mz",)))
    sig = evaluate(_golden_master(), spec)["significance"]

    assert set(sig["test"]) == {"mz"}
    assert {"intercept", "slope", "hac_lags"} <= set(sig.columns)
    assert set(sig["hac_lags"]) == {0}
    assert set(sig["hac_lag_source"]) == {"default_h_minus_1"}


def test_subsamples_split_accuracy_and_significance_by_target_date():
    spec = _spec(
        evaluation=EvalSpec(
            benchmark="AR",
            tests=("dm",),
            subsamples={
                "full": SubsampleWindow(),
                "ex_covid": SubsampleWindow(exclude=(("2020-03-01", "2021-12-31"),)),
                "post_gfc": SubsampleWindow(start="2019-01-01"),
            },
        )
    )

    res = evaluate(_dated_master(n=60), spec)

    assert set(res["accuracy"]["subsample"]) == {"full", "ex_covid", "post_gfc"}
    assert set(res["significance"]["subsample"]) == {"full", "ex_covid", "post_gfc"}
    ex_covid = res["accuracy"].loc[
        (res["accuracy"]["subsample"] == "ex_covid")
        & (res["accuracy"]["contender"] == "OLS")
    ].iloc[0]
    assert int(ex_covid["n_common"]) == 38
    for subsample in {"full", "ex_covid", "post_gfc"}:
        rows = res["significance"].loc[res["significance"]["subsample"] == subsample]
        assert set(rows["contender"]) == {"OLS"}
        assert rows["dm_p"].notna().all()


def test_subsample_validation_rejects_bad_windows():
    with pytest.raises(ValueError, match="parseable date"):
        _spec(
            evaluation=EvalSpec(
                benchmark="AR",
                subsamples={"bad": SubsampleWindow(start="not-a-date")},
            )
        )
    with pytest.raises(ValueError, match="start must be before end"):
        _spec(
            evaluation=EvalSpec(
                benchmark="AR",
                subsamples={"bad": SubsampleWindow(start="2021-01-01", end="2020-01-01")},
            )
        )
    with pytest.raises(ValueError, match="names must be nonempty"):
        _spec(evaluation=EvalSpec(benchmark="AR", subsamples={"": SubsampleWindow()}))


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
