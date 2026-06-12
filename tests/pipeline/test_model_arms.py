"""model_arms: ergonomic multi-model comparison (one Arm per model)."""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import (
    Arm,
    EvalSpec,
    InterpretSpec,
    model_arms,
    pipeline_spec,
    run_pipeline,
)


def test_sequence_of_names_builds_one_arm_per_model():
    arms = model_arms(["ar", "ridge"])
    assert [a.name for a in arms] == ["ar", "ridge"]
    assert [a.model for a in arms] == ["ar", "ridge"]
    assert all(isinstance(a, Arm) for a in arms)
    # shared (None) config by default
    for a in arms:
        assert a.preprocessing is None
        assert a.features is None
        assert a.params is None
        assert a.model_selection is None
        assert a.nested_in_benchmark is False


def test_names_from_mapping():
    arms = model_arms({"benchmark": "ar", "tree": "random_forest"})
    assert [a.name for a in arms] == ["benchmark", "tree"]
    assert [a.model for a in arms] == ["ar", "random_forest"]


def test_custom_names_override_model_names():
    arms = model_arms(["ar", "ols"], names=["A", "B"])
    assert [a.name for a in arms] == ["A", "B"]
    assert [a.model for a in arms] == ["ar", "ols"]


def test_callable_model_default_name():
    def my_model():  # pragma: no cover - never called, only named
        return None

    arms = model_arms([my_model])
    assert arms[0].name == "my_model"
    assert arms[0].model is my_model


def test_shared_config_propagates_to_all_arms():
    feats = mf.feature_engineering.feature_spec(target="y", predictors=["x1"], lags=1)
    arms = model_arms(
        ["ar", "ols"],
        features=feats,
        preprocessing="pp",
        model_selection_metric="rmse",
        interpret=InterpretSpec(methods=("shap",)),
        metadata={"k": "v"},
    )
    for a in arms:
        assert a.features is feats
        assert a.preprocessing == "pp"
        assert a.model_selection_metric == "rmse"
        assert a.interpret.methods == ("shap",)
        assert a.metadata == {"k": "v"}
    # metadata is copied per arm, not shared by reference
    assert arms[0].metadata is not arms[1].metadata


def test_duplicate_names_raise():
    with pytest.raises(ValueError, match="unique"):
        model_arms(["ar", "ols"], names=["X", "X"])


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        model_arms(["ar", "ols"], names=["only_one"])


def test_mapping_with_names_raises():
    with pytest.raises(ValueError, match="Mapping"):
        model_arms({"a": "ar"}, names=["a"])


def test_nested_model_rejected():
    with pytest.raises(ValueError, match="SINGLE model"):
        model_arms([["ar", "ols"]])


def test_shared_params_dict_is_shared_not_per_arm():
    # keys are hyperparameter names, not arm names -> shared by all
    arms = model_arms(["ar", "ols"], params={"alpha": 1.0})
    assert arms[0].params == {"alpha": 1.0}
    assert arms[1].params == {"alpha": 1.0}


def test_per_arm_params_when_keys_match_arm_names():
    arms = model_arms(["ar", "ols"], params={"ar": {"p": 2}, "ols": {"p": 3}})
    assert arms[0].params == {"p": 2}
    assert arms[1].params == {"p": 3}


def test_per_arm_model_selection_when_keys_match():
    arms = model_arms(["ar", "ols"], model_selection={"ar": "grid", "ols": "random"})
    assert arms[0].model_selection == "grid"
    assert arms[1].model_selection == "random"


def test_nested_in_benchmark_bool_marks_all():
    arms = model_arms(["ar", "ols"], nested_in_benchmark=True)
    assert all(a.nested_in_benchmark for a in arms)


def test_nested_in_benchmark_set_marks_only_those():
    arms = model_arms(["ar", "ols", "ridge"], nested_in_benchmark={"ols"})
    flags = {a.name: a.nested_in_benchmark for a in arms}
    assert flags == {"ar": False, "ols": True, "ridge": False}


def test_nested_in_benchmark_unknown_name_raises():
    with pytest.raises(ValueError, match="not"):
        model_arms(["ar", "ols"], nested_in_benchmark={"nope"})


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #


def _bundle(n=96):
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, n)
    frame = pd.DataFrame(
        {"y": 1.0 + 2.0 * x + rng.standard_normal(n) * 0.05, "x1": x}, index=idx
    )
    return mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})


def test_end_to_end_one_contender_per_model():
    feats = mf.feature_engineering.feature_spec(
        target="y", predictors=["x1"], lags=1, target_lags=(0, 1)
    )
    w = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=36),
        val=mf.window.val_last_block(size=12),
        test=mf.window.test_origins(horizon=1, step=3),
    )
    arms = model_arms(["ar", "ols"], features=feats)
    spec = pipeline_spec(
        data=_bundle(),
        targets=["y"],
        horizons=[1],
        window=w,
        arms=arms,
        evaluation=EvalSpec(benchmark="ar"),
        save_models=False,
    )
    report = run_pipeline(spec)
    assert not report.accuracy.empty
    contenders = set(report.accuracy["contender"])
    assert {"ar", "ols"} <= contenders
