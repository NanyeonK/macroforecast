"""Regression tests for deferred-interpretation error contracts."""

from __future__ import annotations

import dataclasses
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.forecasting.task import FeatureRetargetError
from macroforecast.pipeline import Arm, InterpretSpec, interpret_pipeline


def _panel() -> pd.DataFrame:
    index = pd.date_range("2000-01-31", periods=36, freq="ME", name="date")
    x = np.linspace(-1.0, 1.0, len(index))
    return pd.DataFrame(
        {"y": 0.5 + 2.0 * x, "y2": -0.5 + x, "x": x}, index=index
    )


def _features(target: str = "y") -> object:
    return mf.feature_engineering.feature_spec(
        target=target, predictors=["x"], lags=(0,), target_lags=(0,)
    )


def _report(arms: list[Arm], targets: tuple[str, ...] = ("y",)) -> SimpleNamespace:
    spec = SimpleNamespace(
        data=_panel(),
        arms=arms,
        targets=[SimpleNamespace(name=name) for name in targets],
        window=SimpleNamespace(estimation=SimpleNamespace(mode="expanding")),
    )
    return SimpleNamespace(spec=spec, interpretation=None)


def _failing_model(name: str = "broken_model") -> object:
    def fail_fit(X: object, y: object) -> object:
        raise RuntimeError(f"{name} fit failure")

    return mf.models.custom_model(name, fail_fit)


def test_fit_failure_is_model_keyed_error_for_every_method() -> None:
    arm = Arm(
        "DISPLAY_ARM",
        model=_failing_model(),
        features=_features(),
        interpret=InterpretSpec(methods=("shap", "ale")),
    )

    out = interpret_pipeline(_report([arm]))

    assert set(out) == {"DISPLAY_ARM"}
    assert set(out["DISPLAY_ARM"]) == {"broken_model"}
    for method in ("shap", "ale"):
        table = out["DISPLAY_ARM"]["broken_model"][method]
        assert list(table.columns) == ["error"]
        assert "broken_model fit failure" in table.loc[0, "error"]


def test_failed_arm_does_not_abort_healthy_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("macroforecast.pipeline.interpret")
    monkeypatch.setattr(
        module,
        "_run_method",
        lambda *args, **kwargs: pd.DataFrame(
            {"feature": ["x"], "importance": [1.0]}
        ),
    )
    good = Arm(
        "GOOD_ARM",
        model=mf.models.custom_model(
            "good_model", lambda X, y: SimpleNamespace(name="fit")
        ),
        features=_features(),
        interpret=InterpretSpec(methods=("shap",)),
    )
    bad = Arm(
        "BAD_ARM",
        model=_failing_model("bad_model"),
        features=_features(),
        interpret=InterpretSpec(methods=("shap",)),
    )

    out = interpret_pipeline(_report([bad, good]))

    assert "error" in out["BAD_ARM"]["bad_model"]["shap"].columns
    assert "error" not in out["GOOD_ARM"]["good_model"]["shap"].columns


def test_fit_failure_keeps_multi_target_model_suffixes() -> None:
    arm = Arm(
        "DISPLAY_ARM",
        model=_failing_model(),
        features=_features(),
        interpret=InterpretSpec(methods=("shap",)),
    )

    out = interpret_pipeline(_report([arm], targets=("y", "y2")))

    assert set(out["DISPLAY_ARM"]) == {"broken_model:y", "broken_model:y2"}
    assert all(
        "error" in methods["shap"].columns
        for methods in out["DISPLAY_ARM"].values()
    )


def test_origin_mean_fit_failure_is_not_dropped() -> None:
    arm = Arm(
        "DISPLAY_ARM",
        model=_failing_model(),
        features=_features(),
        interpret=InterpretSpec(methods=("shap",)),
    )
    report = _report([arm])
    report.spec.window.estimation.mode = "rolling"
    report.spec.window.plan = lambda index: pd.DataFrame(
        {
            "estimation_start_pos": [0, 0],
            "estimation_end_pos": [20, 30],
        }
    )

    out = interpret_pipeline(report)

    table = out["DISPLAY_ARM"]["broken_model"]["shap"]
    assert "broken_model fit failure" in table.loc[0, "error"]


def test_empty_complete_case_design_is_not_dropped() -> None:
    class EmptyBuilder:
        def transform(self, panel: pd.DataFrame) -> SimpleNamespace:
            return SimpleNamespace(
                X=pd.DataFrame(columns=["x"], index=panel.index[:0]),
                y=pd.Series(dtype=float, index=panel.index[:0]),
            )

    class EmptyFeatures:
        target = "y"
        targets: tuple[str, ...] = ()

        def fit(self, panel: pd.DataFrame) -> EmptyBuilder:
            return EmptyBuilder()

    arm = Arm(
        "DISPLAY_ARM",
        model=mf.models.custom_model(
            "empty_model", lambda X, y: SimpleNamespace(name="fit")
        ),
        features=EmptyFeatures(),
        interpret=InterpretSpec(methods=("shap",)),
    )
    report = _report([arm])

    out = interpret_pipeline(report)

    table = out["DISPLAY_ARM"]["empty_model"]["shap"]
    assert "no complete observations" in table.loc[0, "error"]


@dataclasses.dataclass(frozen=True)
class _RefusingFeatures:
    target: str = "y"
    targets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.target != "y":
            raise ValueError("retarget refused")


def test_align_features_fails_closed_with_arm_context() -> None:
    module = importlib.import_module("macroforecast.pipeline.interpret")

    with pytest.raises(FeatureRetargetError, match="RETARGET_ARM"):
        module._align_features(_RefusingFeatures(), "y2", arm_name="RETARGET_ARM")


def test_retarget_failure_becomes_model_keyed_error() -> None:
    arm = Arm(
        "RETARGET_ARM",
        model=mf.models.custom_model(
            "retarget_model", lambda X, y: SimpleNamespace(name="fit")
        ),
        features=_RefusingFeatures(),
        interpret=InterpretSpec(methods=("shap", "ale")),
    )

    out = interpret_pipeline(_report([arm], targets=("y2",)))

    assert set(out["RETARGET_ARM"]) == {"retarget_model"}
    for method in ("shap", "ale"):
        error = out["RETARGET_ARM"]["retarget_model"][method].loc[0, "error"]
        assert "RETARGET_ARM" in error and "y2" in error
