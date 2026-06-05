"""Stage 0: pipeline spec schema, t-code -> target resolution, validation."""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import (
    Arm, CombinationContender, EvalSpec, InterpretSpec, TargetSpec,
    TCODE_TARGET_MAP, contender_names, pipeline_spec, resolve_target,
)


@pytest.mark.parametrize(
    "tcode,policy,transform",
    [
        (1, "direct", "level"),
        (2, "direct_average", "change"),
        (3, "direct_average", "change"),
        (4, "direct", "level"),
        (5, "direct_average", "log_growth"),
        (6, "direct_average", "log_growth"),
        (7, "direct_average", "growth"),
    ],
)
def test_tcode_resolution(tcode, policy, transform):
    rt = resolve_target("INDPRO", tcode=tcode)
    assert (rt.policy, rt.transform) == (policy, transform)
    assert rt.tcode == tcode


def test_explicit_transform_overrides_tcode():
    rt = resolve_target(TargetSpec("CPI", transform="log_growth", policy="direct"), tcode=1)
    assert rt.transform == "log_growth" and rt.policy == "direct"


def test_resolution_reads_tcode_from_bundle_metadata():
    frame = pd.DataFrame(
        {"y": np.arange(60.0), "x": np.arange(60.0)},
        index=pd.date_range("2000-01-31", periods=60, freq="ME", name="date"),
    )
    bundle = mf.data.custom_dataset(frame, transform_codes={"y": 5, "x": 2})
    rt = resolve_target("y", data=bundle)
    assert (rt.policy, rt.transform) == ("direct_average", "log_growth")


def test_missing_tcode_and_transform_raises():
    with pytest.raises(ValueError):
        resolve_target("unknown_series")  # no tcode, no explicit transform


def test_tcode_map_override():
    rt = resolve_target("z", tcode=5, tcode_map={5: ("direct", "level")})
    assert (rt.policy, rt.transform) == ("direct", "level")


def test_contender_names_single_vs_multi_model():
    assert contender_names(Arm("AR", model="ar")) == ["AR"]
    multi = contender_names(Arm("ML", model=["ridge", "lasso"]))
    assert multi == ["ML:ridge", "ML:lasso"]


def _basic_spec(**over):
    frame = pd.DataFrame(
        {"y": np.arange(60.0)}, index=pd.date_range("2000-01-31", periods=60, freq="ME", name="date")
    )
    bundle = mf.data.custom_dataset(frame, transform_codes={"y": 5})
    kw = dict(
        data=bundle, targets=["y"], horizons=[1, 3], window="dummy",
        arms=[Arm("AR", model="ar"), Arm("FM", model="ridge")],
        evaluation=EvalSpec(benchmark="AR"),
    )
    kw.update(over)
    return pipeline_spec(**kw)


def test_pipeline_spec_builds_and_resolves():
    spec = _basic_spec()
    assert spec.horizons == (1, 3)
    assert spec.targets[0].transform == "log_growth"
    assert spec.save_models is True


def test_validation_benchmark_must_exist():
    with pytest.raises(ValueError):
        _basic_spec(evaluation=EvalSpec(benchmark="does_not_exist"))


def test_validation_unique_arm_names():
    with pytest.raises(ValueError):
        _basic_spec(arms=[Arm("X", model="ar"), Arm("X", model="ridge")])


def test_validation_horizons_positive():
    with pytest.raises(ValueError):
        _basic_spec(horizons=[1, 0])


def test_benchmark_can_be_combination_contender():
    spec = _basic_spec(
        combinations=[CombinationContender(name="POOL", method="mean")],
        evaluation=EvalSpec(benchmark="POOL"),
    )
    assert spec.combinations[0].name == "POOL"


def test_save_models_false_with_interpret_warns_in_provenance():
    spec = _basic_spec(
        arms=[Arm("AR", model="ar"), Arm("RF", model="random_forest", interpret=InterpretSpec(methods=("shap",)))],
        save_models=False,
    )
    assert any("interpret" in w for w in spec.provenance.get("warnings", []))
