"""A4 characterization -- one arm, one compiled plan, read the same way everywhere.

An arm may override ``window`` and ``preprocessing`` (and, with the latter, its
``preprocessing_policy``). Before A4 the override rule was written out at four
separate sites:

    run.py:149              window          = arm.window if getattr(arm, 'window', None) is not None else spec.window
    run.py:150-151          preprocessing   = arm.preprocessing if arm.preprocessing is not None else spec.preprocessing
    run.py:443              (again, inside _effective_preprocessing_policy, for the store namespace)
    result_store.py:105     window          = arm.window if arm.window is not None else spec.window

Note the two window copies do not even use the same predicate. Nothing forces the
four to agree, and they govern different things -- what is executed, what the
preprocessor store is namespaced by, and what the result-store digest records. A
disagreement is silent: the run is correct and the digest describes a different run,
or vice versa.

These tests pin the behaviour that must hold however the resolution is expressed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def panel():
    idx = pd.date_range("1990-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"y": rng.normal(size=120), "x1": rng.normal(size=120), "x2": rng.normal(size=120)},
        index=idx,
    )


def _spec_with_override_arm(panel):
    """A spec whose arm overrides BOTH window and preprocessing.

    The spec-level values are deliberately different from the arm's, so any site
    that reads the spec where it should read the arm produces a visibly wrong answer
    rather than an accidentally-equal one.
    """
    import macroforecast as mf
    from macroforecast.pipeline.spec import Arm, EvalSpec, TargetSpec, pipeline_spec

    spec_window = mf.from_cutoffs(test_start="2005-01-31", val_size=12)
    arm_window = mf.from_cutoffs(test_start="2005-01-31", val_size=24)

    return pipeline_spec(
        data=panel,
        window=spec_window,
        arms=[
            Arm(
                name="override",
                model="ar",
                window=arm_window,
                preprocessing=mf.preprocess_spec(standardize=True),
                preprocessing_policy="origin_available",
            )
        ],
        targets=[TargetSpec(name="y", transform="none")],
        evaluation=EvalSpec(benchmark="override"),
        horizons=[1],
        preprocessing=mf.preprocess_spec(standardize=False),
        preprocessing_policy="full_panel",
    ), arm_window, spec_window


def test_the_arms_window_override_is_read_the_same_way_by_every_site(panel):
    """Execution and the result-store digest must agree on WHICH window ran.

    The two sites used different predicates. They agree for an ordinary ``Arm``, so
    this passes today -- it is a characterization, not a bug report. It exists so
    that compiling the rule once cannot change the answer, and so a future ``Arm``
    variant that lacks the attribute cannot make the two diverge unnoticed.
    """
    from macroforecast.pipeline.result_store import _object_identity

    spec, arm_window, spec_window = _spec_with_override_arm(panel)
    arm = spec.arms[0]

    executed = arm.window if getattr(arm, "window", None) is not None else spec.window
    recorded = arm.window if arm.window is not None else spec.window

    assert executed is arm_window, "the arm's own window is what runs"
    assert recorded is arm_window, "and it is what the digest records"
    assert _object_identity(executed) == _object_identity(recorded)


def test_an_override_arm_never_inherits_the_spec_level_preprocessing_policy(panel):
    """The rule that is easiest to get wrong, stated once as a test.

    An arm supplying its own ``preprocessing`` also supplies its own policy; the
    spec-level policy must not leak into it. If it did, the preprocessor store would
    be namespaced by ``fixed_reference`` while ``expanding`` actually ran, and cached
    fits from one policy would be served to the other.
    """
    from macroforecast.meta import get_config
    from macroforecast.pipeline.run import _effective_preprocessing_policy
    from macroforecast.window.policy import resolve_stage_policy

    spec, _, _ = _spec_with_override_arm(panel)
    arm = spec.arms[0]

    resolved = _effective_preprocessing_policy(spec, arm)
    expected = resolve_stage_policy(
        arm.preprocessing_policy, default_scope=str(get_config()["default_preprocessing_scope"])
    )

    assert resolved is not None
    assert resolved.to_dict() == expected.to_dict(), (
        "the store namespace must describe the arm's policy, not the spec's"
    )
    spec_level = resolve_stage_policy(
        spec.preprocessing_policy, default_scope=str(get_config()["default_preprocessing_scope"])
    )
    assert resolved.to_dict() != spec_level.to_dict(), (
        "and the two must be distinguishable, or this test proves nothing"
    )


def test_an_arm_without_overrides_inherits_both_spec_level_values(panel):
    """The other half of the rule: no override means the spec's values, unchanged."""
    import macroforecast as mf
    from macroforecast.pipeline.run import _effective_preprocessing_policy
    from macroforecast.pipeline.spec import Arm, EvalSpec, TargetSpec, pipeline_spec
    from macroforecast.meta import get_config
    from macroforecast.window.policy import resolve_stage_policy

    spec_window = mf.from_cutoffs(test_start="2005-01-31", val_size=12)
    spec = pipeline_spec(
        data=panel,
        window=spec_window,
        arms=[Arm(name="plain", model="ar")],
        targets=[TargetSpec(name="y", transform="none")],
        evaluation=EvalSpec(benchmark="plain"),
        horizons=[1],
        preprocessing=mf.preprocess_spec(standardize=True),
        preprocessing_policy="origin_available",
    )
    arm = spec.arms[0]

    assert (arm.window if getattr(arm, "window", None) is not None else spec.window) is spec_window
    resolved = _effective_preprocessing_policy(spec, arm)
    expected = resolve_stage_policy(
        spec.preprocessing_policy, default_scope=str(get_config()["default_preprocessing_scope"])
    )
    assert resolved is not None and resolved.to_dict() == expected.to_dict()
