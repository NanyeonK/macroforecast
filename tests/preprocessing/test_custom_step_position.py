"""#453a -- a custom preprocessing step can name where in the chain it runs.

Before this, ``PreprocessSpec.fit`` ran the whole built-in chain and only then
applied custom steps, unconditionally. So a custom outlier filter could not run
before imputation, which is half of what #453 reports.

The design note (``.dev-notes/custom_step_position_design.md``, merged in #510)
names the test that pins this: the same clipping step at ``before:impute`` and at
``last``, on a panel with holes, must produce DIFFERENT panels -- so an
implementation that accepted the parameter and ignored it fails rather than passes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.preprocessing.specs import custom_preprocess_step


def _panel_with_holes() -> pd.DataFrame:
    idx = pd.date_range("2000-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {"a": rng.normal(size=60) * 3.0, "b": rng.normal(size=60)}, index=idx
    )
    # The holes are the point: clipping before imputation changes what the imputer
    # averages over, so the two positions cannot coincide by accident.
    frame.iloc[[7, 20, 33], 0] = np.nan
    return frame


def _clip(panel, metadata=None, **_):
    return panel.clip(lower=-1.0, upper=1.0)


def _fit_transform(frame: pd.DataFrame, position: str) -> pd.DataFrame:
    spec = mf.preprocess_spec(
        transform="none",
        standardize="none",
        impute="mean",
        custom_steps=[
            custom_preprocess_step("clip", func=_clip, row_local=True, position=position)
        ],
    )
    return spec.fit(frame).transform(frame).panel


def test_before_impute_and_last_produce_different_panels():
    """The test the design note specifies, verbatim in intent.

    Both positions clip to the same bounds, so the columns' extremes agree; what
    differs is what the mean imputer saw. If ``position`` were accepted and dropped,
    the two panels would be identical and this fails.
    """
    frame = _panel_with_holes()
    early = _fit_transform(frame, "before:impute")
    late = _fit_transform(frame, "last")

    assert not early.equals(late), "position was accepted and ignored"
    assert float((early - late).abs().max().max()) > 0.0


def test_last_is_the_default_and_reproduces_the_old_behaviour():
    """Every spec written before positions existed must keep its exact meaning."""
    frame = _panel_with_holes()
    step_default = custom_preprocess_step("clip", func=_clip, row_local=True)
    assert step_default["position"] == "last"

    explicit = _fit_transform(frame, "last")
    spec = mf.preprocess_spec(
        transform="none", standardize="none", impute="mean",
        custom_steps=[custom_preprocess_step("clip", func=_clip, row_local=True)],
    )
    implicit = spec.fit(frame).transform(frame).panel
    pd.testing.assert_frame_equal(explicit, implicit)


def test_the_step_ledger_records_the_position():
    """Two runs differing only in position must be distinguishable in provenance.

    The on-disk preprocessing cache keys off the spec, so if the recorded steps did
    not say where a custom step ran, two different pipelines would also collide
    there.
    """
    frame = _panel_with_holes()
    spec = mf.preprocess_spec(
        transform="none", standardize="none", impute="mean",
        custom_steps=[
            custom_preprocess_step("clip", func=_clip, row_local=True, position="before:impute")
        ],
    )
    processed = spec.fit(frame).transform(frame)
    recorded = [
        step for step in processed.metadata["preprocessing"]["steps"]
        if step.get("step") == "custom"
    ]
    assert recorded, "a positioned custom step must appear in the ledger"
    assert recorded[0]["position"] == "before:impute"


def test_an_unknown_boundary_is_refused_at_construction():
    """The vocabulary is the ledger's own stage names, so a typo is catchable early."""
    with pytest.raises(ValueError) as exc:
        custom_preprocess_step("clip", func=_clip, position="before:nonsense")
    message = str(exc.value)
    assert "before:nonsense" in message
    assert "before:impute" in message, "the refusal should list what IS available"


@pytest.mark.parametrize("position", ["before:impute", "after:outliers", "before:transform"])
def test_every_advertised_boundary_actually_runs(position):
    """A boundary in the vocabulary that never fires would be a lie in the API."""
    frame = _panel_with_holes()
    seen: list[int] = []

    def _spy(panel, metadata=None, **_):
        seen.append(int(panel.shape[0]))
        return panel

    spec = mf.preprocess_spec(
        transform="none", standardize="none", impute="mean",
        custom_steps=[custom_preprocess_step("spy", func=_spy, row_local=True, position=position)],
    )
    spec.fit(frame)
    assert seen, f"nothing ran at {position}"
