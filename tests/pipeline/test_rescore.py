"""WP8a: ``rescore(checkpoint_dir, spec)`` -- re-score saved forecasts in ~3 lines.

``evaluate()`` is pure-frame and ``load_checkpoint_frame`` reconstructs a lean
forecast frame from per-origin parquet, but until now users had to hand-walk the
``<checkpoint_dir>/<target>__<arm>/h<h>/`` tree and reassemble the master frame
themselves. ``rescore`` is that glue. These tests pin the round trip: a live
2-arm checkpointed run and a ``rescore`` from its checkpoint directory alone must
produce the SAME accuracy table; an empty/missing/partial checkpoint dir must
fail loudly with an actionable message, never return a silently-empty report.
"""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import (
    Arm,
    EvalSpec,
    SubsampleWindow,
    TargetSpec,
    pipeline_spec,
    rescore,
    run_pipeline,
)


def _toy(checkpoint_dir=None, *, evaluation=None):
    """A cheap 2-arm x 2-horizon pipeline spec with checkpointing optional."""
    idx = pd.date_range("1990-01-01", periods=140, freq="MS")
    rng = np.random.default_rng(5)
    cols = {f"S{i}": rng.normal(size=140) for i in range(5)}
    cols["Y"] = np.cumsum(rng.normal(size=140))
    panel = pd.DataFrame(cols, index=idx)
    panel.index.name = "date"
    bundle = mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})
    win = mf.window.from_cutoffs(
        test_start="2000-01-01",
        test_end="2000-12-01",
        mode="expanding",
        val_method="last_block",
        retrain_every=1,
    )
    feats = mf.feature_engineering.feature_spec(
        target="Y", predictors="all", lags=range(1, 3)
    )
    return pipeline_spec(
        data=bundle,
        targets=[TargetSpec(name="Y")],
        horizons=[1, 3],
        window=win,
        arms=[
            Arm(name="RIDGE", model="ridge", features=feats, is_benchmark=True),
            Arm(name="LASSO", model="lasso", features=feats),
        ],
        evaluation=evaluation or EvalSpec(benchmark="RIDGE", metrics=("rmse",)),
        checkpoint_dir=(str(checkpoint_dir) if checkpoint_dir is not None else None),
        save_models=False,
    )


def _sorted_accuracy(report):
    return (
        report.accuracy.sort_values(["target", "horizon", "contender"])
        .reset_index(drop=True)
    )


def test_rescore_roundtrip_matches_live_run(tmp_path):
    """rescore(checkpoint_dir, spec) == the live run's evaluation, from disk alone."""
    ckpt = tmp_path / "ckpt"
    live = run_pipeline(_toy(ckpt))
    assert not live.forecasts.empty

    # The 3-line user story: build (or reuse) the spec, point rescore at the dir.
    respec = _toy(ckpt)
    rescored = rescore(ckpt, respec)

    # Accuracy tables agree exactly.
    live_acc = _sorted_accuracy(live)
    re_acc = _sorted_accuracy(rescored)
    assert not live_acc.empty
    pd.testing.assert_frame_equal(live_acc, re_acc, atol=1e-12)

    # The reassembled forecasts carry the same (contender, horizon, origin,
    # prediction, actual) content as the live master frame.
    key = ["contender", "horizon", "origin"]
    cols = key + ["prediction", "actual"]
    a = live.forecasts[cols].sort_values(key).reset_index(drop=True)
    b = rescored.forecasts[cols].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, atol=1e-12)

    # Report contract: same type as run_pipeline's output, with the documented
    # evaluation-only fields populated and the live-run-only fields absent.
    assert type(rescored) is type(live)
    assert rescored.interpretation is None
    assert rescored.failed_cells == ()
    assert rescored.provenance.get("rescored_from") == str(ckpt)


def test_rescore_works_when_spec_has_no_checkpoint_dir(tmp_path):
    """The spec passed to rescore need not itself carry checkpoint_dir -- the
    directory argument wins (the caller may have built a fresh spec without
    checkpointing, or be re-scoring a copied directory).
    """
    ckpt = tmp_path / "ckpt"
    live = run_pipeline(_toy(ckpt))
    plain_spec = _toy(None)
    rescored = rescore(ckpt, plain_spec)
    pd.testing.assert_frame_equal(
        _sorted_accuracy(live), _sorted_accuracy(rescored), atol=1e-12
    )


def test_rescore_empty_dir_raises_informatively(tmp_path):
    """A directory with no checkpoints must raise a clear error, not silently
    return an empty report."""
    spec = _toy(None)
    with pytest.raises(ValueError, match="no checkpoint directories found"):
        rescore(tmp_path / "nothing_here", spec)


def test_rescore_partial_checkpoint_raises_when_all_cells_empty(tmp_path):
    """Cell directories that exist but hold zero parquet records (an interrupted
    run that created dirs but never completed an origin) also fail loudly."""
    spec = _toy(None)
    # Fabricate the layout with empty h-subdirectories only.
    for target in spec.targets:
        for arm in spec.arms:
            for h in spec.horizons:
                (tmp_path / f"{target.name}__{arm.name}" / f"h{h}").mkdir(
                    parents=True, exist_ok=True
                )
    with pytest.raises(ValueError, match="every one is empty"):
        rescore(tmp_path, spec)


def test_rescore_missing_arm_surfaces_in_empty_cells(tmp_path):
    """An arm with no checkpoint data (e.g. it failed during the original run, or
    the spec names an arm that never ran) is surfaced on ``empty_cells`` while the
    present arms are still scored."""
    import shutil

    ckpt = tmp_path / "ckpt"
    run_pipeline(_toy(ckpt))
    # Remove one arm's entire cell directory, as if it had failed/never run.
    shutil.rmtree(ckpt / "Y__LASSO")

    rescored = rescore(ckpt, _toy(ckpt))
    assert set(rescored.forecasts["contender"].unique()) == {"RIDGE"}
    assert {c["arm"] for c in rescored.empty_cells} == {"LASSO"}


def test_rescore_can_add_subsamples_to_checkpointed_run(tmp_path):
    ckpt = tmp_path / "ckpt"
    run_pipeline(_toy(ckpt))
    evaluation = EvalSpec(
        benchmark="RIDGE",
        metrics=("rmse",),
        tests=("dm",),
        subsamples={
            "full": SubsampleWindow(),
            "early": SubsampleWindow(end="2000-12-31"),
        },
    )

    rescored = rescore(ckpt, _toy(ckpt, evaluation=evaluation))

    assert set(rescored.accuracy["subsample"]) == {"full", "early"}
    assert set(rescored.significance["subsample"]) == {"full", "early"}
