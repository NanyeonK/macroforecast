"""F-060/F-061: a corrupt checkpoint artifact fails closed on every public read.

A checkpoint directory is read for two very different reasons, and before this
both reasons shared one tolerant reader.

*Resuming* asks "which origins may I skip?". An unreadable origin file is simply
not one of them, so ``completed_origin_positions`` ignores it and the runner
recomputes that origin from scratch. That is the correct answer, it is tested by
``test_checkpoint_resume_identity.py``, and it is deliberately unchanged here.

*Loading* asks "give me the forecasts". There is no recomputation available on
that path -- the checkpoint IS the data source -- so silently dropping an
unreadable origin handed back a forecast table that was short by one origin with
nothing to say so, and every metric derived from it scored as though that origin
had never run (F-060). The selection-history sidecar had the same shape of bug
one level down: each JSONL line was appended to the shared result as it parsed,
so a sidecar truncated mid-write contributed its surviving prefix and undercounted
a selection frequency in silence (F-061).

Everything below pins the split: the public loaders, ``pipeline.rescore``,
``pipeline.selection_history`` and ``pipeline.selection_frequency_table`` raise
:class:`CheckpointCorruptionError` and return nothing, while resume still
self-heals and the runner's own merge stays completable behind one warning.

That last tolerance is bounded, and the tests say where the bound is. The merge
cannot show that what it excluded was out of this run's origin set: origins the
run computed or recomputed are in memory and are safe, but an origin a PREVIOUS
run completed is served from disk, so one damaged between the resume gate's read
and the merge leaves the returned frame short that origin. The warning has to say
that rather than claim the file was irrelevant.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.forecasting import CheckpointCorruptionError
from macroforecast.forecasting import checkpoint as ckpt
from macroforecast.forecasting.runner import _merge_checkpoint_records
from macroforecast.pipeline import Arm, EvalSpec, pipeline_spec, rescore, run_pipeline


def teardown_function() -> None:
    mf.meta.reset_config()


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _records(origin_pos: int) -> list[dict]:
    return [
        {
            "target": "y",
            "horizon": 1,
            "origin": origin_pos,
            "origin_pos": origin_pos,
            "date": pd.Timestamp("2000-01-31"),
            "model": "ridge",
            "prediction": float(origin_pos),
            "actual": 0.0,
        }
    ]


def _origins(directory: Path, *positions: int) -> Path:
    """Write one healthy lean parquet per position, through the real writer."""
    for position in positions:
        ckpt.append_origin_records(directory, position, _records(position))
    return directory


def _corrupt(path: Path) -> Path:
    path.write_bytes(b"not parquet")
    return path


def _sidecar(directory: Path, origin_pos: int, text: str) -> Path:
    path = directory / f"origin_{origin_pos}_selection.jsonl"
    path.write_text(text, encoding="utf-8")
    return path


def _record_line(name: str) -> str:
    return json.dumps(
        {"kind": "feature", "name": name, "origin_pos": 0, "horizon": 1}
    )


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


# Runner fixtures, deliberately the same cheap ridge cell the F-058 suite uses.
def _panel(n: int = 48) -> pd.DataFrame:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    x = np.linspace(0.0, 1.0, n)
    return pd.DataFrame(
        {
            "y": 1.0 + 2.0 * x + 0.1 * np.sin(np.arange(n) / 2.0),
            "x1": x,
            "x2": np.sin(np.arange(n) / 3.0),
        },
        index=idx,
    )


def _run(checkpoint_path: Path):
    return mf.forecasting.run(
        _panel(),
        "ridge",
        window=mf.window.spec(
            estimation=mf.window.estimation_expanding(min_size=24),
            val=mf.window.val_last_block(size=8),
            test=mf.window.test_origins(horizon=1, step=6),
        ),
        features=mf.feature_engineering.feature_spec(target="y", target_lags=[1, 2]),
        params={"alpha": 0.01},
        save_models=False,
        checkpoint_path=checkpoint_path,
    )


def _hdir(cell: Path, horizon: int = 1) -> Path:
    return cell / f"h{horizon}"


def _selection_history_spec(checkpoint_dir: Path):
    """The 3-origin checkpointed cell that actually WRITES sidecars.

    Deliberately the same shape ``tests/pipeline/test_selection_history.py``
    uses, because the point here is the report-object route those tests exercise
    on healthy data.
    """
    idx = pd.date_range("2000-01-01", periods=48, freq="MS", name="date")
    rng = np.random.default_rng(7)
    t = np.arange(len(idx), dtype=float)
    panel = pd.DataFrame(
        {
            "y": np.r_[0.0, t[:-1]] + rng.normal(scale=0.01, size=len(idx)),
            "x1": t,
            "x2": np.tile([1.0, -1.0], len(idx) // 2),
        },
        index=idx,
    )
    bundle = mf.data.custom_dataset(
        panel, transform_codes={column: 1 for column in panel.columns}
    )
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[
            mf.feature_engineering.predictor_screen(
                method="t_stat", top_k=1, min_k=1
            )
        ],
        drop_missing=False,
    )
    return pipeline_spec(
        data=bundle,
        targets=["y"],
        horizons=[1],
        window=mf.window.from_cutoffs(
            test_start="2003-01-01",
            test_end="2003-03-01",
            mode="expanding",
            val_method="last_block",
            retrain_every=1,
        ),
        arms=[
            Arm(
                "RIDGE",
                model="ridge",
                features=features,
                params={"alpha": 0.2},
                model_selection={"ridge": None},
            )
        ],
        evaluation=EvalSpec(benchmark="RIDGE", metrics=("rmse",)),
        checkpoint_dir=str(checkpoint_dir),
        selection_history=True,
        save_models=False,
    )


def _pipeline_spec(checkpoint_dir: Path):
    """One cheap 1-arm x 1-horizon x 3-origin checkpointed pipeline cell."""
    idx = pd.date_range("2000-01-01", periods=48, freq="MS", name="date")
    rng = np.random.default_rng(7)
    t = np.arange(len(idx), dtype=float)
    panel = pd.DataFrame(
        {
            "y": np.r_[0.0, t[:-1]] + rng.normal(scale=0.01, size=len(idx)),
            "x1": t,
            "x2": np.tile([1.0, -1.0], len(idx) // 2),
        },
        index=idx,
    )
    bundle = mf.data.custom_dataset(
        panel, transform_codes={column: 1 for column in panel.columns}
    )
    return pipeline_spec(
        data=bundle,
        targets=["y"],
        horizons=[1],
        window=mf.window.from_cutoffs(
            test_start="2003-01-01",
            test_end="2003-03-01",
            mode="expanding",
            val_method="last_block",
            retrain_every=1,
        ),
        arms=[
            Arm(
                "RIDGE",
                model="ridge",
                features=mf.feature_engineering.feature_spec(
                    target="y", horizon=1, predictors=["x1", "x2"], drop_missing=False
                ),
                params={"alpha": 0.2},
            )
        ],
        evaluation=EvalSpec(benchmark="RIDGE", metrics=("rmse",)),
        checkpoint_dir=str(checkpoint_dir),
        save_models=False,
    )


# --------------------------------------------------------------------------- #
# 1. load_checkpoint_frame: no short frame, ever
# --------------------------------------------------------------------------- #
def test_corrupt_origin_raises_instead_of_returning_a_short_frame(
    tmp_path: Path,
) -> None:
    """Two origins on disk, the second unreadable. The old reader returned the
    first one alone; there is no honest one-origin answer to give here."""
    directory = _origins(tmp_path / "h1", 0, 1)
    assert len(ckpt.load_checkpoint_frame(directory)) == 2, "fixture sanity"
    _corrupt(directory / "origin_1.parquet")

    with pytest.raises(CheckpointCorruptionError):
        ckpt.load_checkpoint_frame(directory)


def test_corrupt_origin_error_names_the_artifact_path_and_route(
    tmp_path: Path,
) -> None:
    directory = _origins(tmp_path / "h1", 0, 1)
    corrupt = _corrupt(directory / "origin_1.parquet")

    with pytest.raises(CheckpointCorruptionError) as excinfo:
        ckpt.load_checkpoint_frame(directory)

    error = excinfo.value
    assert error.path == corrupt
    assert error.artifact == ckpt.CHECKPOINT_ORIGIN_ARTIFACT
    assert error.line is None
    assert isinstance(error, ValueError), "stays catchable as the historical type"

    message = str(error)
    assert repr(str(corrupt)) in message, "the exact path, quoted"
    assert "checkpoint origin file" in message
    assert "selection-history sidecar" not in message
    assert "partial" in message, "says why nothing came back"
    assert "not modified, renamed, or removed" in message
    assert "checkpoint_path" in message, "names a recovery route"
    assert "re-running the same configuration" in message, "the route that works here"
    assert "overwritten in place" in message, "a parquet IS healed by a plain re-run"
    assert "BOTH" not in message, "that two-file recipe belongs to the sidecar only"


def test_corrupt_origin_error_chains_the_reader_error_without_quoting_it(
    tmp_path: Path,
) -> None:
    """The reader's text is pyarrow-version dependent, so it belongs in the
    cause, not in a message users are invited to match on."""
    directory = _origins(tmp_path / "h1", 0)
    _corrupt(directory / "origin_0.parquet")

    with pytest.raises(CheckpointCorruptionError) as excinfo:
        ckpt.load_checkpoint_frame(directory)

    cause = excinfo.value.__cause__
    assert cause is not None
    assert "magic bytes" in str(cause).lower() or "parquet" in str(cause).lower()
    assert "pyarrow" not in str(excinfo.value).lower()
    assert "magic bytes" not in str(excinfo.value).lower()


def test_corrupt_origin_error_reports_the_sorted_first_corrupt_file(
    tmp_path: Path,
) -> None:
    """Fail fast on one file rather than enumerating every corruption, and make
    which one deterministic."""
    directory = _origins(tmp_path / "h1", 0, 1, 2)
    _corrupt(directory / "origin_1.parquet")
    _corrupt(directory / "origin_2.parquet")

    with pytest.raises(CheckpointCorruptionError) as excinfo:
        ckpt.load_checkpoint_frame(directory)

    assert excinfo.value.path == directory / "origin_1.parquet"


def test_failing_to_load_does_not_mutate_the_checkpoint_directory(
    tmp_path: Path,
) -> None:
    directory = _origins(tmp_path / "h1", 0, 1)
    _corrupt(directory / "origin_1.parquet")
    before = _snapshot(directory)

    with pytest.raises(CheckpointCorruptionError):
        ckpt.load_checkpoint_frame(directory)

    assert _snapshot(directory) == before


# --------------------------------------------------------------------------- #
# 2. load_selection_history_frame: a sidecar is accepted whole or not at all
# --------------------------------------------------------------------------- #
def test_truncated_sidecar_raises_and_keeps_no_prefix(tmp_path: Path) -> None:
    """Line 1 parsed and line 2 did not. The old reader kept line 1."""
    directory = _origins(tmp_path / "h1", 0)
    path = _sidecar(directory, 0, _record_line("x1") + "\n" + '{"kind": "feat')

    with pytest.raises(CheckpointCorruptionError) as excinfo:
        ckpt.load_selection_history_frame(directory)

    error = excinfo.value
    assert error.path == path
    assert error.artifact == ckpt.CHECKPOINT_SELECTION_ARTIFACT
    assert error.line == 2
    assert isinstance(error.__cause__, json.JSONDecodeError)

    message = str(error)
    assert repr(str(path)) in message
    assert "selection-history sidecar" in message
    assert "not valid JSON" in message
    assert "line 2" in message
    assert "not modified, renamed, or removed" in message
    # The recovery contract, which is NOT the origin parquet's (F-060/F-061).
    assert "A plain re-run does NOT heal this file" in message
    assert "BOTH" in message, "the sidecar and its parquet have to go together"
    assert repr("origin_0.parquet") in message, "names the matching origin file"
    assert "selection history enabled" in message
    assert "ABSENT rather than reconstructed" in message, "sidecar-only is not a repair"
    assert "recomputed and this file is overwritten in place" not in message


@pytest.mark.parametrize(
    ("payload", "kind"),
    [("42", "number"), ("[1, 2]", "array"), ('"text"', "string"),
     ("true", "boolean"), ("null", "null")],
)
def test_sidecar_line_that_is_not_a_json_object_is_rejected(
    tmp_path: Path, payload: str, kind: str
) -> None:
    """Valid JSON is not enough: a selection record is an object. A bare scalar
    or array used to be fed to ``DataFrame.from_records`` and became a row."""
    directory = _origins(tmp_path / "h1", 0)
    path = _sidecar(directory, 0, _record_line("x1") + "\n" + payload + "\n")

    with pytest.raises(CheckpointCorruptionError) as excinfo:
        ckpt.load_selection_history_frame(directory)

    error = excinfo.value
    assert error.path == path
    assert error.line == 2
    assert error.artifact == ckpt.CHECKPOINT_SELECTION_ARTIFACT
    assert f"JSON {kind}" in str(error)
    assert "rather than an object" in str(error)
    # By design, and the only corruption case without one: the decode SUCCEEDED,
    # so there is no underlying error to chain. Inventing a cause here would
    # point at a reader that did not fail.
    assert error.__cause__ is None
    assert error.__context__ is None


def test_sidecar_that_is_not_utf8_fails_the_whole_file_with_no_line(
    tmp_path: Path,
) -> None:
    """Decoding fails before any line exists, so there is no line to report and
    ``line`` stays None -- the same shape as an unreadable parquet."""
    directory = _origins(tmp_path / "h1", 0)
    path = directory / "origin_0_selection.jsonl"
    path.write_bytes(b'{"kind": "feature", "name": "\xff\xfe"}\n')

    with pytest.raises(CheckpointCorruptionError) as excinfo:
        ckpt.load_selection_history_frame(directory)

    error = excinfo.value
    assert error.path == path
    assert error.artifact == ckpt.CHECKPOINT_SELECTION_ARTIFACT
    assert error.line is None, "no line survived decoding, so none is claimed"
    assert isinstance(error.__cause__, UnicodeDecodeError)
    assert "could not be read" in str(error)
    assert "not modified, renamed, or removed" in str(error)


def test_sidecar_that_cannot_be_opened_fails_the_whole_file_with_no_line(
    tmp_path: Path,
) -> None:
    """An OSError from opening the sidecar is corruption too, not a skip.

    A directory wearing the sidecar's name is the portable way to make the open
    fail: POSIX raises IsADirectoryError and Windows PermissionError, and both
    are OSError, which is what the reader actually catches.
    """
    directory = _origins(tmp_path / "h1", 0)
    path = directory / "origin_0_selection.jsonl"
    path.mkdir()

    with pytest.raises(CheckpointCorruptionError) as excinfo:
        ckpt.load_selection_history_frame(directory)

    error = excinfo.value
    assert error.path == path
    assert error.artifact == ckpt.CHECKPOINT_SELECTION_ARTIFACT
    assert error.line is None
    assert isinstance(error.__cause__, OSError)


def test_sidecar_error_reports_the_sorted_first_corrupt_file(
    tmp_path: Path,
) -> None:
    """Two damaged sidecars must not make WHICH one is reported depend on
    directory iteration order."""
    directory = _origins(tmp_path / "h1", 0, 1)
    _sidecar(directory, 0, "{oops")
    _sidecar(directory, 1, "{oops")

    with pytest.raises(CheckpointCorruptionError) as excinfo:
        ckpt.load_selection_history_frame(directory)

    assert excinfo.value.path == directory / "origin_0_selection.jsonl"


def test_a_healthy_sidecar_before_a_corrupt_one_contributes_nothing(
    tmp_path: Path,
) -> None:
    """No-partial applies across files too, not only within one."""
    directory = _origins(tmp_path / "h1", 0, 1)
    _sidecar(directory, 0, _record_line("x1") + "\n")
    corrupt = _sidecar(directory, 1, "{oops")

    with pytest.raises(CheckpointCorruptionError) as excinfo:
        ckpt.load_selection_history_frame(directory)

    assert excinfo.value.path == corrupt


def test_sidecar_failure_does_not_mutate_the_checkpoint_directory(
    tmp_path: Path,
) -> None:
    directory = _origins(tmp_path / "h1", 0)
    _sidecar(directory, 0, "{oops")
    before = _snapshot(directory)

    with pytest.raises(CheckpointCorruptionError):
        ckpt.load_selection_history_frame(directory)

    assert _snapshot(directory) == before


# --------------------------------------------------------------------------- #
# 3. Preserved behaviour: everything that was not corruption stays as it was
# --------------------------------------------------------------------------- #
def test_missing_directory_still_returns_an_empty_frame(tmp_path: Path) -> None:
    absent = tmp_path / "nope"
    assert ckpt.load_checkpoint_frame(absent).empty
    assert ckpt.load_selection_history_frame(absent).empty


def test_empty_directory_still_returns_an_empty_frame(tmp_path: Path) -> None:
    directory = tmp_path / "h1"
    directory.mkdir()
    assert ckpt.load_checkpoint_frame(directory).empty
    assert ckpt.load_selection_history_frame(directory).empty


def test_an_empty_but_readable_origin_marker_is_still_valid(tmp_path: Path) -> None:
    """An origin with no records still writes a zero-row marker file so it is not
    recomputed. That is a healthy artifact, not a corrupt one."""
    directory = tmp_path / "h1"
    ckpt.append_origin_records(directory, 0, [])
    ckpt.append_origin_records(directory, 1, _records(1))

    frame = ckpt.load_checkpoint_frame(directory)

    assert sorted(frame["origin_pos"].tolist()) == [1]
    assert 0 in ckpt.completed_origin_positions(directory)


def test_nonmatching_and_temporary_filenames_are_still_ignored(
    tmp_path: Path,
) -> None:
    """Membership is by the origin filename grammar. A crashed write leaves a
    dot-prefixed temp file, which is exactly why that name was chosen."""
    directory = _origins(tmp_path / "h1", 0)
    (directory / ".origin_9.parquet.tmp").write_bytes(b"half a parquet file")
    (directory / "origin_x.parquet").write_bytes(b"not parquet")
    (directory / "notes.txt").write_text("hello", encoding="utf-8")

    assert len(ckpt.load_checkpoint_frame(directory)) == 1


def test_an_orphan_sidecar_is_still_skipped_before_it_is_opened(
    tmp_path: Path,
) -> None:
    """A sidecar with no origin parquet beside it is not this directory's data.
    The orphan test must come first, so an orphan that is ALSO corrupt stays
    skipped rather than becoming a new way to fail a healthy load."""
    directory = _origins(tmp_path / "h1", 0)
    _sidecar(directory, 0, _record_line("x1") + "\n")
    _sidecar(directory, 7, "{ truncated orphan")

    frame = ckpt.load_selection_history_frame(directory)

    assert frame["name"].tolist() == ["x1"]


def test_blank_sidecar_lines_are_still_ignored(tmp_path: Path) -> None:
    directory = _origins(tmp_path / "h1", 0)
    _sidecar(
        directory,
        0,
        "\n" + _record_line("x1") + "\n\n   \n" + _record_line("x2") + "\n\n",
    )

    assert ckpt.load_selection_history_frame(directory)["name"].tolist() == ["x1", "x2"]


def test_completed_origin_positions_stays_tolerant(tmp_path: Path) -> None:
    """Unchanged on purpose: this is the question whose right answer is "not
    done, recompute it", which is what keeps a matching resume self-healing."""
    directory = _origins(tmp_path / "h1", 0, 1)
    _corrupt(directory / "origin_1.parquet")

    assert ckpt.completed_origin_positions(directory) == {0}
    assert len(ckpt.final_origin_files(directory)) == 2


# --------------------------------------------------------------------------- #
# 4. The public pipeline surfaces inherit the policy
# --------------------------------------------------------------------------- #
def test_rescore_raises_instead_of_scoring_a_short_checkpoint(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ckpt"
    spec = _pipeline_spec(root)
    run_pipeline(spec)

    h_dir = next(root.glob("*/h1"))
    origins = ckpt.final_origin_files(h_dir)
    assert len(origins) >= 2, "fixture sanity: need a short-by-one to be possible"
    _corrupt(origins[-1])

    with pytest.raises(CheckpointCorruptionError) as excinfo:
        rescore(root, spec)

    assert excinfo.value.path == origins[-1]


def test_selection_history_raises_on_a_corrupt_sidecar(tmp_path: Path) -> None:
    """The bare-path route: a checkpoint tree handed over as a directory."""
    h_dir = tmp_path / "y__RIDGE" / "h1"
    h_dir.mkdir(parents=True)
    (h_dir / "origin_0.parquet").touch()
    corrupt = _sidecar(h_dir, 0, _record_line("x1") + "\n" + "{trunc")

    with pytest.raises(CheckpointCorruptionError) as excinfo:
        mf.pipeline.selection_history(tmp_path)
    assert excinfo.value.path == corrupt

    with pytest.raises(CheckpointCorruptionError):
        mf.pipeline.selection_frequency_table(tmp_path)


def test_selection_history_on_a_report_raises_after_its_sidecar_is_corrupted(
    tmp_path: Path,
) -> None:
    """The route users actually take: ``selection_history(report)``.

    ``selection_history`` resolves a report through its spec/checkpoint
    provenance and re-reads the sidecars from disk on EVERY call -- the report
    carries no history frame of its own. The healthy call first is what pins
    that: if the report were serving a cached in-memory history, it would still
    answer after the file underneath it was destroyed, and this test would fail
    on the second call rather than pass.
    """
    root = tmp_path / "ckpt"
    spec = _selection_history_spec(root)
    report = run_pipeline(spec)

    healthy = mf.pipeline.selection_history(report)
    assert not healthy.empty, "fixture sanity: the report route reads real rows"

    sidecars = sorted((root / "y__RIDGE" / "h1").glob("origin_*_selection.jsonl"))
    assert sidecars, "fixture sanity: this spec writes sidecars"
    corrupt = sidecars[0]
    corrupt.write_text(_record_line("x1") + "\n" + "{trunc", encoding="utf-8")

    with pytest.raises(CheckpointCorruptionError) as excinfo:
        mf.pipeline.selection_history(report)

    error = excinfo.value
    assert error.path == corrupt
    assert error.artifact == ckpt.CHECKPOINT_SELECTION_ARTIFACT
    assert error.line == 2


def test_selection_frequency_table_on_a_report_raises_after_its_sidecar_is_corrupted(
    tmp_path: Path,
) -> None:
    """The aggregate is built on the same read, so it must inherit the refusal
    rather than summarise whatever happened to decode."""
    root = tmp_path / "ckpt"
    spec = _selection_history_spec(root)
    report = run_pipeline(spec)

    healthy = mf.pipeline.selection_frequency_table(report)
    assert not healthy.empty, "fixture sanity: the report route reads real rows"

    sidecars = sorted((root / "y__RIDGE" / "h1").glob("origin_*_selection.jsonl"))
    corrupt = sidecars[-1]
    corrupt.write_text("42\n", encoding="utf-8")

    with pytest.raises(CheckpointCorruptionError) as excinfo:
        mf.pipeline.selection_frequency_table(report)

    assert excinfo.value.path == corrupt
    assert excinfo.value.line == 1


# --------------------------------------------------------------------------- #
# 5. Resume and the runner's own merge stay completable
# --------------------------------------------------------------------------- #
def test_matching_resume_still_self_heals_a_corrupt_origin(tmp_path: Path) -> None:
    """The load path fails closed, but the run that OWNS the directory recomputes
    the damaged origin and overwrites it, so the resumed frame is whole."""
    cell = tmp_path / "cell"
    expected = sorted(_run(cell).to_frame()["origin_pos"].unique().tolist())
    hdir = _hdir(cell)
    _corrupt(hdir / f"origin_{sorted(ckpt.completed_origin_positions(hdir))[-1]}.parquet")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frame = _run(cell).to_frame()

    assert sorted(frame["origin_pos"].unique().tolist()) == expected
    assert not [w for w in caught if "checkpoint" in str(w.message).lower()]
    ckpt.load_checkpoint_frame(hdir)  # healed: the strict loader is happy again


def test_runner_merge_warns_and_completes_rather_than_stranding_a_finished_run(
    tmp_path: Path,
) -> None:
    """A corrupt final-named file must not throw away a run that finished.

    The runner alone reads tolerantly, excludes the file, and says so exactly
    once. Here the damaged file genuinely is a stray this run never computes, so
    the returned frame is whole -- but that is a property of THIS fixture, not
    something the merge can verify, which is the next test's subject.
    """
    cell = tmp_path / "cell"
    expected = sorted(_run(cell).to_frame()["origin_pos"].unique().tolist())
    hdir = _hdir(cell)
    stray = _corrupt(hdir / "origin_999.parquet")

    with pytest.warns(UserWarning, match="origin_999") as caught:
        frame = _run(cell).to_frame()

    assert sorted(frame["origin_pos"].unique().tolist()) == expected
    assert len([w for w in caught if issubclass(w.category, UserWarning)
                and str(stray) in str(w.message)]) == 1, "exactly one warning"


def test_runner_merge_warning_does_not_claim_the_file_was_out_of_this_runs_set(
    tmp_path: Path,
) -> None:
    """The warning is the only thing standing between a short frame and silence,
    so it must not tell the user the excluded file was somebody else's.

    The merge cannot know that. It states what IS true (in-memory origins are
    safe), what MIGHT be true (a previously completed origin is now missing),
    and what to do about it.
    """
    cell = tmp_path / "cell"
    _run(cell)
    hdir = _hdir(cell)
    stray = _corrupt(hdir / "origin_999.parquet")

    with pytest.warns(UserWarning) as caught:
        _run(cell)

    message = str(
        next(w.message for w in caught if str(stray) in str(w.message))
    )
    assert repr(str(stray)) in message, "names the exact path"
    assert "outside this run" not in message
    assert "does not own" not in message
    assert "PREVIOUS run completed" in message, "states the real risk"
    assert "SHORT that origin" in message
    assert "load_checkpoint_frame() and pipeline.rescore() refuse" in message
    assert "re-run the same configuration" in message, "names a recovery route"
    assert "Every file named here is a checkpoint origin file" in message, (
        "the tolerant read only ever names parquets, so the parquet recovery applies"
    )
    assert "overwritten in place" in message


def test_runner_merge_can_return_a_frame_short_a_previously_completed_origin(
    tmp_path: Path,
) -> None:
    """The exact case the prose must not deny.

    ``completed_origin_positions`` runs at the resume gate, before the run. An
    origin a previous run completed is discovered there, skipped, and then
    supplied by the merge read alone. Damage it between those two moments and it
    is dropped: the merge holds nothing in memory for it. This exercises the
    merge directly, because the whole point is a file that was intact when the
    gate looked and is not when the merge does.
    """
    directory = _origins(tmp_path / "h1", 0)  # a PREVIOUS run's completed origin
    assert ckpt.completed_origin_positions(directory) == {0}, "the gate would skip it"
    _corrupt(directory / "origin_0.parquet")

    this_run = _records(1)  # the only origin this run computed
    with pytest.warns(UserWarning) as caught:
        merged = _merge_checkpoint_records(list(this_run), directory)

    assert sorted({record["origin_pos"] for record in merged}) == [1]
    assert 0 not in {record["origin_pos"] for record in merged}, (
        "a previously completed IN-SET origin is missing from the returned frame"
    )
    message = str(next(w.message for w in caught if "origin_0.parquet" in str(w.message)))
    assert "SHORT that origin" in message, "and the warning is what says so"


# --------------------------------------------------------------------------- #
# 6. The recovery contract is artifact-specific, and behaves that way
# --------------------------------------------------------------------------- #
def test_a_plain_rerun_does_not_heal_a_sidecar_but_removing_both_artifacts_does(
    tmp_path: Path,
) -> None:
    """The asymmetry the messages promise, exercised end to end.

    A corrupt sidecar sits beside a HEALTHY ``origin_<pos>.parquet``, and the two
    are discovered independently: the parquet still reads, so that origin is
    still a completed origin. The re-run therefore SKIPS it, and a skipped origin
    writes no sidecar, so the damaged bytes survive a repair that would have
    healed a damaged parquet outright. Only moving or removing BOTH artifacts
    puts the origin back in the work list and writes the sidecar again.
    """
    root = tmp_path / "ckpt"
    run_pipeline(_selection_history_spec(root))

    h_dir = root / "y__RIDGE" / "h1"
    sidecars = sorted(h_dir.glob("origin_*_selection.jsonl"))
    assert sidecars, "fixture sanity: this spec writes sidecars"
    sidecar = sidecars[0]
    position = int(
        sidecar.name.removeprefix("origin_").removesuffix("_selection.jsonl")
    )
    parquet = h_dir / f"origin_{position}.parquet"
    assert parquet.exists(), "fixture sanity: the sidecar has a matching parquet"

    damaged = (_record_line("x1") + "\n" + "{trunc").encode("utf-8")
    sidecar.write_bytes(damaged)
    assert position in ckpt.completed_origin_positions(h_dir), (
        "the parquet was not touched, so its origin is still completed"
    )

    # (a) The repair that heals a damaged parquet does nothing here.
    run_pipeline(_selection_history_spec(root))

    assert sidecar.read_bytes() == damaged, (
        "a skipped origin writes no sidecar, so the corrupt bytes are still there"
    )
    with pytest.raises(CheckpointCorruptionError) as excinfo:
        ckpt.load_selection_history_frame(h_dir)
    assert excinfo.value.path == sidecar
    assert excinfo.value.line == 2

    # (b) Removing BOTH matching artifacts is what actually reconstructs it.
    sidecar.unlink()
    parquet.unlink()
    run_pipeline(_selection_history_spec(root))

    assert sidecar.exists(), "the recomputed origin wrote its sidecar again"
    assert sidecar.read_bytes() != damaged
    frame = ckpt.load_selection_history_frame(h_dir)
    assert not frame.empty, "and the strict loader reads the directory again"
    assert position in {int(value) for value in frame["origin_pos"]}, (
        "the reconstructed sidecar carries this origin's rows"
    )
