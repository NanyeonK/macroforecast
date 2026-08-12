"""F-058: a checkpoint directory may only be resumed by the run that filled it.

``checkpoint_path`` is a PATH, and a path says nothing about configuration. Before
this, handing the same path to a run with a different regularisation strength,
window, panel, or seed made the old per-origin parquet files this run's answer:
``completed_origin_positions`` validated only that they parsed, the origins were
skipped, and ``_merge_checkpoint_records`` merged the stale rows into the returned
frame. ``pipeline/run.py`` writes its ``cell_manifest.json`` only AFTER ``run()``
returns, so it could not protect that decision and would then overwrite the old
manifest with the new identity, erasing the evidence.

Everything here goes through the public entry points -- ``mf.forecasting.run`` and
``run_pipeline`` -- because that is where the defect was reachable. The gate lives
in the forecasting runner, so both inherit it.

The gate FAILS CLOSED and touches nothing: a refused directory is left exactly as
it was, and stays readable by ``load_checkpoint_frame``/``rescore``. It never
adopts, deletes, quarantines, or renames a user artifact.
"""
from __future__ import annotations

import dataclasses as _dc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.forecasting import checkpoint as ckpt
from macroforecast.forecasting import runner


def teardown_function() -> None:
    mf.meta.reset_config()


# --------------------------------------------------------------------------- #
# Fixtures: one cheap direct-policy cell, deliberately identical to the shape
# ``tests/forecasting/test_checkpoint.py`` uses.
# --------------------------------------------------------------------------- #
def _panel(n: int = 48, *, bump: tuple[int, str] | None = None) -> pd.DataFrame:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    x = np.linspace(0.0, 1.0, n)
    panel = pd.DataFrame(
        {
            "y": 1.0 + 2.0 * x + 0.1 * np.sin(np.arange(n) / 2.0),
            "x1": x,
            "x2": np.sin(np.arange(n) / 3.0),
        },
        index=idx,
    )
    if bump is not None:
        row, column = bump
        panel.iloc[row, panel.columns.get_loc(column)] += 1.0
    return panel


def _window(min_size: int = 24) -> mf.window.WindowSpec:
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=min_size),
        val=mf.window.val_last_block(size=8),
        test=mf.window.test_origins(horizon=1, step=6),
    )


def _features() -> mf.feature_engineering.FeatureSpec:
    return mf.feature_engineering.feature_spec(target="y", target_lags=[1, 2])


def _run(checkpoint_path, **overrides):
    """One ridge cell. Every keyword an identity test varies is an override."""
    kwargs = {
        "window": _window(),
        "features": _features(),
        "params": {"alpha": 0.01},
        "save_models": False,
        "checkpoint_path": checkpoint_path,
    }
    panel = overrides.pop("panel", None)
    model = overrides.pop("model", "ridge")
    kwargs.update(overrides)
    return mf.forecasting.run(_panel() if panel is None else panel, model, **kwargs)


def _hdir(cell: Path, horizon: int = 1) -> Path:
    return cell / f"h{horizon}"


def _manifest(cell: Path, horizon: int = 1) -> dict:
    path = _hdir(cell, horizon) / ckpt.CHECKPOINT_IDENTITY_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot(directory: Path) -> dict[str, bytes]:
    """Every file's bytes, so a refusal can be shown to have changed nothing."""
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _spy_on_origin_fits(monkeypatch) -> list[int]:
    import macroforecast.forecasting.runner as runner

    computed: list[int] = []
    original = runner._fit_predict_origin

    def _spy(item, *args, **kwargs):
        position = item["row"].get("origin_pos")
        if position is not None:
            computed.append(int(position))
        return original(item, *args, **kwargs)

    monkeypatch.setattr(runner, "_fit_predict_origin", _spy)
    return computed


#: Every ``metadata=`` payload ``_metadata_sensitive_feature`` was handed, in call
#: order. That step is the only thing in this module that reads metadata, so this
#: list is precisely "what user code received", which is what the identity has to
#: match. Cleared by ``_run_with_metadata`` and by the vintage tests that read it.
_OBSERVED_METADATA: list[dict] = []


def _metadata_sensitive_feature(source, *, metadata=None, **params):
    """A stable custom feature step whose output depends only on ``metadata``.

    ``panel_fingerprint`` covers a panel's index, columns and values, and this
    step touches none of them: it reads ``phase`` out of the ``metadata=``
    argument that ``_FittedFeatureStep`` passes to every custom step, and builds
    its column from the index alone. Two panels with identical values and
    different ``attrs`` therefore yield different features, and different
    forecasts -- which is the whole gap section 9 is about.

    The ``__mf_digest__`` marker keeps the callable itself out of the opaque set,
    so a refusal in these tests can only come from the metadata and never from an
    unrepresentable function object smuggling in an incomplete identity.
    """

    payload = dict(metadata or {})
    _OBSERVED_METADATA.append(payload)
    phase = float(payload.get("phase", 0.0))
    months = (source.index.year - 2000) * 12.0 + source.index.month
    return pd.Series(np.sin((months + phase) / 3.0), index=source.index, name="phased")


_metadata_sensitive_feature.__mf_digest__ = "test:metadata_sensitive_feature/v1"


# --------------------------------------------------------------------------- #
# 1. The manifest is on disk before anything can be trusted or written
# --------------------------------------------------------------------------- #
def test_manifest_is_written_before_the_first_origin_file(tmp_path: Path, monkeypatch) -> None:
    """Ordering, not just presence: a crash after origin 1 must still leave a
    directory whose identity is known, otherwise the very first resume is a
    legacy directory and permanently unresumable."""
    import macroforecast.forecasting.runner as runner

    cell = tmp_path / "cell"
    manifest_present: list[bool] = []
    original = runner.append_origin_records

    def _spy(checkpoint_path, origin_pos, records):
        manifest_present.append(
            (Path(checkpoint_path) / ckpt.CHECKPOINT_IDENTITY_FILENAME).exists()
        )
        return original(checkpoint_path, origin_pos, records)

    monkeypatch.setattr(runner, "append_origin_records", _spy)
    _run(cell)

    assert manifest_present, "expected at least one origin write"
    assert all(manifest_present), "an origin file was written before the manifest"

    manifest = _manifest(cell)
    assert manifest["schema"] == ckpt.CHECKPOINT_IDENTITY_SCHEMA
    assert manifest["version"] == ckpt.CHECKPOINT_IDENTITY_VERSION
    assert manifest["complete"] is True
    assert manifest["opaque_fields"] == []


# --------------------------------------------------------------------------- #
# 2. The matching cases still work: resume, and partial resume
# --------------------------------------------------------------------------- #
def test_identical_identity_resumes_and_skips_completed_work(
    tmp_path: Path, monkeypatch
) -> None:
    cell = tmp_path / "cell"
    full = _run(cell).to_frame().sort_values("origin_pos").reset_index(drop=True)
    digest_before = _manifest(cell)["digest"]

    computed = _spy_on_origin_fits(monkeypatch)
    resumed = _run(cell).to_frame().sort_values("origin_pos").reset_index(drop=True)

    assert computed == [], "an identical run refitted an origin already on disk"
    np.testing.assert_allclose(
        resumed["prediction"].to_numpy(dtype=float),
        full["prediction"].to_numpy(dtype=float),
        rtol=0,
        atol=0,
    )
    # A matching resume does not need to rewrite the manifest, and must not
    # change what it says.
    assert _manifest(cell)["digest"] == digest_before


def test_interrupted_matching_run_recomputes_only_the_missing_origins(
    tmp_path: Path, monkeypatch
) -> None:
    """The case the checkpoint exists for: same configuration, partial output."""
    cell = tmp_path / "cell"
    full = _run(cell).to_frame().sort_values("origin_pos").reset_index(drop=True)

    hdir = _hdir(cell)
    positions = sorted(ckpt.completed_origin_positions(hdir))
    assert len(positions) >= 2
    last = positions[-1]
    (hdir / f"origin_{last}.parquet").unlink()

    computed = _spy_on_origin_fits(monkeypatch)
    resumed = _run(cell).to_frame().sort_values("origin_pos").reset_index(drop=True)

    assert computed == [last]
    np.testing.assert_allclose(
        resumed["prediction"].to_numpy(dtype=float),
        full["prediction"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    )


# --------------------------------------------------------------------------- #
# 3. Changed forecast-affecting input -> fail closed
# --------------------------------------------------------------------------- #
_CHANGED_INPUTS = {
    # The headline case: nothing about the path or the file names changes, and
    # 0.01 vs 1000 is a different forecast in every row.
    "low_level_params": {"params": {"alpha": 1000.0}},
    "window": {"window": _window(min_size=20)},
    "one_panel_cell": {"panel": _panel(bump=(0, "x1"))},
    "model": {"model": "lasso"},
    "features": {"features": mf.feature_engineering.feature_spec(
        target="y", target_lags=[1, 2, 3]
    )},
    "selection_metric": {"model_selection_metric": "mae"},
    "selection_history": {"selection_history": True},
}


@pytest.mark.parametrize("override", sorted(_CHANGED_INPUTS), ids=sorted(_CHANGED_INPUTS))
def test_changed_input_fails_closed_instead_of_returning_stale_forecasts(
    tmp_path: Path, override: str
) -> None:
    cell = tmp_path / "cell"
    _run(cell)

    with pytest.raises(ValueError, match="already holds .* completed origin file"):
        _run(cell, **_CHANGED_INPUTS[override])


def test_refusal_names_the_directory_and_what_differs(tmp_path: Path) -> None:
    cell = tmp_path / "cell"
    _run(cell)

    with pytest.raises(ValueError) as excinfo:
        _run(cell, params={"alpha": 1000.0})

    message = str(excinfo.value)
    assert str(_hdir(cell)) in message
    assert "models" in message  # the component that changed, named
    assert "fresh directory" in message  # the recovery path, named


def test_refusal_changes_nothing_on_disk(tmp_path: Path) -> None:
    """Fail closed is not fail destructive: no purge, no quarantine, no rename,
    and no blessing of the old files with a new identity."""
    cell = tmp_path / "cell"
    _run(cell)
    before = _snapshot(cell)

    with pytest.raises(ValueError):
        _run(cell, params={"alpha": 1000.0})

    assert _snapshot(cell) == before


def test_changed_random_seed_fails_closed(tmp_path: Path) -> None:
    """Seeds reach stochastic search and stochastic models, so two runs at
    different seeds are different forecasts even with every other input equal."""
    cell = tmp_path / "cell"
    mf.meta.configure(random_seed=11)
    _run(cell)
    assert _manifest(cell)["components"]["seeds"]["selection_random_state"] == 11

    mf.meta.configure(random_seed=12)
    with pytest.raises(ValueError, match="differs from theirs in .*seeds"):
        _run(cell)


def test_model_store_path_matters_only_when_model_saving_is_enabled(
    tmp_path: Path,
) -> None:
    """A setting that wrote nothing must not refuse a resume; one that did, must."""
    quiet = tmp_path / "quiet"
    _run(quiet, save_models=False, model_store=str(tmp_path / "store_a"))
    _run(quiet, save_models=False, model_store=str(tmp_path / "store_b"))  # resumes

    saving = tmp_path / "saving"
    _run(saving, save_models=True, model_store=str(tmp_path / "store_a"))
    with pytest.raises(ValueError, match="already holds"):
        _run(saving, save_models=True, model_store=str(tmp_path / "store_b"))


def test_relative_model_store_is_identified_by_path_not_by_string(
    tmp_path: Path, monkeypatch
) -> None:
    """``"trained_model"`` is not one store; it is one store PER working directory.

    ``_store_model_fit`` writes to ``Path(root) / <alias>``, so the default
    relative ``model_store`` denotes a different directory from a different cwd.
    Comparing the raw string made those runs identical, and the resume then
    skipped every fit -- leaving a ``save_models=True`` run whose newly denoted
    store is empty, with nothing on disk saying so.
    """
    cell = tmp_path / "cell"
    first_cwd = tmp_path / "cwd_a"
    second_cwd = tmp_path / "cwd_b"
    first_cwd.mkdir()
    second_cwd.mkdir()

    monkeypatch.chdir(first_cwd)
    _run(cell, save_models=True, model_store="trained_model")
    assert list(first_cwd.glob("trained_model/*/*.pkl")), "the first cwd holds the models"
    assert str(first_cwd) in _manifest(cell)["components"]["model_saving"]["model_store"]

    # Same cwd, same string: still the same store, so this must resume.
    _run(cell, save_models=True, model_store="trained_model")

    monkeypatch.chdir(second_cwd)
    with pytest.raises(ValueError, match="differs from theirs in .*model_saving"):
        _run(cell, save_models=True, model_store="trained_model")
    assert not list(second_cwd.glob("trained_model/**/*.pkl")), (
        "the refused run must not report success over an empty store"
    )


def test_model_store_identity_is_absent_when_saving_is_disabled(tmp_path: Path) -> None:
    """The normalisation must not leak a path into a run that writes no models."""
    cell = tmp_path / "cell"
    monkey_free = tmp_path / "elsewhere"
    monkey_free.mkdir()

    _run(cell, save_models=False, model_store="trained_model")
    assert _manifest(cell)["components"]["model_saving"] == {
        "save_models": False,
        "model_store": None,
    }


# --------------------------------------------------------------------------- #
# 4. Unknowable identity -> fail closed, but stay readable
# --------------------------------------------------------------------------- #
def test_legacy_origins_without_a_manifest_fail_closed_but_stay_loadable(
    tmp_path: Path,
) -> None:
    """A checkpoint written before manifests existed. Its configuration is
    unknowable, so it cannot be resumed INTO -- and adopting it would write this
    run's identity beside artifacts that may not be this run's at all. Reading it
    is untouched: that is what ``rescore`` does."""
    cell = tmp_path / "cell"
    _run(cell)
    hdir = _hdir(cell)
    (hdir / ckpt.CHECKPOINT_IDENTITY_FILENAME).unlink()

    with pytest.raises(ValueError, match="no manifest records which configuration"):
        _run(cell)

    # Still loadable, and still not adopted.
    assert not (hdir / ckpt.CHECKPOINT_IDENTITY_FILENAME).exists()
    frame = ckpt.load_checkpoint_frame(hdir)
    assert not frame.empty
    assert frame["prediction"].notna().any()


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("corrupt", "{not json at all"),
        ("truncated", '{"schema": "macroforecast_checkpoint_run_identity"'),
        ("not_an_object", '["macroforecast_checkpoint_run_identity"]'),
        ("wrong_schema", '{"schema": "macroforecast_checkpoint_cell_manifest", "version": 1}'),
        (
            "wrong_version",
            '{"schema": "macroforecast_checkpoint_run_identity", "version": 99,'
            ' "digest": "x", "complete": true, "components": {}}',
        ),
        (
            "no_digest",
            '{"schema": "macroforecast_checkpoint_run_identity", "version": 1,'
            ' "complete": true, "components": {}}',
        ),
    ],
)
def test_malformed_or_wrong_schema_manifest_fails_closed(
    tmp_path: Path, label: str, text: str
) -> None:
    cell = tmp_path / "cell"
    _run(cell)
    path = _hdir(cell) / ckpt.CHECKPOINT_IDENTITY_FILENAME
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="already holds .* completed origin file"):
        _run(cell)

    # Refused, not repaired: a manifest whose meaning is unknown is not replaced
    # while the artifacts it might describe are still there.
    assert path.read_text(encoding="utf-8") == text


def test_stored_incomplete_identity_fails_closed(tmp_path: Path) -> None:
    cell = tmp_path / "cell"
    _run(cell)
    path = _hdir(cell) / ckpt.CHECKPOINT_IDENTITY_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["complete"] = False
    payload["opaque_fields"] = ["run.models[0].implementation"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="INCOMPLETE identity"):
        _run(cell)


def test_opaque_custom_model_refuses_resume_once_origins_exist(tmp_path: Path) -> None:
    """A user callable cannot be represented canonically, so this run cannot show
    that the stored origins are its own. Asserting equality would be a guess with
    a wrong number on the other side of it."""

    def _custom_ols(X, y, **params):
        class _Fit:
            def __init__(self, beta):
                self.beta = beta

            def predict(self, X_new):
                return np.asarray(X_new, dtype=float) @ self.beta

        beta, *_ = np.linalg.lstsq(
            np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel(), rcond=None
        )
        return _Fit(beta)

    model = mf.models.custom_model("custom_ols", _custom_ols)
    cell = tmp_path / "cell"
    _run(cell, model=model, params=None)

    manifest = _manifest(cell)
    assert manifest["complete"] is False
    assert manifest["opaque_fields"], "an opaque implementation must be recorded"

    with pytest.raises(ValueError, match="INCOMPLETE identity"):
        _run(cell, model=model, params=None)


# --------------------------------------------------------------------------- #
# 5. No origin files -> nothing to protect, so replace and start fresh
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("stale", None),  # filled in below with a real manifest carrying a wrong digest
        ("corrupt", "{not json"),
        ("wrong_schema", '{"schema": "something_else"}'),
    ],
)
def test_no_origin_files_replaces_the_old_manifest_and_starts_fresh(
    tmp_path: Path, label: str, text: str | None
) -> None:
    cell = tmp_path / "cell"
    hdir = _hdir(cell)
    hdir.mkdir(parents=True)
    path = hdir / ckpt.CHECKPOINT_IDENTITY_FILENAME
    if text is None:
        text = json.dumps(
            {
                "schema": ckpt.CHECKPOINT_IDENTITY_SCHEMA,
                "version": ckpt.CHECKPOINT_IDENTITY_VERSION,
                "digest": "0" * 64,
                "complete": True,
                "opaque_fields": [],
                "components": {"run": "some other run"},
            }
        )
    path.write_text(text, encoding="utf-8")

    result = _run(cell)

    assert not result.to_frame().empty
    assert _manifest(cell)["digest"] != "0" * 64
    assert sorted(hdir.glob("origin_*.parquet"))


def test_leftover_temporary_origin_write_does_not_count_as_an_artifact(
    tmp_path: Path,
) -> None:
    """A crashed write leaves ``.origin_N.parquet.tmp``, never a final name. It is
    not a completed origin and must not make a fresh directory unresumable."""
    cell = tmp_path / "cell"
    hdir = _hdir(cell)
    hdir.mkdir(parents=True)
    (hdir / ".origin_3.parquet.tmp").write_bytes(b"half a parquet file")

    result = _run(cell)

    assert not result.to_frame().empty
    assert (hdir / ckpt.CHECKPOINT_IDENTITY_FILENAME).exists()


def test_unreadable_origin_file_is_an_artifact_and_is_recomputed_not_trusted(
    tmp_path: Path,
) -> None:
    """A final-named file that cannot be parsed still counts as a user artifact
    for the gate (so a legacy directory of them is refused, not adopted), and on a
    matching resume its origin is recomputed rather than served."""
    cell = tmp_path / "cell"
    _run(cell)
    hdir = _hdir(cell)
    positions = sorted(ckpt.completed_origin_positions(hdir))
    (hdir / f"origin_{positions[-1]}.parquet").write_bytes(b"not parquet")

    # Same identity: resumes, and the damaged origin is simply recomputed.
    assert not _run(cell).to_frame().empty

    # Without a manifest the same damaged directory is refused, not adopted.
    other = tmp_path / "legacy"
    _hdir(other).mkdir(parents=True)
    (_hdir(other) / "origin_3.parquet").write_bytes(b"not parquet")
    with pytest.raises(ValueError, match="no manifest records which configuration"):
        _run(other)


# --------------------------------------------------------------------------- #
# 6. Multi-horizon: identity is per h<h> directory
# --------------------------------------------------------------------------- #
def test_multi_horizon_identity_is_per_horizon_directory(tmp_path: Path) -> None:
    cell = tmp_path / "cell"
    mf.forecasting.run(
        _panel(),
        "ridge",
        window=_window(),
        features=_features(),
        params={"alpha": 0.01},
        horizons=[1, 3],
        save_models=False,
        checkpoint_path=cell,
    )

    manifests = {h: _manifest(cell, h) for h in (1, 3)}
    assert manifests[1]["digest"] != manifests[3]["digest"], (
        "origin positions are horizon-independent, so two horizons sharing one "
        "digest would let h1's origins be resumed as h3's"
    )

    # Each directory is gated on its own: removing only h3's manifest refuses at
    # h3 and names h3, leaving h1 resumable.
    (_hdir(cell, 3) / ckpt.CHECKPOINT_IDENTITY_FILENAME).unlink()
    with pytest.raises(ValueError) as excinfo:
        mf.forecasting.run(
            _panel(),
            "ridge",
            window=_window(),
            features=_features(),
            params={"alpha": 0.01},
            horizons=[1, 3],
            save_models=False,
            checkpoint_path=cell,
        )
    message = str(excinfo.value)
    assert str(_hdir(cell, 3)) in message
    assert str(_hdir(cell, 1)) not in message


# --------------------------------------------------------------------------- #
# 7. The vintage-aware route carries the same gate
# --------------------------------------------------------------------------- #
class _SyntheticVintageSource:
    kind = "synthetic_vintage"
    dataset = "synthetic"
    frequency = "monthly"

    def __init__(self, bundles: dict) -> None:
        self.bundles = dict(bundles)

    def available_vintages(self):
        return list(self.bundles)

    def resolve(self, origin_date):
        origin = pd.Timestamp(origin_date)
        keys = [key for key in self.bundles if key <= origin]
        if not keys:
            raise mf.data.VintageUnavailableError("no vintage available")
        return self.bundles[max(keys)]


#: Vintages published after this reference position exist but are never resolved
#: as an execution origin's input under ``_run_vintage``'s window (origins land on
#: positions 8, 12, ... 32). Asserted, not assumed, by
#: ``test_vintage_bundles_used_as_origin_inputs_are_the_ones_this_window_reads``.
_LAST_ORIGIN_VINTAGE = 32


def _vintage_spec(
    *,
    shift: float = 0.0,
    revise: tuple[int, float] | None = None,
    metadata_revise: tuple[int, dict] | None = None,
    actuals_vintage: str = "latest",
    provenance: bool = False,
) -> mf.data.VintagePanelSpec:
    """One non-leaky vintage per reference date: the vintage published at
    ``reference[i]`` carries rows ``reference[:i]`` only.

    ``shift`` moves EVERY snapshot, which is the easy case. ``revise=(i, delta)``
    is the hard one: it adds ``delta`` to the newest target row of vintage ``i``
    and touches nothing else, so the available label set and every other bundle --
    the LATEST one included -- stay byte-identical.

    ``metadata_revise=(i, fields)`` is harder still: it merges ``fields`` into
    vintage ``i``'s ``bundle.metadata`` and leaves every panel, every label and
    every ``vintage`` id exactly as they were, so nothing a values-only identity
    can observe moves at all.

    ``provenance=True`` stamps every bundle with the loader-shaped ``artifact``
    block a vintage source backed by real files would carry, with a fresh
    ``downloaded_at`` per call and fixed content fields. It is section 10's case on
    the vintage route: two identical spec constructions differ only in when they
    were built.
    """
    reference = pd.date_range("2000-01-31", periods=36, freq="ME", name="date")
    bundles = {}
    for i in range(1, len(reference)):
        values = np.arange(1, i + 1, dtype=float) + shift
        panel = pd.DataFrame(
            {"A": values, "B": values * 10.0 + np.sin(np.arange(i))},
            index=reference[:i],
        )
        if revise is not None and revise[0] == i:
            panel.iloc[-1, panel.columns.get_loc("A")] += revise[1]
        metadata = {"dataset": "synthetic", "frequency": "monthly", "vintage": f"v{i}"}
        if provenance:
            metadata["artifact"] = {
                "dataset": "synthetic",
                "file_sha256": hashlib.sha256(f"v{i}".encode("utf-8")).hexdigest(),
                "file_format": "csv",
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "cache_hit": i % 2 == 0,
            }
        if metadata_revise is not None and metadata_revise[0] == i:
            metadata.update(metadata_revise[1])
        bundles[reference[i]] = mf.data.DataBundle(panel, metadata)
    return mf.data.VintagePanelSpec(
        _SyntheticVintageSource(bundles),
        reference,
        actuals_vintage=actuals_vintage,
    )


def _vintage_bundle(spec: mf.data.VintagePanelSpec, index: int) -> mf.data.DataBundle:
    """The bundle ``_vintage_spec`` published as vintage ``v{index}``."""
    labels = list(spec.source.available_vintages())
    return spec.source.resolve(pd.Timestamp(labels[index - 1]))


def _run_vintage(
    checkpoint_path,
    *,
    shift: float = 0.0,
    alpha: float = 0.01,
    revise: tuple[int, float] | None = None,
    metadata_revise: tuple[int, dict] | None = None,
    metadata_feature: bool = False,
    metadata_step=None,
    actuals_vintage: str = "latest",
    provenance: bool = False,
):
    """``metadata_feature=True`` adds the metadata-reading step of section 9, so
    a bundle's ``metadata`` can change a forecast while its panel cannot.
    ``metadata_step`` substitutes a different metadata-reading callable in the
    same slot, which is how section 11 reaches this route with a callable
    dataclass instead of a marked function.

    That step needs a predictor column to hang off (it reads only the index, but
    a custom step must name columns that exist in the step source panel), hence
    the ``predictors`` switch below."""
    reads_metadata = metadata_feature or metadata_step is not None
    steps = (
        [
            mf.feature_engineering.custom_step(
                "phased", metadata_step or _metadata_sensitive_feature, columns=["B"]
            )
        ]
        if reads_metadata
        else None
    )
    predictors = ["B"] if reads_metadata else []
    return mf.forecasting.run(
        _vintage_spec(
            shift=shift,
            revise=revise,
            metadata_revise=metadata_revise,
            actuals_vintage=actuals_vintage,
            provenance=provenance,
        ),
        "ridge",
        target="A",
        window=mf.window.spec(
            estimation=mf.window.estimation_expanding(min_size=8),
            val=mf.window.val_last_block(size=3),
            test=mf.window.test_origins(horizon=1, step=4),
        ),
        features=mf.feature_engineering.feature_spec(
            target="A",
            predictors=predictors,
            lags=None,
            target_lags=(1, 2),
            steps=steps,
        ),
        params={"alpha": alpha},
        save_models=False,
        checkpoint_path=checkpoint_path,
    )


def _latest_bundle_bytes(spec: mf.data.VintagePanelSpec) -> tuple:
    """The identity a latest-only fingerprint could possibly see.

    Includes the whole of ``latest.metadata``, not just its ``vintage``, so a test
    asserting "the latest bundle did not move" also rules out a metadata-only
    change to it -- which section 9 shows is enough to change a forecast.
    """
    from macroforecast.data.identity import panel_fingerprint

    labels = list(spec.source.available_vintages())
    latest = spec.source.resolve(pd.Timestamp(labels[-1]))
    return (
        [str(label) for label in labels],
        panel_fingerprint(latest.panel)["value"],
        json.dumps(dict(latest.metadata), sort_keys=True, default=str),
    )


def test_vintage_route_writes_a_truthful_identity_and_resumes(
    tmp_path: Path, monkeypatch
) -> None:
    cell = tmp_path / "cell"
    full = _run_vintage(cell).to_frame().sort_values("origin_pos").reset_index(drop=True)

    manifest = _manifest(cell)
    data = manifest["components"]["data"]
    assert data["kind"] == "vintage"
    # Truthful source identity: the vintages that exist, not just a type name.
    assert data["source"]["available_vintages"], "vintage labels must identify the source"
    assert data["source"]["type"].endswith("_SyntheticVintageSource")
    # ... plus the CONTENT of every bundle the run reads. One row per execution
    # origin, each naming the vintage it resolved and fingerprinting that
    # vintage's work panel -- not merely the latest one.
    origin_inputs = data["origin_inputs"]
    assert origin_inputs["columns"] == [
        "origin_pos",
        "vintage_id",
        "panel_fingerprint",
        "metadata",
    ]
    positions = [row[0] for row in origin_inputs["rows"]]
    assert positions == sorted(set(positions)), "one row per execution origin"
    assert positions == sorted(int(pos) for pos in full["origin_pos"].unique())
    assert all(row[1] and row[2] for row in origin_inputs["rows"])
    assert len({row[2] for row in origin_inputs["rows"]}) == len(positions), (
        "distinct vintage snapshots must not share a fingerprint"
    )
    # ... including the attrs payload each origin's feature steps are handed,
    # which the fingerprint does not cover (section 9).
    assert all(row[3]["vintage"] == row[1] for row in origin_inputs["rows"])
    # ... and the actuals side, which under 'latest' is the one latest panel.
    assert data["actuals"]["policy"] == "latest"
    assert data["actuals"]["base_panel_fingerprint"]
    assert data["actuals"]["base_panel_metadata"]["vintage"] == data["actuals"]["base_vintage_id"]
    assert manifest["complete"] is True

    computed = _spy_on_origin_fits(monkeypatch)
    resumed = (
        _run_vintage(cell).to_frame().sort_values("origin_pos").reset_index(drop=True)
    )
    assert computed == []
    np.testing.assert_allclose(
        resumed["prediction"].to_numpy(dtype=float),
        full["prediction"].to_numpy(dtype=float),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("change", ["params", "vintage_content"])
def test_vintage_route_fails_closed_on_changed_input(tmp_path: Path, change: str) -> None:
    cell = tmp_path / "cell"
    _run_vintage(cell)
    kwargs = {"params": {"alpha": 1000.0}, "vintage_content": {"shift": 5.0}}[change]

    with pytest.raises(ValueError, match="already holds .* completed origin file"):
        _run_vintage(cell, **kwargs)


def test_vintage_bundles_used_as_origin_inputs_are_the_ones_this_window_reads(
    tmp_path: Path,
) -> None:
    """Pins the fixture the two revision tests below aim with.

    Those tests only mean what they claim if ``_LAST_ORIGIN_VINTAGE`` really is
    the last vintage any origin resolves as its fit input -- otherwise the
    "actuals only" case would silently be an origin-input case."""
    cell = tmp_path / "cell"
    _run_vintage(cell)

    rows = _manifest(cell)["components"]["data"]["origin_inputs"]["rows"]
    used = {int(str(row[1]).lstrip("v")) for row in rows}
    assert used == {8, 12, 16, 20, 24, 28, 32}
    assert max(used) == _LAST_ORIGIN_VINTAGE


def test_revised_intermediate_vintage_fails_closed(tmp_path: Path) -> None:
    """The defect a latest-only fingerprint cannot see.

    ``shift`` moves every snapshot at once, so it is caught by ANY fingerprint of
    ANY bundle -- including the latest-only one this identity started with. Here
    exactly one INTERMEDIATE bundle is revised: the available labels and the
    latest bundle stay byte-identical (asserted, so this test keeps failing under
    a latest-only identity rather than passing for the wrong reason), while the
    origins that resolve that vintage would fit on different data.
    """
    revised = 20
    assert revised < _LAST_ORIGIN_VINTAGE

    # What a latest-only identity is able to observe does not move at all.
    assert _latest_bundle_bytes(_vintage_spec()) == _latest_bundle_bytes(
        _vintage_spec(revise=(revised, 7.0))
    )

    # The revision is real: computed fresh, the forecasts differ.
    baseline = _run_vintage(tmp_path / "baseline").to_frame().sort_values("origin_pos")
    revised_run = (
        _run_vintage(tmp_path / "revised", revise=(revised, 7.0))
        .to_frame()
        .sort_values("origin_pos")
    )
    assert not np.allclose(
        baseline["prediction"].to_numpy(dtype=float),
        revised_run["prediction"].to_numpy(dtype=float),
    ), "the revision must be able to change a forecast, or the test proves nothing"

    cell = tmp_path / "cell"
    _run_vintage(cell)
    with pytest.raises(ValueError, match="already holds .* completed origin file"):
        _run_vintage(cell, revise=(revised, 7.0))


def test_revised_first_release_actual_fails_closed(tmp_path: Path) -> None:
    """The same defect on the ACTUALS side of a ``first_release`` run.

    The revised vintage here is published after every execution origin, so it is
    never an origin's fit input; it is only the first bundle to publish the last
    origin's realised value. Nothing else -- labels, latest bundle, every origin
    input -- moves, so this is refused only if the identity covers the resolved
    first-release values and their vintage ids.
    """
    revised = _LAST_ORIGIN_VINTAGE + 2
    assert revised < 35, "must not be the latest vintage, or the old identity sees it"

    assert _latest_bundle_bytes(_vintage_spec()) == _latest_bundle_bytes(
        _vintage_spec(revise=(revised, 9.0))
    )

    baseline = (
        _run_vintage(tmp_path / "baseline", actuals_vintage="first_release")
        .to_frame()
        .sort_values("origin_pos")
    )
    revised_run = (
        _run_vintage(
            tmp_path / "revised", revise=(revised, 9.0), actuals_vintage="first_release"
        )
        .to_frame()
        .sort_values("origin_pos")
    )
    # Only the realised values move; the fits saw identical point-in-time data.
    np.testing.assert_allclose(
        baseline["prediction"].to_numpy(dtype=float),
        revised_run["prediction"].to_numpy(dtype=float),
        rtol=0,
        atol=0,
    )
    assert not np.allclose(
        baseline["actual"].to_numpy(dtype=float),
        revised_run["actual"].to_numpy(dtype=float),
        equal_nan=True,
    ), "the revision must be able to change an actual, or the test proves nothing"

    cell = tmp_path / "cell"
    _run_vintage(cell, actuals_vintage="first_release")
    manifest_rows = _manifest(cell)["components"]["data"]["origin_inputs"]["rows"]
    assert revised not in {int(str(row[1]).lstrip("v")) for row in manifest_rows}, (
        "this vintage must reach the run only through the actuals resolver"
    )

    with pytest.raises(ValueError, match="already holds .* completed origin file"):
        _run_vintage(cell, revise=(revised, 9.0), actuals_vintage="first_release")


def test_vintage_unrelated_to_this_run_does_not_refuse_a_resume(
    tmp_path: Path, monkeypatch
) -> None:
    """Complete is not the same as over-broad.

    Vintage 21 is published between two origins, so under ``actuals_vintage=
    'latest'`` no origin resolves it and no actual comes from it. Revising it
    cannot change a single forecast row, and refusing over it would make every
    ordinary data extension unresumable."""
    cell = tmp_path / "cell"
    first = _run_vintage(cell).to_frame().sort_values("origin_pos").reset_index(drop=True)

    computed = _spy_on_origin_fits(monkeypatch)
    resumed = (
        _run_vintage(cell, revise=(21, 7.0))
        .to_frame()
        .sort_values("origin_pos")
        .reset_index(drop=True)
    )
    assert computed == []
    np.testing.assert_allclose(
        resumed["prediction"].to_numpy(dtype=float),
        first["prediction"].to_numpy(dtype=float),
        rtol=0,
        atol=0,
    )


# --------------------------------------------------------------------------- #
# 8. run_pipeline inherits the gate, and its cell manifest survives a refusal
# --------------------------------------------------------------------------- #
def _pipeline_spec(checkpoint_dir: Path, *, ridge_alpha: float = 1.0):
    from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec

    index = pd.date_range("1990-01-01", periods=120, freq="MS", name="date")
    rng = np.random.default_rng(5)
    columns = {f"S{i}": rng.normal(size=120) for i in range(3)}
    columns["Y"] = np.cumsum(rng.normal(size=120))
    panel = pd.DataFrame(columns, index=index)
    bundle = mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})
    features = mf.feature_engineering.feature_spec(
        target="Y", predictors="all", lags=range(1, 3)
    )
    return pipeline_spec(
        data=bundle,
        targets=[TargetSpec(name="Y")],
        horizons=[1],
        window=mf.window.from_cutoffs(
            test_start="1999-01-01",
            test_end="1999-06-01",
            mode="expanding",
            val_method="last_block",
            retrain_every=1,
        ),
        arms=[
            Arm(
                name="RIDGE",
                model="ridge",
                features=features,
                params={"alpha": ridge_alpha},
                is_benchmark=True,
            ),
            Arm(name="LASSO", model="lasso", features=features),
        ],
        evaluation=EvalSpec(benchmark="RIDGE", metrics=("rmse",)),
        checkpoint_dir=str(checkpoint_dir),
        save_models=False,
    )


def test_run_pipeline_stale_cell_is_refused_and_its_cell_manifest_is_untouched(
    tmp_path: Path,
) -> None:
    """The pipeline's own ``cell_manifest.json`` is written only after ``run()``
    returns, so before this it could neither protect the resume nor survive it:
    the stale forecasts came back AND the manifest was overwritten with the new
    identity, so nothing on disk recorded that a swap had happened."""
    from macroforecast.pipeline import run_pipeline

    checkpoint_dir = tmp_path / "ckpt"
    first = run_pipeline(_pipeline_spec(checkpoint_dir, ridge_alpha=1.0))
    assert not first.forecasts.empty

    ridge_h1 = checkpoint_dir / "Y__RIDGE" / "h1"
    cell_manifest = ridge_h1 / "cell_manifest.json"
    before = cell_manifest.read_bytes()
    ridge_predictions_before = (
        first.forecasts.loc[first.forecasts["contender"] == "RIDGE", "prediction"]
        .to_numpy(dtype=float)
        .copy()
    )

    with pytest.warns(RuntimeWarning, match="pipeline cell failed"):
        second = run_pipeline(_pipeline_spec(checkpoint_dir, ridge_alpha=500.0))

    # The stale cell failed rather than returning the alpha=1.0 forecasts.
    assert second.failed_cells, "the stale RIDGE cell should be reported as failed"
    assert any("RIDGE" in str(cell) for cell in second.failed_cells)
    returned_ridge = second.forecasts.loc[second.forecasts["contender"] == "RIDGE"]
    if not returned_ridge.empty:  # pragma: no cover - would be the F-058 defect
        assert not np.allclose(
            returned_ridge["prediction"].to_numpy(dtype=float),
            ridge_predictions_before,
        )

    # The pipeline's post-run manifest write is unreachable for a refused cell,
    # so the old cell identity is still on disk to be diagnosed.
    assert cell_manifest.read_bytes() == before

    # The unaffected arm still ran.
    assert "LASSO" in set(second.forecasts["contender"])


def test_identity_is_stable_across_the_serial_and_parallel_backends(
    tmp_path: Path,
) -> None:
    """A serial run must be resumable under ``n_jobs>1`` and back.

    The two backends group horizons differently -- serial runs one multi-horizon
    ``run()`` per cell, parallel one single-horizon ``run()`` per process -- and
    the worker rebuilds its config from ``spec.seed`` rather than inheriting the
    parent's. If either made the identity differ, every interrupted run resumed on
    a different backend would refuse for no reason, which is a worse failure than
    the one this gate exists to prevent.
    """
    from macroforecast.pipeline import run_pipeline

    checkpoint_dir = tmp_path / "ckpt"
    serial = run_pipeline(_pipeline_spec(checkpoint_dir))
    assert not serial.forecasts.empty
    digests = {
        str(path.parent.relative_to(checkpoint_dir)): json.loads(
            path.read_text(encoding="utf-8")
        )["digest"]
        for path in sorted(checkpoint_dir.rglob(ckpt.CHECKPOINT_IDENTITY_FILENAME))
    }
    assert digests

    parallel = run_pipeline(_dc.replace(_pipeline_spec(checkpoint_dir), n_jobs=2))

    assert parallel.failed_cells == (), "a backend change must not refuse a resume"
    assert {
        str(path.parent.relative_to(checkpoint_dir)): json.loads(
            path.read_text(encoding="utf-8")
        )["digest"]
        for path in sorted(checkpoint_dir.rglob(ckpt.CHECKPOINT_IDENTITY_FILENAME))
    } == digests


# --------------------------------------------------------------------------- #
# 9. panel.attrs metadata is an INPUT, so it belongs in the input identity
# --------------------------------------------------------------------------- #
# ``panel_fingerprint`` covers index, columns and values and deliberately does not
# cover ``panel.attrs``. The runner does not treat attrs as decoration: it
# re-attaches ``macroforecast_metadata`` to every per-origin slice and hands it to
# each feature step as ``metadata=``. So two inputs whose panels are byte-identical
# and whose metadata differs can produce different forecasts -- and did compare
# EQUAL, which is the silent reuse F-058 exists to stop.
_BASE_METADATA = {"phase": 0.0, "provenance_note": "baseline extract"}


def _panel_with_metadata(metadata: dict) -> pd.DataFrame:
    """``_panel()`` exactly, with a metadata payload and nothing else changed."""
    panel = _panel()
    panel.attrs["macroforecast_metadata"] = dict(metadata)
    return panel


def _run_with_metadata(checkpoint_path, metadata: dict, **overrides):
    _OBSERVED_METADATA.clear()
    return _run(
        checkpoint_path,
        panel=_panel_with_metadata(metadata),
        features=mf.feature_engineering.feature_spec(
            target="y",
            target_lags=[1, 2],
            steps=[
                mf.feature_engineering.custom_step(
                    "phased", _metadata_sensitive_feature, columns=["x1"]
                )
            ],
        ),
        **overrides,
    )


def test_metadata_only_change_fails_closed(tmp_path: Path, monkeypatch) -> None:
    """The ordinary-input case: same values, different metadata, different answer."""
    from macroforecast.data.identity import panel_fingerprint

    changed = {**_BASE_METADATA, "phase": 7.0}
    plain, moved = _panel_with_metadata(_BASE_METADATA), _panel_with_metadata(changed)
    # Nothing the panel fingerprint can see moves at all, so this test keeps
    # failing under a values-only identity rather than passing for another reason.
    assert plain.equals(moved)
    assert panel_fingerprint(plain)["value"] == panel_fingerprint(moved)["value"]

    baseline = (
        _run_with_metadata(tmp_path / "baseline", _BASE_METADATA)
        .to_frame()
        .sort_values("origin_pos")
    )
    revised = (
        _run_with_metadata(tmp_path / "revised", changed)
        .to_frame()
        .sort_values("origin_pos")
    )
    assert not np.allclose(
        baseline["prediction"].to_numpy(dtype=float),
        revised["prediction"].to_numpy(dtype=float),
        rtol=0,
        atol=0,
    ), "the metadata change must be able to move a forecast, or this proves nothing"

    cell = tmp_path / "cell"
    _run_with_metadata(cell, _BASE_METADATA)
    before = _snapshot(cell)
    with pytest.raises(ValueError, match="already holds .* completed origin file"):
        _run_with_metadata(cell, changed)
    assert _snapshot(cell) == before, "a refusal must leave the directory untouched"

    # Complete, not over-broad: the unchanged run still resumes without recomputing.
    computed = _spy_on_origin_fits(monkeypatch)
    resumed = (
        _run_with_metadata(cell, _BASE_METADATA)
        .to_frame()
        .sort_values("origin_pos")
        .reset_index(drop=True)
    )
    assert computed == []
    np.testing.assert_allclose(
        resumed["prediction"].to_numpy(dtype=float),
        baseline.reset_index(drop=True)["prediction"].to_numpy(dtype=float),
        rtol=0,
        atol=0,
    )


def test_identity_records_the_whole_metadata_payload_handed_to_user_code(
    tmp_path: Path,
) -> None:
    """No guesswork about which fields matter: record what user code receives.

    A custom step may branch on any field, and preprocessing and data contracts
    read this payload too, so there is no sound way to decide from the outside
    that a given key is inert. The identity therefore carries the payload the
    runner actually passes -- asserted here against what the step was handed, not
    against a hand-written expectation of it.
    """
    metadata = {
        **_BASE_METADATA,
        # Never read by the step, and recorded anyway.
        "unread_by_the_step": {"ticket": 412, "tags": ["a", "b"]},
    }
    cell = tmp_path / "cell"
    _run_with_metadata(cell, metadata)

    observed = list(_OBSERVED_METADATA)
    assert observed, "the custom step must have run, or nothing is being compared"
    assert all(payload == observed[0] for payload in observed), (
        "every origin sees the same ordinary-input metadata"
    )

    recorded = _manifest(cell)["components"]["data"]["metadata"]
    assert recorded == json.loads(json.dumps(observed[0]))
    assert recorded["unread_by_the_step"] == {"ticket": 412, "tags": ["a", "b"]}
    # ... including the panel-contract block the runner itself attaches upstream
    # of the gate, which is part of what feature steps read.
    assert recorded["panel"]["contract"] == "macroforecast_panel_v1"


def test_vintage_origin_bundle_metadata_only_revision_fails_closed(
    tmp_path: Path,
) -> None:
    """The same gap on the vintage route, aimed at one intermediate bundle.

    Only ``bundle.metadata`` moves, and only for a vintage some origin resolves
    as its fit input. Its panel, its ``vintage`` id, the available labels and the
    latest bundle are all unchanged, so this is refused only if each origin row
    carries the metadata payload that origin's feature steps were handed.
    """
    revised = 20
    assert revised < _LAST_ORIGIN_VINTAGE, "must be an origin input, not an actual"
    revision = {"phase": 7.0}

    plain = _vintage_spec()
    moved = _vintage_spec(metadata_revise=(revised, revision))
    assert _latest_bundle_bytes(plain) == _latest_bundle_bytes(moved)
    before_bundle = _vintage_bundle(plain, revised)
    after_bundle = _vintage_bundle(moved, revised)
    assert before_bundle.panel.equals(after_bundle.panel), "values must not move"
    assert before_bundle.metadata["vintage"] == after_bundle.metadata["vintage"]
    assert set(after_bundle.metadata) - set(before_bundle.metadata) == {"phase"}

    baseline = (
        _run_vintage(tmp_path / "baseline", metadata_feature=True)
        .to_frame()
        .sort_values("origin_pos")
    )
    revised_run = (
        _run_vintage(
            tmp_path / "revised",
            metadata_feature=True,
            metadata_revise=(revised, revision),
        )
        .to_frame()
        .sort_values("origin_pos")
    )
    assert not np.allclose(
        baseline["prediction"].to_numpy(dtype=float),
        revised_run["prediction"].to_numpy(dtype=float),
        rtol=0,
        atol=0,
    ), "the metadata revision must be able to move a forecast"

    cell = tmp_path / "cell"
    _run_vintage(cell, metadata_feature=True)
    rows = _manifest(cell)["components"]["data"]["origin_inputs"]["rows"]
    assert revised in {int(str(row[1]).lstrip("v")) for row in rows}, (
        "this vintage must reach the run as an origin input"
    )
    before = _snapshot(cell)
    with pytest.raises(ValueError, match="already holds .* completed origin file"):
        _run_vintage(
            cell, metadata_feature=True, metadata_revise=(revised, revision)
        )
    assert _snapshot(cell) == before, "a refusal must leave the directory untouched"


def test_vintage_identity_records_each_bundles_metadata_payload(tmp_path: Path) -> None:
    """Again the payload as delivered, per origin, and for the actuals panel."""
    cell = tmp_path / "cell"
    _OBSERVED_METADATA.clear()
    _run_vintage(cell, metadata_feature=True)

    data = _manifest(cell)["components"]["data"]
    observed = {
        payload["vintage"]: payload
        for payload in _OBSERVED_METADATA
        if "vintage" in payload
    }
    recorded = {row[1]: row[3] for row in data["origin_inputs"]["rows"]}
    assert recorded, "the vintage route must record per-origin metadata"
    assert set(recorded) <= set(observed), (
        "every recorded origin payload must be one user code was handed"
    )
    for vintage_id, payload in recorded.items():
        assert payload == json.loads(json.dumps(observed[vintage_id]))

    # The actuals panel's payload is recorded for the same reason: the realised
    # value is produced by transforming that panel WITH that metadata.
    actuals = data["actuals"]
    assert actuals["base_panel_metadata"] == json.loads(
        json.dumps(observed[actuals["base_vintage_id"]])
    )


def test_probe_only_bundle_metadata_revision_does_not_refuse_a_resume(
    tmp_path: Path, monkeypatch
) -> None:
    """The deliberate exclusion, stated as a testable claim.

    Under ``first_release`` a probed bundle contributes exactly two things to the
    run -- the value at the requested date and its ``vintage`` id -- and both are
    already recorded in the actuals rows. Its remaining metadata never reaches
    user code, so refusing over it would make ordinary metadata edits unresumable
    for no correctness gain. This vintage is published after every execution
    origin and is not the latest, so it is reachable only through the resolver.
    """
    probe_only = _LAST_ORIGIN_VINTAGE + 2
    cell = tmp_path / "cell"
    first = (
        _run_vintage(cell, metadata_feature=True, actuals_vintage="first_release")
        .to_frame()
        .sort_values("origin_pos")
        .reset_index(drop=True)
    )
    data = _manifest(cell)["components"]["data"]
    assert probe_only not in {
        int(str(row[1]).lstrip("v")) for row in data["origin_inputs"]["rows"]
    }, "this vintage must not be an origin input"
    assert probe_only != int(str(data["actuals"]["base_vintage_id"]).lstrip("v")), (
        "nor the actuals base panel"
    )

    computed = _spy_on_origin_fits(monkeypatch)
    resumed = (
        _run_vintage(
            cell,
            metadata_feature=True,
            actuals_vintage="first_release",
            metadata_revise=(probe_only, {"phase": 7.0}),
        )
        .to_frame()
        .sort_values("origin_pos")
        .reset_index(drop=True)
    )
    assert computed == []
    np.testing.assert_allclose(
        resumed["prediction"].to_numpy(dtype=float),
        first["prediction"].to_numpy(dtype=float),
        rtol=0,
        atol=0,
    )


# --------------------------------------------------------------------------- #
# 10. Loader provenance is WHEN the bytes arrived, not WHAT they are
# --------------------------------------------------------------------------- #
# Section 9 puts the whole ``macroforecast_metadata`` payload in the identity,
# which is right for everything a step can read and wrong for two fields. Every
# public loader stamps ``artifact.downloaded_at`` (wall clock) and
# ``artifact.cache_hit`` (did the bytes come from cache), so two ``load_*`` calls
# over one unchanged file produce two different payloads. Hashing those makes the
# most ordinary resume there is -- reload the data, restart the run -- impossible
# across two calls or two processes, while neither field can move a forecast that
# no user code reads.
#
# "That no user code reads" is the whole subtlety. A custom feature or
# preprocessing step is handed the payload and may branch on those exact fields,
# so filtering them out of the identity while still passing them in would put back
# the silent stale reuse this gate exists to stop. The policy therefore depends on
# the run: normalised for built-in-only pipelines, identified in full when custom
# code can see it.
def _csv_source(tmp_path: Path) -> Path:
    """``_panel()`` written to disk, so the public loader path is the one tested."""
    path = tmp_path / "source.csv"
    _panel().rename_axis("date").to_csv(path)
    return path


def _load(csv_path: Path, tmp_path: Path) -> mf.data.DataBundle:
    """A fresh public ``load_custom_csv`` call over unchanged bytes.

    ``cache_root`` is redirected into ``tmp_path`` so the loader's raw-manifest
    append cannot touch the user's real cache during the test.
    """
    return mf.data.load_custom_csv(
        csv_path,
        date="date",
        dataset="provenance_probe",
        frequency="monthly",
        cache_root=tmp_path / "cache",
    )


def _artifact(bundle: mf.data.DataBundle) -> dict:
    return dict(bundle.panel.attrs["macroforecast_metadata"]["artifact"])


def _run_loaded(checkpoint_path, bundle, *, metadata_feature: bool, **overrides):
    """One ridge cell over a loaded bundle's panel, built-in or custom features."""
    steps = (
        [
            mf.feature_engineering.custom_step(
                "stamped", _downloaded_at_sensitive_feature, columns=["x1"]
            )
        ]
        if metadata_feature
        else []
    )
    return _run(
        checkpoint_path,
        panel=bundle.panel,
        features=mf.feature_engineering.feature_spec(
            target="y", target_lags=[1, 2], steps=steps
        ),
        **overrides,
    )


def _downloaded_at_sensitive_feature(source, *, metadata=None, **params):
    """A stable custom step that reads the very field the built-in path drops.

    Contrived on purpose: the point is that reading ``artifact.downloaded_at`` is
    *legal*, so an identity that filtered it unconditionally would be unsound. The
    output is a deterministic function of the timestamp, so two loads move the
    forecasts. ``__mf_digest__`` keeps the callable itself out of the opaque set,
    so any refusal here comes from the metadata and not from an unrepresentable
    function object.
    """

    payload = dict(metadata or {})
    _OBSERVED_METADATA.append(payload)
    stamp = str(dict(payload.get("artifact", {})).get("downloaded_at", ""))
    phase = float(int(hashlib.sha256(stamp.encode("utf-8")).hexdigest()[:8], 16) % 997)
    months = (source.index.year - 2000) * 12.0 + source.index.month
    return pd.Series(np.sin((months + phase) / 3.0), index=source.index, name="stamped")


_downloaded_at_sensitive_feature.__mf_digest__ = "test:downloaded_at_sensitive/v1"


def test_reloading_the_same_file_still_resumes_a_builtin_run(
    tmp_path: Path, monkeypatch
) -> None:
    """The defect: two loads of one unchanged file must be the same run.

    The bundle is rebuilt rather than reused, because reusing the object would
    reuse its timestamp and the test would pass without the fix.
    """
    csv_path = _csv_source(tmp_path)
    first, second = _load(csv_path, tmp_path), _load(csv_path, tmp_path)

    # The premise, stated as assertions: identical panels and identical content,
    # differing only in when the loader ran.
    pd.testing.assert_frame_equal(first.panel, second.panel)
    a, b = _artifact(first), _artifact(second)
    assert a["downloaded_at"] != b["downloaded_at"], "two loads must be timestamped apart"
    assert a["file_sha256"] == b["file_sha256"]
    assert {key: value for key, value in a.items() if key not in ("downloaded_at",)} == {
        key: value for key, value in b.items() if key not in ("downloaded_at",)
    }, "nothing but the timestamp may differ between two loads of one file"

    cell = tmp_path / "cell"
    baseline = (
        _run_loaded(cell, first, metadata_feature=False)
        .to_frame()
        .sort_values("origin_pos")
        .reset_index(drop=True)
    )
    computed = _spy_on_origin_fits(monkeypatch)
    resumed = (
        _run_loaded(cell, second, metadata_feature=False)
        .to_frame()
        .sort_values("origin_pos")
        .reset_index(drop=True)
    )
    assert computed == [], "a reload of unchanged bytes must not refit a single origin"
    np.testing.assert_allclose(
        resumed["prediction"].to_numpy(dtype=float),
        baseline["prediction"].to_numpy(dtype=float),
        rtol=0,
        atol=0,
    )

    # Normalised, not dropped: the content fields the artifact carries are still
    # what the identity compares on.
    recorded = _manifest(cell)["components"]["data"]["metadata"]["artifact"]
    assert set(recorded) == set(a) - set(runner._LOADER_PROVENANCE_FIELDS)
    assert recorded["file_sha256"] == a["file_sha256"]


def test_nonvolatile_loader_metadata_change_still_refuses(tmp_path: Path) -> None:
    """Only those two fields are forgiven; the rest of the artifact still binds."""
    csv_path = _csv_source(tmp_path)
    cell = tmp_path / "cell"
    _run_loaded(cell, _load(csv_path, tmp_path), metadata_feature=False)

    revised = _load(csv_path, tmp_path)
    metadata = dict(revised.panel.attrs["macroforecast_metadata"])
    metadata["artifact"] = {**dict(metadata["artifact"]), "file_sha256": "0" * 64}
    revised.panel.attrs["macroforecast_metadata"] = metadata

    before = _snapshot(cell)
    with pytest.raises(ValueError, match="already holds .* completed origin file"):
        _run_loaded(cell, revised, metadata_feature=False)
    assert _snapshot(cell) == before, "a refusal must leave the directory untouched"


def test_custom_step_reading_provenance_keeps_it_in_the_identity(
    tmp_path: Path,
) -> None:
    """The exception, and the reason for it: the output really does move.

    Two fresh runs over two loads of one unchanged file are shown to produce
    different forecasts, because the step reads ``downloaded_at``. The refusal
    that follows is therefore not conservatism -- resuming here would return rows
    this configuration would not compute.
    """
    csv_path = _csv_source(tmp_path)
    first, second = _load(csv_path, tmp_path), _load(csv_path, tmp_path)
    pd.testing.assert_frame_equal(first.panel, second.panel)

    _OBSERVED_METADATA.clear()
    left = (
        _run_loaded(tmp_path / "left", first, metadata_feature=True)
        .to_frame()
        .sort_values("origin_pos")
        .reset_index(drop=True)
    )
    right = (
        _run_loaded(tmp_path / "right", second, metadata_feature=True)
        .to_frame()
        .sort_values("origin_pos")
        .reset_index(drop=True)
    )
    assert _OBSERVED_METADATA, "the custom step must have run"
    assert not np.allclose(
        left["prediction"].to_numpy(dtype=float),
        right["prediction"].to_numpy(dtype=float),
        rtol=0,
        atol=0,
    ), "the timestamp must be able to move a forecast, or the refusal proves nothing"

    # It is kept in the identity, verbatim, as the step was handed it.
    recorded = _manifest(tmp_path / "left")["components"]["data"]["metadata"]
    assert recorded["artifact"]["downloaded_at"] == _artifact(first)["downloaded_at"]
    assert "cache_hit" in recorded["artifact"]

    cell = tmp_path / "cell"
    _run_loaded(cell, _load(csv_path, tmp_path), metadata_feature=True)
    before = _snapshot(cell)
    with pytest.raises(ValueError, match="already holds .* completed origin file"):
        _run_loaded(cell, _load(csv_path, tmp_path), metadata_feature=True)
    assert _snapshot(cell) == before, "a refusal must leave the directory untouched"


def test_custom_model_alone_does_not_widen_the_metadata_identity(
    tmp_path: Path, monkeypatch
) -> None:
    """A custom MODEL is never handed the payload, so it must not change policy.

    True here for two independent reasons, and worth pinning for both. The model
    is not part of ``store_identity.components`` at all -- ``_checkpoint_run_identity``
    adds it separately -- so the detection cannot see it however wide its scan; and
    the detection is in any case restricted to the two surfaces that receive
    ``metadata=``. What this test guards is the observable promise: a resume after
    a reload keeps working when the only user code in the run is an estimator.
    ``test_detection_is_limited_to_the_surfaces_that_receive_metadata`` covers the
    scan width itself, which this one cannot distinguish.
    """

    def _fit_mean(X, y, **params):
        constant = float(np.asarray(y, dtype=float).mean())

        class _Fit:
            def predict(self, X_new):
                return np.full(len(X_new), constant, dtype=float)

        return _Fit()

    model = mf.models.custom_model("mean_only", _fit_mean, mf_digest="test:mean/v1")
    csv_path = _csv_source(tmp_path)
    cell = tmp_path / "cell"
    _run_loaded(cell, _load(csv_path, tmp_path), metadata_feature=False, model=model)

    recorded = _manifest(cell)["components"]["data"]["metadata"]["artifact"]
    assert "downloaded_at" not in recorded and "cache_hit" not in recorded

    computed = _spy_on_origin_fits(monkeypatch)
    _run_loaded(cell, _load(csv_path, tmp_path), metadata_feature=False, model=model)
    assert computed == [], "a custom model must not make a reload unresumable"


def test_provenance_filter_touches_only_the_two_named_fields(tmp_path: Path) -> None:
    """The helper, directly: what it removes, what it keeps, what it leaves alone."""
    payload = {
        "dataset": "custom",
        "artifact": {
            "dataset": "custom",
            "version_mode": "current",
            "vintage": None,
            "source_url": "/tmp/source.csv",
            "local_path": "/tmp/source.csv",
            "file_format": "csv",
            "downloaded_at": "2026-08-12T00:00:00+00:00",
            "file_sha256": "a" * 64,
            "file_size_bytes": 1234,
            "cache_hit": False,
            "manifest_version": "v1",
        },
        "nested": {"downloaded_at": "kept: not the loader artifact block"},
    }
    before = json.loads(json.dumps(payload))

    filtered = runner._identity_metadata_payload(payload, read_by_custom_code=False)
    assert set(payload["artifact"]) - set(filtered["artifact"]) == {
        "downloaded_at",
        "cache_hit",
    }
    assert {key: value for key, value in payload.items() if key != "artifact"} == {
        key: value for key, value in filtered.items() if key != "artifact"
    }, "only the artifact block may be rewritten"
    assert payload == before, "the live panel.attrs payload must not be mutated"
    assert filtered["artifact"] is not payload["artifact"]

    # The custom path is the identity function, object for object.
    assert (
        runner._identity_metadata_payload(payload, read_by_custom_code=True) is payload
    )
    # A payload with no loader artifact is passed straight through.
    plain = {"phase": 0.0}
    assert runner._identity_metadata_payload(plain, read_by_custom_code=False) is plain


@pytest.mark.parametrize(
    ("component", "expected"),
    [
        ({"steps": [{"func": {"__custom__": {"type": "f", "mf_digest": "v1"}}}]}, True),
        ({"steps": [{"func": {"__callable__": "macroforecast.x"}}]}, False),
        ({"steps": [{"func": {"__opaque__": {"type": "f"}}}]}, False),
        ({"__mapping__": [[1, {"__custom__": {}}]]}, True),
        (None, False),
    ],
)
def test_custom_marker_detection_is_recursive_and_marker_specific(
    component, expected
) -> None:
    """Only ``__custom__`` counts: an opaque callable already fails the run closed."""
    assert runner._identity_has_custom_marker(component) is expected


def test_detection_is_limited_to_the_surfaces_that_receive_metadata() -> None:
    """Scan width, asserted where it is decidable: on the components themselves.

    Only ``features`` and ``preprocessing`` steps are called with ``metadata=``.
    Custom code anywhere else in the run-scoped identity -- a future-feature
    policy, a window rule, a stage policy -- never sees the payload, so it must
    not switch the run onto the raw-metadata policy and make every reload
    unresumable.
    """
    custom = {"__custom__": {"type": "m.f", "mf_digest": "v1"}}
    builtin = {"__callable__": "macroforecast.x"}

    def _identity(**components):
        return runner._StoreIdentity(
            components=components, complete=True, opaque_fields=()
        )

    # ``features``/``preprocessing`` are the RAW specs, and are empty here on
    # purpose: this test is about the width of the CANONICAL scan, which section
    # 11 does not change.
    assert not runner._metadata_is_read_by_custom_code(
        _identity(
            features={"steps": [builtin]},
            preprocessing=None,
            window=custom,
            future_feature_policy=custom,
            stage_policies={"model_selection": custom},
            target_transform=custom,
        ),
        features=None,
        preprocessing=None,
    )
    assert runner._metadata_is_read_by_custom_code(
        _identity(features={"steps": [{"func": custom}]}, preprocessing=None),
        features=None,
        preprocessing=None,
    )
    assert runner._metadata_is_read_by_custom_code(
        _identity(features={"steps": [builtin]}, preprocessing={"steps": [custom]}),
        features=None,
        preprocessing=None,
    )


def test_vintage_route_normalises_loader_provenance_too(
    tmp_path: Path, monkeypatch
) -> None:
    """Requirement 4: one policy, applied to every panel the identity records.

    A vintage source backed by real files stamps each snapshot with the same
    ``artifact`` block an ordinary loader does, and the identity records every
    origin's snapshot metadata plus the actuals base panel's. If any of those three
    sites kept the timestamps, rebuilding the spec would refuse -- so this is also
    the test that the policy is threaded rather than re-decided per helper.
    """
    cell = tmp_path / "cell"
    baseline = (
        _run_vintage(cell, provenance=True)
        .to_frame()
        .sort_values("origin_pos")
        .reset_index(drop=True)
    )
    data = _manifest(cell)["components"]["data"]

    # The premise: the stamps really are in the payload these panels carry.
    fresh = _vintage_spec(provenance=True)
    stamped = dict(_vintage_bundle(fresh, 1).metadata)["artifact"]
    assert "downloaded_at" in stamped and "cache_hit" in stamped

    # ... and really are absent from all three recorded sites, content kept.
    for recorded in [row[3]["artifact"] for row in data["origin_inputs"]["rows"]] + [
        data["actuals"]["base_panel_metadata"]["artifact"]
    ]:
        assert "downloaded_at" not in recorded and "cache_hit" not in recorded
        assert len(recorded["file_sha256"]) == 64

    computed = _spy_on_origin_fits(monkeypatch)
    resumed = (
        _run_vintage(cell, provenance=True)
        .to_frame()
        .sort_values("origin_pos")
        .reset_index(drop=True)
    )
    assert computed == [], "rebuilding the same vintage source must not refit"
    np.testing.assert_allclose(
        resumed["prediction"].to_numpy(dtype=float),
        baseline["prediction"].to_numpy(dtype=float),
        rtol=0,
        atol=0,
    )


def test_vintage_route_keeps_provenance_when_a_custom_step_reads_it(
    tmp_path: Path,
) -> None:
    """And the exception applies there as well, at the same three sites."""
    cell = tmp_path / "cell"
    _run_vintage(cell, provenance=True, metadata_feature=True)
    data = _manifest(cell)["components"]["data"]
    for recorded in [row[3]["artifact"] for row in data["origin_inputs"]["rows"]] + [
        data["actuals"]["base_panel_metadata"]["artifact"]
    ]:
        assert "downloaded_at" in recorded and "cache_hit" in recorded

    before = _snapshot(cell)
    with pytest.raises(ValueError, match="already holds .* completed origin file"):
        _run_vintage(cell, provenance=True, metadata_feature=True)
    assert _snapshot(cell) == before, "a refusal must leave the directory untouched"


# --------------------------------------------------------------------------- #
# 11. A callable dataclass is user code even though it canonicalises as a record
# --------------------------------------------------------------------------- #
# Section 10 reads the provenance policy off the CANONICALIZED components, and
# that view has a blind spot. ``_identity_ready`` tests ``is_dataclass`` before it
# looks for ``__mf_digest__``, so a ``@dataclass`` with ``__call__`` -- an
# ordinary way to write a parameterised step, and accepted by both
# ``custom_step`` and ``custom_preprocess_step`` -- renders as
# ``{"__dataclass__": ...}`` and carries no ``__custom__`` marker. It is
# COMPLETE, so no other rule refuses on its behalf; it is called with
# ``metadata=``, so it can read ``artifact.downloaded_at``; and classified as
# built-in it would have that field normalised out from under it. The result is
# the exact failure this file exists to prevent: the forecasts move and the
# digest does not.
#
# The fix scans the RAW ``features`` and ``preprocessing`` objects as well and
# OR-s the two answers. What follows pins the observable consequence on both
# public routes that receive the payload, and the detector's own edges.
@_dc.dataclass(frozen=True)
class _StampedFeatureStep:
    """A callable dataclass feature step whose output depends on ``downloaded_at``.

    Deliberately WITHOUT ``__mf_digest__``: that marker is what would make the
    canonical scan see it, and the run these tests are about is the one that does
    not carry one. Its fields are still inspectable, so the identity stays
    complete and any refusal below comes from the metadata rather than from an
    unrepresentable object.
    """

    scale: float = 1.0

    def __call__(self, source, *, metadata=None, **params):
        payload = dict(metadata or {})
        _OBSERVED_METADATA.append(payload)
        stamp = str(dict(payload.get("artifact", {})).get("downloaded_at", ""))
        phase = float(int(hashlib.sha256(stamp.encode("utf-8")).hexdigest()[:8], 16) % 997)
        months = (source.index.year - 2000) * 12.0 + source.index.month
        return pd.Series(
            self.scale * np.sin((months + phase) / 3.0),
            index=source.index,
            name="stamped",
        )


@_dc.dataclass(frozen=True)
class _StampedPreprocessStep:
    """The same idea on the other public surface that is handed ``metadata=``.

    Row-local, so it is legal under every stage policy, and NON-affine, so the
    rescaling it applies to a predictor cannot be absorbed by a ridge intercept:
    two loads therefore really do produce different forecasts.
    """

    column: str = "x1"

    def __call__(self, panel, *, metadata=None, **params):
        payload = dict(metadata or {})
        _OBSERVED_METADATA.append(payload)
        stamp = str(dict(payload.get("artifact", {})).get("downloaded_at", ""))
        phase = float(int(hashlib.sha256(stamp.encode("utf-8")).hexdigest()[:8], 16) % 997)
        stamped = panel.copy()
        stamped[self.column] = np.sin(stamped[self.column] * (1.0 + phase))
        return stamped


def _dataclass_feature_spec() -> mf.feature_engineering.FeatureSpec:
    return mf.feature_engineering.feature_spec(
        target="y",
        target_lags=[1, 2],
        steps=[
            mf.feature_engineering.custom_step(
                "stamped", _StampedFeatureStep(), columns=["x1"]
            )
        ],
    )


def _dataclass_preprocess_spec() -> mf.preprocessing.PreprocessSpec:
    """Every built-in stage off, so the dataclass step is the only thing acting."""
    return mf.preprocessing.preprocess_spec(
        transform="none",
        outliers="none",
        impute="none",
        standardize="none",
        frame="keep",
        custom_steps=[
            mf.preprocessing.custom_preprocess_step(
                "stamped", _StampedPreprocessStep(), row_local=True
            )
        ],
    )


def _predictions(result) -> np.ndarray:
    return (
        result.to_frame()
        .sort_values("origin_pos")
        .reset_index(drop=True)["prediction"]
        .to_numpy(dtype=float)
    )


def test_callable_dataclass_feature_step_keeps_provenance_in_the_identity(
    tmp_path: Path,
) -> None:
    """The gap, on the feature route: the output moves, so the resume must refuse.

    Two fresh runs over two loads of one unchanged file are shown to disagree
    first. Only then is the refusal meaningful -- resuming the first cell with the
    second bundle would return rows this configuration does not compute.
    """
    csv_path = _csv_source(tmp_path)
    first, second = _load(csv_path, tmp_path), _load(csv_path, tmp_path)
    pd.testing.assert_frame_equal(first.panel, second.panel)

    _OBSERVED_METADATA.clear()
    cell = tmp_path / "cell"
    left = _predictions(_run(cell, panel=first.panel, features=_dataclass_feature_spec()))
    right = _predictions(
        _run(tmp_path / "right", panel=second.panel, features=_dataclass_feature_spec())
    )
    assert _OBSERVED_METADATA, "the callable dataclass must have been handed metadata"
    assert not np.allclose(left, right, rtol=0, atol=0), (
        "the timestamp must be able to move a forecast, or the refusal proves nothing"
    )

    # Complete, unmarked, and identified with the provenance the step can read.
    manifest = _manifest(cell)
    assert manifest["complete"] is True, "a dataclass step is inspectable, not opaque"
    recorded = manifest["components"]["data"]["metadata"]["artifact"]
    assert recorded["downloaded_at"] == _artifact(first)["downloaded_at"]
    assert "cache_hit" in recorded

    before = _snapshot(cell)
    with pytest.raises(ValueError, match="already holds .* completed origin file"):
        _run(cell, panel=second.panel, features=_dataclass_feature_spec())
    assert _snapshot(cell) == before, "a refusal must leave the directory untouched"


def test_callable_dataclass_preprocess_step_keeps_provenance_in_the_identity(
    tmp_path: Path,
) -> None:
    """And on the preprocessing route, which is reached by a different spec.

    ``custom_preprocess_step`` is a separate public entry point with its own
    invocation path (``_invoke_custom_step``), so the feature-route test above
    does not cover it: the detector has to find caller code inside a
    ``PreprocessSpec`` too.
    """
    csv_path = _csv_source(tmp_path)
    first, second = _load(csv_path, tmp_path), _load(csv_path, tmp_path)
    pd.testing.assert_frame_equal(first.panel, second.panel)

    _OBSERVED_METADATA.clear()
    cell = tmp_path / "cell"
    left = _predictions(
        _run(cell, panel=first.panel, preprocessing=_dataclass_preprocess_spec())
    )
    right = _predictions(
        _run(
            tmp_path / "right",
            panel=second.panel,
            preprocessing=_dataclass_preprocess_spec(),
        )
    )
    assert _OBSERVED_METADATA, "the callable dataclass must have been handed metadata"
    assert not np.allclose(left, right, rtol=0, atol=0), (
        "the timestamp must be able to move a forecast, or the refusal proves nothing"
    )

    recorded = _manifest(cell)["components"]["data"]["metadata"]["artifact"]
    assert recorded["downloaded_at"] == _artifact(first)["downloaded_at"]
    assert "cache_hit" in recorded

    before = _snapshot(cell)
    with pytest.raises(ValueError, match="already holds .* completed origin file"):
        _run(cell, panel=second.panel, preprocessing=_dataclass_preprocess_spec())
    assert _snapshot(cell) == before, "a refusal must leave the directory untouched"


def test_a_callable_dataclass_is_invisible_to_the_canonical_scan_alone() -> None:
    """Why the raw scan exists, stated on the two functions themselves."""
    spec = _dataclass_feature_spec()
    identity = runner._new_store_identity(features=spec, preprocessing=None)

    assert identity.complete, "nothing else refuses for us: the dataclass is readable"
    assert not runner._identity_has_custom_marker(identity.components["features"]), (
        "canonicalisation renders it as __dataclass__, carrying no __custom__ marker"
    )
    assert runner._metadata_is_read_by_custom_code(
        identity, features=spec, preprocessing=None
    ), "the raw-spec scan is the half that closes the gap"


def test_raw_spec_detector_ignores_package_specs_and_cannot_spin_on_a_cycle() -> None:
    """Its two edges: no false positive on built-ins, no unbounded walk.

    A false positive here is not silent -- it would make every ordinary reload
    unresumable, which is the defect section 10 fixes -- so the built-in specs
    are asserted directly rather than only through a run.
    """
    fe = mf.feature_engineering
    builtin_features = fe.feature_spec(
        target="y",
        target_lags=[1, 2],
        steps=[
            fe.lag_step(columns=["x1"], lags=[1]),
            fe.pca_step(columns=["x1", "x2"], n_components=1),
            fe.transform_step(columns=["x2"], transform="log"),
            fe.time_step(),
            fe.season_dummy_step(),
        ],
    )
    builtin_preprocessing = mf.preprocessing.preprocess_spec(
        transform="tcode",
        outliers="iqr",
        impute="em_factor",
        standardize="zscore",
        frame="keep",
    )
    assert not runner._spec_contains_custom_callable(builtin_features)
    assert not runner._spec_contains_custom_callable(builtin_preprocessing)
    assert not runner._spec_contains_custom_callable(None)
    assert not runner._spec_contains_custom_callable({"steps": [{"func": fe.lag_step}]}), (
        "a package-owned callable is not caller code however deeply it is nested"
    )

    # Ownership, not shape: found wherever a container can hold it.
    assert runner._spec_contains_custom_callable(_StampedFeatureStep())
    assert runner._spec_contains_custom_callable(_dataclass_feature_spec())
    assert runner._spec_contains_custom_callable(_dataclass_preprocess_spec())
    assert runner._spec_contains_custom_callable({"steps": [(_StampedFeatureStep(),)]})
    assert runner._spec_contains_custom_callable(
        {"steps": frozenset([_StampedFeatureStep()])}
    )

    # A spec that refers to itself terminates, and still answers correctly.
    cyclic: list = [{"steps": None}]
    cyclic[0]["steps"] = cyclic
    assert not runner._spec_contains_custom_callable(cyclic)
    cyclic_custom: list = [_StampedFeatureStep()]
    cyclic_custom.append(cyclic_custom)
    assert runner._spec_contains_custom_callable(cyclic_custom)


def test_vintage_route_keeps_provenance_for_a_callable_dataclass_step(
    tmp_path: Path,
) -> None:
    """The same classification, threaded into the OTHER gate.

    ``_run_vintage_aware`` builds its identity from three panels through its own
    call to the policy decision, so the raw specs have to reach that call too. A
    dataclass step must keep provenance at all three recorded sites, exactly as
    the marked function of section 10 does.
    """
    cell = tmp_path / "cell"
    _run_vintage(cell, provenance=True, metadata_step=_StampedFeatureStep())
    data = _manifest(cell)["components"]["data"]
    for recorded in [row[3]["artifact"] for row in data["origin_inputs"]["rows"]] + [
        data["actuals"]["base_panel_metadata"]["artifact"]
    ]:
        assert "downloaded_at" in recorded and "cache_hit" in recorded

    before = _snapshot(cell)
    with pytest.raises(ValueError, match="already holds .* completed origin file"):
        _run_vintage(cell, provenance=True, metadata_step=_StampedFeatureStep())
    assert _snapshot(cell) == before, "a refusal must leave the directory untouched"
