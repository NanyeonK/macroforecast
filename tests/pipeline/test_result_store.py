from __future__ import annotations

import dataclasses as _dc
import json

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
import macroforecast.pipeline.result_store as result_store_mod
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline
from macroforecast.pipeline.result_store import result_cell_identity
from macroforecast.pipeline.run import _data_identity


FIT_COUNTS: dict[str, int] = {}


class _ConstantFit:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def _recording_fit(X, y, *, offset: float = 0.0):
    FIT_COUNTS["recording"] = FIT_COUNTS.get("recording", 0) + 1
    return _ConstantFit(float(np.nanmean(np.asarray(y, dtype=float))) + float(offset))


def _custom_fit(X, y):
    FIT_COUNTS["custom"] = FIT_COUNTS.get("custom", 0) + 1
    return _ConstantFit(float(np.nanmean(np.asarray(y, dtype=float))))


def _bundle(n: int = 72, *, bump: float = 0.0):
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, n)
    frame = pd.DataFrame(
        {
            "y": 1.0 + 2.0 * x + rng.standard_normal(n) * 0.05,
            "x1": x,
        },
        index=idx,
    )
    frame.iloc[-1, frame.columns.get_loc("x1")] += bump
    return mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})


def _window():
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=30),
        val=mf.window.val_last_block(size=10),
        test=mf.window.test_origins(horizon=1, step=8),
    )


def _features(*, lags=(1,)):
    return mf.feature_engineering.feature_spec(
        target="y",
        predictors=["x1"],
        lags=lags,
        target_lags=(0, 1),
    )


def _recording_model():
    _recording_fit.__mf_digest__ = "recording-fit-v1"
    return mf.models.custom_model("recording_mean", _recording_fit)


def _spec(
    tmp_path,
    *,
    arms=None,
    features=None,
    preprocessing=None,
    preprocessing_policy=None,
    data=None,
):
    feats = _features() if features is None else features
    return pipeline_spec(
        data=_bundle() if data is None else data,
        targets=["y"],
        horizons=[1],
        window=_window(),
        arms=arms
        if arms is not None
        else [
            Arm("A", model=_recording_model(), features=feats, params={"offset": 0.0}),
            Arm("B", model=_recording_model(), features=feats, params={"offset": 0.1}),
        ],
        evaluation=EvalSpec(benchmark="A", metrics=("rmse", "relative_mse")),
        save_models=False,
        preprocessing=preprocessing,
        preprocessing_policy=preprocessing_policy,
        result_store=tmp_path / "results",
    )


def _frame_sort(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["target", "horizon", "contender", "origin", "date"] if c in frame.columns]
    return frame.sort_values(cols).reset_index(drop=True)


def _assert_report_frames_equal(left, right) -> None:
    for name in ("forecasts", "accuracy", "significance", "mcs", "density", "calibration"):
        a = getattr(left, name)
        b = getattr(right, name)
        if isinstance(a, pd.DataFrame):
            pd.testing.assert_frame_equal(_frame_sort(a), _frame_sort(b))


def test_result_store_round_trip_reuses_cells_and_preserves_report_frames(tmp_path):
    FIT_COUNTS.clear()
    first = run_pipeline(_spec(tmp_path))
    assert first.provenance["result_store"]["n_computed"] == 2
    assert FIT_COUNTS["recording"] > 0

    FIT_COUNTS.clear()
    second = run_pipeline(_spec(tmp_path))

    assert second.provenance["result_store"]["n_reused"] == 2
    assert second.provenance["result_store"]["n_computed"] == 0
    assert FIT_COUNTS.get("recording", 0) == 0
    _assert_report_frames_equal(first, second)


def test_result_store_incremental_horse_race_reuses_existing_arms(tmp_path):
    first = run_pipeline(_spec(tmp_path))
    third = Arm("C", model=_recording_model(), features=_features(), params={"offset": 0.2})
    second = run_pipeline(_spec(tmp_path, arms=[*_spec(tmp_path).arms, third]))

    assert second.provenance["result_store"]["n_reused"] == 2
    assert second.provenance["result_store"]["n_computed"] == 1
    assert set(second.accuracy["contender"]) == {"A", "B", "C"}
    for contender in ("A", "B"):
        a = first.accuracy[first.accuracy["contender"] == contender].reset_index(drop=True)
        b = second.accuracy[second.accuracy["contender"] == contender].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)


def test_result_store_reuses_cells_when_only_arm_tags_change(tmp_path):
    FIT_COUNTS.clear()
    base = _spec(tmp_path)
    first = run_pipeline(base)
    assert first.provenance["result_store"]["n_computed"] == 2

    tagged_arms = [
        _dc.replace(arm, tags={"axis": idx, "tagged": True})
        for idx, arm in enumerate(base.arms)
    ]
    tagged = _spec(tmp_path, arms=tagged_arms)

    base_identity = result_cell_identity(
        base,
        base.arms[0],
        base.targets[0],
        horizon=1,
        data_identity=_data_identity(base.data),
    )
    tagged_identity = result_cell_identity(
        tagged,
        tagged.arms[0],
        tagged.targets[0],
        horizon=1,
        data_identity=_data_identity(tagged.data),
    )
    assert tagged_identity.digest == base_identity.digest

    FIT_COUNTS.clear()
    second = run_pipeline(tagged)

    assert second.provenance["result_store"]["n_reused"] == 2
    assert second.provenance["result_store"]["n_computed"] == 0
    assert FIT_COUNTS.get("recording", 0) == 0
    assert {"tag_axis", "tag_tagged"} <= set(second.forecasts.columns)
    assert set(second.forecasts.loc[second.forecasts["arm"] == "A", "tag_axis"]) == {0}
    assert set(second.forecasts.loc[second.forecasts["arm"] == "B", "tag_axis"]) == {1}


def test_result_store_digest_sensitivity_oracle(tmp_path):
    base = _spec(tmp_path)
    identity = result_cell_identity(
        base,
        base.arms[0],
        base.targets[0],
        horizon=1,
        data_identity=_data_identity(base.data),
    )

    changed_param = _spec(
        tmp_path,
        arms=[Arm("A", model=_recording_model(), features=_features(), params={"offset": 9.0})],
    )
    changed_features = _spec(tmp_path, features=_features(lags=(1, 2)))
    changed_prep = _spec(
        tmp_path,
        preprocessing=mf.preprocessing.preprocess_spec(standardize="zscore"),
    )
    changed_data = _spec(tmp_path, data=_bundle(bump=1.0))

    variants = [changed_param, changed_features, changed_prep, changed_data]
    digests = [
        result_cell_identity(
            spec,
            spec.arms[0],
            spec.targets[0],
            horizon=1,
            data_identity=_data_identity(spec.data),
        ).digest
        for spec in variants
    ]
    assert all(digest != identity.digest for digest in digests)

    class Source:
        kind = "toy_vintage"

        def resolve(self, origin_date):
            return _bundle()

        def available_vintages(self):
            return ("2000-01",)

    vintage_spec = pipeline_spec(
        data=mf.data.VintagePanelSpec(
            Source(),
            pd.date_range("2000-01-31", periods=3, freq="ME"),
        ),
        targets=[TargetSpec("y", transform="level", policy="direct")],
        horizons=[1],
        window=_window(),
        arms=[base.arms[0]],
        evaluation=EvalSpec(benchmark="A", metrics=("rmse",)),
        save_models=False,
        result_store=tmp_path / "results",
    )
    vintage_identity = result_cell_identity(
        vintage_spec,
        vintage_spec.arms[0],
        vintage_spec.targets[0],
        horizon=1,
        data_identity=_data_identity(base.data),
    )
    assert vintage_identity.digest != identity.digest


def test_result_store_vintage_content_refresh_recomputes(tmp_path):
    class MutableVintageSource:
        kind = "mutable_vintage"

        def __init__(self) -> None:
            self.bump = 0.0

        def available_vintages(self):
            return ("2006-01-31",)

        def resolve(self, origin_date):
            bundle = _bundle(bump=self.bump)
            panel = bundle.panel.loc[bundle.panel.index < pd.Timestamp(origin_date)]
            metadata = {**bundle.metadata, "vintage": "2006-01-31"}
            return mf.data.DataBundle(panel, metadata)

    source = MutableVintageSource()
    spec = pipeline_spec(
        data=mf.data.VintagePanelSpec(
            source,
            pd.date_range("2000-01-31", periods=72, freq="ME"),
        ),
        targets=[TargetSpec("y", transform="level", policy="direct")],
        horizons=[1],
        window=_window(),
        arms=[Arm("A", model=_recording_model(), features=_features())],
        evaluation=EvalSpec(benchmark="A", metrics=("rmse",)),
        save_models=False,
        result_store=tmp_path / "results",
    )

    first = run_pipeline(spec)
    unchanged = run_pipeline(spec)
    assert first.provenance["result_store"]["n_computed"] == 1
    assert unchanged.provenance["result_store"]["n_reused"] == 1

    source.bump = 5.0
    refreshed = run_pipeline(spec)
    assert refreshed.provenance["result_store"]["n_reused"] == 0
    assert refreshed.provenance["result_store"]["n_computed"] == 1


def test_result_store_callable_vintage_source_requires_digest_opt_in(tmp_path):
    class CallableOnlyVintageSource:
        kind = "callable_only"

        def available_vintages(self):
            return ()

        def resolve(self, origin_date):
            bundle = _bundle()
            metadata = {**bundle.metadata, "vintage": "live"}
            return mf.data.DataBundle(bundle.panel, metadata)

    source = CallableOnlyVintageSource()
    spec = pipeline_spec(
        data=mf.data.VintagePanelSpec(
            source,
            pd.date_range("2000-01-31", periods=72, freq="ME"),
        ),
        targets=[TargetSpec("y", transform="level", policy="direct")],
        horizons=[1],
        window=_window(),
        arms=[Arm("A", model=_recording_model(), features=_features())],
        evaluation=EvalSpec(benchmark="A", metrics=("rmse",)),
        save_models=False,
        result_store=tmp_path / "results",
    )

    identity = result_cell_identity(
        spec,
        spec.arms[0],
        spec.targets[0],
        horizon=1,
        data_identity=_data_identity(spec.data),
    )
    assert identity.digest is None
    assert "available_vintages" in str(identity.reason)

    source.__mf_digest__ = "callable-source-v1"
    digestible = result_cell_identity(
        spec,
        spec.arms[0],
        spec.targets[0],
        horizon=1,
        data_identity=_data_identity(spec.data),
    )
    assert digestible.digest is not None


def test_result_store_digest_includes_backend_version_and_effective_seed(monkeypatch, tmp_path):
    base = _spec(tmp_path, arms=[Arm("A", model="ridge", features=_features())])

    versions = {"scikit-learn": "1.0"}
    real_version = result_store_mod._metadata.version

    def _fake_version(package: str) -> str:
        return versions.get(package, real_version(package))

    monkeypatch.setattr(result_store_mod._metadata, "version", _fake_version)
    with mf.meta.use_config(random_seed=11):
        first = result_cell_identity(
            base,
            base.arms[0],
            base.targets[0],
            horizon=1,
            data_identity=_data_identity(base.data),
        )
    versions["scikit-learn"] = "2.0"
    with mf.meta.use_config(random_seed=11):
        changed_backend = result_cell_identity(
            base,
            base.arms[0],
            base.targets[0],
            horizon=1,
            data_identity=_data_identity(base.data),
        )
    versions["scikit-learn"] = "1.0"
    with mf.meta.use_config(random_seed=99):
        changed_seed = result_cell_identity(
            base,
            base.arms[0],
            base.targets[0],
            horizon=1,
            data_identity=_data_identity(base.data),
        )

    assert changed_backend.digest != first.digest
    assert changed_seed.digest != first.digest
    assert first.cell_echo["backend_versions"]["packages"]["scikit-learn"] == "1.0"
    assert first.cell_echo["effective_selection_seed"] == 11


def test_result_store_custom_callable_requires_digest_opt_in(tmp_path):
    if hasattr(_custom_fit, "__mf_digest__"):
        delattr(_custom_fit, "__mf_digest__")
    custom = mf.models.custom_model("custom_mean", _custom_fit)
    arms = [Arm("A", model=custom, features=_features())]

    FIT_COUNTS.clear()
    first = run_pipeline(_spec(tmp_path, arms=arms))
    second = run_pipeline(_spec(tmp_path, arms=arms))
    assert first.provenance["result_store"]["n_undigestible"] == 1
    assert second.provenance["result_store"]["n_undigestible"] == 1
    assert first.provenance["result_store"]["n_reused"] == 0
    assert second.provenance["result_store"]["n_reused"] == 0
    assert FIT_COUNTS.get("custom", 0) > 0
    assert not list((tmp_path / "results" / "cells").glob("*.json"))

    _custom_fit.__mf_digest__ = "custom-v1"
    FIT_COUNTS.clear()
    run_pipeline(_spec(tmp_path, arms=arms))
    reused = run_pipeline(_spec(tmp_path, arms=arms))
    assert reused.provenance["result_store"]["n_reused"] == 1
    assert FIT_COUNTS.get("custom", 0) > 0

    _custom_fit.__mf_digest__ = "custom-v2"
    missed = run_pipeline(_spec(tmp_path, arms=arms))
    assert missed.provenance["result_store"]["n_computed"] == 1


def test_result_store_digest_tracks_validation_splitter_boundaries(tmp_path):
    model = _recording_model()
    first_search = mf.model_selection.grid(
        {"offset": [0.0]},
        validation_splitter=mf.model_selection.explicit_folds([20, 30, 40]),
    )
    changed_search = mf.model_selection.grid(
        {"offset": [0.0]},
        validation_splitter=mf.model_selection.explicit_folds([21, 30, 40]),
    )
    first = _spec(
        tmp_path,
        arms=[Arm("A", model=model, features=_features(), model_selection=first_search)],
    )
    changed = _spec(
        tmp_path,
        arms=[Arm("A", model=model, features=_features(), model_selection=changed_search)],
    )

    first_identity = result_cell_identity(
        first,
        first.arms[0],
        first.targets[0],
        horizon=1,
        data_identity=_data_identity(first.data),
    )
    changed_identity = result_cell_identity(
        changed,
        changed.arms[0],
        changed.targets[0],
        horizon=1,
        data_identity=_data_identity(changed.data),
    )

    assert first_identity.digest is not None
    assert changed_identity.digest is not None
    assert first_identity.digest != changed_identity.digest


def test_result_store_callable_validation_splitter_requires_digest(tmp_path):
    def splitter(index):
        midpoint = len(index) // 2
        return [(np.arange(midpoint), np.arange(midpoint, len(index)))]

    search = mf.model_selection.grid(
        {"offset": [0.0]},
        validation_splitter=splitter,
    )
    spec = _spec(
        tmp_path,
        arms=[Arm("A", model=_recording_model(), features=_features(), model_selection=search)],
    )

    identity = result_cell_identity(
        spec,
        spec.arms[0],
        spec.targets[0],
        horizon=1,
        data_identity=_data_identity(spec.data),
    )
    assert identity.digest is None
    assert "validation_splitter" in str(identity.reason)

    splitter.__mf_digest__ = "splitter-v1"
    digestible = result_cell_identity(
        spec,
        spec.arms[0],
        spec.targets[0],
        horizon=1,
        data_identity=_data_identity(spec.data),
    )
    assert digestible.digest is not None
    selection_echo = digestible.cell_echo["arm"]["model_selection"]
    assert selection_echo["validation_splitter"]["mf_digest"] == "splitter-v1"


def test_result_store_version_mismatch_warns_once_and_reuses(tmp_path):
    run_pipeline(_spec(tmp_path, arms=[Arm("A", model=_recording_model(), features=_features())]))
    manifest_path = next((tmp_path / "results" / "cells").glob("*.json"))
    manifest = json.loads(manifest_path.read_text())
    manifest["macroforecast_version"] = "0.0-old"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.warns(UserWarning, match="different macroforecast version") as caught:
        report = run_pipeline(
            _spec(tmp_path, arms=[Arm("A", model=_recording_model(), features=_features())])
        )

    assert len(caught) == 1
    assert report.provenance["result_store"]["n_reused"] == 1
    assert report.provenance["result_store"]["version_mismatches"][0]["store_version"] == "0.0-old"


def test_result_store_corrupt_manifest_is_miss_not_crash(tmp_path):
    run_pipeline(_spec(tmp_path, arms=[Arm("A", model=_recording_model(), features=_features())]))
    manifest_path = next((tmp_path / "results" / "cells").glob("*.json"))
    manifest_path.write_text("{")

    report = run_pipeline(_spec(tmp_path, arms=[Arm("A", model=_recording_model(), features=_features())]))

    assert report.provenance["result_store"]["n_reused"] == 0
    assert report.provenance["result_store"]["n_computed"] == 1


def test_result_store_summary_and_purge(tmp_path):
    report = run_pipeline(_spec(tmp_path))
    store = tmp_path / "results"

    summary = mf.pipeline.result_store_summary(store)
    assert set(summary["arm"]) == {"A", "B"}
    assert set(summary["horizon"]) == {1}
    assert set(summary["n_rows"]) == {len(report.forecasts) // 2}

    digest = str(summary.iloc[0]["digest"])
    assert mf.pipeline.purge_result_store(store, digests=[digest]) == 1
    assert digest not in set(mf.pipeline.result_store_summary(store)["digest"])


def _undigestible_mean_fit(X, y):
    # Deliberately NO ``__mf_digest__`` -> the cell is undigestible and is therefore
    # NEVER written to the store (only digestible cells attempt a write).
    return _ConstantFit(float(np.nanmean(np.asarray(y, dtype=float))))


def _write_failure_spec(store):
    feats = _features()
    return pipeline_spec(
        data=_bundle(),
        targets=["y"],
        horizons=[1],
        window=_window(),
        arms=[
            Arm("A", model=_recording_model(), features=feats, is_benchmark=True),
            Arm("B", model=_recording_model(), features=feats, params={"offset": 0.1}),
            Arm(
                "U",
                model=mf.models.custom_model("u_mean", _undigestible_mean_fit),
                features=feats,
            ),
        ],
        evaluation=EvalSpec(benchmark="A", metrics=("rmse", "relative_mse", "r2_oos")),
        save_models=False,
        result_store=store,
    )


def test_result_store_write_failure_does_not_drop_cells(tmp_path):
    """A store WRITE failure must not drop the correctly-computed cells.

    Regression for the ZWW replication bug: when the store's ``cells/`` directory
    was unwritable (disk-full / ENOSPC / read-only / quota / permissions), every
    DIGESTIBLE cell -- including the benchmark -- was discarded, because the write
    failure was caught as a per-cell COMPUTE failure. Only the UNDIGESTIBLE arms
    (never written) survived, with ``benchmark_present=False`` and NaN metrics, and
    the ``cells/`` dir was empty. The report must instead be identical to a no-store
    run; only cache persistence is skipped, and the failure is surfaced.
    """
    import os
    import stat
    import warnings as _warnings

    baseline = run_pipeline(_dc.replace(_write_failure_spec(tmp_path / "unused"), result_store=None))

    store = tmp_path / "ro_store"
    cells = store / "cells"
    cells.mkdir(parents=True)
    os.chmod(cells, stat.S_IRUSR | stat.S_IXUSR)  # r-x------ : file writes fail
    try:
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            report = run_pipeline(_write_failure_spec(store))
    finally:
        os.chmod(cells, stat.S_IRWXU)

    # (1) No cell is dropped; the report equals the no-store baseline.
    assert set(report.accuracy["contender"]) == {"A", "B", "U"}
    assert bool(report.accuracy["benchmark_present"].all())
    assert not report.failed_cells
    pd.testing.assert_frame_equal(_frame_sort(baseline.accuracy), _frame_sort(report.accuracy))

    # (2) The write failure is surfaced (metadata + warning), not silent.
    meta = report.provenance["result_store"]
    failed_write_arms = {rec["arm"] for rec in meta.get("write_failures", [])}
    assert failed_write_arms == {"A", "B"}   # both digestible arms attempted a write
    assert "U" not in failed_write_arms      # the undigestible arm is never written
    assert list(cells.iterdir()) == []       # nothing was persisted
    assert any("could not persist" in str(w.message) for w in caught)


def test_result_store_write_success_still_persists_and_is_bit_identical(tmp_path):
    """Golden path: with a WRITABLE store the report is identical to the no-store run
    and every digestible cell is persisted (no write_failures recorded)."""
    baseline = run_pipeline(_dc.replace(_write_failure_spec(tmp_path / "unused"), result_store=None))
    report = run_pipeline(_write_failure_spec(tmp_path / "ok_store"))

    pd.testing.assert_frame_equal(_frame_sort(baseline.accuracy), _frame_sort(report.accuracy))
    assert not report.failed_cells
    assert report.provenance["result_store"].get("write_failures", []) == []
    assert report.provenance["result_store"]["n_computed"] == 3
    persisted = sorted((tmp_path / "ok_store" / "cells").glob("*.parquet"))
    assert len(persisted) == 2  # A and B are digestible; U is not


# --------------------------------------------------------------------------- #
# Effective stage policies are part of cell identity (F-017)
# --------------------------------------------------------------------------- #
# A stage policy left unset is not "no policy": ``forecasting/runner.py`` resolves it
# against a package default, so the same spec fits differently after
# ``mf.meta.configure(default_preprocessing_scope=...)``. ``origin_available`` and
# ``full_panel`` are the leak-aware and the whole-panel fit, which is why a digest
# that cannot tell them apart can serve one run's forecasts to the other.


def _digest_under(spec, **config):
    with mf.meta.use_config(**config):
        return result_cell_identity(
            spec,
            spec.arms[0],
            spec.targets[0],
            horizon=1,
            data_identity=_data_identity(spec.data),
        ).digest


def _preprocessed_spec(tmp_path, **over):
    return _spec(
        tmp_path,
        arms=[Arm("A", model=_recording_model(), features=_features())],
        preprocessing=mf.preprocess_spec(standardize=True),
        **over,
    )


@pytest.mark.parametrize(
    ("option", "left", "right"),
    [
        ("default_preprocessing_scope", "origin_available", "full_panel"),
        ("default_feature_scope", "fit_window", "full_panel"),
        ("default_selection_scope", "fit_window", "full_panel"),
    ],
)
def test_result_store_digest_tracks_every_effective_stage_default(
    tmp_path, option, left, right
):
    """Each of the three defaults decides a real fit, so each must move the digest."""
    spec = _preprocessed_spec(tmp_path)

    assert _digest_under(spec, **{option: left}) != _digest_under(spec, **{option: right})


def test_result_store_digest_ignores_a_default_the_spec_overrides(tmp_path):
    """An explicit policy wins, so its default no longer describes anything.

    The complement of the test above: identity must follow the RESOLVED policy, which
    means an explicit one pins the digest and leaves it unmoved by a default that can
    no longer reach this stage. Without this, hardening identity would invalidate a
    store on every unrelated config change.
    """
    spec = _preprocessed_spec(
        tmp_path, preprocessing_policy=mf.window.stage_policy(scope="origin_available")
    )

    assert _digest_under(spec, default_preprocessing_scope="origin_available") == _digest_under(
        spec, default_preprocessing_scope="full_panel"
    )


def test_result_store_digest_records_the_arms_own_effective_preprocessing_policy(tmp_path):
    """An arm that overrides preprocessing owns its policy; the spec's must not leak."""
    spec = _spec(
        tmp_path,
        arms=[
            Arm(
                "A",
                model=_recording_model(),
                features=_features(),
                preprocessing=mf.preprocess_spec(standardize=True),
                preprocessing_policy="origin_available",
            )
        ],
        preprocessing=mf.preprocess_spec(standardize=False),
        preprocessing_policy="full_panel",
    )

    identity = result_cell_identity(
        spec,
        spec.arms[0],
        spec.targets[0],
        horizon=1,
        data_identity=_data_identity(spec.data),
    )
    policies = identity.cell_echo["arm"]["stage_policies"]
    assert policies["preprocessing"]["scope"] == "origin_available"
    assert set(policies) == {"preprocessing", "feature_engineering", "model_selection"}


def test_result_store_digest_says_no_policy_when_an_arm_has_no_preprocessing(tmp_path):
    """No stage means no timing rule -- not a fabricated scope."""
    spec = _spec(tmp_path, arms=[Arm("A", model=_recording_model(), features=_features())])

    identity = result_cell_identity(
        spec,
        spec.arms[0],
        spec.targets[0],
        horizon=1,
        data_identity=_data_identity(spec.data),
    )
    policies = identity.cell_echo["arm"]["stage_policies"]
    assert policies["preprocessing"] is None
    assert policies["feature_engineering"]["scope"]
    assert policies["model_selection"]["scope"]


def test_result_store_stage_policies_agree_between_spec_only_and_task_identity(tmp_path):
    """Both identity spellings must resolve the same policies.

    ``tests/pipeline/test_shared_cell_task.py`` pins that the two digests are equal in
    general; this states the same for the newly added block, so a future change that
    resolved policies from the task could not silently give the two paths different
    answers.
    """
    from macroforecast.forecasting.task import resolve_forecast_tasks

    spec = _preprocessed_spec(tmp_path)
    arm, target = spec.arms[0], spec.targets[0]
    task = resolve_forecast_tasks(spec, arm, target, [1])[0]
    data_identity = _data_identity(spec.data)

    without_task = result_cell_identity(
        spec, arm, target, horizon=1, data_identity=data_identity
    )
    with_task = result_cell_identity(
        spec, arm, target, horizon=1, data_identity=data_identity, task=task
    )

    assert without_task.digest == with_task.digest
    assert (
        without_task.cell_echo["arm"]["stage_policies"]
        == with_task.cell_echo["arm"]["stage_policies"]
    )


# --------------------------------------------------------------------------- #
# A panel fingerprint failure disables reuse (F-023)
# --------------------------------------------------------------------------- #


def _break_fingerprint(monkeypatch):
    def _raise(frame):
        raise TypeError("synthetic fingerprint failure")

    import macroforecast.pipeline.run as run_mod

    monkeypatch.setattr(run_mod, "_panel_fingerprint", _raise)


def test_result_store_refuses_a_cell_whose_panel_fingerprint_failed(tmp_path, monkeypatch):
    """A descriptor carrying no content cannot identify a panel.

    Provenance still has to survive -- a failed fingerprint must not take the run down
    -- so the failure is recorded rather than raised. But it is recorded as the
    canonical undigestible marker, because a digest computed over an error string is
    shared by every panel that fails the same way.
    """
    spec = _spec(tmp_path, arms=[Arm("A", model=_recording_model(), features=_features())])
    _break_fingerprint(monkeypatch)

    data_identity = _data_identity(spec.data)
    json.dumps(data_identity)  # provenance must stay serialisable
    assert data_identity["fingerprint"]["method"] == "undigestible"
    assert "synthetic fingerprint failure" in data_identity["fingerprint"]["reason"]

    identity = result_cell_identity(
        spec, spec.arms[0], spec.targets[0], horizon=1, data_identity=data_identity
    )
    assert identity.digest is None
    assert "fingerprint" in str(identity.reason)


def test_result_store_recomputes_when_the_panel_fingerprint_fails(tmp_path, monkeypatch):
    """End to end: no cell is stored, and a second run recomputes rather than reuses."""
    arms = [Arm("A", model=_recording_model(), features=_features())]
    _break_fingerprint(monkeypatch)

    FIT_COUNTS.clear()
    with pytest.warns(RuntimeWarning, match="cannot digest cell"):
        first = run_pipeline(_spec(tmp_path, arms=arms))
    assert first.provenance["result_store"]["n_undigestible"] == 1
    assert first.provenance["result_store"]["n_reused"] == 0
    assert not list((tmp_path / "results" / "cells").glob("*.json"))
    assert FIT_COUNTS.get("recording", 0) > 0

    FIT_COUNTS.clear()
    with pytest.warns(RuntimeWarning, match="cannot digest cell"):
        second = run_pipeline(_spec(tmp_path, arms=arms))
    assert second.provenance["result_store"]["n_reused"] == 0
    assert FIT_COUNTS.get("recording", 0) > 0


def test_result_store_refuses_the_legacy_unavailable_fingerprint_marker(tmp_path):
    """The old marker is refused at the identity boundary, not just at the producer.

    ``_data_identity`` no longer emits ``unavailable``, but a caller can hand
    ``result_cell_identity`` a prebuilt descriptor -- a stored provenance block from an
    older run, say -- and that must not become a reusable digest either.
    """
    spec = _spec(tmp_path, arms=[Arm("A", model=_recording_model(), features=_features())])

    identity = result_cell_identity(
        spec,
        spec.arms[0],
        spec.targets[0],
        horizon=1,
        data_identity={
            "fingerprint": {
                "algorithm": "sha256",
                "method": "unavailable",
                "error": "TypeError: legacy failure",
            }
        },
    )
    assert identity.digest is None
    assert "legacy failure" in str(identity.reason)
