from __future__ import annotations

import dataclasses
import dataclasses as _dc
import datetime
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


# --------------------------------------------------------------------------- #
# Callable and generic value identity (F-020 / F-021)
# --------------------------------------------------------------------------- #
# A digest is a claim that two runs are the same run. Everything below is a way that
# claim used to be made on evidence that could not support it: a callable recorded by
# name (edit the body, keep the name, get the old forecasts), a NumPy array recorded by
# its TRUNCATED repr, a set recorded in hash-iteration order (so it never matched itself
# across processes), and anything else at all recorded as ``repr()``.
#
# The rule now is: identify it fully, or refuse to identify it. Refusing means
# ``digest=None`` and a reason naming the field -- the cell is recomputed, which is
# always safe -- rather than a digest two different configurations could share.


def _identity_for(tmp_path, **arm_kw):
    spec = _spec(tmp_path, arms=[Arm("A", model=_recording_model(), features=_features(), **arm_kw)])
    return result_cell_identity(
        spec,
        spec.arms[0],
        spec.targets[0],
        horizon=1,
        data_identity=_data_identity(spec.data),
    )


def _search_identity(tmp_path, **search_kw):
    return _identity_for(tmp_path, model_selection=mf.model_selection.SearchSpec(**search_kw))


def _custom_search_pair():
    """Two DIFFERENT functions that share one module/qualname."""

    def first(*args, **kwargs):
        return None

    def second(*args, **kwargs):
        return None

    second.__name__ = second.__qualname__ = first.__qualname__
    return first, second


def test_a_custom_search_function_without_a_marker_makes_the_cell_uncacheable(tmp_path):
    """``SearchSpec.to_dict()`` records it by name, which is not an identity.

    The name is right for the public JSON export and wrong for a cache key, so the
    pipeline reads the dataclass fields instead of that export -- and the ordinary
    callable rule then applies.
    """
    search, _other = _custom_search_pair()

    identity = _search_identity(tmp_path, method="custom", custom_func=search)

    assert identity.digest is None
    assert "custom_func" in str(identity.reason)
    assert "__mf_digest__" in str(identity.reason)


def test_two_custom_search_functions_sharing_a_qualname_get_different_digests(tmp_path):
    """The marker is IN the identity, not merely checked for presence."""
    first, second = _custom_search_pair()
    first.__mf_digest__ = "search-v1"
    second.__mf_digest__ = "search-v2"

    one = _search_identity(tmp_path, method="custom", custom_func=first)
    two = _search_identity(tmp_path, method="custom", custom_func=second)

    assert one.digest is not None
    assert one.digest != two.digest

    second.__mf_digest__ = "search-v1"
    same = _search_identity(tmp_path, method="custom", custom_func=second)
    assert same.digest == one.digest, "same marker and same configuration must be stable"


def test_a_callable_nested_in_the_search_configuration_needs_a_marker_too(tmp_path):
    search, _other = _custom_search_pair()
    search.__mf_digest__ = "search-v1"

    def scorer(*args, **kwargs):
        return 0.0

    bare = _search_identity(
        tmp_path, method="custom", custom_func=search, custom_params={"scorer": scorer}
    )
    assert bare.digest is None
    assert "custom_params.scorer" in str(bare.reason)

    scorer.__mf_digest__ = "scorer-v1"
    first = _search_identity(
        tmp_path, method="custom", custom_func=search, custom_params={"scorer": scorer}
    )
    scorer.__mf_digest__ = "scorer-v2"
    second = _search_identity(
        tmp_path, method="custom", custom_func=search, custom_params={"scorer": scorer}
    )
    assert first.digest is not None
    assert first.digest != second.digest


def test_a_callable_in_arm_params_needs_a_marker_and_the_marker_moves_the_digest(tmp_path):
    def callback(*args, **kwargs):
        return None

    assert _identity_for(tmp_path, params={"cb": callback}).digest is None

    callback.__mf_digest__ = "cb-v1"
    first = _identity_for(tmp_path, params={"cb": callback}).digest
    callback.__mf_digest__ = "cb-v2"
    second = _identity_for(tmp_path, params={"cb": callback}).digest

    assert first is not None
    assert first != second


def test_two_arrays_with_the_same_truncated_repr_do_not_share_a_digest(tmp_path):
    """NumPy elides the middle of a large array; the digest must not.

    ``np.arange(2000.0)`` and the same array with one changed element in the elided
    region print identically, so a repr-based identity served one run's forecasts for
    the other.
    """
    left = np.arange(2000.0)
    right = left.copy()
    right[1000] = -12345.0
    assert repr(left) == repr(right), "the premise: their reprs are identical"

    first = _identity_for(tmp_path, params={"w": left})
    second = _identity_for(tmp_path, params={"w": right})

    assert first.digest is not None
    assert first.digest != second.digest


def test_array_identity_follows_content_dtype_and_shape(tmp_path):
    base = np.arange(12.0)

    reference = _identity_for(tmp_path, params={"w": base}).digest
    assert _identity_for(tmp_path, params={"w": base.copy()}).digest == reference
    assert _identity_for(tmp_path, params={"w": base.astype(np.float32)}).digest != reference
    assert _identity_for(tmp_path, params={"w": base.reshape(3, 4)}).digest != reference


def test_set_identity_does_not_depend_on_construction_order(tmp_path):
    """And, by construction, not on the hash seed either.

    The elements are serialized first and then ordered by their canonical JSON text, so
    the answer cannot depend on the iteration order the interpreter happens to give.
    """
    letters = ["alpha", "beta", "gamma", "delta", "epsilon"]

    first = _identity_for(tmp_path, params={"c": set(letters)}).digest
    second = _identity_for(tmp_path, params={"c": set(reversed(letters))}).digest

    assert first is not None
    assert first == second


def test_the_serializer_keeps_the_container_type_of_a_set(tmp_path):
    """A set and a frozenset are different objects to whatever consumes them.

    Note this is asserted at the serializer, not through ``Arm.params``: ``spec.py``'s
    ``_freeze_value`` turns a caller's set into a frozenset before identity ever sees
    it, so through an arm the downstream type genuinely cannot differ.
    """
    from macroforecast.pipeline.result_store import _json_ready

    assert _json_ready({"a", "b"}) != _json_ready(frozenset({"a", "b"}))
    assert _json_ready({"a", "b"})["__set__"]["type"] == "set"
    assert _json_ready(frozenset({"a", "b"}))["__set__"]["type"] == "frozenset"


def test_bytes_identity_is_deterministic_and_type_aware(tmp_path):
    plain = _identity_for(tmp_path, params={"b": b"ab"}).digest
    mutable = _identity_for(tmp_path, params={"b": bytearray(b"ab")}).digest
    changed = _identity_for(tmp_path, params={"b": b"ac"}).digest

    assert plain is not None
    assert plain != mutable
    assert plain != changed


def test_two_unsupported_objects_sharing_a_repr_get_no_digest_rather_than_one(tmp_path):
    """The failure mode a repr fallback produces, stated as a test."""

    class Opaque:
        def __repr__(self):
            return "<same repr>"

    first = _identity_for(tmp_path, params={"o": Opaque()})
    second = _identity_for(tmp_path, params={"o": Opaque()})

    assert first.digest is None and second.digest is None
    assert "Opaque" in str(first.reason)


def test_a_raising_to_dict_makes_the_cell_uncacheable_with_a_path(tmp_path):
    class Exploding:
        def to_dict(self):
            raise RuntimeError("nope")

    identity = _identity_for(tmp_path, params={"e": Exploding()})

    assert identity.digest is None
    assert "to_dict()" in str(identity.reason)
    assert "nope" in str(identity.reason)


def test_mapping_keys_whose_string_forms_collide_get_no_digest(tmp_path):
    """Two keys, one written form: the mapping has no unambiguous canonical shape."""

    class Key:
        def __init__(self, tag):
            self.tag = tag

        def __str__(self):
            return "same"

        def __hash__(self):
            return hash(self.tag)

        def __eq__(self, other):
            return isinstance(other, Key) and other.tag == self.tag

    identity = _identity_for(tmp_path, params={"m": {Key("a"): 1, Key("b"): 2}})

    assert identity.digest is None
    assert "same" in str(identity.reason)


def test_a_self_referential_container_is_refused_rather_than_recursed(tmp_path):
    """Reached through the search configuration, which the spec layer does not freeze."""
    search, _other = _custom_search_pair()
    search.__mf_digest__ = "search-v1"
    cyclic: dict = {}
    cyclic["self"] = cyclic

    identity = _search_identity(
        tmp_path, method="custom", custom_func=search, custom_params={"c": cyclic}
    )

    assert identity.digest is None
    assert "self-referential" in str(identity.reason)


def test_a_numpy_scalar_that_is_not_json_ready_is_refused_not_emitted(tmp_path):
    """``.item()`` on a complex gives a Python complex, which no JSON encoder takes.

    Returning it unchecked put a value into the payload that ``json.dumps`` would then
    reject, ending the run; it goes back through the rules instead.
    """
    assert _identity_for(tmp_path, params={"z": np.complex128(1 + 2j)}).digest is None
    assert _identity_for(tmp_path, params={"z": np.float64(1.5)}).digest is not None


def test_an_unserializable_param_does_not_crash_the_run(tmp_path):
    """The registry model branch embedded ``ModelSpec.to_dict()`` verbatim.

    Any params value the JSON encoder could not take therefore reached ``json.dumps``
    inside ``result_cell_identity`` and raised TypeError out of the whole run. Identity
    may decline to identify a cell; it may not end the run.
    """

    def callback(*args, **kwargs):
        return None

    callback.__mf_digest__ = "cb-v1"

    identity = _identity_for(tmp_path, params={"cb": callback})
    assert identity.digest is not None

    identity = _identity_for(tmp_path, params={"cb": object()})
    assert identity.digest is None


def test_an_ordinary_spec_still_produces_a_digest(tmp_path):
    """The floor: hardening identity must not make a plain configuration uncacheable."""
    identity = _identity_for(tmp_path)

    assert identity.digest is not None
    assert identity.reason is None


# --------------------------------------------------------------------------- #
# Code is identified as code, whatever else it also looks like
# --------------------------------------------------------------------------- #
# A callable OBJECT can also expose ``to_dict()`` or be a dataclass. Serializing it
# through that structure meant it was never asked for ``__mf_digest__`` -- the same
# bypass F-020 closes for a plain function, wearing a different hat. Structure cannot
# stand in for a body: two functors can carry identical fields and compute entirely
# different things.


class _CallableWithDict:
    """Callable whose public export must not stand in for its identity."""

    def __init__(self, marker=None):
        if marker is not None:
            self.__mf_digest__ = marker

    def __call__(self, *args, **kwargs):
        return []

    def to_dict(self):
        return {"kind": "same-for-every-instance"}


@dataclasses.dataclass
class _CallableDataclass:
    """Callable that is ALSO a dataclass, the other structural bypass."""

    setting: int = 1

    def __call__(self, *args, **kwargs):
        return []


def test_a_callable_object_cannot_bypass_the_marker_through_to_dict(tmp_path):
    identity = _identity_for(tmp_path, params={"payload": _CallableWithDict()})

    assert identity.digest is None
    assert "payload" in str(identity.reason)
    assert "__mf_digest__" in str(identity.reason)


def test_a_callable_dataclass_cannot_bypass_the_marker_through_its_fields(tmp_path):
    identity = _identity_for(tmp_path, params={"payload": _CallableDataclass()})

    assert identity.digest is None
    assert "payload" in str(identity.reason)


def test_two_instances_of_one_callable_object_agree_and_the_marker_separates_them(tmp_path):
    """Separate instances, same marker: one cell. Different marker: different cells.

    This is what the address-bearing name fallback used to break -- two instances of one
    functor produced different names and therefore different digests, so a store could
    never hit.
    """
    first = _identity_for(tmp_path, params={"payload": _CallableWithDict("object-v1")})
    repeat = _identity_for(tmp_path, params={"payload": _CallableWithDict("object-v1")})
    second = _identity_for(tmp_path, params={"payload": _CallableWithDict("object-v2")})

    assert first.digest is not None
    assert repeat.digest == first.digest
    assert second.digest != first.digest


def test_a_callable_objects_recorded_name_does_not_carry_its_address(tmp_path):
    from macroforecast.pipeline.result_store import _callable_name

    name = _callable_name(_CallableWithDict("object-v1"))

    assert "0x" not in name
    assert name == _callable_name(_CallableWithDict("object-v2"))
    assert name.endswith("_CallableWithDict")


def test_a_callable_objects_marker_is_recorded_not_merely_checked(tmp_path):
    identity = _identity_for(tmp_path, params={"payload": _CallableWithDict("object-v1")})

    assert "object-v1" in json.dumps(identity.cell_echo, sort_keys=True)


# --------------------------------------------------------------------------- #
# ...but a ModelSpec is not anonymous code
# --------------------------------------------------------------------------- #
# ``ModelSpec`` is callable too, so "callable means marker required" would have made a
# nested registry model uncacheable -- a regression, since the registry already
# identifies its fit function by name and the payload already records the backend
# versions. ``_model_identity`` owns that registry-versus-custom rule; a nested spec
# asks it rather than getting a second one.


def test_a_nested_registry_model_spec_stays_cacheable(tmp_path):
    identity = _identity_for(tmp_path, params={"base": mf.models.get_model("ridge")})

    assert identity.digest is not None


def test_a_nested_custom_model_spec_without_a_marker_is_still_refused(tmp_path):
    def unmarked_fit(X, y):
        return _ConstantFit(0.0)

    custom = mf.models.custom_model("nested_custom", unmarked_fit)

    assert _identity_for(tmp_path, params={"base": custom}).digest is None

    marked = mf.models.custom_model("nested_custom", unmarked_fit, mf_digest="nested-v1")
    assert _identity_for(tmp_path, params={"base": marked}).digest is not None


# --------------------------------------------------------------------------- #
# Only what can be identified completely is identified
# --------------------------------------------------------------------------- #
# Two more ways a digest was minted on evidence that could not support it. Both are the
# same mistake as the repr fallback: accepting a partial view of a value and treating it
# as the whole thing.


def test_a_masked_array_does_not_share_a_digest_with_a_differently_masked_one(tmp_path):
    """The mask decides which elements exist, and the data buffer cannot see it.

    ``isinstance(value, np.ndarray)`` accepted subclasses and hashed only the base
    array's dtype, shape and bytes, so these two -- which behave differently everywhere
    -- were one cell.
    """
    first = np.ma.array([1.0, 2.0], mask=[False, True])
    second = np.ma.array([1.0, 2.0], mask=[True, False])

    left = _identity_for(tmp_path, params={"payload": first})
    right = _identity_for(tmp_path, params={"payload": second})

    assert left.digest is None or right.digest is None or left.digest != right.digest


def test_an_ndarray_subclass_is_refused_by_name(tmp_path):
    """Refused rather than guessed at: which attributes of an unknown subclass matter
    is not something this module can know."""
    identity = _identity_for(tmp_path, params={"payload": np.ma.array([1.0], mask=[True])})

    assert identity.digest is None
    assert "MaskedArray" in str(identity.reason)
    assert "subclass" in str(identity.reason)


def test_a_plain_ndarray_is_still_identified(tmp_path):
    """The floor for the narrowing above: ordinary arrays keep working."""
    identity = _identity_for(tmp_path, params={"payload": np.arange(4.0)})

    assert identity.digest is not None


def test_a_mapping_with_a_non_string_key_is_refused(tmp_path):
    """``str(key)`` on an arbitrary object is a repr carrying its address.

    That is the one fallback F-021 exists to remove, and it had survived on the key side
    of a mapping.
    """
    identity = _identity_for(tmp_path, params={"payload": {object(): "value"}})

    assert identity.digest is None
    assert "key" in str(identity.reason)
    assert "0x" not in str(identity.reason), "the reason must not carry an address either"


def test_an_int_keyed_mapping_is_refused_rather_than_coerced(tmp_path):
    """``1`` and ``"1"`` would otherwise both render as ``"1"``.

    Refusing every non-string key covers that collision by construction, instead of
    detecting it after the fact.
    """
    identity = _identity_for(tmp_path, params={"payload": {1: "integer", "1": "string"}})

    assert identity.digest is None
    assert "key" in str(identity.reason)

    assert _identity_for(tmp_path, params={"payload": {1: "integer"}}).digest is None


def test_a_string_keyed_mapping_is_still_identified(tmp_path):
    """The floor for the narrowing above: ordinary params keep working."""
    identity = _identity_for(tmp_path, params={"payload": {"alpha": 1, "beta": [2, 3]}})

    assert identity.digest is not None


def test_mapping_key_order_does_not_move_the_digest(tmp_path):
    first = _identity_for(tmp_path, params={"payload": {"alpha": 1, "beta": 2}})
    second = _identity_for(tmp_path, params={"payload": {"beta": 2, "alpha": 1}})

    assert first.digest is not None
    assert first.digest == second.digest


def test_a_structured_dtype_holding_objects_is_identified_by_its_elements(tmp_path):
    """A buffer of pointers says nothing about what they point at.

    ``dtype == object`` is False for a structured dtype that merely CONTAINS an object
    field, so such an array was byte-hashed -- comparing addresses. These two arrays hold
    equal elements in separately allocated lists; they are one configuration and must be
    one cell. The dtype string is still carried, so a different structured layout with
    the same elements remains a different cell.
    """
    dtype = [("a", "i4"), ("b", "O")]
    first = np.array([(1, [1, 2])], dtype=dtype)
    second = np.array([(1, [1, 2])], dtype=dtype)
    assert not (first.dtype == object), "the trap: a structured dtype is not == object"
    assert first.dtype.hasobject, "...but it still stores a pointer"

    left = _identity_for(tmp_path, params={"payload": first})
    right = _identity_for(tmp_path, params={"payload": second})

    assert left.digest is not None
    assert left.digest == right.digest

    changed = _identity_for(tmp_path, params={"payload": np.array([(1, [1, 3])], dtype=dtype)})
    assert changed.digest != left.digest

    relabelled = np.array([(1, [1, 2])], dtype=[("a", "i4"), ("c", "O")])
    assert _identity_for(tmp_path, params={"payload": relabelled}).digest != left.digest


# --------------------------------------------------------------------------- #
# Resolved stage policies are identified, not exported (F-044)
# --------------------------------------------------------------------------- #
# ``StagePolicy.to_dict()`` is a readable export and renders a custom selector, and any
# callable in ``metadata``, as a module/qualname string. Packet 08 put the resolved
# policies into the digest by way of that export, so a selector needed no
# ``__mf_digest__`` and two selectors sharing a qualname were one cell -- the bypass this
# module already closes for models, features, preprocessing and search callables.
#
# The export is unchanged. Only the result-store boundary is stricter, and only for the
# callable entries: an ordinary policy still serializes byte-identically to its export,
# which is what keeps existing stores hitting.


def _selector(marker=None):
    """A selector whose qualname is fixed, so only the marker can distinguish two."""

    def implementation(*args, **kwargs):
        return []

    implementation.__module__ = "tests.stage_policy"
    implementation.__name__ = implementation.__qualname__ = "same_name"
    if marker is not None:
        implementation.__mf_digest__ = marker
    return implementation


def _feature_policy_identity(tmp_path, policy):
    return _identity_for(tmp_path, feature_policy=policy)


def _preprocessing_policy_identity(tmp_path, policy):
    spec = _spec(
        tmp_path,
        arms=[Arm("A", model=_recording_model(), features=_features())],
        preprocessing=mf.preprocess_spec(standardize="zscore"),
        preprocessing_policy=policy,
    )
    return result_cell_identity(
        spec,
        spec.arms[0],
        spec.targets[0],
        horizon=1,
        data_identity=_data_identity(spec.data),
    )


@pytest.mark.parametrize(
    ("slot", "identity_for"),
    [
        ("feature_engineering", _feature_policy_identity),
        ("preprocessing", _preprocessing_policy_identity),
    ],
)
def test_a_markerless_stage_selector_makes_the_cell_uncacheable(tmp_path, slot, identity_for):
    identity = identity_for(tmp_path, mf.window.custom_stage_policy(_selector()))

    assert identity.digest is None
    assert "stage_policies" in str(identity.reason)
    assert slot in str(identity.reason)
    assert "selector" in str(identity.reason)


@pytest.mark.parametrize(
    "identity_for", [_feature_policy_identity, _preprocessing_policy_identity]
)
def test_the_stage_selector_marker_is_the_identity(tmp_path, identity_for):
    """Same qualname, different markers: different cells. Same marker: one cell."""
    first = identity_for(tmp_path, mf.window.custom_stage_policy(_selector("selector-v1")))
    repeat = identity_for(tmp_path, mf.window.custom_stage_policy(_selector("selector-v1")))
    second = identity_for(tmp_path, mf.window.custom_stage_policy(_selector("selector-v2")))

    assert first.digest is not None
    assert repeat.digest == first.digest
    assert second.digest != first.digest
    assert "selector-v1" in json.dumps(first.cell_echo, sort_keys=True)


def test_a_callable_in_stage_policy_metadata_obeys_the_same_rule(tmp_path):
    bare = _feature_policy_identity(
        tmp_path,
        mf.window.custom_stage_policy(
            _selector("selector-v1"), metadata={"callback": _selector()}
        ),
    )
    assert bare.digest is None
    assert "metadata" in str(bare.reason)
    assert "callback" in str(bare.reason)

    first = _feature_policy_identity(
        tmp_path,
        mf.window.custom_stage_policy(
            _selector("selector-v1"), metadata={"callback": _selector("metadata-v1")}
        ),
    )
    second = _feature_policy_identity(
        tmp_path,
        mf.window.custom_stage_policy(
            _selector("selector-v1"), metadata={"callback": _selector("metadata-v2")}
        ),
    )
    assert first.digest is not None
    assert first.digest != second.digest


def test_an_ordinary_policy_serializes_exactly_as_it_exports(tmp_path):
    """The floor that keeps existing stores hitting, stated directly.

    Everything the public export already canonicalises — normalized scope and update
    (including an integer or ``DateOffset`` cadence), reference bounds, ``apply_to``,
    ordinary metadata — must come through with the identical representation.
    """
    from macroforecast.pipeline.result_store import _stage_policy_identity

    for policy in (
        mf.window.stage_policy("fit_window"),
        mf.window.stage_policy("full_panel", metadata={"label": "ordinary", "n": 3}),
        mf.window.stage_policy(
            "fixed_reference", update=3, reference_start=pd.Timestamp("2000-01-31")
        ),
        mf.window.stage_policy("fit_window", metadata={"pair": (1, 2)}),
    ):
        assert _stage_policy_identity(policy, path="p") == policy.to_dict()

    # A DateOffset cadence is the one entry that deliberately no longer matches the
    # export: the export records ``freqstr``, which does not identify the offset (see
    # the DateOffset tests below). Stated here rather than quietly dropped, because this
    # loop is the floor that keeps ordinary digests from moving.
    offset_policy = mf.window.stage_policy(
        "fit_window", update=pd.tseries.offsets.MonthEnd(2)
    )
    assert _stage_policy_identity(offset_policy, path="p") != offset_policy.to_dict()
    assert offset_policy.to_dict()["update"] == "2ME"


def test_the_public_stage_policy_export_is_untouched(tmp_path):
    """The marker belongs to the cache key, not to the readable export."""
    selector = _selector("selector-v1")
    exported = mf.window.custom_stage_policy(selector).to_dict()

    assert exported["selector"] == "tests.stage_policy.same_name"
    assert "selector-v1" not in str(exported)


def test_every_stage_slot_is_covered_including_one_the_runner_cannot_reach(tmp_path):
    """``run.py`` never passes a model-selection policy today.

    Covering the slot anyway is the point: a future wiring change must not silently
    reintroduce the bypass in the one place nobody was looking.
    """
    from macroforecast.pipeline.plan import CompiledStagePolicies
    from macroforecast.pipeline.result_store import (
        _UndigestibleCell,
        _stage_policies_identity,
    )

    ordinary = mf.window.stage_policy("fit_window")
    for slot in ("preprocessing", "feature_engineering", "model_selection"):
        policies = CompiledStagePolicies(
            **{
                "preprocessing": ordinary,
                "feature_engineering": ordinary,
                "model_selection": ordinary,
                slot: mf.window.custom_stage_policy(_selector()),
            }
        )
        with pytest.raises(_UndigestibleCell, match=f"{slot}.selector"):
            _stage_policies_identity(policies, path="arm.stage_policies")


def test_the_strict_identity_keeps_the_compiled_slot_shape(tmp_path):
    """Same three slots as the runner metadata and the leakage audit publish."""
    from macroforecast.pipeline.plan import CompiledStagePolicies
    from macroforecast.pipeline.result_store import _stage_policies_identity

    policies = CompiledStagePolicies(
        preprocessing=None,
        feature_engineering=mf.window.stage_policy("fit_window"),
        model_selection=mf.window.stage_policy("fit_window"),
    )

    assert _stage_policies_identity(policies, path="p") == policies.to_dict()


def test_an_unsupported_value_in_a_stage_policy_does_not_crash_the_run(tmp_path):
    """``StagePolicy.to_dict()`` passes a value it does not recognise straight through.

    Embedding that export verbatim put a raw object into the identity payload, and
    ``json.dumps`` then raised out of the whole run. Identity may decline to identify a
    cell; it may not end the run — so the exported payload is serialized like any other
    value, and an unsupported field is named in the reason.
    """
    policy = mf.window.stage_policy("fixed_reference", reference_start=object())

    identity = _identity_for(tmp_path, feature_policy=policy)

    assert identity.digest is None
    assert "reference_start" in str(identity.reason)


# --------------------------------------------------------------------------- #
# A pandas offset is identified by its semantics, not by its label (F-045)
# --------------------------------------------------------------------------- #
# ``freqstr`` is what the readable export records, and it is not an identity: EVERY
# ``CustomBusinessDay`` reports ``"C"`` whatever holidays or week mask it carries. Since
# the update cadence decides when a stage is refit, two policies that refit on different
# days shared one cell.


def _offset_policy_identity(tmp_path, offset):
    return _identity_for(
        tmp_path, feature_policy=mf.window.stage_policy("fit_window", update=offset)
    )


def test_custom_business_days_with_different_holidays_are_different_cells(tmp_path):
    from pandas.tseries.offsets import CustomBusinessDay

    first = _offset_policy_identity(tmp_path, CustomBusinessDay(holidays=["2020-01-01"]))
    same = _offset_policy_identity(tmp_path, CustomBusinessDay(holidays=["2020-01-01"]))
    other = _offset_policy_identity(tmp_path, CustomBusinessDay(holidays=["2020-07-04"]))

    assert first.digest is not None
    assert same.digest == first.digest, "equal offsets, separately constructed, are one cell"
    assert other.digest != first.digest


def test_the_week_mask_and_multiplier_of_a_business_offset_are_part_of_identity(tmp_path):
    from pandas.tseries.offsets import CustomBusinessDay

    reference = _offset_policy_identity(tmp_path, CustomBusinessDay(holidays=["2020-01-01"]))
    weekmask = _offset_policy_identity(
        tmp_path, CustomBusinessDay(weekmask="Mon Tue Wed Thu", holidays=["2020-01-01"])
    )
    multiplier = _offset_policy_identity(
        tmp_path, CustomBusinessDay(n=2, holidays=["2020-01-01"])
    )

    assert weekmask.digest not in {None, reference.digest}
    assert multiplier.digest not in {None, reference.digest}


def test_an_explicitly_passed_calendar_is_what_identity_reads(tmp_path):
    """The reason the calendar object is serialized rather than trusted to ``kwds``.

    Constructing ``CustomBusinessDay(calendar=...)`` leaves ``kwds["weekmask"]``
    reporting the DEFAULT ``'Mon Tue Wed Thu Fri'`` while the calendar holds the mask that
    actually applies — so identifying the offset from kwds alone would make two different
    calendars look alike.
    """
    from pandas.tseries.offsets import CustomBusinessDay

    narrow = np.busdaycalendar(weekmask="Mon Tue Wed")
    wide = np.busdaycalendar(weekmask="Mon Tue Wed Thu")
    assert CustomBusinessDay(calendar=narrow).kwds["weekmask"] == "Mon Tue Wed Thu Fri"

    first = _offset_policy_identity(tmp_path, CustomBusinessDay(calendar=narrow))
    second = _offset_policy_identity(tmp_path, CustomBusinessDay(calendar=wide))

    assert first.digest is not None
    assert first.digest != second.digest


def test_standard_offsets_separate_by_class_and_multiplier(tmp_path):
    from pandas.tseries.offsets import BusinessMonthEnd, MonthEnd

    reference = _offset_policy_identity(tmp_path, MonthEnd(2))

    assert reference.digest is not None
    assert _offset_policy_identity(tmp_path, MonthEnd(2)).digest == reference.digest
    assert _offset_policy_identity(tmp_path, MonthEnd(3)).digest != reference.digest
    assert _offset_policy_identity(tmp_path, BusinessMonthEnd(2)).digest != reference.digest


def test_business_hour_windows_separate(tmp_path):
    from pandas.tseries.offsets import BusinessHour

    first = _offset_policy_identity(tmp_path, BusinessHour(start="09:00", end="17:00"))
    second = _offset_policy_identity(tmp_path, BusinessHour(start="10:00", end="17:00"))

    assert first.digest is not None
    assert first.digest != second.digest


def test_the_offset_identity_is_readable_and_carries_no_address(tmp_path):
    """A ``busdaycalendar``'s only self-description is its address, so it gets one."""
    from pandas.tseries.offsets import CustomBusinessDay
    from macroforecast.pipeline.result_store import _json_ready

    encoded = json.dumps(
        _json_ready(CustomBusinessDay(holidays=["2020-01-01"]), path="p"), sort_keys=True
    )

    assert "0x" not in encoded
    assert "2020-01-01" in encoded
    assert "CustomBusinessDay" in encoded


def test_unsupported_offset_state_fails_closed_with_a_field_path(tmp_path):
    from pandas.tseries.offsets import DateOffset
    from macroforecast.pipeline.result_store import _UndigestibleCell, _json_ready

    class Opaque:
        pass

    class OpaqueStateOffset(DateOffset):
        @property
        def kwds(self):
            return {"gadget": Opaque()}

    with pytest.raises(_UndigestibleCell, match=r"update\.kwds\.gadget"):
        _json_ready(OpaqueStateOffset(), path="policy.update")


def test_an_offset_that_cannot_report_its_state_fails_closed(tmp_path):
    """Identity may decline to identify a cell; it may not end the run."""
    from pandas.tseries.offsets import DateOffset
    from macroforecast.pipeline.result_store import _UndigestibleCell, _json_ready

    class BrokenOffset(DateOffset):
        @property
        def kwds(self):
            raise RuntimeError("no state here")

    with pytest.raises(_UndigestibleCell, match="did not expose its offset state"):
        _json_ready(BrokenOffset(), path="policy.update")


def test_the_public_export_still_records_the_offset_by_label(tmp_path):
    """The stricter representation is private; the readable export is untouched."""
    from pandas.tseries.offsets import CustomBusinessDay

    first = mf.window.stage_policy(
        "fit_window", update=CustomBusinessDay(holidays=["2020-01-01"])
    )
    second = mf.window.stage_policy(
        "fit_window", update=CustomBusinessDay(holidays=["2020-07-04"])
    )

    assert first.to_dict()["update"] == "C"
    assert second.to_dict()["update"] == "C"
    assert first.to_dict() == second.to_dict(), "the export cannot tell them apart"


def test_a_date_offset_in_policy_metadata_is_now_identifiable(tmp_path):
    """A side effect worth having: the packet-11 metadata gap closes with the same helper.

    A ``DateOffset`` inside ``StagePolicy.metadata`` used to make the cell uncacheable,
    because the serializer had no branch for it. It now carries full semantics, so such a
    policy is cacheable and two different cadences are different cells.
    """
    from pandas.tseries.offsets import MonthEnd

    first = _identity_for(
        tmp_path,
        feature_policy=mf.window.stage_policy(
            "fit_window", metadata={"cadence": MonthEnd(2)}
        ),
    )
    second = _identity_for(
        tmp_path,
        feature_policy=mf.window.stage_policy(
            "fit_window", metadata={"cadence": MonthEnd(3)}
        ),
    )

    assert first.digest is not None
    assert first.digest != second.digest


def test_the_datetime_family_is_identified_exactly(tmp_path):
    """The leaves offset state is built from, each with its own unambiguous form."""
    from macroforecast.pipeline.result_store import _json_ready

    assert _json_ready(datetime.time(9, 0)) != _json_ready(datetime.time(10, 0))
    assert _json_ready(datetime.date(2020, 1, 1)) != _json_ready(
        datetime.datetime(2020, 1, 1)
    ), "a date and a midnight datetime are different values"
    assert _json_ready(datetime.timedelta(seconds=1)) != _json_ready(
        datetime.timedelta(seconds=2)
    )
    assert _json_ready(pd.Timedelta("1ns")) != _json_ready(pd.Timedelta("2ns")), (
        "nanoseconds survive: timedelta's own fields stop at microseconds"
    )
    assert _json_ready(datetime.timedelta(seconds=1)) != _json_ready(
        pd.Timedelta(seconds=1)
    ), "same duration, different types: the count alone cannot separate them"
    assert _json_ready(pd.Timestamp("2020-01-01")) == "2020-01-01T00:00:00", (
        "Timestamp keeps its long-standing bare-ISO rendering, so digests do not move"
    )


def test_the_two_sides_of_a_repeated_wall_clock_hour_are_different_values(tmp_path):
    """``fold`` distinguishes them and ``isoformat()`` does not carry it.

    A datetime with ``fold=1`` names the second occurrence of an ambiguous local time --
    genuinely different state, and identical text. Recording the ISO string alone made
    them one value.
    """
    from macroforecast.pipeline.result_store import _json_ready

    first = datetime.datetime(2020, 11, 1, 1, 30, fold=0)
    second = datetime.datetime(2020, 11, 1, 1, 30, fold=1)
    assert first.isoformat() == second.isoformat(), "the premise: the text is identical"

    assert _json_ready(first) != _json_ready(second)
    assert _json_ready(datetime.time(1, 30, fold=0)) != _json_ready(
        datetime.time(1, 30, fold=1)
    )


def test_a_date_is_not_given_a_fold_it_does_not_have(tmp_path):
    """``date`` has no ``fold``, so the field is conditional rather than assumed."""
    from macroforecast.pipeline.result_store import _json_ready

    assert not hasattr(datetime.date(2020, 1, 1), "fold")
    assert "fold" not in _json_ready(datetime.date(2020, 1, 1))["__datetime__"]
    assert "fold" in _json_ready(datetime.datetime(2020, 1, 1))["__datetime__"]


def test_the_datetime_leaves_record_their_concrete_type(tmp_path):
    from macroforecast.pipeline.result_store import _json_ready

    assert (
        _json_ready(datetime.timedelta(seconds=1))["__timedelta__"]["type"]
        == "datetime.timedelta"
    )
    assert (
        _json_ready(pd.Timedelta(seconds=1))["__timedelta__"]["type"]
        == "pandas.Timedelta"
    )
    assert (
        _json_ready(datetime.date(2020, 1, 1))["__datetime__"]["type"] == "datetime.date"
    )


def test_a_public_class_is_named_by_its_public_path_on_every_pandas():
    """The recorded type must not depend on where the library keeps the class.

    ``pandas.Timedelta`` is re-exported from ``pandas`` on every supported version, but
    reports ``pandas._libs.tslibs.timedeltas`` as its own module on pandas 2 and
    ``pandas`` on pandas 3. Recording ``type(value).__module__`` therefore wrote two
    different strings into the same cache digest depending on which pandas read the
    cell, so an upgrade silently missed every cached cell carrying a pandas duration.
    """
    from macroforecast.pipeline.result_store import _type_name

    assert _type_name(pd.Timedelta(seconds=1)) == "pandas.Timedelta"
    assert "_libs" not in _type_name(pd.Timedelta(seconds=1)), (
        "a private submodule path is an implementation detail and must not reach a digest"
    )
    assert pd.Timedelta is getattr(pd, "Timedelta", None), (
        "the premise: the shortened name is only used because pandas exports this class"
    )


def test_a_class_its_package_does_not_export_keeps_its_full_path():
    """Shortening is earned, not assumed, so nothing collapses.

    Two classes with the same ``__qualname__`` in different private submodules would
    become one name under a scheme that shortened by name alone, and they would then
    share a cell digest. The root package has to publish the very same class object.
    """
    from macroforecast.pipeline.result_store import _type_name

    class Hidden(datetime.timedelta):
        pass

    name = _type_name(Hidden(seconds=1))
    assert name.endswith(".Hidden")
    assert name == f"{Hidden.__module__}.{Hidden.__qualname__}", (
        "a class its root package does not export keeps the module it actually lives in"
    )


def test_shortening_requires_the_same_class_object_not_a_matching_name(monkeypatch):
    """The check is identity, so a same-named impostor cannot capture the short path.

    This is what stops the repair from becoming the collapse it replaces: a package that
    publishes SOME other class under the same name has not published this one, and the
    private path is then the only name that still says which class was recorded.
    """
    import sys
    import types

    from macroforecast.pipeline.result_store import _type_name

    root = types.ModuleType("mf_fake_pkg")
    private = types.ModuleType("mf_fake_pkg.private")

    class Widget:
        pass

    class Impostor:
        pass

    Widget.__module__ = "mf_fake_pkg.private"
    Widget.__qualname__ = "Widget"
    Impostor.__qualname__ = "Widget"
    private.Widget = Widget
    root.Widget = Impostor
    monkeypatch.setitem(sys.modules, "mf_fake_pkg", root)
    monkeypatch.setitem(sys.modules, "mf_fake_pkg.private", private)

    assert _type_name(Widget()) == "mf_fake_pkg.private.Widget", (
        "the root publishes a different class under this name, so the short path would "
        "name the wrong thing"
    )

    root.Widget = Widget
    assert _type_name(Widget()) == "mf_fake_pkg.Widget", (
        "and once it really is the same class object, the public path is used"
    )


def test_pandas_and_stdlib_durations_stay_different_values():
    """The shortened pandas name must not become the stdlib one."""
    from macroforecast.pipeline.result_store import _json_ready

    assert _json_ready(pd.Timedelta(seconds=1)) != _json_ready(
        datetime.timedelta(seconds=1)
    )
    assert (
        _json_ready(pd.Timedelta(seconds=1))["__timedelta__"]["ns"]
        == _json_ready(datetime.timedelta(seconds=1))["__timedelta__"]["ns"]
    ), "the premise: only the recorded type separates them"


def test_pandas_timestamp_keeps_its_legacy_rendering(tmp_path):
    """It is a ``datetime``, so it must stay ahead of the tagged branch.

    Its bare ISO string predates this series; tagging it would move every digest that
    carries a timestamp.
    """
    from macroforecast.pipeline.result_store import _json_ready

    assert _json_ready(pd.Timestamp("2020-01-01")) == "2020-01-01T00:00:00"
