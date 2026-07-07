from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, pipeline_spec, run_pipeline
import macroforecast.pipeline.run as run_mod


class _SeededOffsetFit:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value, dtype=float)


def _seeded_offset_fit(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 0,
    offset: float = 0.0,
) -> _SeededOffsetFit:
    rng = np.random.default_rng(int(random_state))
    value = float(np.nanmean(np.asarray(y, dtype=float))) + float(offset)
    return _SeededOffsetFit(value + float(rng.normal(scale=0.01)))


def _runtime_bundle(n: int = 84) -> object:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(123)
    x = np.linspace(-1.0, 1.0, n)
    frame = pd.DataFrame(
        {
            "y": 0.5 + 0.4 * x + rng.normal(scale=0.03, size=n),
            "x1": x,
        },
        index=idx,
    )
    return mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})


def _runtime_window():
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=36),
        val=mf.window.val_last_block(size=8),
        test=mf.window.test_origins(horizon=1, step=8),
    )


def _runtime_features():
    return mf.feature_engineering.feature_spec(
        target="y",
        predictors=["x1"],
        lags=1,
        target_lags=(0, 1),
    )


def _seeded_model(*, with_search: bool = False) -> mf.models.ModelSpec:
    search_spaces = (
        {"standard": {"offset": tuple(np.linspace(-0.25, 0.25, 501))}}
        if with_search
        else {}
    )
    return mf.models.custom_model(
        "seeded_offset",
        _seeded_offset_fit,
        default_params={"random_state": 0, "offset": 0.0},
        search_spaces=search_spaces,
        default_search_method="random",
    )


def _seed_spec(
    *,
    seed: int | None = 11,
    n_jobs: int = 1,
    params: dict[str, object] | None = None,
    with_search: bool = False,
):
    model = _seeded_model(with_search=with_search)
    features = _runtime_features()
    selection = None if with_search else {"seeded_offset": None}
    return pipeline_spec(
        data=_runtime_bundle(),
        targets=["y"],
        horizons=[1],
        window=_runtime_window(),
        arms=[
            Arm(
                "A",
                model=model,
                features=features,
                params=params,
                model_selection=selection,
                is_benchmark=True,
            ),
            Arm("B", model=model, features=features, params=params, model_selection=selection),
        ],
        evaluation=EvalSpec(benchmark="A", metrics=("rmse",), tests=()),
        save_models=False,
        seed=seed,
        n_jobs=n_jobs,
    )


def _sorted_forecasts(report) -> pd.DataFrame:
    frame = report.forecasts
    keys = [col for col in ["arm", "target", "horizon", "origin", "date"] if col in frame]
    return frame.sort_values(keys, kind="mergesort").reset_index(drop=True)


def test_pipeline_seed_drives_model_random_state_serial_and_parallel() -> None:
    first = _sorted_forecasts(run_pipeline(_seed_spec(seed=77, n_jobs=1)))
    second = _sorted_forecasts(run_pipeline(_seed_spec(seed=77, n_jobs=1)))
    parallel = _sorted_forecasts(run_pipeline(_seed_spec(seed=77, n_jobs=2)))
    changed = _sorted_forecasts(run_pipeline(_seed_spec(seed=78, n_jobs=1)))

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first, parallel)
    assert not np.allclose(first["prediction"], changed["prediction"])


def test_pipeline_seed_derives_independent_streams_per_arm() -> None:
    report = run_pipeline(_seed_spec(seed=91))
    states = report.provenance["effective_seeds"]["model_random_states"]

    assert states["A"]["source"] == "derived_from_pipeline_seed"
    assert states["B"]["source"] == "derived_from_pipeline_seed"
    assert states["A"]["random_state"] != states["B"]["random_state"]

    forecasts = _sorted_forecasts(report)
    by_arm = {
        arm: group["prediction"].to_numpy(dtype=float)
        for arm, group in forecasts.groupby("arm", sort=True)
    }
    assert not np.allclose(by_arm["A"], by_arm["B"])


def test_explicit_random_state_overrides_pipeline_derived_seed() -> None:
    first = run_pipeline(_seed_spec(seed=101, params={"random_state": 123}))
    second = run_pipeline(_seed_spec(seed=202, params={"random_state": 123}))

    pd.testing.assert_frame_equal(_sorted_forecasts(first), _sorted_forecasts(second))
    states = first.provenance["effective_seeds"]["model_random_states"]
    assert states["A"]["source"] == "explicit_param"
    assert states["A"]["random_state"] == 123


def test_pipeline_seed_drives_model_owned_random_search_draws() -> None:
    first = run_pipeline(_seed_spec(seed=303, with_search=True))
    second = run_pipeline(_seed_spec(seed=303, with_search=True))
    changed = run_pipeline(_seed_spec(seed=304, with_search=True))

    first_param = _sorted_forecasts(first)["model_selection"].iloc[0]["best_params"]["offset"]
    second_param = _sorted_forecasts(second)["model_selection"].iloc[0]["best_params"]["offset"]
    changed_param = _sorted_forecasts(changed)["model_selection"].iloc[0]["best_params"]["offset"]

    assert first_param == second_param
    assert first_param != changed_param


def test_parallel_worker_initializer_sets_threads_seed_and_data_once() -> None:
    with ProcessPoolExecutor(
        max_workers=1,
        initializer=run_mod._parallel_worker_initializer,
        initargs=("probe-token", {"payload": 1}, 3, 5150),
    ) as executor:
        state = executor.submit(run_mod._parallel_worker_probe).result(timeout=30)

    assert state["blas_env"] == {var: "1" for var in run_mod._BLAS_THREAD_ENV_VARS}
    assert state["n_jobs"] == 3
    assert state["random_seed"] == 5150
    assert state["pipeline_seed"] == 5150
    assert state["data_tokens"] == ["probe-token"]


def test_save_models_default_is_off_and_large_store_warns(tmp_path, monkeypatch) -> None:
    spec = _seed_spec(seed=1)
    assert spec.save_models is False

    enabled = pipeline_spec(
        data=_runtime_bundle(),
        targets=["y"],
        horizons=[1],
        window=_runtime_window(),
        arms=[
            Arm(
                "A",
                model=_seeded_model(),
                features=_runtime_features(),
                model_selection={"seeded_offset": None},
                is_benchmark=True,
            )
        ],
        evaluation=EvalSpec(benchmark="A", metrics=("rmse",), tests=()),
        save_models=True,
        model_store=tmp_path / "models",
    )
    monkeypatch.setattr(run_mod, "_projected_origin_count", lambda _spec: 5_001)
    with pytest.warns(UserWarning, match="projected to write"):
        run_mod._warn_large_model_store(enabled, [run_mod._Cell(0, 0, (1,))])


def test_purge_model_store_removes_sidecars_and_pickles(tmp_path) -> None:
    spec = pipeline_spec(
        data=_runtime_bundle(),
        targets=["y"],
        horizons=[1],
        window=_runtime_window(),
        arms=[
            Arm(
                "A",
                model=_seeded_model(),
                features=_runtime_features(),
                model_selection={"seeded_offset": None},
                is_benchmark=True,
            )
        ],
        evaluation=EvalSpec(benchmark="A", metrics=("rmse",), tests=()),
        save_models=True,
        model_store=tmp_path / "models",
    )
    run_pipeline(spec)

    assert list((tmp_path / "models" / "seeded_offset").glob("*.json"))
    assert list((tmp_path / "models" / "seeded_offset").glob("*.pkl"))

    deleted = mf.pipeline.purge_model_store(tmp_path / "models")

    assert deleted > 0
    assert not list((tmp_path / "models").glob("*/*.json"))
    assert not list((tmp_path / "models").glob("*/*.pkl"))
