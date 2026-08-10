"""Serial and parallel must produce identical forecasts on this operating system.

``n_jobs > 1`` uses ``spawn`` on Windows and macOS and ``fork`` on Linux. Under
spawn the worker re-imports this module and re-pickles whatever it was handed, so
a payload that survives fork can still fail here -- and that failure would be a
wrong number or a crash on someone's laptop, never on the Ubuntu suite.

Run as a FILE, not piped on stdin: under spawn the child re-imports ``__main__``
to unpickle its payload, and a ``__main__`` of ``<stdin>`` does not exist on disk.
A stdin heredoc therefore fails with ``BrokenProcessPool`` for a reason that has
nothing to do with the package and that no real user would hit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline


def build():
    n = 70
    idx = pd.date_range("1990-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(0)
    panel = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(3)}, index=idx)
    panel["y"] = 0.5 * panel["x0"] + rng.normal(size=n) * 0.3
    bundle = mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})
    features = mf.feature_engineering.feature_spec(
        target="y", predictors=[f"x{i}" for i in range(3)], lags=0, target_lags=1
    )
    return idx, bundle, features


def run(n_jobs: int) -> pd.DataFrame:
    idx, bundle, features = build()
    spec = pipeline_spec(
        data=bundle,
        targets=[TargetSpec("y", transform="level")],
        horizons=[1],
        window=mf.window.from_cutoffs(
            test_start=idx[50],
            horizon=1,
            embargo=0,
            val_method="expanding",
            val_min_train_size=10,
        ),
        arms=[
            Arm("AR", model="ar", is_benchmark=True),
            Arm("OLS", model="ols", features=features),
        ],
        evaluation=EvalSpec(benchmark="AR", metrics=("rmse",), tests=()),
        save_models=False,
        n_jobs=n_jobs,
        seed=42,
    )
    out = run_pipeline(spec).forecasts
    return out.sort_values(["contender", "date"]).reset_index(drop=True)


def main() -> None:
    serial = run(1)
    parallel = run(2)
    assert not serial.empty, "the serial path produced no forecasts"
    assert not parallel.empty, "the parallel path produced no forecasts"
    np.testing.assert_allclose(
        serial["prediction"].to_numpy(float),
        parallel["prediction"].to_numpy(float),
        rtol=0,
        atol=1e-12,
        err_msg="serial and parallel disagree on this OS",
    )
    print(f"serial == parallel on {len(serial)} forecasts")


if __name__ == "__main__":
    main()
