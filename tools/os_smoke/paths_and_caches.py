"""A cache written and re-read on this filesystem must not change the answer.

Windows is case-insensitive and uses backslashes; a cache directory or a saved
model written with one convention and read with another fails here and nowhere
else. The directory name carries a space on purpose -- unquoted path handling is
the other half of the same failure mode.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline

CACHE = Path(tempfile.mkdtemp()) / "preprocessing cache"


def run() -> pd.DataFrame:
    n = 70
    idx = pd.date_range("1990-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(0)
    panel = pd.DataFrame({"x": rng.normal(size=n)}, index=idx)
    panel["y"] = 0.5 * panel["x"] + rng.normal(size=n) * 0.3
    bundle = mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})
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
        arms=[Arm("AR", model="ar", is_benchmark=True)],
        evaluation=EvalSpec(benchmark="AR", metrics=("rmse",), tests=()),
        preprocessing_cache_dir=str(CACHE),
        save_models=False,
    )
    return run_pipeline(spec).forecasts.sort_values("date").reset_index(drop=True)


def main() -> None:
    cold = run()
    warm = run()
    assert not cold.empty, "the cold run produced no forecasts"
    np.testing.assert_allclose(
        cold["prediction"].to_numpy(float),
        warm["prediction"].to_numpy(float),
        rtol=0,
        atol=1e-12,
        err_msg="a warm on-disk cache changed the answer on this filesystem",
    )
    print(f"cold == warm through {CACHE}")


if __name__ == "__main__":
    main()
