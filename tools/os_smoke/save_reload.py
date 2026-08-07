"""A model pickled on this operating system must reload and predict identically."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import macroforecast as mf


def main() -> None:
    X = pd.DataFrame({"a": np.arange(40.0), "b": np.arange(40.0) ** 0.5})
    y = pd.Series(2.0 * X["a"] + X["b"], name="y")
    fit = mf.models.ols(X, y)

    root = Path(tempfile.mkdtemp()) / "model store"
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "ols.pkl"
    saved = mf.models.save_fit(fit, model_path)
    assert Path(saved.model_path).exists(), f"nothing was written: {saved}"

    reloaded = mf.models.load_fit(saved.model_path)
    np.testing.assert_allclose(
        np.asarray(fit.predict(X), dtype=float),
        np.asarray(reloaded.predict(X), dtype=float),
        rtol=0,
        atol=1e-12,
        err_msg="a reloaded model predicts differently on this OS",
    )
    print(f"save/load round-trip through {root}")


if __name__ == "__main__":
    main()
