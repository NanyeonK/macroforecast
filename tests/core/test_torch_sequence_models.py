"""Issue #198 -- pin the torch-missing behaviour for lstm/gru/transformer.

When ``torch`` is not installed, ``_TorchSequenceModel`` raises a clear
``NotImplementedError`` (no silent MLP fallback) at fit time *and* at
predict time. The ``[deep]`` extra installs torch and unblocks both
paths; those tests live behind the ``deep`` marker so the default lane
remains lightweight.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from macroforecast.core.runtime import _TorchSequenceModel


HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(HAS_TORCH, reason="torch is installed; covered by the deep marker")
class TestTorchMissing:
    def test_fit_raises_with_actionable_message(self):
        model = _TorchSequenceModel(kind="lstm", n_epochs=1, hidden_size=4)
        X = pd.DataFrame({"x1": np.arange(10.0)})
        y = pd.Series(np.arange(10.0))
        with pytest.raises(NotImplementedError, match="macroforecast\\[deep\\]"):
            model.fit(X, y)

    def test_predict_without_fit_raises(self):
        model = _TorchSequenceModel(kind="gru", n_epochs=1, hidden_size=4)
        X = pd.DataFrame({"x1": np.arange(5.0)})
        with pytest.raises(NotImplementedError, match="install macroforecast\\[deep\\]"):
            model.predict(X)


@pytest.mark.deep
@pytest.mark.skipif(not HAS_TORCH, reason="requires torch (install `macroforecast[deep]`)")
class TestTorchPresent:
    @pytest.mark.parametrize("kind", ["lstm", "gru"])
    def test_fit_predict_shape(self, kind):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(20, 3)), columns=["x1", "x2", "x3"])
        y = pd.Series(rng.normal(size=20))
        model = _TorchSequenceModel(kind=kind, n_epochs=2, hidden_size=4, random_state=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (20,)

    def test_seed_makes_fits_deterministic(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(20, 3)), columns=list("abc"))
        y = pd.Series(rng.normal(size=20))
        a = _TorchSequenceModel(kind="lstm", n_epochs=2, hidden_size=4, random_state=42)
        b = _TorchSequenceModel(kind="lstm", n_epochs=2, hidden_size=4, random_state=42)
        a.fit(X, y)
        b.fit(X, y)
        np.testing.assert_allclose(a.predict(X), b.predict(X), rtol=1e-5)
