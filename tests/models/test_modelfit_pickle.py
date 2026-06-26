"""ModelFit must survive a pickle round-trip on every supported Python.

Regression guard for a Python 3.10 specific bug: ModelFit.__getattr__ delegated
attribute lookups to the wrapped estimator, including dunders. On 3.10 (where
``object`` has no default ``__getstate__`` / ``__setstate__``) that delegation
hijacked pickling -- the estimator's ``__getstate__`` ran instead of the
dataclass default, so an unpickled ModelFit carried the estimator's ``__dict__``
and lost its own fields. Saving and reloading a trained model then raised
``AttributeError: 'ModelFit' object has no attribute 'model'``. 3.11+ ships a
default ``object.__getstate__`` so the bug was invisible there.
"""
import pickle

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from macroforecast.models.types import ModelFit


def _fitted():
    est = Ridge().fit(np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 3.0]))
    return ModelFit(
        estimator=est,
        model="ridge",
        feature_names=("x",),
        target_name="y",
        metadata={"k": 1},
    )


def test_modelfit_pickle_preserves_its_own_fields():
    restored = pickle.loads(pickle.dumps(_fitted()))
    assert restored.model == "ridge"
    assert type(restored.estimator).__name__ == "Ridge"
    assert restored.feature_names == ("x",)
    assert restored.target_name == "y"
    assert restored.metadata == {"k": 1}


def test_modelfit_still_delegates_real_estimator_attributes():
    # Non-dunder attributes are still forwarded to the estimator after unpickling.
    restored = pickle.loads(pickle.dumps(_fitted()))
    assert hasattr(restored, "coef_")
    assert np.all(np.isfinite(np.asarray(restored.coef_, dtype=float)))


def test_modelfit_does_not_delegate_dunders_to_estimator():
    # A dunder that ModelFit/object does not define must raise AttributeError,
    # not resolve to the estimator's method; that delegation is what corrupts
    # pickling on 3.10.
    m = _fitted()
    with pytest.raises(AttributeError):
        m.__some_missing_dunder__
