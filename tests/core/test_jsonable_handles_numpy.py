"""Cycle 14 L3 regression test: _jsonable must handle numpy scalars and arrays.

Prior to the L3 fix, numpy types raised RepresenterError when passed through
yaml.safe_dump (activated by the L2-4 output_directory manifest write path).
"""
import numpy as np
from macroforecast.core.runtime import _jsonable


def test_jsonable_handles_np_generic():
    assert _jsonable(np.float64(0.5)) == 0.5
    assert _jsonable(np.int64(42)) == 42
    assert _jsonable(np.bool_(True)) is True


def test_jsonable_handles_np_ndarray():
    arr = np.array([1.0, 2.0, 3.0])
    assert _jsonable(arr) == [1.0, 2.0, 3.0]
