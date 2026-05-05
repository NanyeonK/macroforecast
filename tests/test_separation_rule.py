from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from macroforecast.preprocessing.separation import LeakError, apply_separation_rule


@pytest.fixture
def xy():
    rng = np.random.default_rng(0)
    Xtr = pd.DataFrame(rng.normal(size=(20, 3)), columns=['a', 'b', 'c'])
    Xte = pd.DataFrame(rng.normal(size=(8, 3)), columns=['a', 'b', 'c'])
    ytr = pd.Series(rng.normal(size=20), name='y')
    yte = pd.Series(rng.normal(size=8), name='y')
    return Xtr, Xte, ytr, yte


@pytest.mark.parametrize('rule', [
    'strict_separation',
    'shared_transform_then_split',
    'X_only_transform',
    'target_only_transform',
])
def test_modes_callable_without_preprocessor(xy, rule):
    Xtr, Xte, ytr, yte = xy
    out = apply_separation_rule(rule=rule, X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte, preprocessor=None)
    Xtr2, Xte2, ytr2, yte2 = out
    assert Xtr2.shape == Xtr.shape
    assert Xte2.shape == Xte.shape
    assert ytr2.shape == ytr.shape
    assert yte2.shape == yte.shape


def test_strict_separation_with_scaler_no_leak(xy):
    Xtr, Xte, ytr, yte = xy
    sc = StandardScaler()
    Xtr2, Xte2, _, _ = apply_separation_rule(
        rule='strict_separation', X_train=Xtr, X_test=Xte,
        y_train=ytr, y_test=yte, preprocessor=sc,
    )
    sc_ref = StandardScaler().fit(Xtr)
    assert np.allclose(sc.mean_, sc_ref.mean_)
    assert np.allclose(sc.scale_, sc_ref.scale_)
    assert abs(Xtr2.mean().mean()) < 1e-9


def test_shared_transform_then_split_uses_combined_X(xy):
    Xtr, Xte, ytr, yte = xy
    sc = StandardScaler()
    apply_separation_rule(
        rule='shared_transform_then_split', X_train=Xtr, X_test=Xte,
        y_train=ytr, y_test=yte, preprocessor=sc,
    )
    sc_ref = StandardScaler().fit(pd.concat([Xtr, Xte]))
    assert np.allclose(sc.mean_, sc_ref.mean_)


def test_target_only_transform_leaves_X_unchanged(xy):
    Xtr, Xte, ytr, yte = xy
    sc = StandardScaler()
    Xtr2, Xte2, ytr2, yte2 = apply_separation_rule(
        rule='target_only_transform', X_train=Xtr, X_test=Xte,
        y_train=ytr, y_test=yte, preprocessor=sc,
    )
    pd.testing.assert_frame_equal(Xtr2, Xtr)
    pd.testing.assert_frame_equal(Xte2, Xte)
    assert abs(ytr2.mean()) < 1e-9


def test_X_only_transform_leaves_y_unchanged(xy):
    Xtr, Xte, ytr, yte = xy
    sc = StandardScaler()
    Xtr2, Xte2, ytr2, yte2 = apply_separation_rule(
        rule='X_only_transform', X_train=Xtr, X_test=Xte,
        y_train=ytr, y_test=yte, preprocessor=sc,
    )
    pd.testing.assert_series_equal(ytr2, ytr)
    pd.testing.assert_series_equal(yte2, yte)


def test_joint_preprocessor_requires_pipeline(xy):
    Xtr, Xte, ytr, yte = xy
    with pytest.raises(LeakError):
        apply_separation_rule(
            rule='joint_preprocessor', X_train=Xtr, X_test=Xte,
            y_train=ytr, y_test=yte, preprocessor=None,
        )


def test_joint_preprocessor_with_scaler(xy):
    Xtr, Xte, ytr, yte = xy
    sc = StandardScaler()
    out = apply_separation_rule(
        rule='joint_preprocessor', X_train=Xtr, X_test=Xte,
        y_train=ytr, y_test=yte, preprocessor=sc,
    )
    Xtr2, Xte2, ytr2, yte2 = out
    assert Xtr2.shape == Xtr.shape
    assert Xte2.shape == Xte.shape


def test_invalid_rule_raises():
    Xtr = pd.DataFrame({'a': [1.0, 2.0]})
    Xte = pd.DataFrame({'a': [3.0]})
    ytr = pd.Series([0.1, 0.2], name='y')
    yte = pd.Series([0.3], name='y')
    with pytest.raises(ValueError):
        apply_separation_rule(
            rule='nonexistent_rule', X_train=Xtr, X_test=Xte,
            y_train=ytr, y_test=yte, preprocessor=None,
        )


def test_output_index_alignment_preserved(xy):
    Xtr, Xte, ytr, yte = xy
    sc = StandardScaler()
    Xtr2, Xte2, _, _ = apply_separation_rule(
        rule='strict_separation', X_train=Xtr, X_test=Xte,
        y_train=ytr, y_test=yte, preprocessor=sc,
    )
    assert list(Xtr2.index) == list(Xtr.index)
    assert list(Xte2.index) == list(Xte.index)
    assert list(Xtr2.columns) == list(Xtr.columns)
