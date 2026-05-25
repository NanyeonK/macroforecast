from __future__ import annotations

import hashlib
from typing import Protocol

import pandas as pd


class LeakError(RuntimeError):
    """Raised when separation_rule invariants are violated."""


class _Preprocessor(Protocol):
    def fit(self, X): ...
    def transform(self, X): ...


_SUPPORTED = {
    'strict_separation',
    'shared_transform_then_split',
    'joint_preprocessor',
    'target_only_transform',
    'X_only_transform',
}


def _hash_frame(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def apply_separation_rule(
    *,
    rule: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor=None,
):
    if rule not in _SUPPORTED:
        raise ValueError(f'unsupported separation_rule={rule!r}')

    if rule == 'strict_separation':
        if preprocessor is None:
            return X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
        train_hash_before = _hash_frame(X_train)
        preprocessor.fit(X_train)
        sentinel_state_a = repr(getattr(preprocessor, '__dict__', {}))
        preprocessor.fit(X_train)
        sentinel_state_b = repr(getattr(preprocessor, '__dict__', {}))
        if sentinel_state_a != sentinel_state_b:
            raise LeakError('preprocessor fit is non-deterministic on X_train alone')
        if _hash_frame(X_train) != train_hash_before:
            raise LeakError('strict_separation: X_train mutated during fit')
        Xtr = pd.DataFrame(preprocessor.transform(X_train), index=X_train.index, columns=X_train.columns)
        Xte = pd.DataFrame(preprocessor.transform(X_test), index=X_test.index, columns=X_test.columns)
        return Xtr, Xte, y_train.copy(), y_test.copy()

    if rule == 'shared_transform_then_split':
        if preprocessor is None:
            return X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
        joint = pd.concat([X_train, X_test])
        preprocessor.fit(joint)
        Xtr = pd.DataFrame(preprocessor.transform(X_train), index=X_train.index, columns=X_train.columns)
        Xte = pd.DataFrame(preprocessor.transform(X_test), index=X_test.index, columns=X_test.columns)
        return Xtr, Xte, y_train.copy(), y_test.copy()

    if rule == 'joint_preprocessor':
        if preprocessor is None:
            raise LeakError('joint_preprocessor requires user-supplied pipeline')
        joint_X = pd.concat([X_train, X_test])
        try:
            varnames = preprocessor.fit.__code__.co_varnames
        except AttributeError:
            varnames = ()
        if 'y' in varnames:
            joint_y = pd.concat([y_train, y_test])
            preprocessor.fit(joint_X, joint_y)
        else:
            preprocessor.fit(joint_X)
        Xtr = pd.DataFrame(preprocessor.transform(X_train), index=X_train.index, columns=X_train.columns)
        Xte = pd.DataFrame(preprocessor.transform(X_test), index=X_test.index, columns=X_test.columns)
        return Xtr, Xte, y_train.copy(), y_test.copy()

    if rule == 'target_only_transform':
        if preprocessor is None:
            return X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
        preprocessor.fit(y_train.to_frame())
        ytr = pd.Series(preprocessor.transform(y_train.to_frame()).ravel(), index=y_train.index, name=y_train.name)
        yte = pd.Series(preprocessor.transform(y_test.to_frame()).ravel(), index=y_test.index, name=y_test.name)
        return X_train.copy(), X_test.copy(), ytr, yte

    if rule == 'X_only_transform':
        if preprocessor is None:
            return X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
        preprocessor.fit(X_train)
        Xtr = pd.DataFrame(preprocessor.transform(X_train), index=X_train.index, columns=X_train.columns)
        Xte = pd.DataFrame(preprocessor.transform(X_test), index=X_test.index, columns=X_test.columns)
        return Xtr, Xte, y_train.copy(), y_test.copy()

    raise AssertionError('unreachable')
