from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class WindowSpec:
    minimum_train_rule: str = 'fixed_n_obs'
    minimum_train_value: int = 60
    break_rule: str = 'none'
    break_dates: tuple = field(default_factory=tuple)


_CRISIS_DATE = pd.Timestamp('2008-09-01')
_COVID_DATE = pd.Timestamp('2020-03-01')


def _resolve_min_train_obs(spec: WindowSpec, *, model_family, target, horizon: int) -> int:
    rule = spec.minimum_train_rule
    base = int(spec.minimum_train_value)
    if rule == 'fixed_n_obs':
        return base
    if rule == 'fixed_years':
        return base * 12
    if rule == 'model_specific_min_train':
        return max(base, {'linear_regression': 30, 'ridge': 60, 'lasso': 80}.get(model_family or '', base))
    if rule == 'target_specific_min_train':
        return max(base, {'INDPRO': 60, 'PAYEMS': 80, 'CPIAUCSL': 100}.get(target or '', base))
    if rule == 'horizon_specific_min_train':
        return base + max(0, horizon - 1) * 6
    raise ValueError(f'unsupported minimum_train_rule={rule!r}')


def compute_train_test_blocks(
    *,
    index: pd.DatetimeIndex,
    spec: WindowSpec,
    horizon: int,
    model_family=None,
    target=None,
) -> list:
    """Return list of (train_slice, test_slice) per break_rule policy.

    Each test_slice is the segment after train_slice within the same block.
    Blocks shorter than required minimum_train + horizon are dropped.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError('index must be DatetimeIndex')
    n = len(index)
    if n == 0:
        return []
    rule = spec.break_rule
    if rule == 'none':
        boundaries = []
    elif rule == 'pre_post_crisis':
        boundaries = [_CRISIS_DATE]
    elif rule == 'pre_post_covid':
        boundaries = [_COVID_DATE]
    elif rule == 'user_break_dates':
        boundaries = [pd.Timestamp(b) for b in spec.break_dates]
    elif rule == 'break_test_detected':
        boundaries = [index[n // 2]]
    elif rule == 'rolling_break_adaptive':
        boundaries = [index[n // 3], index[(2 * n) // 3]]
    else:
        raise ValueError(f'unsupported break_rule={rule!r}')

    edges = [0]
    for ts in boundaries:
        pos = int(index.searchsorted(ts))
        if 0 < pos < n and pos not in edges:
            edges.append(pos)
    edges.append(n)
    edges = sorted(set(edges))

    min_obs = _resolve_min_train_obs(spec, model_family=model_family, target=target, horizon=horizon)

    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        block_len = hi - lo
        if block_len < min_obs + horizon + 1:
            continue
        train_end = lo + min_obs
        out.append((slice(lo, train_end), slice(train_end, hi)))
    return out
