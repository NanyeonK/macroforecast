"""The MCS fixed-block bootstrap must be able to draw the last block.

Regression for an off-by-one: the block start was drawn from
``rng.integers(0, n - block_length)`` (exclusive upper bound), so the last valid
start ``n - block_length`` was never sampled and the final observation never
appeared in any resample. R's MCS GetIndices draws 1..(T - block + 1), i.e.
0..(n - block_length) inclusive.
"""
import numpy as np

from macroforecast.tests import _mcs_bootstrap_indices


def test_last_observation_is_reachable():
    n, block_length = 20, 4
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(2000):
        seen.update(_mcs_bootstrap_indices(n, block_length, method="mcs_fixed_block", rng=rng).tolist())
    # Every observation, including the last, must be reachable.
    assert seen == set(range(n)), f"unreachable indices: {sorted(set(range(n)) - seen)}"


def test_block_starts_cover_the_full_range():
    # The block starting at n - block_length covers the tail; it must be drawn.
    n, block_length = 12, 3
    rng = np.random.default_rng(1)
    last_index_hits = 0
    for _ in range(1000):
        idx = _mcs_bootstrap_indices(n, block_length, method="mcs_fixed_block", rng=rng)
        if (idx == n - 1).any():
            last_index_hits += 1
    assert last_index_hits > 0
