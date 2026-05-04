"""Tests for v0.25 #257 (L3 cascade enforcement) + #259 (HAC kernels)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macrocast.core.runtime import _diebold_mariano_test, _long_run_variance


# ---------------------------------------------------------------------------
# #259 HAC kernels
# ---------------------------------------------------------------------------

def test_long_run_variance_three_kernels_run_on_clean_series():
    rng = np.random.default_rng(0)
    series = rng.normal(size=200)
    nw = _long_run_variance(series, kernel="newey_west", lag=4)
    andrews = _long_run_variance(series, kernel="andrews")
    parzen = _long_run_variance(series, kernel="parzen", lag=4)
    for variance in (nw, andrews, parzen):
        assert variance > 0
        assert np.isfinite(variance)


def test_long_run_variance_andrews_picks_data_driven_lag():
    """Strongly autocorrelated series -> Andrews picks a larger bandwidth
    than naive Newey-West rule of thumb -> larger long-run variance."""

    rng = np.random.default_rng(0)
    n = 200
    raw = rng.normal(size=n)
    autocorrelated = np.zeros(n)
    autocorrelated[0] = raw[0]
    for i in range(1, n):
        autocorrelated[i] = 0.8 * autocorrelated[i - 1] + raw[i]
    nw = _long_run_variance(autocorrelated, kernel="newey_west", lag=2)
    andrews = _long_run_variance(autocorrelated, kernel="andrews")
    assert andrews > nw  # data-driven bandwidth captures more autocorrelation


def test_long_run_variance_rejects_unknown_kernel():
    with pytest.raises(ValueError, match="unknown HAC kernel"):
        _long_run_variance(np.ones(10), kernel="totally_made_up")


def test_diebold_mariano_kernel_choice_changes_p_value():
    rng = np.random.default_rng(0)
    n = 100
    diff = pd.Series(rng.normal(size=n))
    diff_autocor = diff.copy()
    for i in range(1, n):
        diff_autocor.iloc[i] += 0.6 * diff_autocor.iloc[i - 1]
    stat_nw, p_nw = _diebold_mariano_test(diff_autocor, horizon=4, kernel="newey_west")
    stat_p, p_p = _diebold_mariano_test(diff_autocor, horizon=4, kernel="parzen")
    # Different kernel -> different statistic.
    assert stat_nw != stat_p
    assert isinstance(p_nw, float) and isinstance(p_p, float)


# ---------------------------------------------------------------------------
# #257 L3 cascade β -- depth enforcement
# ---------------------------------------------------------------------------

def test_l3_cascade_max_depth_unit_level():
    """Issue #257 -- ``_execute_l3_dag`` raises with a clear message when
    the cascade depth exceeds the configured cap. Unit-level test so we
    don't fight the layer validator.
    """

    from macrocast.core.runtime import _execute_l3_dag
    from types import SimpleNamespace

    n = 12

    # Build a synthetic DAG: 1 source + n step nodes chained linearly.
    class _Ref:
        def __init__(self, node_id):
            self.node_id = node_id

    class _Node:
        def __init__(self, node_id, node_type, op, inputs, params=None):
            self.id = node_id
            self.type = node_type
            self.op = op
            self.inputs = inputs
            self.params = params or {}
            self.selector = SimpleNamespace(
                layer_ref="l2",
                sink_name="l2_clean_panel_v1",
                subset={"role": "predictors"},
            )

    nodes_dict: dict[str, _Node] = {
        "src": _Node("src", "source", None, []),
    }
    prev = "src"
    for i in range(n):
        nid = f"step_{i}"
        nodes_dict[nid] = _Node(nid, "step", "identity", [_Ref(prev)])
        prev = nid

    dag = SimpleNamespace(
        nodes=nodes_dict,
        layer_id="l3",
        leaf_config={"cascade_max_depth": 5},
    )
    frame = pd.DataFrame({"y": np.arange(10.0), "x1": np.arange(10.0)})

    with pytest.raises(ValueError, match="cascade_max_depth"):
        _execute_l3_dag(dag, frame, target_name="y")


def test_l3_cascade_pipeline_id_propagates_through_steps():
    """Issue #257 -- pipeline_id stamped on a deep step is inherited by
    its descendants when they don't override it."""

    from macrocast.core.runtime import _execute_l3_dag
    from types import SimpleNamespace

    class _Ref:
        def __init__(self, node_id):
            self.node_id = node_id

    class _Node:
        def __init__(self, node_id, node_type, op, inputs, params=None):
            self.id = node_id
            self.type = node_type
            self.op = op
            self.inputs = inputs
            self.params = params or {}
            self.selector = SimpleNamespace(
                layer_ref="l2", sink_name="l2_clean_panel_v1", subset={"role": "predictors"}
            )

    nodes_dict = {
        "src": _Node("src", "source", None, []),
        "stamped": _Node("stamped", "step", "identity", [_Ref("src")], params={"pipeline_id": "marx_block"}),
        "child": _Node("child", "step", "identity", [_Ref("stamped")]),
    }
    dag = SimpleNamespace(nodes=nodes_dict, layer_id="l3", leaf_config={})
    frame = pd.DataFrame({"y": np.arange(8.0), "x1": np.arange(8.0)})
    _execute_l3_dag(dag, frame, target_name="y")
    assert dag.runtime_pipeline_by_node["stamped"] == "marx_block"
    assert dag.runtime_pipeline_by_node["child"] == "marx_block"
