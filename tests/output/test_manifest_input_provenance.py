"""A replication manifest has to identify the inputs, not only the machine.

`collect_provenance()` answers "which build, which platform, which commit". A
replication package also has to answer "from what" -- and until now the manifest
did not, even though `run_pipeline` already computes a full-content SHA-256 of
the panel and a spec echo and puts both on `PipelineReport.provenance`.

The point is not that a hash is written down. It is that the hash **separates**:
a manifest whose data fingerprint is identical for two different panels records
nothing useful. So the tests below check discrimination, not presence.

Issue #447.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline

N = 50


def _report(seed: int = 0, n: int = N):
    idx = pd.date_range("1990-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(seed)
    panel = pd.DataFrame({"x": rng.normal(size=n)}, index=idx)
    panel["y"] = panel["x"] * 2.0 + rng.normal(size=n) * 0.2
    bundle = mf.data.custom_dataset(
        panel, transform_codes={c: 1 for c in panel.columns}
    )
    spec = pipeline_spec(
        data=bundle,
        targets=[TargetSpec("y", transform="level")],
        horizons=[1],
        window=mf.window.from_cutoffs(
            test_start=idx[n - 10], horizon=1, embargo=0,
            val_method="expanding", val_min_train_size=10,
        ),
        arms=[Arm("AR", model="ar", is_benchmark=True)],
        evaluation=EvalSpec(benchmark="AR", metrics=("rmse",), tests=()),
        save_models=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_pipeline(spec)


def _manifest(report, *, carry: bool) -> dict:
    artifacts = {"forecasts": report.forecasts, "accuracy": report.accuracy}
    kwargs = {"run_provenance": report.provenance} if carry else {}
    manifest = mf.output.write_artifacts(artifacts, tempfile.mkdtemp(), **kwargs)
    return manifest.to_dict() if hasattr(manifest, "to_dict") else dict(manifest)


def test_without_run_provenance_the_manifest_is_unchanged() -> None:
    """Nothing is invented. No run handed in means no input block."""
    provenance = _manifest(_report(), carry=False)["provenance"]
    assert "macroforecast_version" in provenance, "environment block should still be there"
    assert "data" not in provenance
    assert "spec_echo" not in provenance


def test_with_run_provenance_the_manifest_identifies_the_input() -> None:
    provenance = _manifest(_report(), carry=True)["provenance"]
    data = provenance.get("data")
    assert data, "no data block in the manifest"
    for field in ("dataset", "n_rows", "n_columns", "start", "end", "fingerprint"):
        assert field in data, f"data block is missing {field}"
    assert data["fingerprint"]["algorithm"] == "sha256"
    assert data["fingerprint"]["value"]
    assert "spec_echo" in provenance, "the specification is not identified"


def test_the_fingerprint_separates_two_different_panels() -> None:
    """The whole value of recording it.

    Same shape, same columns, same date range -- only the values differ. A
    fingerprint that did not move here would let a package be reproduced against
    the wrong data and still look verified.
    """
    a = _manifest(_report(seed=0), carry=True)["provenance"]["data"]
    b = _manifest(_report(seed=99), carry=True)["provenance"]["data"]
    assert [a["n_rows"], a["n_columns"]] == [b["n_rows"], b["n_columns"]], (
        "this test is only meaningful when the two panels have the same shape"
    )
    assert a["fingerprint"]["value"] != b["fingerprint"]["value"], (
        "two different panels produced the same fingerprint"
    )


def test_the_fingerprint_is_stable_for_the_same_panel() -> None:
    """The other half: it must not move for no reason."""
    a = _manifest(_report(seed=0), carry=True)["provenance"]["data"]
    b = _manifest(_report(seed=0), carry=True)["provenance"]["data"]
    assert a["fingerprint"]["value"] == b["fingerprint"]["value"]


def test_a_different_row_count_is_visible() -> None:
    a = _manifest(_report(seed=0, n=N), carry=True)["provenance"]["data"]
    b = _manifest(_report(seed=0, n=N + 10), carry=True)["provenance"]["data"]
    assert a["n_rows"] != b["n_rows"]
    assert a["fingerprint"]["value"] != b["fingerprint"]["value"]


def test_environment_provenance_is_not_overwritten() -> None:
    """The input block is additive; it must not displace what was already there."""
    report = _report()
    plain = _manifest(report, carry=False)["provenance"]
    carried = _manifest(report, carry=True)["provenance"]
    for key, value in plain.items():
        assert carried.get(key) == value, f"{key} changed when input provenance was added"


def test_the_manifest_stays_json_serializable() -> None:
    """It is written to disk as JSON; a non-serializable value would fail there,
    not here, which is a bad place to find out."""
    import json

    provenance = _manifest(_report(), carry=True)["provenance"]
    json.dumps(provenance)  # must not raise
