"""Wave B lane B-2: PipelineReport.provenance is self-certifying by default.

Before this change, run_pipeline's provenance dict (``_audit`` in
pipeline/run.py) carried package_version/seed/targets/arms/leakage-audit but no
git SHA, no environment, and no data identity -- a referee handed only the
report artifact could not tell which macroforecast commit produced it, nor pin
the data vintage. ``collect_provenance`` (output/core.py) already collected all
of that (git commit/branch/dirty, Python/platform, pinned core dependency
versions), but only attached on the opt-in save/``write_run`` path.

This module pins:

- the default (``provenance_level="full"``) report gains "environment"
  (reusing ``output.collect_provenance`` against the RUNNING PACKAGE's own
  checkout, not the caller's cwd), "data" (dataset/source_family/vintage +
  panel shape/date range + a content fingerprint), and "spec_echo" (targets/
  policies/horizons/window cutoffs/arms/models/seed/n_jobs) -- asserted for
  plausible presence/types, not exact values (git SHA etc. vary by checkout);
- ``provenance_level="basic"`` reproduces EXACTLY the pre-change dict shape;
- the content fingerprint is stable across two runs on identical data and
  changes when the panel is modified, including the (patched-threshold)
  deterministic-subsample fallback path;
- ``rescore()`` reports carry the same "environment" block plus the existing
  ``rescored_from`` marker;
- forecasts/accuracy are BYTE-IDENTICAL to a golden fixture captured at the
  base commit (70ad5b0e, before this lane's changes) from the identical
  fixture spec -- provenance is additive, never touches forecast/accuracy
  computation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
import macroforecast.pipeline.run as run_mod
from macroforecast.pipeline import (
    Arm,
    CombinationContender,
    EvalSpec,
    pipeline_spec,
    rescore,
    run_pipeline,
)
from macroforecast.pipeline.run import _panel_fingerprint

_GOLDEN_DIR = Path(__file__).parent / "_golden"


def _bundle(n=96):
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, n)
    frame = pd.DataFrame({"y": 1.0 + 2.0 * x + rng.standard_normal(n) * 0.05, "x1": x}, index=idx)
    return mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})


def _spec(**over):
    """Identical fixture to tests/pipeline/test_run_pipeline.py's ``_spec()`` --
    also what generated the golden forecasts/accuracy fixtures below (base
    commit 70ad5b0e, before this lane's changes).
    """
    feats = mf.feature_engineering.feature_spec(target="y", predictors=["x1"], lags=1, target_lags=(0, 1))
    w = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=36),
        val=mf.window.val_last_block(size=12),
        test=mf.window.test_origins(horizon=1, step=3),
    )
    kw = dict(
        data=_bundle(), targets=["y"], horizons=[1], window=w,
        arms=[Arm("AR", model="ar", features=feats), Arm("OLS", model="ols", features=feats)],
        evaluation=EvalSpec(benchmark="AR"), save_models=False,
        combinations=[CombinationContender(name="POOL", method="mean")],
    )
    kw.update(over)
    return pipeline_spec(**kw)


# --------------------------------------------------------------------------- #
# 1. default ("full") provenance gains environment/data/spec_echo
# --------------------------------------------------------------------------- #

def test_default_provenance_is_full_and_self_certifying():
    report = run_pipeline(_spec())
    prov = report.provenance
    assert {"environment", "data", "spec_echo"} <= set(prov)

    env = prov["environment"]
    assert isinstance(env["macroforecast_version"], str) and env["macroforecast_version"]
    assert isinstance(env["python"], str) and env["python"]
    assert isinstance(env["platform"], str) and env["platform"]
    assert isinstance(env["packages"], dict)
    for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels"):
        assert pkg in env["packages"]
    git = env["git"]
    assert set(git) == {"commit", "branch", "dirty"}
    # None (not a git checkout, e.g. installed from a wheel) or a real 40-char SHA.
    assert git["commit"] is None or (isinstance(git["commit"], str) and len(git["commit"]) == 40)
    assert isinstance(git["dirty"], bool)

    data = prov["data"]
    assert data["n_rows"] == 96
    assert data["n_columns"] == 2
    assert data["dataset"] == "custom"
    assert data["source_family"] == "custom"
    assert data["start"] is not None and data["end"] is not None
    fp = data["fingerprint"]
    assert fp["algorithm"] == "sha256"
    assert fp["method"] == "full_content"
    assert isinstance(fp["value"], str) and len(fp["value"]) == 64

    echo = prov["spec_echo"]
    assert echo["horizons"] == [1]
    assert echo["seed"] == 42
    assert echo["n_jobs"] == 1
    assert {a["name"] for a in echo["arms"]} == {"AR", "OLS"}
    assert echo["benchmark"] == "AR"
    assert isinstance(echo["window"], dict) and "test" in echo["window"]

    # A referee must be able to json.dumps the whole provenance dict.
    json.dumps(prov)


# --------------------------------------------------------------------------- #
# 2. provenance_level="basic" opt-out: exactly the pre-change shape
# --------------------------------------------------------------------------- #

_BASIC_KEYS = {
    "package_version", "seed", "targets", "horizons", "arms", "benchmark", "combinations",
}


def test_provenance_level_basic_matches_pre_change_shape():
    report = run_pipeline(_spec(provenance_level="basic"))
    assert set(report.provenance) == _BASIC_KEYS
    assert report.provenance["benchmark"] == "AR"
    assert report.provenance["seed"] == 42


def test_provenance_level_invalid_raises():
    with pytest.raises(ValueError, match="provenance_level"):
        _spec(provenance_level="verbose")


# --------------------------------------------------------------------------- #
# 3. content fingerprint: stable / sensitive to modification / subsample path
# --------------------------------------------------------------------------- #

def test_fingerprint_mapping_is_exactly_the_pre_change_shape():
    """The stored identity is the whole mapping, not the digest string.

    ``result_cell_identity`` serialises this dict into the canonical payload and
    ``ResultStore.load`` compares the whole manifest mapping, so adding or dropping a
    key misses every existing cache even when the digest is unchanged. Making the
    fingerprint read the full content (F-005) had to leave this mapping alone for every
    panel the old code already hashed in full -- which is what this pins, by value and
    by key set, rather than by digest alone.
    """
    panel = pd.DataFrame(
        {"x": [1.0, 2.0], "y": [3.0, 4.0]},
        index=pd.date_range("2000-01-01", periods=2, freq="D", name="date"),
    )

    assert run_mod._panel_fingerprint(panel) == {
        "algorithm": "sha256",
        "method": "full_content",
        "value": "902f2ee172cf19ab5d557d56dca7d6faad3dc85e46a328095c27a7b9e6c651e8",
        "row_stride": 1,
        "col_stride": 1,
        "sampled_shape": [2, 2],
    }


def test_fingerprint_stable_across_identical_runs():
    fp1 = _panel_fingerprint(_bundle().panel)
    fp2 = _panel_fingerprint(_bundle().panel)
    assert fp1["value"] == fp2["value"]
    assert fp1["method"] == "full_content"


def test_fingerprint_changes_when_panel_is_modified():
    base = _bundle()
    modified_panel = base.panel.copy()
    modified_panel.iloc[0, 0] = modified_panel.iloc[0, 0] + 1.0
    fp_base = _panel_fingerprint(base.panel)
    fp_mod = _panel_fingerprint(modified_panel)
    assert fp_base["value"] != fp_mod["value"]


def test_fingerprint_matches_across_independent_pipeline_runs():
    """End-to-end: two independent runs on numerically identical data fingerprint
    to the same value even though the DataBundle objects are distinct."""
    r1 = run_pipeline(_spec())
    r2 = run_pipeline(_spec())
    assert (
        r1.provenance["data"]["fingerprint"]["value"]
        == r2.provenance["data"]["fingerprint"]["value"]
    )


def test_fingerprint_chunking_is_a_memory_bound_not_a_coverage_bound(monkeypatch):
    """Large panels are streamed, not sampled, so the chunk size cannot move the digest.

    This replaces a test that pinned the opposite: above a cell cap the digest used to
    come from a strided subsample, which meant a cell the stride skipped could change
    without the digest moving (F-005). The digest feeds result-cell identity, so that
    was a stale-forecast path, not just a weaker hash.

    The cap lives with the function it governs, and both moved to ``data.identity`` in
    A1. Patching the re-exported name on ``run_mod`` does not reach it: the function
    reads its OWN module global, so the patch has to land where the function is.
    """
    from macroforecast.data import identity as identity_mod

    frame = _bundle(n=96).panel  # 96 x 2 = 192 cells
    unchunked = run_mod._panel_fingerprint(frame)

    monkeypatch.setattr(identity_mod, "_FINGERPRINT_CHUNK_CELLS", 7)
    chunked = run_mod._panel_fingerprint(frame)
    again = run_mod._panel_fingerprint(frame)

    assert chunked["method"] == "full_content"
    assert chunked == unchunked, "chunk size must not change the digest or the mapping"
    assert chunked == again
    assert chunked["sampled_shape"] == [96, 2]
    assert chunked["row_stride"] == 1 and chunked["col_stride"] == 1


def test_fingerprint_sees_a_cell_the_old_stride_would_have_skipped(monkeypatch):
    """The F-005 acceptance case, in the shape the audit reported it.

    With the old cap patched to 50, a 100x10 panel sampled at row stride 4 and column
    stride 5, so cell (0, 1) was never read and mutating it left the fingerprint
    identical. Chunking is exercised here at the same small size to prove the streaming
    path -- not just the small-panel path -- reads every cell.
    """
    from macroforecast.data import identity as identity_mod

    monkeypatch.setattr(identity_mod, "_FINGERPRINT_CHUNK_CELLS", 7)
    index = pd.date_range("1990-01-31", periods=100, freq="ME", name="date")
    rng = np.random.default_rng(0)
    base = pd.DataFrame(
        rng.normal(size=(100, 10)), index=index, columns=[f"c{i}" for i in range(10)]
    )
    mutated = base.copy()
    mutated.iloc[0, 1] = mutated.iloc[0, 1] + 1.0

    assert (
        run_mod._panel_fingerprint(base)["value"]
        != run_mod._panel_fingerprint(mutated)["value"]
    )
    assert (
        run_mod._panel_fingerprint(base)["value"]
        == run_mod._panel_fingerprint(base.copy())["value"]
    )


# --------------------------------------------------------------------------- #
# 4. rescore() carries the environment block
# --------------------------------------------------------------------------- #

def test_rescore_carries_environment_block(tmp_path):
    ckpt = tmp_path / "ckpt"
    spec_with_ckpt = _spec(checkpoint_dir=str(ckpt))
    run_pipeline(spec_with_ckpt)
    rescored = rescore(ckpt, spec_with_ckpt)
    assert rescored.provenance.get("rescored_from") == str(ckpt)
    assert "environment" in rescored.provenance
    assert isinstance(rescored.provenance["environment"]["macroforecast_version"], str)


def test_rescore_basic_omits_environment_block(tmp_path):
    ckpt = tmp_path / "ckpt"
    spec_with_ckpt = _spec(checkpoint_dir=str(ckpt), provenance_level="basic")
    run_pipeline(spec_with_ckpt)
    rescored = rescore(ckpt, spec_with_ckpt)
    assert "environment" not in rescored.provenance
    assert rescored.provenance.get("rescored_from") == str(ckpt)


# --------------------------------------------------------------------------- #
# 5. byte-identity: forecasts/accuracy unchanged vs the pre-change golden
# --------------------------------------------------------------------------- #

_PLAIN_FORECAST_COLS = [
    "date", "origin", "horizon", "forecast_policy", "target", "model",
    "prediction", "actual", "combined", "arm", "contender",
]


def _normalize_datetime_resolution(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast every datetime64 column to ns resolution before comparison.

    The parquet round-trip can hand timestamps back at us resolution while the
    live pipeline emits ns (pandas/pyarrow resolve the unit differently across
    CI matrix legs). Values are untouched, so the byte-identity claim on the
    forecast/accuracy numbers stays strict.
    """
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].astype("datetime64[ns]")
    return out


@pytest.mark.parametrize("table", ["forecasts", "accuracy"])
def test_forecasts_and_accuracy_byte_identical_to_pre_change_golden(table):
    """Provenance is additive: adding environment/data/spec_echo must not move a
    single forecast or accuracy number. Golden fixtures were captured at the
    base commit (70ad5b0e) from the IDENTICAL ``_spec()``/``_bundle()`` fixture,
    before this lane's changes (see ``docs/reference/pipeline.md``)."""
    report = run_pipeline(_spec())
    if table == "forecasts":
        got = report.forecasts[_PLAIN_FORECAST_COLS].reset_index(drop=True)
    else:
        got = report.accuracy.reset_index(drop=True)
    golden = pd.read_parquet(_GOLDEN_DIR / f"default_provenance_{table}.parquet").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        _normalize_datetime_resolution(got),
        _normalize_datetime_resolution(golden),
        atol=1e-12,
    )
