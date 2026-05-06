"""v0.8.6 Gap 3 -- ``fred_md+fred_sd`` combined-dataset smoke test.

Locks in the L1 dispatch added in v0.2 / refined in v0.8.5: when the
caller passes ``dataset='fred_md+fred_sd'`` (or ``fred_qd+fred_sd``)
the loader fetches both the national FRED-MD/QD panel and the regional
FRED-SD panel and joins them on the date index, so L2 sees one merged
panel with both national series (e.g. ``INDPRO``) and state-level
series (e.g. ``UR_CA``, ``UR_TX``).

The test pre-populates the raw cache with the small CSV fixtures already
checked in under ``tests/fixtures/`` so no network access is required.
Marked ``slow`` to mirror the rest of the integration suite -- gated
in the default fast matrix via ``pytest -m slow``.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.core.execution import execute_recipe


_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures"
_MD_FIXTURE = _FIXTURE_DIR / "fred_md_sample.csv"
_SD_FIXTURE = _FIXTURE_DIR / "fred_sd_sample.csv"

pytestmark = pytest.mark.slow


def _seed_cache(tmp_path: Path) -> Path:
    """Pre-populate the raw cache with the FRED-MD + FRED-SD fixtures.

    The L1 dispatch invokes ``load_fred_md(cache_root=...)`` and
    ``load_fred_sd(cache_root=...)``; ``get_raw_file_path`` resolves to
    ``<cache_root>/fred_md/current/raw.csv`` and
    ``<cache_root>/fred_sd/current/raw.csv``. By placing the fixtures
    at those slots before the loader runs we exercise the cache-hit
    branch and avoid any network call.
    """

    cache_root = tmp_path / "raw_cache"
    md_target = cache_root / "fred_md" / "current" / "raw.csv"
    sd_target = cache_root / "fred_sd" / "current" / "raw.csv"
    md_target.parent.mkdir(parents=True, exist_ok=True)
    sd_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_MD_FIXTURE, md_target)
    shutil.copyfile(_SD_FIXTURE, sd_target)
    return cache_root


def test_fred_md_plus_fred_sd_loader_dispatches_combined(tmp_path):
    """L1 must merge FRED-MD national + FRED-SD regional columns."""

    cache_root = _seed_cache(tmp_path)
    # Use the low-level loader path directly (bypasses the simple-API
    # builder so we verify the L1 runtime dispatch end-to-end).
    from macroforecast.core.runtime import _load_official_raw_result

    resolved = {"dataset": "fred_md+fred_sd"}
    leaf = {"cache_root": str(cache_root)}
    result = _load_official_raw_result(resolved, leaf)
    columns = set(result.data.columns)
    # FRED-MD fixture columns
    assert {"INDPRO", "RPI", "UNRATE", "CPIAUCSL"}.issubset(columns)
    # FRED-SD fixture columns (variable_state)
    assert {"BPPRIVSA_CA", "UR_CA", "BPPRIVSA_TX", "UR_TX"}.issubset(columns)


def test_fred_md_plus_fred_sd_l1_sink_carries_both_panels(tmp_path):
    """End-to-end (L1 + L2 only): the L1 sink must carry both
    fred_md columns and fred_sd state columns.

    We invoke ``materialize_l1`` directly so we sidestep the L4 fit
    minimum-observation requirement (the tiny fixtures only have 4
    rows). The point of this test is to confirm that
    ``Experiment(dataset='fred_md+fred_sd', ...)`` lowers to a recipe
    whose L1 block routes through the combined dispatch, not that the
    full pipeline fits a model on a 4-row toy panel.
    """

    cache_root = _seed_cache(tmp_path)

    exp = mf.Experiment(
        dataset="fred_md+fred_sd",
        target="INDPRO",
        horizons=[1],
        frequency="monthly",
        model_family="ridge",
    )
    # Force the cache_root so the loader reads our fixtures.
    exp._builder.l1.set_leaf(cache_root=str(cache_root))

    from macroforecast.core.runtime import materialize_l1

    recipe = exp.to_recipe_dict()
    l1_artifact, _regime, _axes = materialize_l1(recipe)
    columns = set(l1_artifact.raw_panel.data.columns)
    assert "INDPRO" in columns  # FRED-MD national
    assert {"UR_CA", "UR_TX"}.issubset(columns)  # FRED-SD regional


def test_fred_qd_plus_fred_sd_dispatches_combined(tmp_path):
    """The ``fred_qd+fred_sd`` route uses the same dispatch logic."""

    from macroforecast.core.runtime import _load_official_raw_result

    cache_root = tmp_path / "raw_cache"
    qd_target = cache_root / "fred_qd" / "current" / "raw.csv"
    sd_target = cache_root / "fred_sd" / "current" / "raw.csv"
    qd_target.parent.mkdir(parents=True, exist_ok=True)
    sd_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_FIXTURE_DIR / "fred_qd_sample.csv", qd_target)
    shutil.copyfile(_SD_FIXTURE, sd_target)

    resolved = {"dataset": "fred_qd+fred_sd"}
    leaf = {"cache_root": str(cache_root)}
    result = _load_official_raw_result(resolved, leaf)
    columns = set(result.data.columns)
    # FRED-QD fixture columns
    assert {"GDPC1", "CPIAUCSL", "FEDFUNDS"}.issubset(columns)
    # FRED-SD fixture columns
    assert {"UR_CA", "UR_TX"}.issubset(columns)
