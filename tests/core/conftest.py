from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macroforecast.core import ModelArtifact, Panel, Series


@pytest.fixture
def mock_fred_md() -> Panel:
    index = pd.date_range("2000-01-01", periods=60, freq="MS")
    return Panel(
        shape=(60, 5),
        column_names=("INDPRO", "CPIAUCSL", "UNRATE", "PAYEMS", "FEDFUNDS"),
        index=index,
    )


@pytest.fixture
def mock_fred_qd() -> Panel:
    index = pd.date_range("2000-01-01", periods=40, freq="QS")
    return Panel(
        shape=(40, 4),
        column_names=("GDPC1", "PCECC96", "INDPRO", "PAYEMS"),
        index=index,
    )


@pytest.fixture
def mock_clean_panel(mock_fred_md: Panel) -> Panel:
    return mock_fred_md


@pytest.fixture
def mock_target_series() -> Series:
    return Series(shape=(60,), name="INDPRO")


@pytest.fixture
def mock_model_artifact_xgboost() -> ModelArtifact:
    return ModelArtifact(
        model_id="xgb_full",
        family="xgboost",
        fitted_object={"booster": "mock"},
        framework="xgboost",
        fit_metadata={"n_obs": 60},
        feature_names=("CPIAUCSL", "UNRATE"),
    )
