from __future__ import annotations

from collections.abc import Mapping
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastResult:
    """Forecast runner output."""

    forecasts: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Return a copy of the forecast table."""

        return self.forecasts.copy()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready forecast result."""

        return {
            "forecasts": _json_ready(self.forecasts.to_dict(orient="records")),
            "metadata": _json_ready(self.metadata),
        }

    def evaluate(self, **kwargs: Any) -> pd.DataFrame:
        """Evaluate this forecast result with ``macroforecast.metrics``."""

        from macroforecast.metrics import evaluate_forecasts

        return evaluate_forecasts(self, **kwargs)

    def to_json(self, path: str | Path | None = None, *, indent: int | None = 2) -> str:
        """Return JSON text, and optionally write it to ``path``."""

        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text


def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, pd.Series):
        return {
            "name": value.name,
            "index": [_json_ready(item) for item in value.index],
            "data": [_json_ready(item) for item in value.to_list()],
        }
    if isinstance(value, pd.DataFrame):
        return {
            "columns": [str(column) for column in value.columns],
            "index": [_json_ready(item) for item in value.index],
            "data": _json_ready(value.to_dict(orient="list")),
        }
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


__all__ = ["ForecastResult"]
