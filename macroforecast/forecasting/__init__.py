from __future__ import annotations

from macroforecast.forecasting.combination import (
    CombinationSpec,
    combine_best_n,
    combine_dmspe,
    combine_inverse_mspe,
    combine_mean,
    combine_median,
    combine_trimmed_mean,
    combine_winsorized_mean,
    combination_spec,
    custom_combination,
)
from macroforecast.forecasting.runner import run
from macroforecast.forecasting.types import ForecastResult

run_forecast = run

__all__ = [
    "ForecastResult",
    "CombinationSpec",
    "combine_best_n",
    "combine_dmspe",
    "combine_inverse_mspe",
    "combine_mean",
    "combine_median",
    "combine_trimmed_mean",
    "combine_winsorized_mean",
    "combination_spec",
    "custom_combination",
    "run",
    "run_forecast",
]
