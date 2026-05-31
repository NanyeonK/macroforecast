from __future__ import annotations

from importlib import import_module

from .report import (
    BENCHMARK_METRICS,
    DEFAULT_METRICS,
    DEFAULT_SCORE_BY,
    EvaluationReport,
    aggregate_scores,
    benchmark_comparison,
    evaluate_report,
    regime_scores,
)

metrics = import_module("macroforecast.metrics")
tests = import_module("macroforecast.tests")

__all__ = [
    "BENCHMARK_METRICS",
    "DEFAULT_METRICS",
    "DEFAULT_SCORE_BY",
    "EvaluationReport",
    "aggregate_scores",
    "benchmark_comparison",
    "evaluate_report",
    "metrics",
    "regime_scores",
    "tests",
]
