"""Evaluation utilities for benchmark forecasts and relative metrics."""
from .benchmark_resolver import (
    BenchmarkResolverError,
    BenchmarkSpec,
    resolve_benchmark_forecasts,
    resolve_benchmark_suite,
)
from .metrics import (
    compute_metrics_dict,
    compute_relative_metrics,
    compute_relative_metrics_suite,
)

__all__ = [
    "resolve_benchmark_forecasts",
    "resolve_benchmark_suite",
    "BenchmarkResolverError",
    "BenchmarkSpec",
    "compute_relative_metrics",
    "compute_relative_metrics_suite",
    "compute_metrics_dict",
]
