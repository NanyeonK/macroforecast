from macrocast.meta import load_benchmark_registry, validate_benchmark_registry
from macrocast.meta.exceptions import BenchmarkRegistryError


def test_benchmark_registry_valid() -> None:
    reg = load_benchmark_registry()
    validate_benchmark_registry(reg)


def test_benchmark_registry_requires_denominator_rule() -> None:
    reg = load_benchmark_registry()
    reg['benchmarks'][0].pop('denominator_rule')
    try:
        validate_benchmark_registry(reg)
    except BenchmarkRegistryError:
        return
    raise AssertionError('expected BenchmarkRegistryError')
