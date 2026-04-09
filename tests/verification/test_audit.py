from macrocast.data import load_dataset_registry
from macrocast.meta import load_benchmark_registry
from macrocast.preprocessing import load_preprocessing_registry
from macrocast.verification import (
    audit_benchmark_registry,
    audit_preprocessing_registry,
    audit_target_mapping,
    load_verification_registry,
    validate_verification_registry,
)


def test_benchmark_audit_passes_for_known_benchmark() -> None:
    assert audit_benchmark_registry(load_benchmark_registry(), 'ar_bic_expanding') is True


def test_preprocessing_audit_passes_for_known_recipe() -> None:
    assert audit_preprocessing_registry(load_preprocessing_registry(), 'mcng_em', family='x') is True


def test_target_mapping_audit_simple_case() -> None:
    mapping = {'INDPRO': 'INDPRO', 'PAYEMS': 'PAYEMS'}
    assert audit_target_mapping(mapping, 'INDPRO') is True
    assert audit_target_mapping(mapping, 'UNRATE') is False


def test_verification_registry_valid() -> None:
    validate_verification_registry(load_verification_registry())


def test_dataset_registry_still_loads() -> None:
    assert load_dataset_registry()['datasets']
