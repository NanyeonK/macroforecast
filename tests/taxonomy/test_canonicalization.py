from pathlib import Path
import yaml

from macrocast.taxonomy import load_taxonomy_file

ROOT = Path(__file__).resolve().parents[2]


def test_treepath_migration_map_exists() -> None:
    assert (ROOT / 'docs' / 'planning' / 'treepath-migration-map.md').exists()
    assert (ROOT / 'config' / 'plans' / 'treepath_migration_map.yaml').exists()


def test_benchmark_taxonomy_is_family_plus_options() -> None:
    data = load_taxonomy_file('5_evaluation/benchmark_registry.yaml')
    for key in ['benchmark_families', 'lag_selection_rules', 'estimation_window_rules', 'target_construction_rules', 'denominator_rules', 'benchmark_specificity', 'support_status']:
        assert key in data
    assert 'ar' in data['benchmark_families']
    assert 'bic' in data['lag_selection_rules']
    assert 'expanding' in data['estimation_window_rules']


def test_migration_map_has_expected_buckets() -> None:
    data = yaml.safe_load((ROOT / 'config' / 'plans' / 'treepath_migration_map.yaml').read_text())
    assert 'migration_map' in data
    for key in ['taxonomy', 'registries', 'recipes', 'runs', 'legacy_migration', 'critical_redesigns']:
        assert key in data['migration_map']
