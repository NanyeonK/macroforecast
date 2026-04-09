from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
PLAN_DIR = ROOT / 'config' / 'plans' / 'package_rebuild'

REQUIRED_FILES = [
    '00_meta_layer.yaml',
    '01_whole_package_plan.yaml',
    '02_separate_plans.yaml',
    '03_implementation_issues.yaml',
    '04_reverse_plan_check.yaml',
    '05_registry_map.yaml',
    '06_contracts.yaml',
    '07_test_matrix.yaml',
    '08_e2e_plan.yaml',
]


def _load(name: str):
    with (PLAN_DIR / name).open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def test_required_plan_files_exist() -> None:
    for name in REQUIRED_FILES:
        assert (PLAN_DIR / name).exists(), name


def test_meta_layer_contains_required_sections() -> None:
    data = _load('00_meta_layer.yaml')
    for key in ['package_mode_lock', 'unit_of_run', 'axis_classes', 'config_inheritance', 'fit_scope_rule', 'failure_policy']:
        assert key in data


def test_whole_package_plan_contains_closure_criteria() -> None:
    data = _load('01_whole_package_plan.yaml')
    assert 'package_plan' in data
    assert 'end_to_end_closure_criteria' in data['package_plan']
    assert data['package_plan']['end_to_end_closure_criteria']


def test_separate_plans_cover_core_streams() -> None:
    data = _load('02_separate_plans.yaml')
    ids = {item['id'] for item in data['separate_plans']}
    for needed in ['SP-0', 'SP-1', 'SP-2', 'SP-3', 'SP-4', 'SP-5', 'SP-6', 'SP-7', 'SP-8', 'SP-9', 'SP-10', 'SP-11', 'SP-12']:
        assert needed in ids


def test_implementation_issues_are_dependency_ordered() -> None:
    data = _load('03_implementation_issues.yaml')
    issues = data['implementation_issues']
    seen = set()
    for issue in issues:
        for dep in issue['dependencies']:
            assert dep in seen
        seen.add(issue['id'])


def test_contracts_and_e2e_exist() -> None:
    contracts = _load('06_contracts.yaml')
    e2e = _load('08_e2e_plan.yaml')
    assert contracts['contracts']
    assert e2e['e2e_plan']['baseline_runs']
