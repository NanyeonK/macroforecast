from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
SEPARATE = ROOT / 'config' / 'plans' / 'treepath_overhaul_separate_plan.yaml'
ISSUES = ROOT / 'config' / 'plans' / 'treepath_overhaul_issue_stack.yaml'
DOC = ROOT / 'docs' / 'planning' / 'treepath-package-overhaul.md'


def test_treepath_plan_files_exist() -> None:
    assert DOC.exists()
    assert SEPARATE.exists()
    assert ISSUES.exists()


def test_treepath_separate_plan_has_core_streams() -> None:
    data = yaml.safe_load(SEPARATE.read_text())
    assert 'treepath_overhaul' in data
    for key in ['sp_t0_architecture_lock','sp_t1_taxonomy_canonicalization','sp_t2_registries_layer','sp_t3_recipe_schema_and_resolver','sp_t4_benchmark_model_symmetry','sp_t5_execution_and_runs','sp_t6_verification_migration','sp_t7_docs_examples_migration']:
        assert key in data['treepath_overhaul']


def test_treepath_issue_stack_ordered() -> None:
    data = yaml.safe_load(ISSUES.read_text())
    seen = set()
    for issue in data['issue_stack']:
        for dep in issue['depends_on']:
            assert dep in seen
        seen.add(issue['id'])
