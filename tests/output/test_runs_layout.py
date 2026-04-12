from pathlib import Path

import yaml

from macrocast.output import build_run_manifest, ensure_output_dirs


def test_runs_layout_for_recipe() -> None:
    dirs = ensure_output_dirs('/tmp/macrocast_test_runs', 'run_1', recipe_id='minimal_fred_md')
    assert 'runs/recipes/minimal_fred_md/run_1' in str(dirs['root'])


def test_runs_layout_for_taxonomy_path() -> None:
    dirs = ensure_output_dirs('/tmp/macrocast_test_runs', 'run_2', taxonomy_path={'data': 'fred_md', 'model': 'random_forest'})
    assert 'runs/paths/data=fred_md/model=random_forest/run_2' in str(dirs['root'])


def test_manifest_includes_recipe_and_taxonomy_path(tmp_path: Path) -> None:
    manifest = build_run_manifest(
        run_id='run_1',
        experiment_id='exp_1',
        recipe_id='minimal_fred_md',
        taxonomy_path={'data': 'fred_md', 'model': 'random_forest'},
        config_hash='abc123',
        code_version='v1',
        dataset_ids=['fred_md'],
        benchmark_ids=['ar_bic_expanding'],
        artifact_paths={'forecasts': 'x.parquet'},
    )
    p = tmp_path / 'manifest.yaml'
    p.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding='utf-8')
    loaded = yaml.safe_load(p.read_text())
    assert loaded['recipe_id'] == 'minimal_fred_md'
    assert loaded['taxonomy_path']['data'] == 'fred_md'
