from pathlib import Path

import yaml

from macrocast.output import build_run_manifest, write_run_manifest


def test_manifest_build_and_write(tmp_path: Path) -> None:
    manifest = build_run_manifest(
        run_id='run_1',
        experiment_id='exp_1',
        config_hash='abc123',
        code_version='v1',
        dataset_ids=['fred_md'],
        benchmark_ids=['ar_bic_expanding'],
        artifact_paths={'forecasts': 'x.parquet'},
    )
    out = write_run_manifest(manifest, tmp_path / 'manifest.yaml')
    loaded = yaml.safe_load(out.read_text())
    assert loaded['run_id'] == 'run_1'
    assert loaded['artifact_paths']['forecasts'] == 'x.parquet'
