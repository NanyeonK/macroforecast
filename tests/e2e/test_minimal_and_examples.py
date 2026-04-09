from pathlib import Path

import yaml

from macrocast.config import load_config, load_config_from_dict
from macrocast.evaluation import load_evaluation_registry, load_test_registry, validate_evaluation_registry, validate_test_registry
from macrocast.interpretation.registry import load_interpretation_registry, validate_interpretation_registry
from macrocast.output import load_output_registry, validate_output_registry
from macrocast.specs import compile_experiment_spec_from_dict
from macrocast.verification import load_verification_registry, validate_verification_registry


def test_minimal_mode_resolves() -> None:
    raw = {
        'experiment_id': 'minimal-baseline',
        'dataset': 'fred_md',
        'target': 'INDPRO',
        'horizons': [1],
        'window': 'expanding',
        'oos_start': '2012-01-01',
        'oos_end': '2012-03-01',
        'models': ['RF'],
    }
    cfg = load_config_from_dict(raw)
    compiled = compile_experiment_spec_from_dict(raw, preset_id='practitioner_minimal')
    assert cfg.data.dataset == 'fred_md'
    assert compiled.meta_config['benchmark_id'] == 'historical_mean_expanding'


def test_example_yaml_files_parse() -> None:
    root = Path(__file__).resolve().parents[2] / 'config' / 'examples'
    for name in ['fred_md_baseline.yaml', 'fred_qd_baseline.yaml', 'minimal_baseline.yaml', 'verification_audit.yaml', 'custom_extension_smoke.yaml']:
        path = root / name
        raw = yaml.safe_load(path.read_text())
        assert raw
        load_config(path)


def test_registry_layers_load() -> None:
    validate_evaluation_registry(load_evaluation_registry())
    validate_test_registry(load_test_registry())
    validate_interpretation_registry(load_interpretation_registry())
    validate_output_registry(load_output_registry())
    validate_verification_registry(load_verification_registry())
