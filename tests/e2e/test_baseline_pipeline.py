import numpy as np
import pandas as pd

from macrocast.config import load_config_from_dict
from macrocast.evaluation.dm import dm_test
from macrocast.evaluation.metrics import msfe
from macrocast.interpretation.variable_importance import extract_vi_dataframe
from macrocast.output import build_run_manifest, ensure_output_dirs, write_eval_table, write_failure_log, write_forecast_table, write_interpretation_table, write_run_manifest, write_test_table
from macrocast.pipeline.experiment import ForecastExperiment
from macrocast.specs import compile_experiment_spec_from_dict


def _synthetic_panel():
    rng = np.random.default_rng(7)
    dates = pd.date_range('2005-01', periods=120, freq='MS')
    X = rng.standard_normal((120, 10))
    y = X[:, 0] + 0.3 * X[:, 1] + rng.standard_normal(120) * 0.2
    panel = pd.DataFrame(X, index=dates, columns=[f'x{i}' for i in range(10)])
    target = pd.Series(y, index=dates, name='target')
    return panel, target


def test_end_to_end_baseline_pipeline(tmp_path):
    raw = {
        'experiment': {'id': 'e2e-baseline', 'output_dir': str(tmp_path), 'horizons': [1], 'window': 'expanding', 'n_jobs': 1, 'oos_start': '2014-01-01', 'oos_end': '2014-03-01'},
        'data': {'dataset': 'fred_md', 'target': 'INDPRO'},
        'features': {'factor_type': 'X', 'n_factors': 2, 'n_lags': 2},
        'models': [{'name': 'rf', 'kwargs': {'n_estimators': 10, 'min_samples_leaf_grid': [5], 'cv_folds': 2}}, {'name': 'krr', 'kwargs': {'alpha_grid': [0.1], 'gamma_grid': [0.1], 'cv_folds': 2}}],
    }
    compiled = compile_experiment_spec_from_dict(raw, preset_id='researcher_explicit')
    cfg = load_config_from_dict(raw)
    panel, target = _synthetic_panel()
    exp = ForecastExperiment(panel=panel, target=target, horizons=cfg.horizons, model_specs=cfg.model_specs, feature_spec=cfg.feature_spec, window=cfg.window, oos_start=cfg.oos_start, oos_end=cfg.oos_end, n_jobs=1, output_dir=tmp_path)
    rs = exp.run()
    df = rs.to_dataframe()
    assert not df.empty
    pivot = df.pivot(index='forecast_date', columns='model_id', values='y_hat').dropna()
    y_true = df.drop_duplicates('forecast_date').set_index('forecast_date').loc[pivot.index, 'y_true'].values
    dm = dm_test(y_true, pivot.iloc[:, 0].values, pivot.iloc[:, 1].values)
    vi = extract_vi_dataframe(rs)
    dirs = ensure_output_dirs(tmp_path, compiled.experiment_config.experiment_id)
    f_fore = write_forecast_table(df, dirs['forecasts'] / 'forecasts.parquet')
    f_eval = write_eval_table(pd.DataFrame([{'metric': 'msfe_model_1', 'value': float(msfe(y_true, pivot.iloc[:, 0].values))}]), dirs['evaluation'] / 'eval.csv')
    f_test = write_test_table(pd.DataFrame([{'test': 'dm_default', 'p_value': dm.p_value}]), dirs['tests'] / 'tests.parquet')
    f_int = write_interpretation_table(vi if not vi.empty else pd.DataFrame([{'feature_name': 'none', 'importance': 0.0}]), dirs['interpretation'] / 'interp.parquet')
    f_fail = write_failure_log(rs.failures_dataframe(), dirs['manifests'] / 'failures.parquet')
    manifest = build_run_manifest(run_id=compiled.experiment_config.experiment_id, experiment_id=compiled.experiment_config.experiment_id, config_hash='e2e', code_version='local', dataset_ids=['fred_md'], benchmark_ids=[compiled.meta_config['benchmark_id']], artifact_paths={'forecasts': str(f_fore), 'evaluation': str(f_eval), 'tests': str(f_test), 'interpretation': str(f_int), 'failures': str(f_fail)}, degraded=rs.degraded, failure_summary=rs.failure_summary(), provenance_fields=['failure_stage', 'cell_id', 'model_id', 'horizon'])
    man = write_run_manifest(manifest, dirs['manifests'] / 'manifest.yaml')
    assert man.exists()
