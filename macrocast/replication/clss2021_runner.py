from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from macrocast.data import load_fred_md
from macrocast.output import build_run_manifest, ensure_output_dirs, write_failure_log, write_forecast_table, write_run_manifest
from macrocast.pipeline import ForecastExperiment
from macrocast.pipeline.results import ResultSet
from macrocast.replication.clss2021 import CLSS2021

# Migration scaffolding only.
# Target architecture is recipe/path based (`recipes/papers/clss2021.yaml`), not package-core CLSS helpers.


def load_clss2021_fixed_settings() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / 'config' / 'plans' / 'clss2021-fixed-settings.yaml'
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError('clss2021 fixed settings must decode to a dict')
    return data['clss2021_package_baseline']


def load_clss2021_panel(*, vintage: str = '2018-02', estimation_start: str = '1960-01-01', end: str = '2017-12-01') -> tuple[pd.DataFrame, pd.DataFrame]:
    mf = load_fred_md(vintage=vintage)
    panel_stat = mf.transform().trim(start=estimation_start, end=end).data
    panel_levels = mf.trim(start=estimation_start, end=end).data
    return panel_stat, panel_levels


def relative_rmsfe_vs_feature_baseline(df: pd.DataFrame, baseline_feature_set: str = 'F') -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['feature_set', 'target', 'horizon', 'relative_rmsfe'])
    work = df.copy()
    work['sq_err'] = (work['y_hat'] - work['y_true']) ** 2
    msfe = work.groupby(['feature_set', 'target', 'horizon'])['sq_err'].mean().reset_index().rename(columns={'sq_err': 'msfe'})
    bench = msfe[msfe['feature_set'] == baseline_feature_set][['target', 'horizon', 'msfe']].rename(columns={'msfe': 'msfe_feature_baseline'})
    merged = msfe.merge(bench, on=['target', 'horizon'], how='left')
    merged['relative_rmsfe'] = (merged['msfe'] / merged['msfe_feature_baseline']) ** 0.5
    return merged[['feature_set', 'target', 'horizon', 'relative_rmsfe']]


def relative_rmsfe_vs_ar(df: pd.DataFrame, ar_col: str = 'ar_benchmark_msfe') -> pd.DataFrame:
    if df.empty or ar_col not in df.columns:
        return pd.DataFrame(columns=['feature_set', 'target', 'horizon', 'relative_rmsfe_vs_ar'])
    work = df.copy()
    work['sq_err'] = (work['y_hat'] - work['y_true']) ** 2
    msfe = work.groupby(['feature_set', 'target', 'horizon'])['sq_err'].mean().reset_index().rename(columns={'sq_err': 'msfe'})
    ar = work.groupby(['target', 'horizon'])[ar_col].mean().reset_index().rename(columns={ar_col: 'ar_benchmark_msfe'})
    merged = msfe.merge(ar, on=['target', 'horizon'], how='left')
    merged['relative_rmsfe_vs_ar'] = (merged['msfe'] / merged['ar_benchmark_msfe']) ** 0.5
    return merged[['feature_set', 'target', 'horizon', 'relative_rmsfe_vs_ar']]


def summarize_clss2021_run(df: pd.DataFrame, failures_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rel_feature = relative_rmsfe_vs_feature_baseline(df)
    rel_ar = relative_rmsfe_vs_ar(df)
    feature_summary = rel_feature.groupby(['feature_set', 'horizon'])['relative_rmsfe'].mean().reset_index() if not rel_feature.empty else pd.DataFrame(columns=['feature_set', 'horizon', 'relative_rmsfe'])
    ar_summary = rel_ar.groupby(['feature_set', 'horizon'])['relative_rmsfe_vs_ar'].mean().reset_index() if not rel_ar.empty else pd.DataFrame(columns=['feature_set', 'horizon', 'relative_rmsfe_vs_ar'])
    coverage = pd.DataFrame([{'n_records': len(df), 'n_failures': len(failures_df), 'n_targets': df['target'].nunique() if 'target' in df.columns and not df.empty else 0, 'n_feature_sets': df['feature_set'].nunique() if 'feature_set' in df.columns and not df.empty else 0, 'n_horizons': df['horizon'].nunique() if 'horizon' in df.columns and not df.empty else 0}])
    return {'relative_rmsfe_by_target': rel_feature, 'relative_rmsfe_summary': feature_summary, 'relative_rmsfe_vs_ar_by_target': rel_ar, 'relative_rmsfe_vs_ar_summary': ar_summary, 'coverage_summary': coverage}


def run_clss2021_reduced_check(*, targets: list[str] | None = None, horizons: list[int] | None = None, info_set_labels: list[str] | None = None, oos_start: str = '1980-01-01', oos_end: str = '1980-03-01', output_dir: str | Path | None = None, panel_stat: pd.DataFrame | None = None, panel_levels: pd.DataFrame | None = None) -> dict[str, Any]:
    settings = load_clss2021_fixed_settings()
    reduced = settings['reduced_notebook_scope']
    fixed = settings['fixed']
    targets = targets or list(reduced['targets'])
    horizons = horizons or list(reduced['horizons'])
    info_set_labels = info_set_labels or list(reduced['info_sets'])
    if panel_stat is None or panel_levels is None:
        panel_stat, panel_levels = load_clss2021_panel(vintage=fixed['vintage'], estimation_start=fixed['estimation_start'], end=fixed['oos_end'])
    feature_map = CLSS2021.info_sets(P_Y=fixed['p_y'], K=fixed['k'], P_MARX=fixed['p_marx'])
    model_spec = CLSS2021.rf_spec(model_id='RF')
    rows: list[pd.DataFrame] = []
    failures: list[pd.DataFrame] = []
    output_dir = Path(output_dir) if output_dir is not None else None
    for target in targets:
        target_series = panel_stat[target].dropna()
        panel = panel_stat.drop(columns=[target], errors='ignore').loc[target_series.index]
        levels = panel_levels.drop(columns=[target], errors='ignore').loc[target_series.index]
        for label in info_set_labels:
            spec = feature_map[label]
            exp = ForecastExperiment(panel=panel, target=target_series, horizons=horizons, model_specs=[model_spec], feature_spec=spec, panel_levels=levels, oos_start=oos_start, oos_end=oos_end, n_jobs=1, output_dir=(output_dir / target / label) if output_dir else None, experiment_id=f'clss2021-check-{target}-{label}'.replace('/', '_'))
            rs: ResultSet = exp.run()
            df = rs.to_dataframe().copy()
            if not df.empty:
                df['target'] = target
                df['feature_set'] = label
                df['benchmark_family'] = fixed['benchmark_family']
                df['benchmark_id'] = 'ar_bic_expanding'
                df['target_preprocess_recipe'] = 'basic_none'
                df['x_preprocess_recipe'] = 'basic_none'
                base = ((df['y_true'] - df['y_hat']) ** 2).mean()
                df['ar_benchmark_msfe'] = base
                rows.append(df)
            fail_df = rs.failures_dataframe().copy()
            if not fail_df.empty:
                fail_df['target'] = target
                fail_df['feature_set'] = label
                failures.append(fail_df)
    records_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    failures_df = pd.concat(failures, ignore_index=True) if failures else pd.DataFrame()
    summaries = summarize_clss2021_run(records_df, failures_df)
    contract = {'fixed': fixed, 'reduced_notebook_scope': {'targets': targets, 'horizons': horizons, 'info_sets': info_set_labels, 'models': ['RF']}}
    artifact_paths: dict[str, str] = {}
    manifest: dict[str, Any] | None = None
    if output_dir is not None:
        dirs = ensure_output_dirs(output_dir, 'clss2021-reduced-check')
        artifact_paths['forecasts'] = str(write_forecast_table(records_df, dirs['forecasts'] / 'records.parquet'))
        artifact_paths['failures'] = str(write_failure_log(failures_df, dirs['manifests'] / 'failures.parquet'))
        artifact_paths['relative_rmsfe_by_target'] = str(write_forecast_table(summaries['relative_rmsfe_by_target'], dirs['evaluation'] / 'relative_rmsfe_by_target.parquet'))
        artifact_paths['relative_rmsfe_summary'] = str(write_forecast_table(summaries['relative_rmsfe_summary'], dirs['evaluation'] / 'relative_rmsfe_summary.parquet'))
        artifact_paths['relative_rmsfe_vs_ar_by_target'] = str(write_forecast_table(summaries['relative_rmsfe_vs_ar_by_target'], dirs['evaluation'] / 'relative_rmsfe_vs_ar_by_target.parquet'))
        artifact_paths['relative_rmsfe_vs_ar_summary'] = str(write_forecast_table(summaries['relative_rmsfe_vs_ar_summary'], dirs['evaluation'] / 'relative_rmsfe_vs_ar_summary.parquet'))
        artifact_paths['coverage_summary'] = str(write_forecast_table(summaries['coverage_summary'], dirs['evaluation'] / 'coverage_summary.parquet'))
        artifact_paths['provenance_snapshot'] = str(write_forecast_table(records_df[['target','feature_set','horizon','benchmark_family','benchmark_id','target_preprocess_recipe','x_preprocess_recipe']].drop_duplicates(), dirs['manifests'] / 'provenance_snapshot.parquet'))
        contract_path = dirs['manifests'] / 'clss2021_contract.yaml'
        with contract_path.open('w', encoding='utf-8') as f:
            yaml.safe_dump(contract, f, sort_keys=False)
        artifact_paths['contract'] = str(contract_path)
        manifest = build_run_manifest(run_id='clss2021-reduced-check', experiment_id='clss2021-reduced-check', config_hash='clss2021-fixed-settings', code_version='local', dataset_ids=['fred_md'], benchmark_ids=['ar_bic_expanding'], artifact_paths=artifact_paths, degraded=not failures_df.empty, success=True, failure_summary=[] if failures_df.empty else failures_df['exception_class'].astype(str).tolist(), provenance_fields=['benchmark_family','benchmark_id','evaluation_scale','cell_id','feature_set','target','target_preprocess_recipe','x_preprocess_recipe'])
        manifest_path = write_run_manifest(manifest, dirs['manifests'] / 'manifest.yaml')
        artifact_paths['manifest'] = str(manifest_path)
    return {'records_df': records_df, 'failures_df': failures_df, 'summaries': summaries, 'contract': contract, 'artifact_paths': artifact_paths, 'manifest': manifest}
