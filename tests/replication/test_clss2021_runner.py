from pathlib import Path

import numpy as np
import pandas as pd

from macrocast.replication import (
    load_clss2021_fixed_settings,
    relative_rmsfe_vs_ar,
    relative_rmsfe_vs_feature_baseline,
    run_clss2021_reduced_check,
)


def _synthetic_panel(T: int = 180, N: int = 12):
    rng = np.random.default_rng(0)
    dates = pd.date_range('1960-01', periods=T, freq='MS')
    panel = pd.DataFrame(rng.standard_normal((T, N)), index=dates, columns=[f'X{i:02d}' for i in range(N)])
    for target in ['INDPRO', 'UNRATE', 'CPIAUCSL']:
        panel[target] = rng.standard_normal(T)
    return panel, panel.copy()


def test_fixed_settings_load() -> None:
    settings = load_clss2021_fixed_settings()
    assert settings['fixed']['dataset'] == 'fred_md'
    assert settings['fixed']['p_y'] == 12


def test_relative_rmsfe_helpers() -> None:
    df = pd.DataFrame([
        {'feature_set': 'F', 'target': 'INDPRO', 'horizon': 1, 'y_hat': 1.0, 'y_true': 2.0, 'ar_benchmark_msfe': 1.0},
        {'feature_set': 'F', 'target': 'INDPRO', 'horizon': 1, 'y_hat': 1.0, 'y_true': 1.0, 'ar_benchmark_msfe': 1.0},
        {'feature_set': 'F-MARX', 'target': 'INDPRO', 'horizon': 1, 'y_hat': 1.5, 'y_true': 2.0, 'ar_benchmark_msfe': 1.0},
        {'feature_set': 'F-MARX', 'target': 'INDPRO', 'horizon': 1, 'y_hat': 1.0, 'y_true': 1.0, 'ar_benchmark_msfe': 1.0},
    ])
    rel_feature = relative_rmsfe_vs_feature_baseline(df)
    rel_ar = relative_rmsfe_vs_ar(df)
    assert set(rel_feature['feature_set']) == {'F', 'F-MARX'}
    assert set(rel_ar['feature_set']) == {'F', 'F-MARX'}


def test_run_clss2021_reduced_check_with_synthetic_panel(tmp_path: Path) -> None:
    panel_stat, panel_levels = _synthetic_panel()
    out = run_clss2021_reduced_check(
        targets=['INDPRO'],
        horizons=[1],
        info_set_labels=['F', 'F-MARX'],
        oos_start='1970-01-01',
        oos_end='1970-02-01',
        output_dir=tmp_path,
        panel_stat=panel_stat,
        panel_levels=panel_levels,
    )
    assert not out['records_df'].empty
    assert 'relative_rmsfe_summary' in out['summaries']
    assert 'relative_rmsfe_vs_ar_summary' in out['summaries']
    assert 'manifest' in out['artifact_paths']
    assert out['manifest']['benchmark_ids'] == ['ar_bic_expanding']
