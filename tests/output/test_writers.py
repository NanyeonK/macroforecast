from pathlib import Path

import pandas as pd

from macrocast.output import ensure_output_dirs, load_output_registry, validate_output_registry, write_eval_table, write_failure_log, write_forecast_table, write_interpretation_table, write_test_table


def test_output_paths_and_writers(tmp_path: Path) -> None:
    dirs = ensure_output_dirs(tmp_path, 'run_1')
    assert dirs['forecasts'].exists()
    df = pd.DataFrame({'a': [1], 'b': [2]})
    f1 = write_forecast_table(df, dirs['forecasts'] / 'f.parquet')
    f2 = write_eval_table(df, dirs['evaluation'] / 'e.csv')
    f3 = write_test_table(df, dirs['tests'] / 't.parquet')
    f4 = write_interpretation_table(df, dirs['interpretation'] / 'i.parquet')
    f5 = write_failure_log(df, dirs['manifests'] / 'failures.parquet')
    for fp in [f1, f2, f3, f4, f5]:
        assert fp.exists()


def test_output_registry_valid() -> None:
    validate_output_registry(load_output_registry())
