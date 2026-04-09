import pytest

from macrocast.design import resolve_experiment_spec_from_dict
from macrocast.meta.exceptions import IllegalOverrideError


MINIMAL_CONFIG = {
    'experiment': {'id': 'design-test', 'output_dir': '/tmp/macrocast_test', 'horizons': [1], 'window': 'expanding', 'n_jobs': 1},
    'data': {'dataset': 'fred_md', 'target': 'INDPRO'},
    'features': {'factor_type': 'X', 'n_factors': 2, 'n_lags': 1},
    'models': [{'name': 'rf'}],
}


def test_invalid_invariant_override_fails() -> None:
    with pytest.raises(IllegalOverrideError):
        resolve_experiment_spec_from_dict(MINIMAL_CONFIG, experiment_overrides={'no_lookahead_rule': False})
