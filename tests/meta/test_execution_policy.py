from macrocast.meta import load_execution_policy_registry, validate_execution_policy


def test_execution_policy_valid() -> None:
    validate_execution_policy(load_execution_policy_registry())


def test_execution_policy_requires_hard_benchmark_failure() -> None:
    reg = load_execution_policy_registry()
    reg['policies'][0]['benchmark_failure_policy'] = 'soft'
    try:
        validate_execution_policy(reg)
    except ValueError:
        return
    raise AssertionError('expected ValueError')


def test_execution_policy_requires_failure_provenance_fields() -> None:
    reg = load_execution_policy_registry()
    reg['policies'][0]['provenance_fields'] = ['failure_stage']
    try:
        validate_execution_policy(reg)
    except ValueError:
        return
    raise AssertionError('expected ValueError')


def test_execution_policy_requires_degraded_run_severity() -> None:
    reg = load_execution_policy_registry()
    reg['policies'][0]['severity_levels'] = ['warning', 'hard_error']
    try:
        validate_execution_policy(reg)
    except ValueError:
        return
    raise AssertionError('expected ValueError')
