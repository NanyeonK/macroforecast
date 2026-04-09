from macrocast.meta import load_naming_policy, validate_naming_policy
from macrocast.meta.exceptions import NamingPolicyError


def test_naming_policy_valid() -> None:
    validate_naming_policy(load_naming_policy())


def test_naming_policy_rejects_bad_slug_style() -> None:
    reg = load_naming_policy()
    reg['naming']['slug_style'] = 'camelCase'
    try:
        validate_naming_policy(reg)
    except NamingPolicyError:
        return
    raise AssertionError('expected NamingPolicyError')
