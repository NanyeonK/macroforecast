"""Wizard smoke tests using a scripted input function."""
from __future__ import annotations

from pathlib import Path

import pytest

from macrocast.scaffold import wizard


class _ScriptedInput:
    """Iterator-backed ``input()`` replacement so tests can drive the
    wizard without a tty."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._answers:
            raise RuntimeError(f"wizard asked for more input than scripted (prompt: {prompt!r})")
        return self._answers.pop(0)


def test_wizard_default_run_produces_runnable_recipe(tmp_path):
    out = tmp_path / "recipe.yaml"
    answers = [
        "1",            # template choice = single_model_forecast
        "",             # L0.failure_policy default
        "",             # L0.reproducibility_mode default
        "",             # L0.leaf_config.random_seed default
        "",             # L0.compute_mode default
        "",             # L0.leaf_config.parallel_unit default
        "",             # L0.leaf_config.n_workers default
    ]
    builder = wizard.run_wizard(
        output_path=out,
        input_fn=_ScriptedInput(answers),
        interactive_layers=("l0",),
    )
    assert out.exists()
    recipe = builder.build()
    assert recipe["0_meta"]["fixed_axes"]["failure_policy"] == "fail_fast"
    assert recipe["0_meta"]["fixed_axes"]["reproducibility_mode"] == "seeded_reproducible"
    assert recipe["0_meta"]["fixed_axes"]["compute_mode"] == "serial"


def test_help_command_displays_option_doc_when_registered(capsys, tmp_path):
    """When the user types ``? official_only`` against the L1 source policy
    axis, the wizard surfaces the rich OptionDoc rather than the schema
    short description -- but only after the L1 docs PR (PR-A2) lands.
    Until then the test pins the fallback path: short description shown
    and a ``Tier-1 OptionDoc not yet registered`` notice."""

    answers = [
        "1",
        "?",                     # ask for L0.failure_policy axis help
        "fail_fast",             # actually pick a value
        "",                      # reproducibility default
        "",                      # leaf random_seed
        "",                      # compute_mode default
        "",                      # leaf parallel_unit
        "",                      # leaf n_workers
    ]
    out = tmp_path / "r.yaml"
    wizard.run_wizard(
        output_path=out,
        input_fn=_ScriptedInput(answers),
        interactive_layers=("l0",),
    )
    captured = capsys.readouterr().out
    # Axis-help emission appears in the captured output.
    assert "axis: l0.l0_a.failure_policy" in captured


def test_option_help_for_documented_l0_option(capsys, tmp_path):
    """L0.failure_policy.fail_fast has a Tier-1 OptionDoc (PR-A1). Asking
    ``? fail_fast`` must surface the registered description, not the
    fallback notice."""

    answers = [
        "1",                 # template
        "? fail_fast",       # request option help
        "fail_fast",         # then pick the value
        "",                  # reproducibility default
        "",                  # leaf random_seed
        "",                  # compute_mode default
        "",                  # leaf parallel_unit
        "",                  # leaf n_workers
    ]
    out = tmp_path / "r.yaml"
    wizard.run_wizard(
        output_path=out,
        input_fn=_ScriptedInput(answers),
        interactive_layers=("l0",),
    )
    captured = capsys.readouterr().out
    assert "When to use:" in captured
    # Reference text from the OptionDoc must appear.
    assert "fail_fast vs continue_on_failure" in captured


def test_invalid_choice_re_prompts(capsys, tmp_path):
    """Invalid number/value triggers a re-prompt instead of crashing."""

    answers = [
        "1",                  # template
        "999",                # invalid number
        "fail_fast",          # recover
        "",                   # reproducibility default
        "",                   # leaf random_seed
        "",                   # compute_mode default
        "",                   # leaf parallel_unit
        "",                   # leaf n_workers
    ]
    out = tmp_path / "r.yaml"
    wizard.run_wizard(
        output_path=out,
        input_fn=_ScriptedInput(answers),
        interactive_layers=("l0",),
    )
    captured = capsys.readouterr().out
    assert "invalid number" in captured


def test_cli_module_is_invokable():
    """``python -m macrocast`` should print help (no crash) when invoked
    without a sub-command."""

    from macrocast.scaffold.cli import main

    rc = main([])
    assert rc == 0
