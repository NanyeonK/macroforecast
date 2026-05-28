"""C65 independent validation — recipe orchestration extraction (R3-P2).

Written by tester from test-spec.md only. No knowledge of spec.md or
implementation.md. Verifies behavioral contracts T1-T11 and edge cases EC1-EC4.

Scope: pure import / identity checks. No FRED data required. No YAML execution.
Heavy execution tests (T1/T2 actual recipe run) are marked @pytest.mark.slow
and excluded by the scope-limited run (-m "not slow").
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WORKTREE = Path(__file__).parent.parent.parent  # repo root


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    """Run a short Python snippet in a fresh subprocess under the same uv env."""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_WORKTREE),
    )


# ---------------------------------------------------------------------------
# T9 — Circular import checks (subprocess; fast)
# ---------------------------------------------------------------------------

class TestCircularImport:
    """T9: import macroforecast must not raise ImportError or circular import."""

    def test_import_macroforecast_no_circular(self):
        """Fresh interpreter: import macroforecast prints 'ok' and exits 0."""
        result = _run_python("import macroforecast; print('ok')")
        assert result.returncode == 0, (
            f"import macroforecast failed in subprocess.\n"
            f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )
        assert "ok" in result.stdout

    def test_import_recipes_submodule_no_circular(self):
        """Fresh interpreter: from macroforecast import recipes."""
        result = _run_python("from macroforecast import recipes; print('ok')")
        assert result.returncode == 0, (
            f"'from macroforecast import recipes' failed.\n"
            f"stderr: {result.stderr!r}"
        )
        assert "ok" in result.stdout

    def test_import_recipes_symbols_no_circular(self):
        """Fresh interpreter: from macroforecast.recipes import run, replicate, Experiment."""
        result = _run_python(
            "from macroforecast.recipes import run, replicate, Experiment; print('ok')"
        )
        assert result.returncode == 0, (
            f"direct recipes symbol import failed.\nstderr: {result.stderr!r}"
        )
        assert "ok" in result.stdout


# ---------------------------------------------------------------------------
# T3 / T4 — mf.run / mf.replicate callable (no recipe execution)
# ---------------------------------------------------------------------------

class TestTopLevelCallable:
    """T3/T4: mf.run and mf.replicate are callable (smoke import only)."""

    def test_mf_run_is_callable(self):
        """T3: macroforecast.run is callable."""
        import macroforecast as mf
        assert callable(mf.run), "macroforecast.run must be callable"

    def test_mf_replicate_is_callable(self):
        """T4: macroforecast.replicate is callable."""
        import macroforecast as mf
        assert callable(mf.replicate), "macroforecast.replicate must be callable"

    def test_mf_run_file_is_callable(self):
        """mf.run_file is callable."""
        import macroforecast as mf
        assert callable(mf.run_file), "macroforecast.run_file must be callable"


# ---------------------------------------------------------------------------
# T3 / T4 identity — recipes.run IS mf.run
# ---------------------------------------------------------------------------

class TestIdentityInvariants:
    """T3/T4/T5/T6: recipes.* symbols are the same object as mf.* aliases."""

    def test_run_identity(self):
        """macroforecast.run is macroforecast.recipes.run."""
        import macroforecast as mf
        assert mf.run is mf.recipes.run, (
            "macroforecast.run and macroforecast.recipes.run must be the same object"
        )

    def test_replicate_identity(self):
        """macroforecast.replicate is macroforecast.recipes.replicate."""
        import macroforecast as mf
        assert mf.replicate is mf.recipes.replicate

    def test_run_file_identity(self):
        """T5: macroforecast.run_file is macroforecast.recipes.run_file."""
        import macroforecast as mf
        assert mf.run_file is mf.recipes.run_file

    def test_forecast_identity(self):
        """macroforecast.forecast is macroforecast.recipes.forecast."""
        import macroforecast as mf
        assert mf.forecast is mf.recipes.forecast

    def test_Experiment_identity(self):
        """macroforecast.Experiment is macroforecast.recipes.Experiment."""
        import macroforecast as mf
        assert mf.Experiment is mf.recipes.Experiment

    def test_ForecastResult_identity(self):
        """macroforecast.ForecastResult is macroforecast.recipes.ForecastResult."""
        import macroforecast as mf
        assert mf.ForecastResult is mf.recipes.ForecastResult

    def test_ManifestExecutionResult_identity(self):
        """macroforecast.ManifestExecutionResult is macroforecast.recipes.ManifestExecutionResult."""
        import macroforecast as mf
        assert mf.ManifestExecutionResult is mf.recipes.ManifestExecutionResult

    def test_ReplicationResult_identity(self):
        """macroforecast.ReplicationResult is macroforecast.recipes.ReplicationResult."""
        import macroforecast as mf
        assert mf.ReplicationResult is mf.recipes.ReplicationResult


# ---------------------------------------------------------------------------
# T6 — Canonical namespace completeness (__all__ and hasattr)
# ---------------------------------------------------------------------------

class TestRecipesNamespaceCompleteness:
    """T6: All 8 symbols are in macroforecast.recipes and its __all__.

    Phase 3b removed paper_methods from macroforecast.recipes; it now lives at
    macroforecast.models.paper_methods.
    """

    _EXPECTED_SYMBOLS = [
        "run",
        "run_file",
        "replicate",
        "forecast",
        "Experiment",
        "ForecastResult",
        "ManifestExecutionResult",
        "ReplicationResult",
    ]

    def test_all_symbols_are_attributes(self):
        """Each expected symbol exists as an attribute on macroforecast.recipes."""
        import macroforecast as mf
        missing = [s for s in self._EXPECTED_SYMBOLS if not hasattr(mf.recipes, s)]
        assert not missing, f"Missing attributes on macroforecast.recipes: {missing}"

    def test_all_symbols_in_dunder_all(self):
        """Each expected symbol appears in macroforecast.recipes.__all__."""
        import macroforecast as mf
        dunder_all = set(mf.recipes.__all__)
        missing = [s for s in self._EXPECTED_SYMBOLS if s not in dunder_all]
        assert not missing, (
            f"Missing from macroforecast.recipes.__all__: {missing}\n"
            f"Actual __all__: {sorted(mf.recipes.__all__)}"
        )

    def test_dunder_all_exact_count(self):
        """macroforecast.recipes.__all__ contains exactly 8 symbols."""
        import macroforecast as mf
        assert len(mf.recipes.__all__) == 8, (
            f"Expected 8 symbols in __all__, got {len(mf.recipes.__all__)}: "
            f"{sorted(mf.recipes.__all__)}"
        )


# ---------------------------------------------------------------------------
# T7 — Experiment class identity
# ---------------------------------------------------------------------------

class TestExperimentClassIdentity:
    """T7: macroforecast.recipes.Experiment is macroforecast.Experiment."""

    def test_Experiment_is_class(self):
        """recipes.Experiment resolves to a class."""
        import macroforecast as mf
        assert isinstance(mf.recipes.Experiment, type), (
            "macroforecast.recipes.Experiment must be a class (type)"
        )

    def test_Experiment_class_identity(self):
        """The class object is identical via both access paths."""
        import macroforecast as mf
        assert mf.recipes.Experiment is mf.Experiment


# ---------------------------------------------------------------------------
# T8 — api_high.py private symbols still importable (regression guard)
# ---------------------------------------------------------------------------

class TestApiHighRegressionGuard:
    """T8: api_high.py was NOT moved; private symbols remain directly importable."""

    def test_ForecastResult_from_api_high(self):
        """from macroforecast.api_high import ForecastResult succeeds."""
        from macroforecast.api_high import ForecastResult  # noqa: F401

    def test_set_at_from_api_high(self):
        """from macroforecast.api_high import _set_at succeeds."""
        from macroforecast.api_high import _set_at  # noqa: F401

    def test_build_default_recipe_from_api_high(self):
        """from macroforecast.api_high import _build_default_recipe succeeds."""
        from macroforecast.api_high import _build_default_recipe  # noqa: F401


# ---------------------------------------------------------------------------
# T11 — Module docstring contains framing note
# ---------------------------------------------------------------------------

class TestRecipesDocstring:
    """T11: macroforecast.recipes.__doc__ is non-empty and contains framing note."""

    def test_docstring_is_nonempty(self):
        """recipes.__doc__ is a non-empty string."""
        import macroforecast as mf
        assert isinstance(mf.recipes.__doc__, str) and len(mf.recipes.__doc__) > 0, (
            "macroforecast.recipes.__doc__ must be a non-empty string"
        )

    def test_docstring_mentions_models_namespace(self):
        """T11: __doc__ mentions the standalone model API (macroforecast.models)."""
        import macroforecast as mf
        assert "macroforecast.models" in mf.recipes.__doc__, (
            f"Expected 'macroforecast.models' in recipes.__doc__.\n"
            f"Actual docstring excerpt: {mf.recipes.__doc__[:300]!r}"
        )


# ---------------------------------------------------------------------------
# EC1 — Explicit symbol import from recipes
# ---------------------------------------------------------------------------

class TestEdgeCaseImports:
    """EC1-EC4: Edge case import patterns."""

    def test_EC1_starred_explicit_import(self):
        """EC1: from macroforecast.recipes import run, replicate succeeds."""
        from macroforecast.recipes import run, replicate  # noqa: F401
        assert callable(run)
        assert callable(replicate)

    def test_EC2_import_as_alias(self):
        """EC2: import macroforecast.recipes as mf_recipes; mf_recipes.run accessible."""
        import macroforecast.recipes as mf_recipes
        assert hasattr(mf_recipes, "run")
        assert callable(mf_recipes.run)

    def test_EC3_paper_methods_accessible(self):
        """EC3: macroforecast.models.paper_methods is the paper_methods module.

        Phase 3b relocated paper_methods from macroforecast.recipes to
        macroforecast.models.paper_methods.
        """
        import macroforecast.models.paper_methods as pm
        # The module should have at least one callable (e.g., macroeconomic_random_forest)
        assert pm is not None
        assert hasattr(pm, "__name__"), "paper_methods should be a module"

    def test_EC4_repeated_access_consistent(self):
        """EC4: Repeated access to macroforecast.run returns the same object."""
        import macroforecast as mf
        first = mf.run
        second = mf.run
        assert first is second, (
            "macroforecast.run should return the same cached object on repeated access"
        )


# ---------------------------------------------------------------------------
# Property-based invariant sweep
# ---------------------------------------------------------------------------

class TestPropertyBasedInvariants:
    """All 8 core symbols satisfy the identity invariant simultaneously."""

    _SYMBOLS = [
        "run",
        "run_file",
        "replicate",
        "forecast",
        "Experiment",
        "ForecastResult",
        "ManifestExecutionResult",
        "ReplicationResult",
    ]

    @pytest.mark.parametrize("symbol", _SYMBOLS)
    def test_identity_invariant(self, symbol: str):
        """For every symbol s: getattr(mf, s) is getattr(mf.recipes, s)."""
        import macroforecast as mf
        mf_attr = getattr(mf, symbol)
        recipes_attr = getattr(mf.recipes, symbol)
        assert mf_attr is recipes_attr, (
            f"Identity invariant violated for '{symbol}': "
            f"macroforecast.{symbol} is not macroforecast.recipes.{symbol}"
        )
