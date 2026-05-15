"""Tests for Cycle 14 L2 fixes (P2 batch 2).

Covers:
- L2-1: markdown extra / tabulate declared in pyproject.toml
- L2-2: FileNotFoundError on missing recipe path
- L2-3: CLI prints manifest path on success; clean error on invalid YAML
- L2-4: output_directory kwarg propagates to L8 leaf_config
- L2-5: SHAP subsamples at >2000 rows with UserWarning
"""
from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import macroforecast
from macroforecast.core.execution import execute_recipe


# ---------------------------------------------------------------------------
# Minimal inline recipe used across multiple tests
# ---------------------------------------------------------------------------

_MINI_RECIPE = {
    "0_meta": {
        "fixed_axes": {
            "failure_policy": "fail_fast",
            "reproducibility_mode": "seeded_reproducible",
        },
        "leaf_config": {"random_seed": 1},
    },
    "1_data": {
        "fixed_axes": {
            "custom_source_policy": "custom_panel_only",
            "frequency": "monthly",
            "horizon_set": "custom_list",
        },
        "leaf_config": {
            "target": "y",
            "target_horizons": [1],
            "custom_panel_inline": {
                "date": [
                    "2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01",
                    "2020-05-01", "2020-06-01", "2020-07-01", "2020-08-01",
                ],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            },
        },
    },
}


# ---------------------------------------------------------------------------
# L2-1: tabulate declared in pyproject extras
# ---------------------------------------------------------------------------

class TestL21TabulateDeclared:
    """Cycle 14 L2-1: tabulate listed in pyproject.toml optional-dependencies."""

    def test_tabulate_in_markdown_extra(self, tmp_path: Path) -> None:
        """pyproject.toml must declare a 'markdown' extra containing tabulate."""
        import importlib.util

        # Find pyproject.toml at repo root (two levels above macroforecast package)
        import macroforecast as mf_mod
        pkg_dir = Path(mf_mod.__file__).parent
        pyproject = pkg_dir.parent / "pyproject.toml"
        if not pyproject.exists():
            pytest.skip("pyproject.toml not found at repo root")

        text = pyproject.read_text(encoding="utf-8")
        # Verify 'markdown' extra exists and includes tabulate
        assert 'markdown = [' in text, (
            "pyproject.toml must have a 'markdown' extra declaring tabulate"
        )
        assert 'tabulate' in text, (
            "tabulate must appear in pyproject.toml optional-dependencies"
        )

    def test_tabulate_in_all_extra(self) -> None:
        """'all' extra must include macroforecast[...markdown...]."""
        import macroforecast as mf_mod
        pkg_dir = Path(mf_mod.__file__).parent
        pyproject = pkg_dir.parent / "pyproject.toml"
        if not pyproject.exists():
            pytest.skip("pyproject.toml not found at repo root")

        text = pyproject.read_text(encoding="utf-8")
        # The all extra line should contain 'markdown'
        for line in text.splitlines():
            if line.startswith("all = ") and "macroforecast[" in line:
                assert "markdown" in line, (
                    "'all' extra must include 'markdown' so tabulate is available via macroforecast[all]"
                )
                return
        pytest.fail("Could not find 'all = [...]' line in pyproject.toml")


# ---------------------------------------------------------------------------
# L2-2: FileNotFoundError on missing recipe path
# ---------------------------------------------------------------------------

class TestL22FileNotFoundError:
    """Cycle 14 L2-2: mf.run(Path('/nonexistent')) raises FileNotFoundError."""

    def test_missing_path_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="recipe path does not exist"):
            execute_recipe(Path("/nonexistent_recipe_cycle14.yaml"))

    def test_missing_path_message_contains_path(self) -> None:
        bad_path = Path("/no/such/file.yaml")
        with pytest.raises(FileNotFoundError) as exc_info:
            execute_recipe(bad_path)
        assert str(bad_path) in str(exc_info.value), (
            "FileNotFoundError message must include the missing path"
        )

    def test_existing_path_does_not_raise(self, tmp_path: Path) -> None:
        """Existing path should not trigger L2-2 guard (may raise for other reasons)."""
        recipe_file = tmp_path / "recipe.yaml"
        recipe_file.write_text("0_meta: null\n", encoding="utf-8")
        # Should raise something (recipe is incomplete) but NOT FileNotFoundError
        with pytest.raises(Exception) as exc_info:
            execute_recipe(recipe_file)
        assert not isinstance(exc_info.value, FileNotFoundError), (
            "Existing path must not raise FileNotFoundError from L2-2 guard"
        )


# ---------------------------------------------------------------------------
# L2-3: CLI UX — manifest path on success + clean error
# ---------------------------------------------------------------------------

class TestL23CLIUx:
    """Cycle 14 L2-3: CLI prints manifest path; clean error on invalid YAML."""

    def _make_cli_args(self, recipe: str, output_dir: str) -> object:
        """Build a minimal argparse.Namespace-like object."""
        from types import SimpleNamespace
        return SimpleNamespace(recipe=recipe, output_directory=output_dir)

    def test_missing_recipe_returns_2(self, tmp_path: Path) -> None:
        """CLI returns 2 for a nonexistent recipe file."""
        from macroforecast.scaffold.cli import _cmd_run
        args = self._make_cli_args(
            str(tmp_path / "nonexistent.yaml"), str(tmp_path / "out")
        )
        rc = _cmd_run(args)
        assert rc == 2

    def test_clean_error_on_invalid_yaml(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """CLI catches YAML errors and prints a clean 1-3 line message."""
        from macroforecast.scaffold.cli import _cmd_run

        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("key: {bad: yaml: :\n", encoding="utf-8")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        args = self._make_cli_args(str(bad_yaml), str(out_dir))

        rc = _cmd_run(args)
        captured = capsys.readouterr()
        # Should exit non-zero
        assert rc != 0
        # Error message should be short (no raw Python traceback)
        err_lines = [l for l in captured.err.splitlines() if l.strip()]
        # The error output from CLI should be concise — at most 3 lines
        # (preamble line + error line; full traceback would be 20+)
        assert len(err_lines) <= 3, (
            f"CLI should print clean short error (≤3 lines), got {len(err_lines)}: {captured.err!r}"
        )
        assert any("Error" in l or "error" in l for l in err_lines), (
            "CLI error output must contain 'Error' or 'error'"
        )


# ---------------------------------------------------------------------------
# L2-4: output_directory kwarg propagates to L8 leaf_config
# ---------------------------------------------------------------------------

class TestL24OutputDirectoryWiring:
    """Cycle 14 L2-4: mf.run(recipe, output_directory=...) injects into L8 leaf_config."""

    def test_output_directory_injected_into_l8(self, tmp_path: Path) -> None:
        """When output_directory is passed and recipe has no L8, it is injected."""
        import copy
        import yaml
        from macroforecast.core.execution import execute_recipe, _canonicalize_keys

        recipe = copy.deepcopy(_MINI_RECIPE)
        # Ensure no 8_output block in recipe
        recipe.pop("8_output", None)

        # We patch execute_recipe internals: just verify the recipe_root gets L8 injected
        # by reading the execution.py code path directly
        from macroforecast.core import execution as exec_mod

        original_run_cells = exec_mod._run_cells_serial

        captured_recipe = {}

        def fake_run_cells(cell_jobs, **kwargs):
            # Capture the concrete_root from first job
            if cell_jobs:
                captured_recipe["l8_leaf"] = (
                    cell_jobs[0][1].get("8_output", {}).get("leaf_config", {})
                )
            return original_run_cells(cell_jobs, **kwargs)

        out_dir = tmp_path / "test_out"
        out_dir.mkdir()

        with patch.object(exec_mod, "_run_cells_serial", fake_run_cells):
            try:
                execute_recipe(copy.deepcopy(recipe), output_directory=out_dir)
            except Exception:
                pass  # Recipe may fail; we only care about L8 injection

        # The L8 leaf_config should have output_directory = str(out_dir)
        l8_leaf = captured_recipe.get("l8_leaf", {})
        assert "output_directory" in l8_leaf, (
            "L8 leaf_config must have output_directory after injection"
        )
        assert l8_leaf["output_directory"] == str(out_dir), (
            f"L8 leaf_config.output_directory must equal {out_dir!s}, got {l8_leaf.get('output_directory')!r}"
        )

    def test_existing_l8_output_directory_not_overwritten(self, tmp_path: Path) -> None:
        """If recipe already has L8 leaf_config.output_directory, kwarg must not overwrite it."""
        import copy
        from macroforecast.core import execution as exec_mod

        original_run_cells = exec_mod._run_cells_serial

        captured_recipe = {}
        existing_out = "/explicit/recipe/path"

        def fake_run_cells(cell_jobs, **kwargs):
            if cell_jobs:
                captured_recipe["l8_leaf"] = (
                    cell_jobs[0][1].get("8_output", {}).get("leaf_config", {})
                )
            return original_run_cells(cell_jobs, **kwargs)

        recipe = copy.deepcopy(_MINI_RECIPE)
        recipe["8_output"] = {"leaf_config": {"output_directory": existing_out}}

        out_dir = tmp_path / "kwarg_out"
        out_dir.mkdir()

        with patch.object(exec_mod, "_run_cells_serial", fake_run_cells):
            try:
                execute_recipe(copy.deepcopy(recipe), output_directory=out_dir)
            except Exception:
                pass

        l8_leaf = captured_recipe.get("l8_leaf", {})
        assert l8_leaf.get("output_directory") == existing_out, (
            "Explicit L8 output_directory in recipe must not be overwritten by kwarg"
        )


# ---------------------------------------------------------------------------
# L2-5: SHAP subsampling with UserWarning for >2000 rows
# ---------------------------------------------------------------------------

class TestL25ShapSubsampling:
    """Cycle 14 L2-5: SHAP subsamples to 2000 rows with UserWarning for large panels."""

    def _make_large_X(self, n: int = 2500) -> pd.DataFrame:
        import numpy as np
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            rng.standard_normal((n, 5)),
            columns=[f"f{i}" for i in range(5)],
        )

    def _make_model_artifact(self, X: pd.DataFrame) -> object:
        """Build a minimal ModelArtifact using the correct dataclass fields."""
        from macroforecast.core.types import ModelArtifact
        import numpy as np

        coef = [0.1] * X.shape[1]

        class FakeModel:
            coef_ = coef
            def predict(self, Xp):
                return np.dot(Xp, self.coef_)

        return ModelArtifact(
            model_id="test_model",
            family="ridge",
            fitted_object=FakeModel(),
            framework="sklearn",
            fit_metadata={},
            feature_names=tuple(X.columns),
        )

    def test_large_panel_emits_userwarning(self) -> None:
        """_shap_importance_frame should emit UserWarning when len(X) > 2000."""
        from macroforecast.core.runtime import _shap_importance_frame, _SHAP_SUBSAMPLE_THRESHOLD

        pytest.importorskip("shap")
        X = self._make_large_X(n=_SHAP_SUBSAMPLE_THRESHOLD + 100)
        artifact = self._make_model_artifact(X)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                _shap_importance_frame(artifact, X, kind="shap_linear")
            except Exception:
                pass  # we only care about the warning being emitted

        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) >= 1, "UserWarning must be emitted for X with >2000 rows"
        assert any("subsam" in str(w.message).lower() or "slow" in str(w.message).lower()
                   for w in user_warnings), (
            "UserWarning must mention subsampling or slow SHAP"
        )

    def test_small_panel_no_warning(self) -> None:
        """_shap_importance_frame should NOT emit UserWarning when len(X) <= 2000."""
        from macroforecast.core.runtime import _shap_importance_frame, _SHAP_SUBSAMPLE_THRESHOLD

        pytest.importorskip("shap")
        X = self._make_large_X(n=_SHAP_SUBSAMPLE_THRESHOLD)
        artifact = self._make_model_artifact(X)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                _shap_importance_frame(artifact, X, kind="shap_linear")
            except Exception:
                pass

        subsample_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and ("subsam" in str(w.message).lower() or "slow" in str(w.message).lower())
        ]
        assert len(subsample_warnings) == 0, (
            f"No subsampling UserWarning should be emitted for X with {len(X)} rows (≤ threshold)"
        )

    def test_threshold_constant_is_2000(self) -> None:
        """_SHAP_SUBSAMPLE_THRESHOLD must equal 2000 per spec."""
        from macroforecast.core.runtime import _SHAP_SUBSAMPLE_THRESHOLD
        assert _SHAP_SUBSAMPLE_THRESHOLD == 2000, (
            f"Default SHAP subsample threshold must be 2000, got {_SHAP_SUBSAMPLE_THRESHOLD}"
        )
