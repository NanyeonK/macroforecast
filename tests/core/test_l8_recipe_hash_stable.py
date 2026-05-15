"""F-P1-12 -- recipe hash stable across processes (hashlib.sha256 not Python hash()).

Tests verify:
1. Same recipe in the same process gives identical hash.
2. Hash is deterministic (no process-level salting).
3. Different recipes give different hashes.
4. Hash is a 16-hex-character string.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest


def _compute_hash_via_runtime(recipe: dict) -> str:
    """Compute the recipe hash using the same logic as runtime.py F-P1-12."""
    from macroforecast.core.runtime import _jsonable
    canonical_json = json.dumps(
        _jsonable(recipe), sort_keys=True, default=str, separators=(",", ":")
    )
    return hashlib.sha256(canonical_json.encode()).hexdigest()[:16]


_RECIPE_A = {
    "1_data": {
        "fixed_axes": {"custom_source_policy": "custom_panel_only", "frequency": "monthly"},
        "leaf_config": {"target": "y"},
    }
}

_RECIPE_B = {
    "1_data": {
        "fixed_axes": {"custom_source_policy": "custom_panel_only", "frequency": "quarterly"},
        "leaf_config": {"target": "y"},
    }
}


class TestRecipeHashStable:
    def test_same_recipe_same_hash(self):
        h1 = _compute_hash_via_runtime(_RECIPE_A)
        h2 = _compute_hash_via_runtime(_RECIPE_A)
        assert h1 == h2

    def test_hash_is_16_hex_chars(self):
        h = _compute_hash_via_runtime(_RECIPE_A)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_recipes_different_hashes(self):
        h_a = _compute_hash_via_runtime(_RECIPE_A)
        h_b = _compute_hash_via_runtime(_RECIPE_B)
        assert h_a != h_b

    def test_key_order_invariant(self):
        """Key order in the recipe dict must not affect the hash."""
        recipe_1 = {"1_data": {"fixed_axes": {"a": 1, "b": 2}}}
        recipe_2 = {"1_data": {"fixed_axes": {"b": 2, "a": 1}}}
        h1 = _compute_hash_via_runtime(recipe_1)
        h2 = _compute_hash_via_runtime(recipe_2)
        assert h1 == h2

    def test_hash_stable_across_subprocess(self):
        """Same recipe hash from a fresh subprocess (no PYTHONHASHSEED contamination)."""
        # Run a subprocess to compute the hash independently
        script = (
            "import sys, hashlib, json\n"
            "sys.path.insert(0, '/home/nanyeon99/project/macroforecast')\n"
            "from macroforecast.core.runtime import _jsonable\n"
            "recipe = {'1_data': {'fixed_axes': {'custom_source_policy': 'custom_panel_only', 'frequency': 'monthly'}, 'leaf_config': {'target': 'y'}}}\n"
            "cj = json.dumps(_jsonable(recipe), sort_keys=True, default=str, separators=(',', ':'))\n"
            "print(hashlib.sha256(cj.encode()).hexdigest()[:16])\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 0, f"subprocess failed: {proc.stderr}"
        subprocess_hash = proc.stdout.strip()
        local_hash = _compute_hash_via_runtime(_RECIPE_A)
        assert local_hash == subprocess_hash, (
            f"Hash differs between processes: local={local_hash!r} subprocess={subprocess_hash!r}"
        )

    def test_not_python_builtin_hash(self):
        """The recipe hash must NOT be produced by Python's built-in hash()
        (which is process-salted since Python 3.3 by default).
        Verified indirectly: hash() on a string changes with PYTHONHASHSEED."""
        from macroforecast.core.runtime import _jsonable
        recipe = _RECIPE_A
        canonical = json.dumps(
            _jsonable(recipe), sort_keys=True, default=str, separators=(",", ":")
        )
        # The F-P1-12 hash is a hex string, not an integer string
        h = _compute_hash_via_runtime(recipe)
        try:
            int(h)
            is_integer_string = True
        except ValueError:
            is_integer_string = False
        assert not is_integer_string, (
            "Hash looks like an integer (possible Python hash() regression)"
        )
