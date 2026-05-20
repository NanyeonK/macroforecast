"""Introspection-based catalog generator for standalone callables.

Produces verified signatures and result attributes for every callable in
``macroforecast.functions.__all__``, layered by module.

Used by Cycle 41 v0.9.2b1 docs rewrite — outputs ground truth that
``docs/standalone_functions/{l2..l7}.md`` MUST verbatim-transcribe.
"""
from __future__ import annotations

import dataclasses
import inspect
import os as _os
import sys
_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import traceback
from typing import Any

import numpy as np
import pandas as pd

import macroforecast.functions as f


# Module-to-layer mapping (verified by Cycle 41 layer audit)
LAYER_MAP = {
    "macroforecast.functions.clean": "L2",
    "macroforecast.functions.transforms": "L3",
    "macroforecast.functions.linear": "L4",
    "macroforecast.functions.ridge": "L4",
    "macroforecast.functions.tree": "L4",
    "macroforecast.functions.deep": "L4",
    "macroforecast.functions.timeseries": "L4",
    "macroforecast.functions.misc": "L4",
    "macroforecast.functions.metrics": "L5",
    "macroforecast.functions.theil_u": "L5",
    "macroforecast.functions.tests": "L6",
    "macroforecast.functions.importance": "L7",
}


def first_doc_line(obj: Any) -> str:
    doc = inspect.getdoc(obj) or ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def render_signature(name: str, obj: Any) -> str:
    """Render signature stripping quotes around type annotations for readability."""
    sig = inspect.signature(obj)
    # Get the canonical string form (with quotes)
    text = f"{name}{sig}"
    # Strip the forward-ref quotes added by from __future__ import annotations
    text = text.replace(chr(39), "")
    return text


def try_instantiate_result(name: str, obj: Any) -> Any | None:
    """Try to call the callable with minimal valid args to get a live result."""
    rng = np.random.default_rng(0)

    # L4 fit callables: (X, y)
    if name.endswith("_fit"):
        X = rng.standard_normal((80, 5))
        y = rng.standard_normal(80)
        try:
            if name == "realized_garch_fit":
                rv = np.abs(rng.standard_normal(80))
                return obj(X, y, rv)
            return obj(X, y)
        except Exception:
            return None

    # L6 tests
    if name in ("dm_test", "gw_test"):
        return obj(rng.standard_normal(100)**2, rng.standard_normal(100)**2)
    if name == "dmp_test":
        return obj([rng.standard_normal(100), rng.standard_normal(100)])
    if name == "hn_test":
        return obj(rng.standard_normal(100), rng.standard_normal(100))
    if name == "cw_test":
        return obj(rng.standard_normal(100)**2, rng.standard_normal(100)**2,
                   rng.standard_normal(100), rng.standard_normal(100))
    if name in ("enc_new_test", "enc_t_test"):
        return obj(rng.standard_normal(100)**2, rng.standard_normal(100)**2)

    # L7 importance: need a fitted result + X (+y)
    if "_importance" in name:
        X = rng.standard_normal((80, 5))
        y = rng.standard_normal(80)
        if "tree" in name:
            res = f.random_forest_fit(X, y, n_estimators=10)
        else:
            res = f.ols_fit(X, y)
        try:
            if name in ("permutation_importance", "cond_permutation_importance"):
                return obj(res, X, y)
            return obj(res, X)
        except Exception:
            return None

    return None


def public_attrs(result: Any) -> list[str]:
    if result is None:
        return []
    return sorted(a for a in dir(result) if not a.startswith("_"))


def main() -> int:
    by_layer: dict[str, list[tuple[str, Any]]] = {f"L{i}": [] for i in range(2, 8)}
    for name in sorted(f.__all__):
        if not name[0].islower():
            continue
        obj = getattr(f, name)
        layer = LAYER_MAP.get(obj.__module__)
        if layer is None:
            print(f"WARN: unmapped module {obj.__module__} for {name}", file=sys.stderr)
            continue
        by_layer[layer].append((name, obj))

    for layer in sorted(by_layer):
        entries = by_layer[layer]
        print(f"### {layer} ({len(entries)} callables)")
        print()
        for name, obj in entries:
            sig_text = render_signature(name, obj)
            return_anno = inspect.signature(obj).return_annotation
            ret_name = (
                return_anno.__name__ if hasattr(return_anno, "__name__")
                else str(return_anno).replace(chr(39), "")
            )
            doc1 = first_doc_line(obj)
            print(f"#### {name}")
            print(f"  module: {obj.__module__}")
            print(f"  signature: {sig_text}")
            print(f"  return_annotation: {ret_name}")
            print(f"  doc_first_line: {doc1}")
            # Try to instantiate result and dump public attrs
            result = try_instantiate_result(name, obj)
            attrs = public_attrs(result)
            if attrs:
                print(f"  result_attrs: {attrs}")
            else:
                print(f"  result_attrs: (not introspected — optional extra or X,y-only fit)")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
