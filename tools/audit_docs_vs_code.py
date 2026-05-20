"""Bidirectional drift detector for macroforecast documentation.

Scans ``.md`` files under a root directory for code-symbol-shaped tokens
(function references, result type names, version strings, YAML recipe keys,
dotted import paths, result attribute accesses) and resolves each token
against the live codebase.

Emits a JSON report with one entry per (file, line, token, verdict, evidence).
Verdicts: PASS / DRIFT / UNRESOLVABLE.

Usage::

    # Scan docs/ with default output:
    python tools/audit_docs_vs_code.py --root docs/

    # Scan and fail CI if drift found:
    python tools/audit_docs_vs_code.py --root docs/ --fail-on-drift

    # Scan standalone_functions only (expected near-zero drift):
    python tools/audit_docs_vs_code.py --root docs/standalone_functions/ --out audit.json

    # Exclude archive and changelog:
    python tools/audit_docs_vs_code.py --root docs/ \\
        --exclude 'docs/archive/**' --exclude 'docs/CHANGELOG*'
"""
from __future__ import annotations

import argparse
import datetime
import fnmatch
import importlib
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Step 1.1 — Resolve repo root and insert into sys.path
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _bootstrap_ops() -> None:
    """Import all ops modules to populate the registry."""
    import macroforecast.core.ops.l3_ops  # noqa: F401
    import macroforecast.core.ops.l4_ops  # noqa: F401
    import macroforecast.core.ops.l5_ops  # noqa: F401
    import macroforecast.core.ops.l6_ops  # noqa: F401
    import macroforecast.core.ops.l7_ops  # noqa: F401
    import macroforecast.core.ops.l8_ops  # noqa: F401
    import macroforecast.core.ops.diagnostic_ops  # noqa: F401
    import macroforecast.core.ops.universal  # noqa: F401


# ---------------------------------------------------------------------------
# Step 3.2 — Skip-context fence language tags
# Lines inside fences with these languages are NOT scanned for tokens.
# ---------------------------------------------------------------------------

SKIP_FENCE_LANGS: set[str] = {"text", "console", "bash", "shell", "output", ""}

# ---------------------------------------------------------------------------
# Step 4 — Token-class extractors (compiled regexes at module level)
# ---------------------------------------------------------------------------

# 4a. standalone_callable: `mf.functions.<NAME>` or `mf.functions.<NAME>(`
RE_STANDALONE_CALLABLE = re.compile(
    r'`mf\.functions\.([A-Za-z_][A-Za-z0-9_]*)`'
    r'|`mf\.functions\.([A-Za-z_][A-Za-z0-9_]*)\('
    r'|mf\.functions\.([A-Za-z_][A-Za-z0-9_]*)'
)

# 4b. result_type: names ending in FitResult / TestResult / ImportanceResult
RE_RESULT_TYPE = re.compile(
    r'`([A-Za-z][A-Za-z0-9]*(?:Fit|Test|Importance)Result)`'
)

# 4c. version_string: v<digits>.<digits>.<digits> with optional pre-release
RE_VERSION_STRING = re.compile(
    r'\bv(\d+\.\d+\.\d+(?:[a-z]\d+)?)\b'
)

# 4d. yaml_recipe_key: in yaml-fenced blocks
RE_YAML_RECIPE_KEY = re.compile(
    r'^\s{0,4}(op|family|point_metrics|density_metrics|direction_metrics'
    r'|relative_metrics|transform_policy|outlier_policy|imputation_policy'
    r'|frame_edge_policy|stationarity_test|failure_policy|reproducibility_mode'
    r'|compute_mode|export_format|compression):\s+["\']?([A-Za-z_][A-Za-z0-9_]*)["\']?'
)

# 4e. dotted_import_path: `macroforecast.X.Y` or `mf.X.Y` (2+ segments after root)
RE_DOTTED_IMPORT = re.compile(
    r'`(macroforecast(?:\.[A-Za-z_][A-Za-z0-9_]*){2,})`'
    r'|`(mf(?:\.[A-Za-z_][A-Za-z0-9_]*){2,})`'
)

# 4f. result_attribute: `.attr_name` in backticks
RE_RESULT_ATTR = re.compile(
    r'`(\.[A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?)`'
)

# Detect fenced code block openings: 3+ backticks + optional language tag
RE_FENCE_OPEN = re.compile(r'^(`{3,})([A-Za-z0-9_\-+]*)$')
RE_FENCE_CLOSE = re.compile(r'^(`{3,})$')

# Regex to detect standalone_callable pattern (reused in 4e exclusion check)
RE_STANDALONE_SIMPLE = re.compile(r'^mf\.functions\.[A-Za-z_][A-Za-z0-9_]*$')

# Known YAML axes and their expected layer scopes for validation
# Used in Step 5d to check if a value is a valid op name in the registry
YAML_KEY_TO_LAYER: dict[str, str | None] = {
    "op": "l3",           # L3 op axis
    "family": None,       # L4 family (checked against MODEL_FAMILY_STATUS)
    "point_metrics": "l5",
    "density_metrics": "l5",
    "direction_metrics": "l5",
    "relative_metrics": "l5",
    "transform_policy": "l2",
    "outlier_policy": "l2",
    "imputation_policy": "l2",
    "frame_edge_policy": "l2",
    # Other keys: hard to validate without richer schema; mark UNRESOLVABLE
    "stationarity_test": None,
    "failure_policy": None,
    "reproducibility_mode": None,
    "compute_mode": None,
    "export_format": None,
    "compression": None,
}


def _extract_standalone_callables(line: str) -> list[str]:
    """4a: Extract mf.functions.<NAME> tokens from a line."""
    tokens = []
    for m in RE_STANDALONE_CALLABLE.finditer(line):
        name = m.group(1) or m.group(2) or m.group(3)
        if name:
            tokens.append(f"mf.functions.{name}")
    return tokens


def _extract_result_types(line: str) -> list[str]:
    """4b: Extract result type tokens from a line."""
    return [m.group(1) for m in RE_RESULT_TYPE.finditer(line)]


def _extract_version_strings(line: str) -> list[str]:
    """4c: Extract version string tokens from a line."""
    return [f"v{m.group(1)}" for m in RE_VERSION_STRING.finditer(line)]


def _extract_yaml_keys(line: str) -> list[str]:
    """4d: Extract YAML recipe key:value tokens from a line (only inside yaml fences)."""
    m = RE_YAML_RECIPE_KEY.match(line)
    if m:
        return [f"{m.group(1)}:{m.group(2)}"]
    return []


def _extract_dotted_imports(line: str, already_standalone: list[str]) -> list[str]:
    """4e: Extract dotted import path tokens, excluding standalone_callable tokens."""
    tokens = []
    for m in RE_DOTTED_IMPORT.finditer(line):
        path = m.group(1) or m.group(2)
        if path:
            # Normalize mf. prefix
            normalized = path.replace("mf.", "macroforecast.", 1) if path.startswith("mf.") else path
            # Exclude tokens already captured as standalone_callables
            if RE_STANDALONE_SIMPLE.match(path) or path.replace("mf.", "macroforecast.functions.").endswith(path.split(".")[-1]):
                # Check if this is a standalone callable
                if any(f"mf.functions.{tok.split('.')[-1]}" == f"mf.functions.{path.split('.')[-1]}" for tok in already_standalone):
                    continue
            tokens.append(normalized)
    return tokens


def _extract_result_attributes(line: str) -> list[str]:
    """4f: Extract result attribute access tokens from a line."""
    return [m.group(1) for m in RE_RESULT_ATTR.finditer(line)]


# ---------------------------------------------------------------------------
# Step 5 — Token resolution functions
# ---------------------------------------------------------------------------


def resolve_standalone_callable(
    token: str,
    functions_all: set[str],
) -> tuple[str, str]:
    """5a: Resolve mf.functions.<NAME> token.

    Returns (verdict, evidence) tuple.
    """
    name = token.split(".")[-1]
    # Strip any trailing parenthesis if present
    name = name.rstrip("(")
    if name in functions_all:
        return "PASS", f"found in macroforecast.functions.__all__"
    return "DRIFT", f"{name!r} not in macroforecast.functions.__all__"


def resolve_result_type(
    token: str,
    mf_functions: Any,
) -> tuple[str, str]:
    """5b: Resolve FitResult / TestResult / ImportanceResult class name token."""
    if len(token) < 5:
        return "UNRESOLVABLE", "token too short to distinguish"
    if hasattr(mf_functions, token):
        return "PASS", f"found in macroforecast.functions"
    return "DRIFT", f"class {token!r} not found in macroforecast.functions"


def resolve_version_string(
    token: str,
    version: str,
    file_path: str,
    line_text: str,
) -> tuple[str, str]:
    """5c: Resolve version string token against macroforecast.__version__."""
    ver_from_token = token.lstrip("v")
    ver_normalized = version.lstrip("v")

    # Check if this is a changelog context (intentionally historical)
    is_changelog = (
        "CHANGELOG" in file_path.upper()
        or "changelog" in file_path.lower()
        or "Unreleased" in line_text
        or "## v" in line_text
        or "Added" in line_text
        or "Fixed" in line_text
        or "Changed" in line_text
    )

    if is_changelog:
        return "UNRESOLVABLE", "version string in changelog context — intentionally historical"

    if ver_from_token == ver_normalized:
        return "PASS", f"matches macroforecast.__version__ = {version!r}"

    return "DRIFT", f"version {token!r} != package version v{version!r}"


def resolve_yaml_recipe_key(
    token: str,
    ops: dict[str, Any],
    mf_functions: Any,
) -> tuple[str, str]:
    """5d: Resolve YAML recipe key:value token against the op registry."""
    parts = token.split(":", 1)
    if len(parts) != 2:
        return "UNRESOLVABLE", "malformed key:value token"

    key, value = parts[0], parts[1]

    if key == "family":
        # Check against MODEL_FAMILY_STATUS
        try:
            from macroforecast.core.ops.l4_ops import MODEL_FAMILY_STATUS
            if value in MODEL_FAMILY_STATUS:
                return "PASS", f"family {value!r} in MODEL_FAMILY_STATUS"
            return "DRIFT", f"family {value!r} not in MODEL_FAMILY_STATUS"
        except ImportError:
            return "UNRESOLVABLE", "could not import MODEL_FAMILY_STATUS"

    layer_id = YAML_KEY_TO_LAYER.get(key)

    if layer_id is None:
        # Keys like failure_policy, compression are not easily validated
        return "UNRESOLVABLE", f"no validation rule for YAML key {key!r}"

    # Check if value is a known op in the registry with the right layer scope
    if value in ops:
        op_spec = ops[value]
        scope = op_spec.layer_scope
        if scope == "universal" or layer_id in scope:
            return "PASS", f"op {value!r} in registry with scope including {layer_id!r}"
        return (
            "DRIFT",
            f"op {value!r} found in registry but scope {scope} does not include {layer_id!r}",
        )

    # For L5/L6 virtual ops, check the standalone functions namespace
    if layer_id in ("l5", "l6"):
        if hasattr(mf_functions, value):
            return "PASS", f"value {value!r} found in macroforecast.functions"
        # Check OP_TO_STANDALONE mappings
        from tools.gen_encyclopedia_docs import OP_TO_STANDALONE, L5_VIRTUAL_OPS, L6_VIRTUAL_OPS
        if value in L5_VIRTUAL_OPS or value in L6_VIRTUAL_OPS:
            return "PASS", f"value {value!r} is a known virtual op for {layer_id!r}"
        return "DRIFT", f"value {value!r} not found in registry or functions namespace"

    return "DRIFT", f"op {value!r} not found in registry"


def resolve_dotted_import(
    token: str,
) -> tuple[str, str]:
    """5e: Resolve dotted import path token via importlib.

    Uses static resolution only: importlib.import_module + getattr.
    Never calls eval or exec.
    """
    # Split into module path and attribute
    parts = token.split(".")
    if len(parts) < 2:
        return "UNRESOLVABLE", "too short to resolve"

    # Try progressively shorter module paths + attribute
    for split_idx in range(len(parts) - 1, 0, -1):
        module_path = ".".join(parts[:split_idx])
        attr_chain = parts[split_idx:]

        try:
            mod = importlib.import_module(module_path)
        except (ImportError, ModuleNotFoundError) as e:
            continue
        except Exception as e:
            # Unexpected import error (e.g. optional dep missing)
            return "UNRESOLVABLE", f"import error for {module_path!r}: {e}"

        # Walk the attribute chain
        obj = mod
        try:
            for attr in attr_chain:
                obj = getattr(obj, attr)
            return "PASS", f"resolved via importlib: {token}"
        except AttributeError:
            return "DRIFT", f"attribute chain failed at {attr!r} in {token}"

    return "DRIFT", f"could not import any module prefix of {token}"


def resolve_result_attribute(
    token: str,
    context_type_names: list[str],
    mf_functions: Any,
) -> tuple[str, str]:
    """5f: Resolve result attribute access token given preceding result type context.

    For dataclasses, checks ``__dataclass_fields__`` since plain ``hasattr``
    does not see per-instance fields. Also checks ``__annotations__`` and
    callable attributes (predict, summary, etc.) via ``hasattr``.
    """
    import dataclasses as _dc

    if not context_type_names:
        return "UNRESOLVABLE", "no result type context in preceding lines"

    # Strip leading dot and trailing (...) for attribute lookup
    attr_name = token.lstrip(".")
    attr_name = attr_name.split("(")[0]

    for type_name in context_type_names:
        cls = getattr(mf_functions, type_name, None)
        if cls is None:
            continue

        # 1. Check dataclass fields (covers instance fields not visible via hasattr on class)
        if _dc.is_dataclass(cls):
            dc_field_names = {f.name for f in _dc.fields(cls)}
            if attr_name in dc_field_names:
                return "PASS", f"dataclass field {attr_name!r} found on {type_name}"

        # 2. Check __annotations__ (catches typed attributes on non-dataclass classes)
        if attr_name in getattr(cls, "__annotations__", {}):
            return "PASS", f"annotation {attr_name!r} found on {type_name}"

        # 3. Check class-level attribute or method (hasattr works for class methods)
        if hasattr(cls, attr_name):
            return "PASS", f"attribute {attr_name!r} found on {type_name}"

    return "DRIFT", (
        f"attribute {attr_name!r} not found on result type(s): "
        + ", ".join(context_type_names)
    )


# ---------------------------------------------------------------------------
# Step 2 + 3 — File collection and per-file token scanning
# ---------------------------------------------------------------------------


def collect_md_files(root: Path, exclude_patterns: list[str]) -> list[Path]:
    """Collect .md files under root, sorted lexicographically, excluding patterns."""
    files = []
    for path in sorted(root.rglob("*.md")):
        # Check exclude patterns
        path_str = str(path)
        excluded = False
        for pattern in exclude_patterns:
            # Use fnmatch for glob patterns
            if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(path.name, pattern):
                excluded = True
                break
            # Also try relative path matching
            try:
                rel = str(path.relative_to(root))
                if fnmatch.fnmatch(rel, pattern):
                    excluded = True
                    break
            except ValueError:
                pass
        if not excluded:
            files.append(path)
    return files


def scan_file(
    file_path: Path,
    root: Path,
    functions_all: set[str],
    version: str,
    ops: dict[str, Any],
    mf_functions: Any,
) -> list[dict[str, Any]]:
    """Scan a single .md file and return a list of token entry dicts."""
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str]] = set()

    # Read file content
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError:
        return entries

    lines = content.splitlines()
    rel_path = str(file_path.relative_to(root))

    # State machine for fenced code blocks
    in_skip_fence = False       # True when inside a non-yaml skip-context fence
    in_yaml_fence = False       # True when inside a yaml-labeled fence
    fence_marker = ""           # The backtick characters that opened the current fence

    # Rolling window of recently seen result type names (for 4f context)
    recent_result_types: list[str] = []  # recent result type names (last 3 lines)

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Detect fence open/close
        m_open = RE_FENCE_OPEN.match(stripped)
        m_close = RE_FENCE_CLOSE.match(stripped)

        if m_open and not in_skip_fence and not in_yaml_fence:
            # Opening a new fence
            fence_marker = m_open.group(1)
            lang = m_open.group(2).lower().strip()
            if lang in SKIP_FENCE_LANGS:
                in_skip_fence = True
            elif lang == "yaml":
                in_yaml_fence = True
            # Don't scan the fence-opening line itself
            continue

        if (in_skip_fence or in_yaml_fence) and m_close:
            # Check if this close matches the opening fence marker length
            if len(m_close.group(1)) >= len(fence_marker):
                in_skip_fence = False
                in_yaml_fence = False
                fence_marker = ""
                continue

        # Skip lines inside skip-context fences (bash/console/text/shell/output)
        if in_skip_fence:
            continue

        # Extract result types on this line first so they're available as context
        # for attribute resolution on the SAME line (common pattern:
        # "Returns `LassoFitResult`: `.alpha`, `.coef_`").
        types_on_line = _extract_result_types(line)

        # Include this line's result types in the context for attribute resolution,
        # then roll the window for the next line.
        current_attr_context = types_on_line + recent_result_types

        # --- 4a: standalone_callable ---
        standalone_tokens = _extract_standalone_callables(line)
        for token in standalone_tokens:
            key = (rel_path, lineno, token, "standalone_callable")
            if key in seen:
                continue
            seen.add(key)
            verdict, evidence = resolve_standalone_callable(token, functions_all)
            entries.append({
                "file": rel_path,
                "line": lineno,
                "token": token,
                "token_class": "standalone_callable",
                "verdict": verdict,
                "evidence": evidence,
            })

        # --- 4b: result_type ---
        for token in types_on_line:
            key = (rel_path, lineno, token, "result_type")
            if key in seen:
                continue
            seen.add(key)
            verdict, evidence = resolve_result_type(token, mf_functions)
            entries.append({
                "file": rel_path,
                "line": lineno,
                "token": token,
                "token_class": "result_type",
                "verdict": verdict,
                "evidence": evidence,
            })

        # --- 4c: version_string ---
        for token in _extract_version_strings(line):
            key = (rel_path, lineno, token, "version_string")
            if key in seen:
                continue
            seen.add(key)
            verdict, evidence = resolve_version_string(
                token, version, rel_path, line
            )
            entries.append({
                "file": rel_path,
                "line": lineno,
                "token": token,
                "token_class": "version_string",
                "verdict": verdict,
                "evidence": evidence,
            })

        # --- 4d: yaml_recipe_key (only inside yaml fences) ---
        if in_yaml_fence:
            for token in _extract_yaml_keys(line):
                key = (rel_path, lineno, token, "yaml_recipe_key")
                if key in seen:
                    continue
                seen.add(key)
                verdict, evidence = resolve_yaml_recipe_key(token, ops, mf_functions)
                entries.append({
                    "file": rel_path,
                    "line": lineno,
                    "token": token,
                    "token_class": "yaml_recipe_key",
                    "verdict": verdict,
                    "evidence": evidence,
                })

        # --- 4e: dotted_import_path (not in skip fences) ---
        dotted_tokens = _extract_dotted_imports(line, standalone_tokens)
        for token in dotted_tokens:
            key = (rel_path, lineno, token, "dotted_import_path")
            if key in seen:
                continue
            seen.add(key)
            # Only resolve if inside a code fence or import statement
            if "import" in line or line.strip().startswith("`"):
                verdict, evidence = resolve_dotted_import(token)
                entries.append({
                    "file": rel_path,
                    "line": lineno,
                    "token": token,
                    "token_class": "dotted_import_path",
                    "verdict": verdict,
                    "evidence": evidence,
                })

        # --- 4f: result_attribute ---
        # Use current_attr_context which includes types found on this line
        # (so "Returns `LassoFitResult`: `.alpha`" works correctly).
        if current_attr_context:
            for token in _extract_result_attributes(line):
                key = (rel_path, lineno, token, "result_attribute")
                if key in seen:
                    continue
                seen.add(key)
                verdict, evidence = resolve_result_attribute(
                    token, current_attr_context, mf_functions
                )
                entries.append({
                    "file": rel_path,
                    "line": lineno,
                    "token": token,
                    "token_class": "result_attribute",
                    "verdict": verdict,
                    "evidence": evidence,
                })

        # Update rolling result-type window with types found on this line.
        # The window tracks the last 3 lines of result types for cross-line context.
        recent_result_types = types_on_line + recent_result_types
        # Cap at 3 lines worth of types (estimate 3 types per line max)
        if len(recent_result_types) > 9:
            recent_result_types = recent_result_types[:9]

    return entries


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point for audit_docs_vs_code.py.

    Returns 0 on success (or drift without --fail-on-drift),
    1 on drift with --fail-on-drift, 2 on error.
    """
    parser = argparse.ArgumentParser(
        prog="audit_docs_vs_code.py",
        description="Detect drift between macroforecast docs and live codebase.",
    )
    parser.add_argument(
        "--root",
        default="docs/",
        metavar="DIR",
        help="Root directory to scan for .md files. Default: docs/.",
    )
    parser.add_argument(
        "--out",
        default="audit-report.json",
        metavar="FILE",
        help="Output JSON report path. Default: audit-report.json.",
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit with code 1 if any DRIFT verdict is found.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="Glob pattern for files to exclude. Repeatable.",
    )

    args = parser.parse_args()

    # Validate --root
    root = Path(args.root)
    if not root.is_dir():
        print(
            f"ERROR: --root {args.root!r} is not an existing directory.",
            file=sys.stderr,
        )
        return 2

    # Validate --out parent directory exists
    out_path = Path(args.out)
    if not out_path.parent.exists():
        print(
            f"ERROR: output directory {out_path.parent!r} does not exist.",
            file=sys.stderr,
        )
        return 2

    # Bootstrap ops registry
    try:
        _bootstrap_ops()
    except Exception as e:
        print(f"ERROR: Failed to import ops modules: {e}", file=sys.stderr)
        return 2

    # Import macroforecast package
    try:
        import macroforecast
        import macroforecast.functions as mf_functions
    except Exception as e:
        print(f"ERROR: Failed to import macroforecast: {e}", file=sys.stderr)
        return 2

    from macroforecast.core.ops.registry import list_ops

    # Cache live codebase state
    functions_all: set[str] = set(mf_functions.__all__)
    version: str = macroforecast.__version__
    ops: dict[str, Any] = list_ops()

    # Collect .md files
    md_files = collect_md_files(root, args.exclude)

    # Process all files
    all_entries: list[dict[str, Any]] = []

    for file_path in md_files:
        file_entries = scan_file(
            file_path=file_path,
            root=root,
            functions_all=functions_all,
            version=version,
            ops=ops,
            mf_functions=mf_functions,
        )
        all_entries.extend(file_entries)

    # Sort entries by (file, line, token) for deterministic output
    all_entries.sort(key=lambda e: (e["file"], e["line"], e["token"]))

    # Compute summary counts
    n_pass = sum(1 for e in all_entries if e["verdict"] == "PASS")
    n_drift = sum(1 for e in all_entries if e["verdict"] == "DRIFT")
    n_unresolvable = sum(1 for e in all_entries if e["verdict"] == "UNRESOLVABLE")
    n_total = len(all_entries)

    # Build report
    report: dict[str, Any] = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "root": str(root.resolve()),
        "package_version": version,
        "summary": {
            "files_scanned": len(md_files),
            "total_tokens": n_total,
            "pass": n_pass,
            "drift": n_drift,
            "unresolvable": n_unresolvable,
        },
        "entries": all_entries,
    }

    # Write report
    try:
        out_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as e:
        print(f"ERROR: Cannot write report to {out_path}: {e}", file=sys.stderr)
        return 2

    # Print summary line
    print(
        f"Scanned {len(md_files)} files, {n_total} tokens: "
        f"{n_pass} PASS, {n_drift} DRIFT, {n_unresolvable} UNRESOLVABLE"
    )

    # Exit code logic
    if args.fail_on_drift and n_drift > 0:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
