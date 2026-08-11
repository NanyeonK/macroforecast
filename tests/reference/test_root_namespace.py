"""The discovered root surface is exactly ``__all__``, with nothing beside it.

``macroforecast/__init__.py`` publishes its own module globals: a name bound at
that scope is reachable as ``mf.<name>`` whether or not anyone meant it to be.
Three names arrived that way and none was ever API -- ``Any`` and
``import_module`` from the lazy loader's imports, and ``annotations`` from the
future statement, which binds a name at runtime as a side effect of a directive
whose real effect is compiled in. The root-namespace cleanup on this branch
removed all three.

The invariant asserted here is about the **discovered** surface: the
non-underscore names in ``dir(mf)`` equal ``mf.__all__``. It is deliberately not
"everything public equals ``__all__``", because ``mf.__version__`` is public on
purpose and sits outside ``__all__`` -- an underscore-prefixed name a discovery
scan skips, which is exactly why the inventory carries it as its own ``special``
surface. A test that conflated the two would forbid a name the policy keeps.

The assertions are two-sided on purpose: the three names must be *gone*, and the
376 supported names, ``__version__``, the lazy loading, the caching and the
resolvable ``__getattr__`` annotations must be *unchanged*. Both sides are
load-bearing -- a cleanup that imports ``Any`` publicly and deletes it with
``del`` at the end of the module also empties the surface, and breaks
``get_type_hints`` while doing so.

The namespace assertions run in a fresh interpreter on purpose. Within one
pytest session an earlier test can have triggered a lazy import, so an
in-process ``dir(mf)`` measures the accumulated state of the session rather than
what a user gets on ``import macroforecast``.
"""
from __future__ import annotations

import subprocess
import sys
import typing
from pathlib import Path

import macroforecast as mf

#: Bound at root module scope before this cleanup; never exported, never documented.
REMOVED = ("Any", "annotations", "import_module")

SUPPORTED = 376
LAZY_EXPORTS = 353
LAZY_MODULES = 23

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Asserts that the fresh interpreter measured *this* checkout rather than an
# installed copy, so a passing run cannot be an artefact of import shadowing.
_PREAMBLE = f"""
import macroforecast as mf
from pathlib import Path

root = Path(mf.__file__).resolve().parents[1]
assert root == Path({str(_PROJECT_ROOT)!r}), root
"""


def _fresh(body: str) -> None:
    """Run ``body`` in a fresh interpreter; fail with its stderr if it exits non-zero."""

    result = subprocess.run(
        [sys.executable, "-c", _PREAMBLE + body],
        cwd=_PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_fresh_import_leaves_no_discovered_name_outside_all() -> None:
    """The non-underscore ``dir(mf)`` scan equals ``__all__`` -- no more, no less."""

    _fresh(
        f"""
discovered = {{name for name in dir(mf) if not name.startswith("_")}}
assert discovered == set(mf.__all__), sorted(discovered ^ set(mf.__all__))
assert len(discovered) == {SUPPORTED}, len(discovered)
"""
    )


def test_version_stays_public_outside_the_discovered_surface() -> None:
    """``__version__`` is deliberately public and deliberately not in ``__all__``.

    Removing the three globals must not be over-applied into "the root exposes
    nothing but ``__all__``": this name is the documented exception, and it is
    invisible to the scan above only because it is underscore-prefixed.
    """

    _fresh(
        """
assert isinstance(mf.__version__, str) and mf.__version__
assert "__version__" not in mf.__all__
assert "__version__" in dir(mf)
assert "__version__" in mf.__dict__
"""
    )


def test_fresh_import_removes_the_three_non_api_globals() -> None:
    _fresh(
        f"""
for name in {REMOVED!r}:
    assert not hasattr(mf, name), name
    assert name not in dir(mf), name
    assert name not in mf.__dict__, name
    assert name not in mf.__all__, name
    try:
        getattr(mf, name)
    except AttributeError as exc:
        assert name in str(exc), (name, str(exc))
    else:
        raise AssertionError(name + " resolved instead of raising AttributeError")
"""
    )


def test_fresh_import_preserves_every_supported_behaviour() -> None:
    _fresh(
        f"""
import typing

assert isinstance(mf.__version__, str) and mf.__version__

assert len(mf.__all__) == {SUPPORTED}
assert len(mf._LAZY_EXPORTS) == {LAZY_EXPORTS}
assert len(mf._LAZY_MODULES) == {LAZY_MODULES}

# configure is the one lazy export that stays a stable root name at 1.0.
assert mf.configure.__name__ == "configure"
assert mf.configure is mf.meta.configure

# Namespace export.
assert mf.models.__name__ == "macroforecast.models"

# Lazy export: not resolved until asked for, then cached into root globals.
assert "random_forest" not in mf.__dict__
assert mf.random_forest is mf.models.random_forest
assert mf.__dict__["random_forest"] is mf.random_forest

hints = typing.get_type_hints(mf.__getattr__)
assert hints == {{"name": str, "return": typing.Any}}, hints
"""
    )


def test_all_is_the_unchanged_supported_surface() -> None:
    assert len(mf.__all__) == SUPPORTED
    assert len(mf._LAZY_EXPORTS) == LAZY_EXPORTS
    assert len(mf._LAZY_MODULES) == LAZY_MODULES
    assert SUPPORTED == LAZY_EXPORTS + LAZY_MODULES

    assert set(mf.__all__) == set(mf._LAZY_EXPORTS) | set(mf._LAZY_MODULES)
    assert set(mf._LAZY_EXPORTS) & set(mf._LAZY_MODULES) == set()
    assert mf.__all__ == sorted(mf.__all__)
    assert len(set(mf.__all__)) == len(mf.__all__)

    for name in REMOVED:
        assert name not in mf.__all__


def test_getattr_annotations_stay_resolvable() -> None:
    """``Any`` is aliased to ``_Any``; the alias must still resolve to ``typing.Any``.

    A cleanup that dropped the import, or aliased it to a name the module does
    not bind, would leave a ``__getattr__`` whose annotations cannot be
    evaluated -- and under PEP 563 that failure is invisible until something
    calls ``get_type_hints``.
    """

    hints = typing.get_type_hints(mf.__getattr__)

    assert hints["name"] is str
    assert hints["return"] is typing.Any
    assert hints == {"name": str, "return": typing.Any}


def test_unknown_root_attribute_raises_attributeerror() -> None:
    for name in (*REMOVED, "definitely_not_an_export"):
        try:
            getattr(mf, name)
        except AttributeError as exc:
            assert "macroforecast" in str(exc)
            assert name in str(exc)
        else:  # pragma: no cover - only reachable on a regression
            raise AssertionError(f"mf.{name} resolved unexpectedly")


def test_dir_reports_the_supported_surface_without_duplicates() -> None:
    listing = dir(mf)

    assert listing == sorted(listing)
    assert len(listing) == len(set(listing))

    discovered = [name for name in listing if not name.startswith("_")]
    assert set(discovered) == set(mf.__all__)

    # Public, listed, and outside both of the sets above -- by design.
    assert "__version__" in listing
    assert "__version__" not in discovered
