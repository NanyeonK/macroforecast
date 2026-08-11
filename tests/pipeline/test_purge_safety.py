"""Destructive purge safety: fail closed, stay inside the store, count honestly.

Both public purge APIs delete files, and every defect this module pins was a way for
one of them to delete the wrong thing or lie about what it did:

- an unparseable ``before`` was read as "no cutoff", so a filter meant to spare recent
  entries removed all of them (``purge_result_store(store, before="not-a-date")``
  returned 1 and removed a cell it had been asked to keep);
- an alias or digest naming a path rather than an entry was interpolated straight into
  the store path, so ``aliases=["../outside"]`` deleted a sidecar outside the store, and
  ``aliases=[""]`` made the store ROOT the alias directory and ``rmdir``'d it;
- a symlinked alias directory was globbed through, whether it was named or reached by
  enumerating them all;
- the returned count incremented once per candidate CONSIDERED, so a purge that deleted
  nothing at all still reported deletions;
- a store written with a relative ``model_store=`` records a relative ``model_path``,
  which a later process resolved against ITS OWN working directory -- deleting the JSON
  sidecar and orphaning the pickle.

Validation happens before enumeration, so a refused call must leave every file in
place; that is what most of these assert alongside the ``ValueError``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import macroforecast as mf
from macroforecast.pipeline import purge_model_store, purge_result_store


def _result_cell(store: Path, digest: str = "deadbeef") -> tuple[Path, Path]:
    """One result-store cell: manifest plus parquet payload."""
    cells = store / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    manifest = cells / f"{digest}.json"
    payload = cells / f"{digest}.parquet"
    manifest.write_text(
        json.dumps(
            {
                "digest": digest,
                "created_at": "2020-01-01T00:00:00Z",
                "macroforecast_version": "1.0",
            }
        )
    )
    payload.write_bytes(b"payload")
    return manifest, payload


def _model_fit(
    store: Path,
    *,
    alias: str = "arm",
    stem: str = "origin_0_h1",
    model_path: str | None = None,
) -> tuple[Path, Path]:
    """One stored fit, laid out exactly as ``_store_model_fit`` writes it.

    ``model_path`` defaults to the absolute pickle path the writer would record; pass a
    string to simulate a relative record (a store created from another cwd) or a
    hand-edited one.
    """
    directory = store / alias
    directory.mkdir(parents=True, exist_ok=True)
    pickle_path = directory / f"{stem}.pkl"
    sidecar = directory / f"{stem}.json"
    pickle_path.write_bytes(b"fit")
    sidecar.write_text(
        json.dumps({"model_path": model_path if model_path is not None else str(pickle_path)})
    )
    return sidecar, pickle_path


def _escaping_alias(tmp_path: Path) -> tuple[Path, Path]:
    """A model store holding a symlink to a directory of sidecars outside it."""
    store = tmp_path / "store"
    store.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim.json"
    victim.write_text("{}")
    (outside / "victim.pkl").write_bytes(b"precious")
    os.symlink(outside, store / "escape")
    return store, victim


# --------------------------------------------------------------------------- #
# An unreadable cutoff is refused, not treated as no cutoff
# --------------------------------------------------------------------------- #


def test_result_purge_refuses_an_unparseable_before_without_deleting(tmp_path):
    manifest, payload = _result_cell(tmp_path)

    with pytest.raises(ValueError, match="before="):
        purge_result_store(tmp_path, before="not-a-date")

    assert manifest.exists() and payload.exists()


def test_model_purge_refuses_an_unparseable_before_without_deleting(tmp_path):
    sidecar, pickle_path = _model_fit(tmp_path)

    with pytest.raises(ValueError, match="before="):
        purge_model_store(tmp_path, before="not-a-date")

    assert sidecar.exists() and pickle_path.exists()


def test_a_parseable_before_still_filters(tmp_path):
    """The guard must not have made the ordinary cutoff stricter."""
    manifest, payload = _result_cell(tmp_path)

    assert purge_result_store(tmp_path, before="2019-01-01") == 0
    assert manifest.exists()
    assert purge_result_store(tmp_path, before="2021-01-01") == 1
    assert not manifest.exists() and not payload.exists()


# --------------------------------------------------------------------------- #
# A filter names one entry, and cannot address anything else
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("alias", ["", ".", "..", "../outside", "/etc", "a/b", "a\\b"])
def test_model_purge_refuses_an_alias_that_is_not_a_plain_name(tmp_path, alias):
    sidecar, pickle_path = _model_fit(tmp_path)

    with pytest.raises(ValueError, match="aliases"):
        purge_model_store(tmp_path, aliases=[alias])

    assert sidecar.exists() and pickle_path.exists()


@pytest.mark.parametrize("digest", ["", ".", "..", "../outside", "/etc/passwd", "a/b"])
def test_result_purge_refuses_a_digest_that_is_not_a_plain_name(tmp_path, digest):
    manifest, payload = _result_cell(tmp_path)

    with pytest.raises(ValueError, match="digests"):
        purge_result_store(tmp_path, digests=[digest])

    assert manifest.exists() and payload.exists()


def test_one_bad_entry_refuses_the_whole_call_before_any_deletion(tmp_path):
    """The reason validation runs before enumeration, stated as a test.

    Validating lazily would delete every entry listed before the bad one and then
    raise, leaving the caller with a partially purged store and an exception.
    """
    manifest, payload = _result_cell(tmp_path)

    with pytest.raises(ValueError, match="digests"):
        purge_result_store(tmp_path, digests=["deadbeef", "../outside"])

    assert manifest.exists() and payload.exists()


def test_model_purge_one_bad_alias_refuses_the_whole_call(tmp_path):
    sidecar, pickle_path = _model_fit(tmp_path)

    with pytest.raises(ValueError, match="aliases"):
        purge_model_store(tmp_path, aliases=["arm", ".."])

    assert sidecar.exists() and pickle_path.exists()


# --------------------------------------------------------------------------- #
# A name is not a containment boundary; symlinks are resolved
# --------------------------------------------------------------------------- #


def test_a_named_symlink_alias_is_not_followed(tmp_path):
    store, victim = _escaping_alias(tmp_path)

    assert purge_model_store(store, aliases=["escape"]) == 0
    assert victim.exists()


def test_enumerating_all_aliases_does_not_follow_a_symlink_out_of_the_store(tmp_path):
    """``root.glob("*/*.json")`` walked through the link; the containment check does not."""
    store, victim = _escaping_alias(tmp_path)

    assert purge_model_store(store) == 0
    assert victim.exists()


def test_result_purge_refuses_a_cells_directory_that_leaves_the_store(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "x.json"
    victim.write_text("{}")
    os.symlink(outside, store / "cells")

    with pytest.raises(ValueError, match="result store"):
        purge_result_store(store)

    assert victim.exists()


def test_an_empty_alias_can_no_longer_remove_the_store_root(tmp_path):
    store = tmp_path / "store"
    sidecar, _pickle = _model_fit(store)

    with pytest.raises(ValueError, match="aliases"):
        purge_model_store(store, aliases=[""])

    assert store.exists() and sidecar.exists()


# --------------------------------------------------------------------------- #
# The count describes deletions that happened
# --------------------------------------------------------------------------- #


def test_a_digest_naming_nothing_counts_zero(tmp_path):
    _result_cell(tmp_path)

    assert purge_result_store(tmp_path, digests=["cafebabe"]) == 0


def test_a_cell_whose_files_cannot_be_removed_counts_zero(tmp_path, monkeypatch):
    """A swallowed unlink failure must not read as a deletion.

    Cleanup stays best effort -- an unremovable file is still not an exception -- so the
    count is the only thing that can tell the caller nothing happened.
    """
    manifest, payload = _result_cell(tmp_path)

    def _deny(self, *args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "unlink", _deny)

    assert purge_result_store(tmp_path) == 0
    assert manifest.exists() and payload.exists()


def test_a_partially_present_cell_still_counts_once(tmp_path):
    """One logical entry, counted once, even though only one of its files was there.

    Counting files instead would report 1 here and 2 for a complete cell, which is a
    different quantity from the one the docstring promises. A residual file that could
    not be removed is possible and is not reported separately.
    """
    manifest, payload = _result_cell(tmp_path)
    payload.unlink()

    assert purge_result_store(tmp_path, digests=["deadbeef"]) == 1
    assert not manifest.exists()


def test_a_complete_cell_counts_once_not_twice(tmp_path):
    _result_cell(tmp_path)

    assert purge_result_store(tmp_path, digests=["deadbeef"]) == 1


# --------------------------------------------------------------------------- #
# The pickle is found without knowing the cwd the store was written from
# --------------------------------------------------------------------------- #


def test_a_relative_model_path_is_purged_from_a_different_cwd(tmp_path, monkeypatch):
    """The F-043 case: a store written with a relative ``model_store=``.

    ``save_fit`` records ``model_path`` verbatim, so the sidecar holds
    ``trained_model/arm/origin_0_h1.pkl``. Resolving that in a later process addressed
    that process's cwd instead -- the JSON went and the pickle stayed. The writer always
    puts the pickle beside its sidecar, so the sibling answers it without a cwd at all.
    """
    store = tmp_path / "trained_model"
    sidecar, pickle_path = _model_fit(
        store, model_path=str(Path("trained_model") / "arm" / "origin_0_h1.pkl")
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert purge_model_store(store) == 1
    assert not sidecar.exists()
    assert not pickle_path.exists()


def test_an_in_root_absolute_model_path_is_purged(tmp_path):
    sidecar, pickle_path = _model_fit(tmp_path)

    assert purge_model_store(tmp_path) == 1
    assert not sidecar.exists() and not pickle_path.exists()


def test_a_sidecar_with_no_recorded_model_path_still_purges_its_sibling(tmp_path):
    """``save_fit`` records ``None`` when the fit could not be pickled."""
    sidecar, pickle_path = _model_fit(tmp_path, model_path=None)
    sidecar.write_text(json.dumps({"model_path": None}))

    assert purge_model_store(tmp_path) == 1
    assert not sidecar.exists() and not pickle_path.exists()


def test_an_out_of_store_model_path_is_never_unlinked(tmp_path):
    """The sidecar belongs to this store; the file it points at does not.

    Deleting the sidecar keeps the purge convergent -- the entry the caller asked to
    remove really is gone from the store -- while the outside file is left untouched.
    The count reports 1 because a file was removed; the in-store pickle beside the
    sidecar is NOT removed, because the record explicitly names something else.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim.pkl"
    victim.write_bytes(b"precious")
    store = tmp_path / "store"
    sidecar, sibling = _model_fit(store, model_path=str(victim))

    assert purge_model_store(store) == 1
    assert victim.exists()
    assert not sidecar.exists()
    assert sibling.exists()


def test_a_hand_edited_relative_model_path_is_not_redirected_onto_the_sibling(tmp_path):
    """A relative record that names another file is refused, not reinterpreted.

    The sibling rule exists to recover a path the writer produced. Applying it to a
    record that names something else would silently delete a file the manifest never
    referred to.
    """
    store = tmp_path / "store"
    sidecar, sibling = _model_fit(store, model_path="somewhere/else.pkl")

    assert purge_model_store(store) == 1
    assert not sidecar.exists()
    assert sibling.exists()


def test_an_alias_directory_is_dropped_once_it_is_empty(tmp_path):
    sidecar, _pickle = _model_fit(tmp_path)
    alias_dir = sidecar.parent

    assert purge_model_store(tmp_path) == 1
    assert not alias_dir.exists()
    assert tmp_path.exists()


def test_purge_is_idempotent(tmp_path):
    _model_fit(tmp_path)
    _result_cell(tmp_path / "results")

    assert purge_model_store(tmp_path) == 1
    assert purge_model_store(tmp_path) == 0
    assert purge_result_store(tmp_path / "results") == 1
    assert purge_result_store(tmp_path / "results") == 0


def test_the_public_helpers_are_the_ones_under_test():
    """Guard against testing a shadowed import rather than the exported API."""
    assert purge_model_store is mf.pipeline.purge_model_store
    assert purge_result_store is mf.pipeline.purge_result_store
