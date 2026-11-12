"""mount_resolver: redacted input listing, datasets located by file hash, fail-closed on absent / wrong-hash / ambiguous mounts."""
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "experiment_packaging"))

import dependency_gate as DG  # noqa: E402
import mount_resolver as MR  # noqa: E402

USER, SLUG_C, SLUG_X = "someowner", "some-ckpt-slug", "some-x-slug"
DATA = {"C": {"checkpoint_300.pt": b"checkpoint bytes"}, "X": {"a.pt": b"aaa", "b.json": b"{}"}}


def _lock(inp):
    lock = {"assets": {}, "datasets": {}}
    for role, slug in (("C", SLUG_C), ("X", SLUG_X)):
        lock["datasets"][role] = {"id": f"{USER}/{slug}", "mount": str(inp / "datasets" / USER / slug),
                                  "files": {n: hashlib.sha256(b).hexdigest() for n, b in DATA[role].items()}}
    return lock


def _put(d, role, corrupt=False):
    d.mkdir(parents=True, exist_ok=True)
    for n, b in DATA[role].items():
        (d / n).write_bytes(b + (b"!" if corrupt else b""))


def _setup(tmp_path, c_at=None, c_corrupt=False):
    inp = tmp_path / "input"
    lock = _lock(inp)
    _put(Path(lock["datasets"]["X"]["mount"]), "X")
    if c_at is not None:
        _put(inp / c_at, "C", corrupt=c_corrupt)
    lp = tmp_path / "lock.json"
    lp.write_text(json.dumps(lock))
    (tmp_path / "work").mkdir()
    return inp, lock, lp


def _check(resolved_lock_path):
    return DG.verify_lock(resolved_lock_path)                # unchanged gate: hashes every locked file at the mount in the lock


def test_pinned_mount_used_as_before(tmp_path):
    inp, lock, lp = _setup(tmp_path, c_at=Path("datasets") / USER / SLUG_C)
    out = MR.resolve_and_write(lp, inp, tmp_path / "work")
    assert {k: v["method"] for k, v in out["report"].items()} == {"C": "pinned", "X": "pinned"}
    assert _check(out["resolved_lock_path"])["checked_files"] == 3


def test_relocated_checkpoint_is_found_by_hash_and_rehashed_by_the_gate(tmp_path):
    inp, lock, lp = _setup(tmp_path, c_at=Path(SLUG_C))                 # the other Kaggle layout: <input>/<slug>
    out = MR.resolve_and_write(lp, inp, tmp_path / "work")
    assert out["report"]["C"]["method"] == "hash" and out["roots"]["C"] == str(inp / SLUG_C)
    v = _check(out["resolved_lock_path"])
    assert str(inp / SLUG_C / "checkpoint_300.pt") in v["hashes"]
    assert json.loads((tmp_path / "work" / "mount_resolution.json").read_text())["C"]["actual_equals_pinned"] is False


def test_pinned_path_with_wrong_hash_stays_fatal_and_is_not_searched(tmp_path):
    inp, lock, lp = _setup(tmp_path, c_at=Path("datasets") / USER / SLUG_C, c_corrupt=True)
    _put(inp / "elsewhere", "C")                                          # a good copy elsewhere must NOT rescue a bad pinned mount
    out = MR.resolve_and_write(lp, inp, tmp_path / "work")
    assert out["report"]["C"]["method"] == "pinned"
    with pytest.raises(ValueError, match="CORRUPT_DATASET_FILE"):
        _check(out["resolved_lock_path"])


def test_relocated_wrong_hash_is_missing(tmp_path):
    inp, lock, lp = _setup(tmp_path, c_at=Path(SLUG_C), c_corrupt=True)
    with pytest.raises(MR.MountError, match="MISSING_DATASET_ROOT C .*1 directories had the names with a wrong hash"):
        MR.resolve_and_write(lp, inp, tmp_path / "work")
    assert (tmp_path / "work" / "input_listing.json").is_file()        # listing written before resolution, even on failure


def test_absent_is_missing(tmp_path):
    inp, lock, lp = _setup(tmp_path, c_at=None)
    with pytest.raises(MR.MountError, match="MISSING_DATASET_ROOT C"):
        MR.resolve_and_write(lp, inp, tmp_path / "work")


def test_two_distinct_matching_copies_are_ambiguous(tmp_path):
    inp, lock, lp = _setup(tmp_path, c_at=Path("copy1"))
    _put(inp / "copy2", "C")
    with pytest.raises(MR.MountError, match="AMBIGUOUS_DATASET_ROOT C"):
        MR.resolve_and_write(lp, inp, tmp_path / "work")


def test_symlinked_duplicate_of_one_directory_counts_once(tmp_path):
    inp, lock, lp = _setup(tmp_path, c_at=Path(SLUG_C))
    os.symlink(inp / SLUG_C, inp / "alias")
    out = MR.resolve_and_write(lp, inp, tmp_path / "work")
    assert out["report"]["C"]["method"] == "hash"


def test_partial_file_set_does_not_match(tmp_path):
    inp, lock, lp = _setup(tmp_path, c_at=None)
    lock["datasets"]["X"]["mount"] = str(inp / "nowhere")
    lp.write_text(json.dumps(lock))
    (inp / "partial").mkdir(parents=True)
    (inp / "partial" / "a.pt").write_bytes(b"aaa")                        # one of the two X files only
    with pytest.raises(MR.MountError, match="MISSING_DATASET_ROOT"):
        MR.resolve(lock, inp)


def test_listing_is_redacted_and_printed(tmp_path, capsys):
    inp, lock, lp = _setup(tmp_path, c_at=Path(SLUG_C))
    MR.resolve_and_write(lp, inp, tmp_path / "work")
    printed = capsys.readouterr().out
    saved = (tmp_path / "work" / "input_listing.json").read_text()
    resolved = (tmp_path / "work" / "mount_resolution.json").read_text()
    for text in (printed, saved, resolved):
        assert USER not in text and SLUG_C not in text and SLUG_X not in text
    assert "INPUT_LISTING " in printed and "<slug:C>/checkpoint_300.pt" in printed and "MOUNT_RESOLUTION " in printed


def test_failure_message_is_redacted(tmp_path, capsys):
    inp, lock, lp = _setup(tmp_path, c_at=None)
    with pytest.raises(MR.MountError) as exc:
        MR.resolve_and_write(lp, inp, tmp_path / "work")
    out = capsys.readouterr().out
    assert USER not in str(exc.value) and SLUG_C not in str(exc.value) and "MOUNT_RESOLUTION_FAILED" in out and USER not in out


def test_resolver_source_has_no_kaggle_literal_paths():
    src = (ROOT / "tools" / "experiment_packaging" / "mount_resolver.py").read_text()
    assert "/" + "kag" + "gle" not in src and "@@" not in src
