"""Find each locked dataset by file hash where Kaggle actually mounted it, and print a redacted listing of the inputs.

Why: the rehearsal2 kernel (notebook 3d5756c8) stopped at MISSING_DATASET_ROOT C. Its lock pinned the checkpoint dataset at
<KROOT>/input/datasets/<user>/<slug>, the same pinned path that worked for cont400, and Kaggle's kernel metadata listed the
dataset. Kaggle is publicly reported to use two layouts, <KROOT>/input/datasets/<owner>/<slug> and <KROOT>/input/<slug>.
The old notebook never listed its inputs, so the miss could not be diagnosed.

Rules (fail closed; nothing is weakened):
* The listing (relative paths and sizes under the input root, with the lock's user and slugs replaced by <user> and <slug:ROLE>)
  is printed and written to input_listing.json BEFORE any resolution, so a miss is always diagnosable from the kernel log.
* If the pinned mount is a directory, it is used as before. A missing or wrong file there stays fatal: the unchanged
  dependency_gate raises MISSING_DATASET_FILE or CORRUPT_DATASET_FILE. No search happens.
* Only if the pinned directory is absent, every directory under the input root is checked. A candidate must contain ALL locked file
  names with ALL locked SHA-256 values. Symlinked duplicates of one real directory count once. Exactly one candidate means
  method "hash". Zero means MISSING_DATASET_ROOT. Two or more means AMBIGUOUS_DATASET_ROOT.
* The roots found are written into a resolved copy of the lock (same files, same hashes, mount = actual). The unchanged
  dependency_gate.verify_lock then hashes every locked file at the actual mount.
"""
import hashlib
import json
import os
from pathlib import Path

MAX_DEPTH = 6
MAX_ENTRIES = 4000


class MountError(ValueError):
    pass


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def redactor(lock):
    pairs = []
    for role, entry in lock["datasets"].items():
        user, _, slug = entry["id"].partition("/")
        pairs.append((slug, "<slug:%s>" % role))
        pairs.append((user, "<user>"))
    pairs = sorted(set(pairs), key=lambda p: -len(p[0]))

    def red(text):
        for old, new in pairs:
            if old:
                text = text.replace(old, new)
        return text
    return red


def list_inputs(input_root, red):
    root = Path(input_root)
    out = {"input_root_exists": root.is_dir(), "entries": [], "truncated": False}
    if not root.is_dir():
        return out
    base = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        depth = len(Path(dirpath).parts) - base
        dirnames.sort()
        if depth >= MAX_DEPTH:
            dirnames[:] = []
        rel = Path(dirpath).relative_to(root)
        for d in dirnames:
            p = Path(dirpath) / d
            out["entries"].append({"path": red(str(rel / d)), "kind": "symlink-dir" if p.is_symlink() else "dir"})
        for f in sorted(filenames):
            p = Path(dirpath) / f
            try:
                size = p.stat().st_size
            except OSError:
                size = None
            out["entries"].append({"path": red(str(rel / f)), "kind": "file", "bytes": size})
        if len(out["entries"]) > MAX_ENTRIES:
            out["truncated"] = True
            break
    return out


def _candidates(input_root, names):
    seen, found = set(), []
    root = Path(input_root)
    if not root.is_dir():
        return found
    base = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        if len(Path(dirpath).parts) - base >= MAX_DEPTH:
            dirnames[:] = []
        if names <= set(filenames):
            real = os.path.realpath(dirpath)
            if real not in seen:
                seen.add(real)
                found.append(Path(dirpath))
    return found


def resolve(lock, input_root):
    """Return {role: {"pinned", "actual", "method"}}; raise MountError when a dataset cannot be found by hash."""
    out = {}
    for role, entry in lock["datasets"].items():
        pinned = Path(entry["mount"])
        if pinned.is_dir():
            out[role] = {"pinned": str(pinned), "actual": str(pinned), "method": "pinned"}
            continue
        files = entry["files"]
        matches, near = [], 0
        for cand in _candidates(input_root, set(files)):
            if all(_sha(cand / n) == h for n, h in files.items()):
                matches.append(cand)
            else:
                near += 1
        if not matches:
            raise MountError("MISSING_DATASET_ROOT %s %s (no directory under the input root holds all %d locked files at the locked "
                             "SHA-256; %d directories had the names with a wrong hash)" % (role, pinned, len(files), near))
        if len(matches) > 1:
            raise MountError("AMBIGUOUS_DATASET_ROOT %s (%d distinct directories match by hash)" % (role, len(matches)))
        out[role] = {"pinned": str(pinned), "actual": str(matches[0]), "method": "hash"}
    actual = [os.path.realpath(v["actual"]) for v in out.values()]
    if len(set(actual)) != len(actual):
        raise MountError("DATASET_ROOTS_NOT_DISTINCT")
    return out


def resolve_and_write(lock_path, input_root, working):
    """Print and save the listing, resolve every locked dataset, and write the resolved lock. Returns roots and paths."""
    lock = json.loads(Path(lock_path).read_text())
    red = redactor(lock)
    listing = list_inputs(input_root, red)
    Path(working, "input_listing.json").write_text(json.dumps(listing, indent=1))
    print("INPUT_LISTING " + json.dumps(listing, sort_keys=True), flush=True)
    try:
        mounts = resolve(lock, input_root)
    except MountError as exc:
        print("MOUNT_RESOLUTION_FAILED " + red(str(exc)), flush=True)
        raise MountError(red(str(exc))) from None
    resolved = json.loads(json.dumps(lock))
    for role, m in mounts.items():
        resolved["datasets"][role]["mount"] = m["actual"]
    rpath = Path(working, "dependency_lock_resolved.json")
    rpath.write_text(json.dumps(resolved, indent=2, sort_keys=True))
    report = {role: {"method": m["method"], "actual_equals_pinned": m["actual"] == m["pinned"], "actual": red(m["actual"])}
              for role, m in mounts.items()}
    Path(working, "mount_resolution.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print("MOUNT_RESOLUTION " + json.dumps(report, sort_keys=True), flush=True)
    return {"roots": {role: m["actual"] for role, m in mounts.items()}, "resolved_lock_path": str(rpath), "report": report}
