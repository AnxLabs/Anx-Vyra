"""Fail closed against the build's audited archive and asset lock before imports."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_lock(lock_path="@@KROOT@@/working/dependency_lock.json"):
    lock = json.loads(Path(lock_path).read_text())
    checked = {}
    for rel, expected in lock["assets"].items():
        path = Path("@@KROOT@@/working") / rel
        if not path.is_file():
            raise ValueError("MISSING_ASSET " + rel)
        if sha(path) != expected:
            raise ValueError("CORRUPT_ASSET " + rel)
        checked[str(path)] = expected
    for role, entry in lock["datasets"].items():
        root = Path(entry["mount"])
        if not root.is_dir():
            raise ValueError("MISSING_DATASET_ROOT " + role + " " + str(root))
        for name, expected in entry["files"].items():
            path = root / name
            if not path.is_file():
                raise ValueError("MISSING_DATASET_FILE " + str(path))
            if sha(path) != expected:
                raise ValueError("CORRUPT_DATASET_FILE " + str(path))
            checked[str(path)] = expected
    return {"status": "PASS", "checked_files": len(checked), "hashes": checked}
