"""No tracked file may contain the private Kaggle identifiers.

Scans every file `git ls-files` lists (raw bytes, so binary files count too) for:
  * the Kaggle username, every dataset/kernel slug and the Kaggle root, read from the local
    gitignored kaggle_config.json when it exists (a clean clone has no real values, so the structural patterns below
    and the VYRA_PRIVATE_HANDLES list carry the check there);
  * structural patterns: a Kaggle filesystem path, a Kaggle URL, and the shapes of the dataset and kernel slugs;
  * the handles in the VYRA_PRIVATE_HANDLES environment variable (comma separated).

Only kaggle_config.example.json (placeholders) is allowlisted. One narrow, explicit exemption exists, see
FROZEN_EVIDENCE: the bare bundle directory names inside the hash-pinned pre-registered evidence of the held-out set
(prereg/y72_heldout_v2*). Those files cannot be edited without breaking their pins. They are still scanned for
everything else.
The patterns are assembled from pieces so this file does not match itself.
"""
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = {"kaggle_config.example.json"}
FROZEN_EVIDENCE = ("prereg/y72_heldout_v2/", "prereg/y72_heldout_v2_results/")

_K = "kag" + "gle"
_V = "vy" + "ra"
STRUCTURAL = [                       # (name, bytes regex, frozen_exempt)
    ("kaggle_path", re.compile(("/" + _K + r"(?![A-Za-z0-9_-])").encode()), False),
    ("kaggle_url", re.compile((_K + r"\.com/\S").encode()), False),
    ("bundle_slug", re.compile((_V + r"-[a-z0-9]+-bundle").encode()), True),
    ("kernel_slug", re.compile((_V + r"-[a-z0-9-]+-packaged").encode()), False),
    ("checkpoint_slug", re.compile((_V + r"-(?:[a-z0-9]+-ckpt[0-9]+|expanded[0-9]+-[a-z0-9-]+-checkpoint)").encode()), False),
    ("approved_slug", re.compile((_V + r"-replay[0-9]+-approved").encode()), False),
]


def _literals():
    """(name, bytes, frozen_exempt) for each real private value, from the local config and the handles variable."""
    out = []
    real = ROOT / "kaggle_config.json"
    if real.is_file():
        cfg = json.loads(real.read_text())
        out.append(("username", cfg["username"].encode(), False))
        out.append(("root", (cfg["root"].rstrip("/") + "/").encode(), False))
        for key, value in cfg["slugs"].items():
            out.append(("slug_" + key, value.encode(), key in ("x_bundle", "y_bundle")))
    for handle in filter(None, (h.strip() for h in os.environ.get("VYRA_PRIVATE_HANDLES", "").split(","))):
        out.append(("handle", handle.encode(), False))
    return out


def scan_bytes(path, data):
    frozen = path.startswith(FROZEN_EVIDENCE)
    hits = []
    for name, rx, exempt in STRUCTURAL:
        if rx.search(data) and not (frozen and exempt):
            hits.append(name)
    for name, lit, exempt in _literals():
        if lit and lit.lower() in data.lower() and not (frozen and exempt):
            hits.append(name)
    return hits


def _tracked():
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout; the scan covers `git ls-files`")
    raw = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return [p for p in raw.decode().split("\0") if p and (ROOT / p).is_file()]


def test_no_tracked_file_contains_a_private_kaggle_identifier():
    offenders = {}
    for rel in _tracked():
        if rel in ALLOWLIST:
            continue
        hits = scan_bytes(rel, (ROOT / rel).read_bytes())
        if hits:
            offenders[rel] = sorted(set(hits))
    assert not offenders, "private Kaggle identifiers found in tracked files: " + json.dumps(offenders, indent=1)


def test_the_real_config_is_never_tracked_and_is_ignored():
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    tracked = subprocess.run(["git", "ls-files", "kaggle_config.json"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert tracked.strip() == ""
    ignored = subprocess.run(["git", "check-ignore", "-q", "kaggle_config.json"], cwd=ROOT).returncode
    assert ignored == 0


def test_the_example_template_is_the_only_allowlisted_file():
    assert ALLOWLIST == {"kaggle_config.example.json"}
    cfg = json.loads((ROOT / "kaggle_config.example.json").read_text())
    assert cfg["example"] is True


def test_scanner_catches_each_kind_of_identifier_and_the_frozen_exemption_is_narrow():
    # positive controls, assembled from pieces
    assert "kaggle_path" in scan_bytes("a.py", ("Path('/" + _K + "/working')").encode())
    assert "kaggle_url" in scan_bytes("a.md", ("https://www." + _K + ".com/co" + "de/someone/run").encode())
    assert "bundle_slug" in scan_bytes("a.py", (_V + "-x-bundle").encode())
    assert "kernel_slug" in scan_bytes("a.py", (_V + "-y72fix2-pertask-packaged").encode())
    assert "checkpoint_slug" in scan_bytes("a.py", (_V + "-resume300-ckpt225").encode())
    assert "checkpoint_slug" in scan_bytes("a.py", (_V + "-expanded40-200u-checkpoint").encode())
    assert "approved_slug" in scan_bytes("a.py", (_V + "-replay175-approved").encode())
    # innocent text and the placeholder names are clean
    assert scan_bytes("a.py", b"/kaggle_config.json  example-user /example-root/input") == []
    # frozen evidence: the bundle directory name is tolerated there only, nothing else is
    bundle = (_V + "-y-bundle/rows").encode()
    assert scan_bytes("prereg/y72_heldout_v2/overlap_check.json", bundle) == []
    assert "bundle_slug" in scan_bytes("prereg/other/overlap_check.json", bundle)
    assert "kaggle_path" in scan_bytes("prereg/y72_heldout_v2/overlap_check.json", ("/" + _K + "/working").encode())
    assert "kernel_slug" in scan_bytes("prereg/y72_heldout_v2_results/x.json", (_V + "-z-packaged").encode())


def test_handles_in_the_environment_are_scanned(monkeypatch):
    monkeypatch.setenv("VYRA_PRIVATE_HANDLES", "some-handle, other-handle")
    assert "handle" in scan_bytes("a.txt", b"written by Some-Handle today")
    assert scan_bytes("a.txt", b"nothing here") == []
