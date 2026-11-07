"""Fail-closed check for the external 40-row Y72 rehearsal artifact.

The artifact is not in this repository. Callers pass a path. The manifest
committed beside this file pins the file hash, shapes, and per-row hashes.
It does not contain payload plaintext. Nothing here regenerates rows.
"""
import hashlib
import json
from pathlib import Path

import torch

PINNED_SHA256 = "abd8d94d7e105deab2a19c461266c89ab308f7edf67b4394b638afdb7cb474d7"
PINNED_BYTES = 88749
MANIFEST_PATH = Path(__file__).with_name("rehearsal_Y72_40_manifest.json")


def load_manifest(path=None):
    raw = Path(path or MANIFEST_PATH).read_text()
    manifest = json.loads(raw)
    if manifest.get("artifact_sha256") != PINNED_SHA256:
        raise ValueError("REPLAY_MANIFEST_PIN")
    if manifest.get("artifact_bytes") != PINNED_BYTES:
        raise ValueError("REPLAY_MANIFEST_PIN")
    if "hex" in raw:
        raise ValueError("REPLAY_MANIFEST_PIN")
    return manifest


def replay_indices(updates=175, rows_per_update=2, row_count=40):
    """Return the (update, slot, row) presentations. Formula is (2*s+j) % 40."""
    out = []
    for step in range(updates):
        for slot in range(rows_per_update):
            out.append((step, slot, (rows_per_update * step + slot) % row_count))
    return out


def exposure_histogram(updates=175, rows_per_update=2, row_count=40):
    counts = [0] * row_count
    for _step, _slot, row in replay_indices(updates, rows_per_update, row_count):
        counts[row] += 1
    histogram = {}
    for count in counts:
        histogram[str(count)] = histogram.get(str(count), 0) + 1
    return counts, histogram


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_rehearsal_file(path, manifest=None, unseen_payloads=None, *, require_pin=True):
    """Load and check one artifact. Raises ValueError on any pin failure."""
    path = Path(path)
    if not path.is_file():
        raise ValueError("REPLAY_ARTIFACT_MISSING")
    manifest = load_manifest() if manifest is None else manifest
    blob = path.read_bytes()
    digest = _sha(blob)
    if digest != manifest["artifact_sha256"]:
        raise ValueError("REPLAY_ARTIFACT_HASH")
    if require_pin and digest != PINNED_SHA256:
        raise ValueError("REPLAY_ARTIFACT_HASH")
    if len(blob) != manifest["artifact_bytes"]:
        raise ValueError("REPLAY_ARTIFACT_BYTES")
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("schema") != manifest["schema"]:
        raise ValueError("REPLAY_ARTIFACT_SCHEMA")
    rows, mask = artifact["rows"], artifact["mask"]
    if list(rows.shape) != manifest["rows_shape"] or list(mask.shape) != manifest["mask_shape"]:
        raise ValueError("REPLAY_ARTIFACT_SHAPE")
    if str(rows.dtype) != manifest["rows_dtype"] or str(mask.dtype) != manifest["mask_dtype"]:
        raise ValueError("REPLAY_ARTIFACT_DTYPE")
    payloads = list(artifact["payloads"])
    if len(payloads) != manifest["payload_count"] or len(set(payloads)) != len(payloads):
        raise ValueError("REPLAY_ARTIFACT_PAYLOAD")
    original = manifest["original_row_count"]
    per_row = manifest["payloads_per_row"]
    if len(payloads) != rows.shape[0] * per_row:
        raise ValueError("REPLAY_ARTIFACT_PAYLOAD")
    payload_hashes = []
    for text in payloads:
        raw = bytes.fromhex(text)
        if len(raw) != 8:
            raise ValueError("REPLAY_ARTIFACT_PAYLOAD")
        payload_hashes.append(_sha(raw))
    if len(set(payload_hashes)) != len(payload_hashes):
        raise ValueError("REPLAY_ARTIFACT_PAYLOAD")
    original_hashes = set()
    added_hashes = set()
    for spec, row, row_mask in zip(manifest["rows"], rows, mask):
        storage = _sha(row.numpy().tobytes())
        canonical = _sha(row.long().numpy().tobytes())
        mask_hash = _sha(row_mask.numpy().tobytes())
        got_payloads = payload_hashes[spec["index"] * per_row:(spec["index"] + 1) * per_row]
        if int((row_mask != 0).sum()) != 40:
            raise ValueError("REPLAY_ARTIFACT_SHAPE")
        if (
            storage != spec["row_sha256_storage"]
            or canonical != spec["row_sha256_int64_canonical"]
            or mask_hash != spec["mask_sha256_storage"]
            or got_payloads != spec["payload_sha256"]
        ):
            code = "REPLAY_ARTIFACT_ORIGINAL8" if spec["index"] < original else "REPLAY_ARTIFACT_ROW_PIN"
            raise ValueError(code)
        bucket = original_hashes if spec["index"] < original else added_hashes
        bucket.update(got_payloads)
    if original_hashes & added_hashes:
        raise ValueError("REPLAY_ARTIFACT_LEAK")
    if unseen_payloads is not None and set(payloads[original * per_row:]) & set(unseen_payloads):
        raise ValueError("REPLAY_HELDOUT_LEAK")
    counts, histogram = exposure_histogram(manifest["updates"], manifest["replay_rows_per_update"], rows.shape[0])
    if histogram != manifest["repeat_histogram"] or sum(counts) != manifest["presentations"]:
        raise ValueError("REPLAY_ARTIFACT_EXPOSURE")
    return artifact


def load_for_training(path, unseen_payloads=None):
    """Training replay loader. Evaluation rows are not taken from this file."""
    return validate_rehearsal_file(path, unseen_payloads=unseen_payloads)
