"""Fail-closed checks for the external 40-row rehearsal pin.

Synthetic tensors stand in for shape and leak failures. They are not the
pinned artifact and they are not a training run.
"""
import hashlib
import sys
from pathlib import Path

import pytest
import torch

PACKAGING = Path(__file__).resolve().parents[1] / "tools" / "experiment_packaging"
sys.path.insert(0, str(PACKAGING))

from rehearsal_artifact import (  # noqa: E402
    PINNED_BYTES,
    PINNED_SHA256,
    exposure_histogram,
    load_manifest,
    replay_indices,
    validate_rehearsal_file,
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _payloads():
    return [f"{i:016x}" for i in range(200)]


def _tensors():
    rows = torch.arange(40 * 512, dtype=torch.int16).reshape(40, 512)
    mask = torch.zeros(40, 510, dtype=torch.int16)
    mask[:, :40] = 1
    return rows, mask


def _manifest_for(rows, mask, payloads):
    base = load_manifest()
    specs = []
    for index in range(rows.shape[0]):
        specs.append({
            "index": index,
            "origin": "preserved_original" if index < 8 else "generated_additional",
            "row_sha256_int64_canonical": _sha(rows[index].long().numpy().tobytes()),
            "row_sha256_storage": _sha(rows[index].numpy().tobytes()),
            "mask_sha256_storage": _sha(mask[index].numpy().tobytes()),
            "payload_sha256": [_sha(bytes.fromhex(item)) for item in payloads[index * 5:(index + 1) * 5]],
        })
    manifest = dict(base)
    manifest["rows"] = specs
    return manifest


def _save(path, rows, mask, payloads):
    torch.save({
        "schema": "vyra-derived-Y72-rehearsal40-v1",
        "rows": rows,
        "mask": mask,
        "payloads": payloads,
    }, path)
    return path.read_bytes()


def test_pinned_exposure_is_eight_or_nine_and_the_manifest_hash_matches():
    manifest = load_manifest()
    assert manifest["artifact_sha256"] == PINNED_SHA256
    assert manifest["artifact_bytes"] == PINNED_BYTES == 88749
    counts, histogram = exposure_histogram()
    assert sum(counts) == 350
    assert histogram == {"9": 30, "8": 10}
    assert counts[:30] == [9] * 30 and counts[30:] == [8] * 10
    assert [row for _step, _slot, row in replay_indices(2)] == [0, 1, 2, 3]


def test_missing_and_wrong_hash_are_refused(tmp_path):
    with pytest.raises(ValueError, match="REPLAY_ARTIFACT_MISSING"):
        validate_rehearsal_file(tmp_path / "absent.pt")
    blob = tmp_path / "wrong.pt"
    blob.write_bytes(b"not the rehearsal artifact")
    with pytest.raises(ValueError, match="REPLAY_ARTIFACT_HASH"):
        validate_rehearsal_file(blob)


def test_shape_original8_and_heldout_leaks_are_refused(tmp_path):
    rows, mask = _tensors()
    payloads = _payloads()
    path = tmp_path / "synthetic.pt"
    blob = _save(path, rows, mask, payloads)
    manifest = _manifest_for(rows, mask, payloads)
    manifest["artifact_sha256"] = _sha(blob)
    manifest["artifact_bytes"] = len(blob)
    manifest["rows_shape"] = [41, 512]
    with pytest.raises(ValueError, match="REPLAY_ARTIFACT_SHAPE"):
        validate_rehearsal_file(path, manifest, require_pin=False)
    manifest["rows_shape"] = [40, 512]
    changed = rows.clone()
    changed[0, 0] += 1
    changed_path = tmp_path / "changed.pt"
    changed_blob = _save(changed_path, changed, mask, payloads)
    changed_manifest = dict(manifest)
    changed_manifest["artifact_sha256"] = _sha(changed_blob)
    changed_manifest["artifact_bytes"] = len(changed_blob)
    with pytest.raises(ValueError, match="REPLAY_ARTIFACT_ORIGINAL8"):
        validate_rehearsal_file(changed_path, changed_manifest, require_pin=False)
    with pytest.raises(ValueError, match="REPLAY_HELDOUT_LEAK"):
        validate_rehearsal_file(path, manifest, unseen_payloads={payloads[40]}, require_pin=False)
