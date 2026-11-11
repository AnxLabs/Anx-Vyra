"""Pre-registered larger Y72 held-out set (120 rows x 5 segments = 600) and its integer gate. Specification + verification only.

Registered in prereg/y72_heldout_v2.json BEFORE any scoring on this set. No float is ever compared in the gate.
Set identity: SET_CANONICAL_SHA256 = sha256(rows int16 little-endian bytes + mask int16 little-endian bytes); `verify_set` recomputes it
(fail closed) before anything is scored.
"""
import hashlib
from pathlib import Path

RULE_ID = "vyra-y72-heldout-v2-gate-v1"
N_ROWS = 120
SEGMENTS_PER_ROW = 5
N_SEGMENTS = N_ROWS * SEGMENTS_PER_ROW           # 600
BLOCK_ROWS = 40
SEEDS = (20263672, 20263772, 20263872)           # base + distance 72, bases 20263600/20263700/20263800
SET_CANONICAL_SHA256 = "16ff53a839c5c68e2230fe4f96e794f11d7ff7f885cac3e184e4f4770f3ee71b"
SET_FILE_SHA256 = "12e034022357e230685a956e47658315797f164b5b11c30bd35ba4f6ca04a826"
SET_PAYLOAD_LIST_SHA256 = "448e33d083368652f2be11c7e8b1fe958e9e6d0d111fea55d639e9c277eb693a"
CHECKPOINTS = (250, 275, 300)
FLOOR_TENTHS_PCT = 800        # 80.0 % of N_SEGMENTS at the sum level: sum >= 3 * 480 = 1440
GUARD_TENTHS_PCT = 720        # every checkpoint >= 72.0 % = 432
SUM3_FLOOR = 1440
GUARD_FLOOR = 432


class SetGateError(ValueError):
    pass


def pct_ge(count, total, tenths_pct):
    return count * 1000 >= tenths_pct * total


def canonical_sha256(rows, mask):
    return hashlib.sha256(rows.numpy().astype("<i2").tobytes() + mask.numpy().astype("<i2").tobytes()).hexdigest()


def verify_set(path, *, expected_file_sha=SET_FILE_SHA256, expected_canonical=SET_CANONICAL_SHA256):
    """File sha256 BEFORE torch.load, then the canonical content hash and shape checks. Raises SetGateError."""
    import torch
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_file_sha:
        raise SetGateError("SET_FILE_SHA256_MISMATCH")
    s = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(s["rows"].shape) != (N_ROWS, 512) or tuple(s["mask"].shape) != (N_ROWS, 510):
        raise SetGateError("SET_SHAPE")
    if canonical_sha256(s["rows"], s["mask"]) != expected_canonical:
        raise SetGateError("SET_CANONICAL_SHA256_MISMATCH")
    if int(s["mask"].sum()) != N_ROWS * SEGMENTS_PER_ROW * 8 or tuple(s["seeds"]) != SEEDS:
        raise SetGateError("SET_CONTENT")
    return s


def evaluate(segments_exact_by_update, n=N_SEGMENTS):
    """segments_exact_by_update: {250: int, 275: int, 300: int} on the 600-segment set. Integer arithmetic only."""
    seen = sorted(u for u in segments_exact_by_update if u in CHECKPOINTS)
    out = {"rule": RULE_ID, "n_segments": n, "checkpoints_seen": seen}
    if len(seen) != len(CHECKPOINTS):
        out["verdict"] = "NOT_EVALUATED_BEFORE_300"
        return out
    vals = [segments_exact_by_update[u] for u in CHECKPOINTS]
    if any((not isinstance(v, int)) or v < 0 or v > n for v in vals):
        raise SetGateError("COUNT_RANGE")
    total = sum(vals)
    sum_ok = total * 1000 >= 3 * FLOOR_TENTHS_PCT * n          # == total >= 1440 for n = 600
    guard_ok = all(pct_ge(v, n, GUARD_TENTHS_PCT) for v in vals)  # each >= 432
    out.update({"sum3": total, "R_sum3_ge_floor": bool(sum_ok), "R_every_checkpoint_ge_guard": bool(guard_ok),
                "R_retention": bool(sum_ok and guard_ok), "verdict": "R_PASS" if (sum_ok and guard_ok) else "R_FAIL"})
    return out
