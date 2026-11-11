"""Pre-registered larger Y72 held-out set: identity, overlap evidence, integer gate boundaries, pinned to prereg/y72_heldout_v2.json."""
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "experiment_packaging"))
import y72_heldout_v2 as g  # noqa: E402

D = ROOT / "prereg" / "y72_heldout_v2"
PREREG = json.loads((ROOT / "prereg" / "y72_heldout_v2.json").read_text())
SET = D / "y72_heldout_v2.pt"


def test_module_constants_match_the_committed_prereg():
    p = PREREG
    assert p["rule_id"] == g.RULE_ID
    s = p["set"]
    assert (s["rows"], s["segments"], s["blocks"], s["seeds"]) == (g.N_ROWS, g.N_SEGMENTS, 3, list(g.SEEDS))
    assert s["canonical_rows_mask_sha256"] == g.SET_CANONICAL_SHA256 and s["file_sha256"] == g.SET_FILE_SHA256 and s["payload_list_sha256"] == g.SET_PAYLOAD_LIST_SHA256
    gate = p["gate"]
    assert gate["sum3_floor_segments"] == g.SUM3_FLOOR == 3 * 480 and gate["per_checkpoint_guard_segments"] == g.GUARD_FLOOR == 432
    assert g.SUM3_FLOOR * 1000 == 3 * g.FLOOR_TENTHS_PCT * g.N_SEGMENTS and g.GUARD_FLOOR * 1000 == g.GUARD_TENTHS_PCT * g.N_SEGMENTS
    assert p["gate"]["checkpoints"] == list(g.CHECKPOINTS)


def test_committed_set_verifies_and_matches_generator_evidence():
    s = g.verify_set(SET)
    info = json.loads((D / "gen_info.json").read_text())
    assert info["canonical_rows_mask_sha256"] == g.SET_CANONICAL_SHA256 and info["pt_sha256"] == g.SET_FILE_SHA256
    assert info["reproduction_check"]["rows_equal"] and info["reproduction_check"]["payloads_equal"]
    assert len(set(s["payloads"])) == len(s["payloads"]) == g.N_SEGMENTS
    assert s["rows"].dtype == torch.int16 and int(s["copies"]) == 5 and int(s["distance"]) == 72


def test_verify_set_fails_closed(tmp_path):
    bad = tmp_path / "set.pt"
    raw = bytearray(SET.read_bytes()); raw[100] ^= 1
    bad.write_bytes(bytes(raw))
    with pytest.raises(g.SetGateError, match="SET_FILE_SHA256_MISMATCH"):
        g.verify_set(bad)
    with pytest.raises(g.SetGateError, match="SET_CANONICAL_SHA256_MISMATCH"):
        g.verify_set(SET, expected_canonical="0" * 64)


def test_overlap_evidence_is_committed_and_clean_and_the_checker_has_a_positive_control():
    o = json.loads((D / "overlap_check.json").read_text())
    assert o["pass"] is True and o["seed_level"]["pass"] is True and o["seed_level"]["new_seeds_in_used"] == []
    c = o["content_level"]
    assert c["exact_row_matches_total"] == c["payload_matches_total"] == c["filler_string_matches_total"] == c["16byte_window_matches_total"] == 0
    assert c["old_rows_scanned"] > 9000 and o["vs_current_200_heldout"]["exact_row_matches"] == 0
    assert "overall pass = False" in (D / "overlap_positive_control.log").read_text()


def test_seeds_are_new_under_the_original_policy():
    used = {b + d for b in (20261500, 20261600, 20261700) for d in [t + j for t in (32, 64, 109) for j in (-8, -4, 0, 4, 8)] + [44, 48, 52, 76, 80, 84, 88, 92, 96, 121, 128]} | {20262672}
    assert not (set(g.SEEDS) & used) and [s - 72 for s in g.SEEDS] == [20263600, 20263700, 20263800]


def test_gate_boundaries_sum_and_guard():
    ev = g.evaluate
    assert ev({250: 480, 275: 480, 300: 480})["R_retention"] is True            # sum 1440
    assert ev({250: 480, 275: 480, 300: 479})["R_retention"] is False           # sum 1439
    assert ev({250: 432, 275: 504, 300: 504})["R_retention"] is True            # guard 432 passes, sum 1440
    r = ev({250: 431, 275: 505, 300: 504})                                         # sum 1440 but guard 431 fails
    assert r["R_sum3_ge_floor"] is True and r["R_every_checkpoint_ge_guard"] is False and r["R_retention"] is False
    assert ev({250: 452, 275: 520, 300: 468})["R_retention"] is True              # sum rule passes a dip below 480 (452 < 480)


def test_gate_not_evaluated_and_range_checks():
    assert g.evaluate({250: 500, 275: 500})["verdict"] == "NOT_EVALUATED_BEFORE_300"
    with pytest.raises(g.SetGateError, match="COUNT_RANGE"):
        g.evaluate({250: 601, 275: 500, 300: 500})
    with pytest.raises(g.SetGateError, match="COUNT_RANGE"):
        g.evaluate({250: 500.0, 275: 500, 300: 500})


def test_no_float_arithmetic_in_the_gate_module():
    import ast
    tree = ast.parse((ROOT / "tools" / "experiment_packaging" / "y72_heldout_v2.py").read_text())
    assert not [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
    assert not any(isinstance(n, ast.Div) for n in ast.walk(tree))
