"""The committed scoring record of the existing checkpoints on the pre-registered 600-segment set is self-consistent."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "experiment_packaging"))
import y72_heldout_v2 as g  # noqa: E402

R = ROOT / "prereg" / "y72_heldout_v2_results"
SC = json.loads((R / "y72_v2_scores.json").read_text())
NF = json.loads((R / "y72_v2_noise_floor.json").read_text())


def test_scores_refer_to_the_registered_set():
    assert SC["set_canonical_sha256"] == g.SET_CANONICAL_SHA256
    assert SC["set_file_sha256"] == g.SET_FILE_SHA256


def test_counts_are_derivable_from_the_recorded_token_matches():
    assert len(SC["ckpt"]) == 8
    for name, rec in SC["ckpt"].items():
        tok = rec["new"]["tok"]
        assert len(tok) == 120 and all(len(r) == 5 for r in tok)
        seg = sum(1 for r in tok for t in r if t == 8)
        assert seg == rec["new"]["segment_exact"] == NF["counts"][name]["new600"], name
        assert len(rec["sha256_verified"]) == 64 and set(rec["sha256_verified"]) <= set("0123456789abcdef")
    assert SC["ckpt"]["l025_300"]["sha256_verified"].startswith("efce99e8")
    assert SC["ckpt"]["l05_300"]["sha256_verified"].startswith("c962c8ae")


def test_registered_gate_applied_to_the_observed_runs():
    for arm in ("l025", "l05"):
        counts = {c: NF["counts"][f"{arm}_{c}"]["new600"] for c in (250, 275, 300)}
        ev = g.evaluate(counts)
        assert ev == NF["gate_on_observed"][arm]
        assert ev["verdict"] == "R_FAIL"
