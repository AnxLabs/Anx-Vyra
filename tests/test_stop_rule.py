"""Pre-registered stop rule: exact integer boundaries and honest ambiguity."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "experiment_packaging"))
import stop_rule as sr  # noqa: E402


def c(tc, se=0, tokens=1920, segs=240):
    return {"tokens_correct": tc, "tokens": tokens, "segments_exact": se, "segments": segs}


def y72(se=170):
    return {"Y72_unseen": {"tokens_correct": 1550, "tokens": 1600, "segments_exact": se, "segments": 200}}


def hist(h250_train, h300_held, h300_train, *, held_exact=0, train_exact=0, y=170, h250_held=300):
    return {
        200: {"x64": {k: dict(v) for k, v in sr.BASELINE_X64.items()}, "counts": y72(165)},
        225: {"x64": {"train": c(700), "heldout": c(280)}, "counts": y72(y)},
        250: {"x64": {"train": c(h250_train), "heldout": c(h250_held)}, "counts": y72(y)},
        275: {"x64": {"train": c(h250_train + 5), "heldout": c(h250_held + 5)}, "counts": y72(y)},
        300: {"x64": {"train": c(h300_train, train_exact), "heldout": c(h300_held, held_exact)}, "counts": y72(y)},
    }


def test_heldout_gain_boundary_is_exact_not_rounded():
    # gain 191 tokens = 9.947pp < 10  ->  A true ; 192 tokens = exactly 10.000pp -> A false
    a_true = sr.evaluate(hist(900, 257 + 191, 1000))
    a_false = sr.evaluate(hist(900, 257 + 192, 1000))
    assert a_true["terms"]["A_heldout_gain_200_to_300_below_10pp"] is True
    assert a_false["terms"]["A_heldout_gain_200_to_300_below_10pp"] is False
    assert a_false["terms"]["heldout_gain_200_to_300_pp_exact"] == f"{192 * 100}/1920"


def test_train_gain_boundary_is_exact():
    # 3pp of 1920 = 57.6 tokens: 57 -> B true (2.96875), 58 -> false (3.0208)
    assert sr.evaluate(hist(900, 700, 900 + 57))["terms"]["B_train_gain_250_to_300_below_3pp"] is True
    assert sr.evaluate(hist(900, 700, 900 + 58))["terms"]["B_train_gain_250_to_300_below_3pp"] is False


def test_verdict_stuck_not_stuck_and_ambiguous():
    stuck = sr.evaluate(hist(900, 300, 905))                 # A true, B true, C true
    assert stuck["stuck_verdict"] == "STUCK" and stuck["always_stops_at"] == 300 and stuck["automatic_followup"] is None
    not_stuck = sr.evaluate(hist(900, 257 + 400, 900 + 200, held_exact=3))   # A false, B false
    assert not_stuck["stuck_verdict"] == "NOT_STUCK"
    # A true but segment-exact > 0: R1 (A or (B and C)) true, R2 ((A or B) and C) false
    amb = sr.evaluate(hist(900, 300, 905, held_exact=2))
    assert amb["terms"]["reading_R1_A_or_(B_and_C)"] is True and amb["terms"]["reading_R2_(A_or_B)_and_C"] is False
    assert amb["stuck_verdict"].startswith("AMBIGUOUS") and amb["overlap_note"]


def test_slow_learner_evidence_requires_same_checkpoint_y72_and_overlap_is_reported():
    h = hist(900, 300, 905, held_exact=1, y=170)
    h[275]["x64"]["heldout"]["segments_exact"] = 1
    out = sr.evaluate(h)
    assert 275 in out["slow_learner_evidence_at"] and 300 in out["slow_learner_evidence_at"]
    # same exact>0 but Y72 below 80% at that checkpoint: no slow-learner evidence there, retention failure flagged
    h2 = hist(900, 300, 905, held_exact=1, y=159)             # 159/200 = 79.5%
    out2 = sr.evaluate(h2)
    assert out2["retention_failure"] is True and out2["success_claim_allowed"] is False
    assert out2["slow_learner_evidence_at"] == [] and out2["retention_failure_at"] == [225, 250, 275, 300]
    h3 = hist(900, 300, 905, held_exact=1, y=160)             # exactly 80%
    assert sr.evaluate(h3)["retention_failure"] is False
    # STUCK verdict + slow evidence = overlapping/inconclusive note
    h4 = hist(900, 300, 905, held_exact=0)
    h4[250]["x64"]["heldout"]["segments_exact"] = 1
    out4 = sr.evaluate(h4)
    assert out4["stuck_verdict"] == "STUCK" and "OVERLAPPING" in out4["overlap_note"]


def test_partial_history_does_not_decide_and_baseline_is_pinned():
    h = hist(900, 300, 905)
    del h[300]
    out = sr.evaluate(h)
    assert out["final"] is False and out["stuck_verdict"] == "NOT_EVALUATED_BEFORE_300"
    bad = hist(900, 300, 905)
    bad[200]["x64"]["heldout"]["tokens_correct"] = 258
    with pytest.raises(ValueError, match="BASELINE_MISMATCH"):
        sr.evaluate(bad)
    assert sr.BASELINE_X64["heldout"]["tokens_correct"] == 257 and sr.BASELINE_X64["train"]["tokens_correct"] == 669
    # no float rounding anywhere in the rule source
    src = (Path(sr.__file__)).read_text()
    assert "round(" not in src and "float(" not in src
