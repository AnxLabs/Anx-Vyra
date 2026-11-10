"""Pre-registered stop rule: one rule, every branch, exact boundaries at 10.0 and 3.0 pp."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "experiment_packaging"))
import stop_rule as sr  # noqa: E402

HELD200, TRAIN200 = 257, 669
# 10.0 pp of 1920 tokens = 192.0 tokens exactly; 3.0 pp = 57.6 tokens (not an integer).


def c(tc, se=0, tokens=1920, segs=240):
    return {"tokens_correct": tc, "tokens": tokens, "segments_exact": se, "segments": segs}


def y72(se=170):
    return {"Y72_unseen": {"tokens_correct": 1550, "tokens": 1600, "segments_exact": se, "segments": 200}}


def hist(*, held300=300, train300=905, train250=900, held_exact=0, train_exact=0, y=170, held_exact_other=None, y_by=None):
    h = {
        200: {"x64": {k: dict(v) for k, v in sr.BASELINE_X64.items()}, "counts": y72(165)},
        225: {"x64": {"train": c(700), "heldout": c(280)}, "counts": y72(y)},
        250: {"x64": {"train": c(train250), "heldout": c(300)}, "counts": y72(y)},
        275: {"x64": {"train": c(train250 + 5), "heldout": c(305)}, "counts": y72(y)},
        300: {"x64": {"train": c(train300, train_exact), "heldout": c(held300, held_exact)}, "counts": y72(y)},
    }
    for u, e in (held_exact_other or {}).items():
        h[u]["x64"]["heldout"]["segments_exact"] = e
    for u, v in (y_by or {}).items():
        h[u]["counts"] = y72(v)
    return h


# ---- the STUCK conjunction: every branch --------------------------------------------------------------
@pytest.mark.parametrize("held_gain_below, train_gain_below, held_exact, train_exact, expected", [
    (True, True, 0, 0, "STUCK"),
    (True, False, 0, 0, "STUCK"),         # held-out clause alone is enough (OR)
    (False, True, 0, 0, "STUCK"),         # train clause alone is enough (OR)
    (False, False, 0, 0, "INCONCLUSIVE"),
    (True, True, 1, 0, "INCONCLUSIVE"),   # exact-0 gates BOTH clauses: held-out exact > 0
    (True, True, 0, 1, "INCONCLUSIVE"),   # train exact > 0
    (False, True, 3, 2, "INCONCLUSIVE"),
    (True, False, 2, 0, "INCONCLUSIVE"),
])
def test_stuck_truth_table(held_gain_below, train_gain_below, held_exact, train_exact, expected):
    held300 = HELD200 + (191 if held_gain_below else 192)         # 191 -> 9.948pp (<10), 192 -> 10.000pp
    train300 = 900 + (57 if train_gain_below else 58)             # 57 -> 2.969pp (<3), 58 -> 3.021pp
    out = sr.evaluate(hist(held300=held300, train300=train300, held_exact=held_exact, train_exact=train_exact))
    assert out["terms"]["heldout_gain_200_to_300_below_10_0_pp"] is held_gain_below
    assert out["terms"]["train_gain_250_to_300_below_3_0_pp"] is train_gain_below
    assert out["terms"]["segment_exact_both_zero_at_300"] is (held_exact == 0 and train_exact == 0)
    assert out["verdict"] == expected
    assert out["always_stops_at"] == 300 and out["automatic_followup"] is None and out["final"] is True


def test_heldout_boundary_exactly_10_0_pp_is_not_below():
    assert sr.evaluate(hist(held300=HELD200 + 192, train300=1500, train250=900))["terms"]["heldout_gain_200_to_300_below_10_0_pp"] is False
    assert sr.evaluate(hist(held300=HELD200 + 191, train300=1500, train250=900))["terms"]["heldout_gain_200_to_300_below_10_0_pp"] is True
    out = sr.evaluate(hist(held300=HELD200 + 192, train300=1500, train250=900))
    assert out["terms"]["heldout_gain_200_to_300_pp_exact"] == f"{192 * 100}/1920"     # exactly 10 pp
    assert out["verdict"] == "INCONCLUSIVE"                                              # 10.0 is NOT < 10.0
    assert sr.evaluate(hist(held300=HELD200 + 191, train300=1500, train250=900))["verdict"] == "STUCK"


def test_train_boundary_at_3_0_pp_is_exact_with_a_non_integer_threshold():
    # 3.0 pp of 1920 tokens is 57.6 tokens: 57 tokens is below, 58 is not; there is no integer exactly at 3.0.
    assert sr.evaluate(hist(train300=957, held300=HELD200 + 400))["terms"]["train_gain_250_to_300_below_3_0_pp"] is True
    assert sr.evaluate(hist(train300=958, held300=HELD200 + 400))["terms"]["train_gain_250_to_300_below_3_0_pp"] is False
    # the helper itself at an exact tie, with a denominator where 3.0 pp is an integer count (1000 tokens -> 30)
    assert sr.gain_below(130, 1000, 100, 1000, 30) is False     # exactly 3.0 pp
    assert sr.gain_below(129, 1000, 100, 1000, 30) is True
    assert sr.gain_below(1100 + 192 * 0, 1000, 900, 1000, 200) is False   # exactly 20.0 pp tie is not below
    assert sr.gain_below(257 + 192, 1920, 257, 1920, 100) is False       # exactly 10.0 pp
    assert sr.gain_below(257 + 191, 1920, 257, 1920, 100) is True
    assert sr.gain_below(900 - 5, 1920, 900, 1920, 30) is True            # negative gain is below


def test_train_gain_is_measured_against_update_250_not_200():
    # train 250 == 1000, train 300 == 1010: gain 10 tokens (0.52 pp) < 3 -> clause true although 300 vs 200 is large
    out = sr.evaluate(hist(train250=1000, train300=1010, held300=HELD200 + 400))
    assert out["terms"]["train_gain_250_to_300_below_3_0_pp"] is True and out["verdict"] == "STUCK"
    assert out["terms"]["train_gain_250_to_300_pp_exact"] == f"{10 * 100}/1920"


# ---- slow learner and retention ------------------------------------------------------------------------
def test_slow_learner_needs_same_checkpoint_y72_at_least_80_0():
    out = sr.evaluate(hist(held_exact_other={275: 1}, y=170))
    assert out["slow_learner_evidence"] is True and out["slow_learner_evidence_at"] == [275]
    # exactly 160/200 = 80.0% passes, 159/200 = 79.5% does not
    assert sr.evaluate(hist(held_exact_other={275: 1}, y_by={275: 160}))["slow_learner_evidence_at"] == [275]
    low = sr.evaluate(hist(held_exact_other={275: 1}, y_by={275: 159}))
    assert low["slow_learner_evidence"] is False and low["retention_failure_at"] == [275]
    # exact > 0 at 250 but Y72 only low at 275: evidence at 250 stays (same-checkpoint rule)
    mix = sr.evaluate(hist(held_exact_other={250: 1, 275: 1}, y_by={275: 100}))
    assert mix["slow_learner_evidence_at"] == [250]
    # held-out exact 0 never counts, however good Y72 is
    assert sr.evaluate(hist(y=200))["slow_learner_evidence"] is False


def test_retention_failure_flag_blocks_success_and_boundaries():
    ok = sr.evaluate(hist(y=160))
    assert ok["retention_failure"] is False and ok["success_claim_allowed"] is True
    bad = sr.evaluate(hist(y=159))
    assert bad["retention_failure"] is True and bad["success_claim_allowed"] is False
    assert bad["retention_failure_at"] == [225, 250, 275, 300]
    one = sr.evaluate(hist(y=170, y_by={225: 159}))
    assert one["retention_failure_at"] == [225] and one["success_claim_allowed"] is False
    # retention failure does not change the verdict by itself
    assert sr.evaluate(hist(y=159, held300=HELD200 + 400, train300=1500))["verdict"] == "INCONCLUSIVE"
    assert any("retention failure" in n for n in bad["notes"])


def test_stuck_and_slow_learner_are_both_reported_not_merged():
    out = sr.evaluate(hist(held_exact_other={250: 2}, y=170))        # stuck at 300 (exact 0 at 300), evidence at 250
    assert out["verdict"] == "STUCK" and out["slow_learner_evidence_at"] == [250]
    assert any("slow-learner" in n for n in out["notes"])


# ---- structure, pins, honesty ---------------------------------------------------------------------------
def test_only_two_verdicts_and_no_ambiguous_path_remains():
    verdicts = {sr.evaluate(hist(held300=HELD200 + d, train300=900 + t, held_exact=e))["verdict"]
                for d in (0, 191, 192, 400) for t in (0, 57, 58, 300) for e in (0, 1)}
    assert verdicts == {"STUCK", "INCONCLUSIVE"}
    src = Path(sr.__file__).read_text()
    for banned in ("AMBIGUOUS", "reading_R", "R1", "R2", "NOT_STUCK", "round(", "float("):
        assert banned not in src, banned


def test_partial_history_does_not_decide_and_baseline_is_pinned():
    h = hist()
    del h[300]
    out = sr.evaluate(h)
    assert out["final"] is False and out["verdict"] == "NOT_EVALUATED_BEFORE_300" and "terms" not in out
    bad = hist()
    bad[200]["x64"]["heldout"]["tokens_correct"] = 258
    with pytest.raises(ValueError, match="BASELINE_MISMATCH"):
        sr.evaluate(bad)
    h2 = hist()
    del h2[250]
    with pytest.raises(ValueError, match="NEEDS_250"):
        sr.evaluate(h2)
    assert sr.BASELINE_X64["heldout"]["tokens_correct"] == 257 and sr.BASELINE_X64["train"]["tokens_correct"] == 669
    assert sr.CHECKPOINTS == (225, 250, 275, 300) and sr.FINAL == 300


def test_committed_prereg_matches_the_implementation():
    reg = json.loads((ROOT / "prereg" / "stop_rule_300.json").read_text())
    md = (ROOT / "prereg" / "stop_rule_300.md").read_text()
    assert reg["rule_id"] == sr.RULE_ID
    assert reg["data"]["baseline_counts"] == sr.BASELINE_X64
    assert reg["data"]["checkpoints"] == list(sr.CHECKPOINTS)
    assert reg["data"]["tokens_per_split"] == sr.X64_TOKENS == 1920
    assert (sr.HELDOUT_GAIN_TENTHS_PP, sr.TRAIN_GAIN_TENTHS_PP, sr.Y72_FLOOR_TENTHS_PCT) == (100, 30, 800)
    assert reg["STUCK"]["heldout_gain_threshold_pp"] == "10.0" and reg["STUCK"]["train_gain_threshold_pp"] == "3.0"
    assert reg["arithmetic"]["y72_floor_pct"] == "80.0" and reg["arithmetic"]["gain_clause_operator"] == "strict <"
    for needle in ("< 10.0 pp", "< 3.0 pp", "257/1920", "669/1920", "always stops at 300", "INCONCLUSIVE", "both"):
        assert needle in md, needle
    assert "ambiguous" not in md.lower() and "ambiguous" not in json.dumps(reg).lower().replace("the earlier dual-reading/ambiguous path was removed", "")
