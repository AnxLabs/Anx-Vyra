"""Pre-registered gates for Y72 rehearsal2 300 -> 400 (prereg/y72_rehearsal2.json, rule vyra-y72-rehearsal2-fresh-v1).

The gates are the cont400 gates, unchanged (the registration reuses them verbatim); only the rule id and the pinning test differ.

SPECIFICATION ONLY: nothing in this module builds, uploads or runs anything. Integer arithmetic only (cross-multiplication, `>=`).
tests/test_y72_rehearsal2.py pins this module to the registered JSON.

R hold:     Y72 600-set segment-exact, sum over 350/375/400 >= 1440 of 1800 AND every checkpoint >= 432 of 600
            (vyra-y72-heldout-v2-gate-v1 percentages: 80.0 % sum level, 72.0 % guard).
A progress: X64 held-out segment-exact >= 44 of 240 (value at 300) at every gate checkpoint AND token gain 300 -> 400 >= 3.0 pp
            (tokens_correct(400) >= 1723 of 1920, from 1665).
N:          X32 >= 128/320 (40.0 %) and X109 >= 90/120 (75.0 %) at every gate checkpoint (unchanged).
"""
from y72fix_gates import X32_FLOOR_TENTHS_PCT, X109_FLOOR_TENTHS_PCT, gain_ge, pct_ge

RULE_ID = "vyra-y72-rehearsal2-fresh-v1"
RESUME_UPDATE = 300
FINAL = 400
CHECKPOINTS = (350, 375, 400)
OBSERVATION_ONLY = (325,)
REPLAY_LAMBDA_NUM_DEN = (1, 4)       # 0.25, unchanged (replay rows come from the fresh pool F200)
TASK_CLIP_TENTHS = 5                 # per-task clip 0.5, unchanged
Y600_SEGMENTS = 600
Y600_FLOOR_TENTHS_PCT = 800          # sum level: sum >= 3 * 480 = 1440
Y600_GUARD_TENTHS_PCT = 720          # every checkpoint >= 432
RESUME_X64_HELDOUT = {"tokens_correct": 1665, "tokens": 1920, "segments_exact": 44, "segments": 240}
X64_HELDOUT_SEG_FLOOR = 44
X64_HELDOUT_GAIN_TENTHS_PP = 30      # 3.0 pp => tokens_correct(400) >= 1723
X64_SEG_GAIN_NONGATING = 12          # reported, not gating
VERDICTS = {
    (True, True, True): "SUCCESS",
    (False, True, True): "FAIL_HOLD",
    (True, False, True): "FAIL_PROGRESS",
    (True, True, False): "FAIL_NONREGRESSION",
    (False, False, True): "FAIL_HOLD_PROGRESS",
    (False, True, False): "FAIL_HOLD_NONREGRESSION",
    (True, False, False): "FAIL_PROGRESS_NONREGRESSION",
    (False, False, False): "FAIL_HOLD_PROGRESS_NONREGRESSION",
}
NOT_EVALUATED = "NOT_EVALUATED_BEFORE_400"


def _int(v, hi):
    if isinstance(v, bool) or not isinstance(v, int) or v < 0 or v > hi:
        raise ValueError("COUNT_RANGE")
    return v


def evaluate(h):
    """h: {update: {"y600", "x64_seg", "x64_tokens", "x32", "x109"}} (600 / 240 / 1920 / 320 / 120). Returns gate values + verdict."""
    seen = sorted(u for u in h if u in CHECKPOINTS)
    if len(seen) != len(CHECKPOINTS):
        return {"rule": RULE_ID, "checkpoints_seen": seen, "verdict": NOT_EVALUATED}
    rows = {u: {"y600": _int(h[u]["y600"], 600), "x64_seg": _int(h[u]["x64_seg"], 240), "x64_tokens": _int(h[u]["x64_tokens"], 1920),
                "x32": _int(h[u]["x32"], 320), "x109": _int(h[u]["x109"], 120)} for u in CHECKPOINTS}
    ys = [rows[u]["y600"] for u in CHECKPOINTS]
    total = sum(ys)
    r_sum = total * 1000 >= len(CHECKPOINTS) * Y600_FLOOR_TENTHS_PCT * Y600_SEGMENTS
    r_guard = all(pct_ge(v, Y600_SEGMENTS, Y600_GUARD_TENTHS_PCT) for v in ys)
    R = bool(r_sum and r_guard)
    tok = rows[CHECKPOINTS[-1]]["x64_tokens"]
    a_seg = all(rows[u]["x64_seg"] >= X64_HELDOUT_SEG_FLOOR for u in CHECKPOINTS)
    a_gain = gain_ge(tok, RESUME_X64_HELDOUT["tokens"], RESUME_X64_HELDOUT["tokens_correct"], RESUME_X64_HELDOUT["tokens"],
                     X64_HELDOUT_GAIN_TENTHS_PP)
    A = bool(a_seg and a_gain)
    N = bool(all(pct_ge(rows[u]["x32"], 320, X32_FLOOR_TENTHS_PCT) and pct_ge(rows[u]["x109"], 120, X109_FLOOR_TENTHS_PCT) for u in CHECKPOINTS))
    seg_final = rows[CHECKPOINTS[-1]]["x64_seg"]
    return {"rule": RULE_ID, "checkpoints": list(CHECKPOINTS), "y600": ys, "sum3": total, "R_sum3_ge_floor": bool(r_sum),
            "R_every_checkpoint_ge_guard": bool(r_guard), "R_hold": R, "x64_seg": [rows[u]["x64_seg"] for u in CHECKPOINTS],
            "x64_tokens_final": tok, "A_seg_every_checkpoint_ge_resume": bool(a_seg), "A_token_gain_ge_3pp": bool(a_gain),
            "A_progress": A, "N_nonregression": N,
            "nongating_x64_seg_gain": seg_final - RESUME_X64_HELDOUT["segments_exact"],
            "nongating_x64_seg_gain_ge_12": seg_final - RESUME_X64_HELDOUT["segments_exact"] >= X64_SEG_GAIN_NONGATING,
            "nongating_y600_sum3_minus_3x496": total - 3 * 496,
            "verdict": VERDICTS[(R, A, N)]}
