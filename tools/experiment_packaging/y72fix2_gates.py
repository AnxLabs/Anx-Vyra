"""Pre-registered gates for the Y72 retention fix 2 (per-task clip, resume at 225, replay weight unchanged), integer arithmetic only.

SPECIFICATION ONLY: nothing in this module builds, uploads or runs anything. Registered in prereg/y72fix2.json (+ .md) before any run
of this change exists; tests/test_y72fix2_gates.py pins this module to that file.

Same change as vyra-pertask-clip-v1 (per-task clip tau 0.5, lambda 0.25), but retention is now judged on the pre-registered 600-segment
Y72 held-out set (vyra-y72-heldout-v2-gate-v1: sum over 250/275/300 >= 1440 of 1800 AND every checkpoint >= 432 of 600). The acquisition
and non-regression floors are those of vyra-pertask-clip-v1 (7.0 pp token gain) unchanged.
"""
from y72fix_gates import RESUME_X64_HELDOUT, VERDICTS, X32_FLOOR_TENTHS_PCT, X64_HELDOUT_SEG_FLOOR, X109_FLOOR_TENTHS_PCT, gain_ge, pct_ge
import y72_heldout_v2 as heldout

RULE_ID = "vyra-y72fix2-pertask-v2"
CHECKPOINTS = (250, 275, 300)
REPLAY_LAMBDA_NUM_DEN = (1, 4)       # 0.25, unchanged
TASK_CLIP_TENTHS = 5                 # per-task clip 0.5 on gx and on lambda*gy
X64_HELDOUT_GAIN_TENTHS_PP = 70      # 7.0 pp => tokens_correct(300) >= 1553 of 1920 (from 1418)
X64_GAIN_NONGATING_TENTHS_PP = 100   # 10.0 pp (the lambda 0.5 rule), reported, not gating


def evaluate(h):
    """h: {update: {"y600": int, "x64_seg": int, "x64_tokens": int, "x32": int, "x109": int}} for updates 250/275/300 (x64 on 240 segments /
    1920 tokens, x32 on 320, x109 on 120). Returns the gate values and the verdict. Integer arithmetic only."""
    seen = sorted(u for u in h if u in CHECKPOINTS)
    if len(seen) != len(CHECKPOINTS):
        return {"rule": RULE_ID, "checkpoints_seen": seen, "verdict": "NOT_EVALUATED_BEFORE_300"}
    r = heldout.evaluate({u: h[u]["y600"] for u in CHECKPOINTS})
    R = bool(r["R_retention"])
    tok300 = h[CHECKPOINTS[-1]]["x64_tokens"]
    A = all(h[u]["x64_seg"] >= X64_HELDOUT_SEG_FLOOR for u in CHECKPOINTS) and gain_ge(
        tok300, RESUME_X64_HELDOUT["tokens"], RESUME_X64_HELDOUT["tokens_correct"], RESUME_X64_HELDOUT["tokens"], X64_HELDOUT_GAIN_TENTHS_PP)
    N = all(pct_ge(h[u]["x32"], 320, X32_FLOOR_TENTHS_PCT) and pct_ge(h[u]["x109"], 120, X109_FLOOR_TENTHS_PCT) for u in CHECKPOINTS)
    return {"rule": RULE_ID, "sum3": r["sum3"], "R_retention": R, "A_acquisition": bool(A), "N_nonregression": bool(N),
            "nongating_gain_ge_10pp": gain_ge(tok300, 1920, 1418, 1920, X64_GAIN_NONGATING_TENTHS_PP), "verdict": VERDICTS[(R, bool(A), bool(N))]}
