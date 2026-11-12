"""Pre-registered gates for the per-task-clip experiment (resume at 225, per-task clip 0.5, lambda stays 0.25), integer arithmetic only.

SPECIFICATION ONLY: nothing in this module builds, uploads or runs anything. Registered in prereg/pertask_clip_v1.json (+ .md)
before any run of this experiment exists; tests/test_pertask_clip_gates.py pins this module to that file.

Difference from y72fix_gates (the lambda 0.5 rule), chosen from the CPU noise-floor study (vyra-noise-clip-study-20261110):
  R  retention is judged on the SUM of the three checkpoint Y72 segment-exact counts, sum(250,275,300) >= 480 (mean >= 160/200 = 80.0%),
     not on "every checkpoint >= 160": one checkpoint has SD ~5.3 segments of pure iterate noise, the sum of three has SD ~9-12
     (mean ~3-4), so the old rule failed a model whose true level is 168/200 7-19% of the time (checkpoints fully correlated .. independent). A per-checkpoint collapse guard is kept:
     every checkpoint >= 72.0% (>= 144/200).
  A  X64 held-out segment-exact >= 11 at every checkpoint AND token gain 225 -> 300 >= 7.0 pp (>= 1553 of 1920 from 1418); the control
     gained 14.0 pp, lambda 0.5 gained 10.4 pp, the one-step probe shows the per-task clip makes ~0.60x the X progress per step.
  N  unchanged (X32 >= 128/320, X109 >= 90/120 at every checkpoint).
No float and no rounded value is ever compared.
"""
from y72fix_gates import (CONTROL, RESUME_X64_HELDOUT, VERDICTS, X32_FLOOR_TENTHS_PCT, X64_HELDOUT_SEG_FLOOR, X109_FLOOR_TENTHS_PCT,
                          gain_ge, gain_pp_exact, pct_ge)

RULE_ID = "vyra-pertask-clip-v1"
RESUME_UPDATE = 225
CHECKPOINTS = (250, 275, 300)
FINAL = 300
REPLAY_LAMBDA = 0.25          # unchanged
TASK_CLIP = 0.5               # per-task clip max-norm (each of gx and lambda*gy); combined norm <= 2*TASK_CLIP = 1.0
Y72_SUM3_FLOOR = 480          # sum over 250/275/300 of Y72 segments_exact (600 segment-evaluations), i.e. mean 160/200
Y72_GUARD_TENTHS_PCT = 720    # every checkpoint >= 72.0% (>= 144/200)
X64_HELDOUT_GAIN_TENTHS_PP = 70   # 7.0 pp
# lambda 0.5 (observed, FAIL_RETENTION run) Y72 segment-exact at 250/275/300 and X64 held-out tokens at 300: non-gating comparison
LAMBDA05 = {250: 152, 275: 147, 300: 152}
LAMBDA05_X64_HELDOUT_TOKENS_300 = 1618


def checkpoint_gates(h):
    c, x = h["counts"], h["x64"]
    y, x32, x109 = c["Y72_unseen"], c["X32_unseen"], c["X109_unseen"]
    return {
        "R_y72_ge_72_0_guard": pct_ge(y["segments_exact"], y["segments"], Y72_GUARD_TENTHS_PCT),
        "A_x64_heldout_seg_ge_resume": x["heldout"]["segments_exact"] >= X64_HELDOUT_SEG_FLOOR,
        "N_x32_ge_40_0": pct_ge(x32["segments_exact"], x32["segments"], X32_FLOOR_TENTHS_PCT),
        "N_x109_ge_75_0": pct_ge(x109["segments_exact"], x109["segments"], X109_FLOOR_TENTHS_PCT),
    }


def evaluate(history):
    """history as in y72fix_gates.evaluate. Needs all of 250/275/300 for a verdict."""
    if RESUME_UPDATE in history:
        r = history[RESUME_UPDATE]["x64"]["heldout"]
        for k in ("tokens_correct", "tokens", "segments_exact"):
            if r[k] != RESUME_X64_HELDOUT[k]:
                raise ValueError("PERTASK_CLIP_RESUME_BASELINE_MISMATCH")
    seen = sorted(u for u in history if u in CHECKPOINTS)
    per = {}
    for u in seen:
        g = checkpoint_gates(history[u])
        y = history[u]["counts"]["Y72_unseen"]
        per[str(u)] = {**g, "y72_segments_exact": f"{y['segments_exact']}/{y['segments']}",
                       "x64_heldout_tokens_correct": history[u]["x64"]["heldout"]["tokens_correct"],
                       "x64_heldout_segments_exact": history[u]["x64"]["heldout"]["segments_exact"]}
    sum3 = sum(history[u]["counts"]["Y72_unseen"]["segments_exact"] for u in seen)
    out = {"rule": RULE_ID, "checkpoints_seen": seen, "per_checkpoint": per, "final": FINAL in history, "always_stops_at": FINAL,
           "automatic_followup": None, "y72_sum_over_seen_checkpoints": sum3,
           "note": "single seed, single run; one-checkpoint Y72 SD ~5.3 segments; controls are observed trajectories; not a replication"}
    paired = {}
    for u in seen:
        if u in CONTROL:
            h, ctl = history[u], CONTROL[u]
            paired[str(u)] = {"y72_seg_minus_lambda025": h["counts"]["Y72_unseen"]["segments_exact"] - ctl["y72"],
                              "y72_seg_minus_lambda05": h["counts"]["Y72_unseen"]["segments_exact"] - LAMBDA05[u],
                              "x64_heldout_tokens_minus_lambda025": h["x64"]["heldout"]["tokens_correct"] - ctl["x64_heldout"][0],
                              "x64_heldout_seg_minus_lambda025": h["x64"]["heldout"]["segments_exact"] - ctl["x64_heldout"][1]}
    out["paired_nongating"] = paired
    if FINAL not in history or any(u not in history for u in CHECKPOINTS):
        out["verdict"] = "NOT_EVALUATED_BEFORE_300"
        return out
    R_sum = sum3 >= Y72_SUM3_FLOOR
    R_guard = all(per[str(u)]["R_y72_ge_72_0_guard"] for u in CHECKPOINTS)
    R = bool(R_sum and R_guard)
    A_seg = all(per[str(u)]["A_x64_heldout_seg_ge_resume"] for u in CHECKPOINTS)
    h300 = history[FINAL]["x64"]["heldout"]
    A_gain = gain_ge(h300["tokens_correct"], h300["tokens"], RESUME_X64_HELDOUT["tokens_correct"], RESUME_X64_HELDOUT["tokens"], X64_HELDOUT_GAIN_TENTHS_PP)
    A = bool(A_seg and A_gain)
    N = all(per[str(u)]["N_x32_ge_40_0"] and per[str(u)]["N_x109_ge_75_0"] for u in CHECKPOINTS)
    out["gates"] = {"R_retention": R, "R_sum3_ge_480": bool(R_sum), "R_every_checkpoint_ge_144": bool(R_guard),
                    "A_acquisition": A, "A_seg_floor_every_checkpoint": bool(A_seg), "A_heldout_gain_225_to_300_ge_7_0_pp": bool(A_gain),
                    "A_heldout_gain_pp_exact": gain_pp_exact(h300["tokens_correct"], h300["tokens"], RESUME_X64_HELDOUT["tokens_correct"], RESUME_X64_HELDOUT["tokens"]),
                    "N_nonregression": bool(N)}
    out["verdict"] = VERDICTS[(R, A, bool(N))]
    out["success_claim_allowed"] = out["verdict"] == "SUCCESS"
    return out
