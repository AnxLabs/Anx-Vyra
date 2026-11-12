"""Pre-registered gates for the replay-weight experiment (resume at 225, lambda 0.25 -> 0.5), integer arithmetic only.

Pre-registered in prereg/y72fix_lambda05.json (+ .md) BEFORE any run of this experiment; a test pins this module to
that file. Nothing here has any effect on training: the run always stops at update 300.

Registered gates (R = retention, A = X64 held-out acquisition, N = non-regression), checkpoints 250, 275, 300:
  R  Y72 segment-exact >= 80.0 %  (>= 160/200) at EVERY checkpoint 250, 275, 300.
  A  X64 held-out segment-exact >= 11 (the resume point's value) at EVERY checkpoint, AND
     X64 held-out token gain 225 -> 300 >= 10.0 pp (tokens_correct(300) - 1418 >= 192 of 1920), strict >= on integers.
  N  X32 segment-exact >= 40.0 % (>= 128/320) and X109 segment-exact >= 75.0 % (>= 90/120) at EVERY checkpoint.
Verdict: SUCCESS iff R and A and N; otherwise FAIL_<names of the failed gates joined by '_'> (see VERDICTS).
The control (lambda 0.25, same plan, same state) is the already-observed 225 -> 300 trajectory of the 300 run; its numbers
are pinned below for the paired, NON-gating comparison. One seed, one run: not a replication.
No float and no rounded value is ever compared.
"""
RULE_ID = "vyra-y72fix-lambda05-v1"
RESUME_UPDATE = 225
CHECKPOINTS = (250, 275, 300)
FINAL = 300
NEW_LAMBDA = 0.5
SAVED_LAMBDA = 0.25
# thresholds in tenths of a percent
Y72_FLOOR_TENTHS_PCT = 800
X32_FLOOR_TENTHS_PCT = 400
X109_FLOOR_TENTHS_PCT = 750
X64_HELDOUT_SEG_FLOOR = 11           # resume point (225) value: 11/240
X64_HELDOUT_GAIN_TENTHS_PP = 100     # 10.0 pp
RESUME_X64_HELDOUT = {"tokens_correct": 1418, "tokens": 1920, "segments_exact": 11, "segments": 240}
VERDICTS = {
    (True, True, True): "SUCCESS",
    (False, True, True): "FAIL_RETENTION",
    (True, False, True): "FAIL_ACQUISITION",
    (True, True, False): "FAIL_NONREGRESSION",
    (False, False, True): "FAIL_RETENTION_ACQUISITION",
    (False, True, False): "FAIL_RETENTION_NONREGRESSION",
    (True, False, False): "FAIL_ACQUISITION_NONREGRESSION",
    (False, False, False): "FAIL_RETENTION_ACQUISITION_NONREGRESSION",
}
# Observed lambda 0.25 trajectory (downloaded outputs of the 300 run): X64 train/held-out tokens & segment-exact, Y72/X32/X109 segment-exact.
CONTROL = {
    250: {"x64_train": (1701, 80), "x64_heldout": (1542, 29), "y72": 137, "x32": 206, "x109": 99},
    275: {"x64_train": (1785, 126), "x64_heldout": (1603, 46), "y72": 137, "x32": 235, "x109": 106},
    300: {"x64_train": (1854, 182), "x64_heldout": (1687, 92), "y72": 153, "x32": 264, "x109": 109},
}


def pct_ge(count, total, tenths_pct):
    """count/total*100 >= tenths_pct/10, exact."""
    return count * 1000 >= tenths_pct * total


def gain_ge(new, new_tokens, old, old_tokens, tenths_pp):
    """(new/new_tokens - old/old_tokens)*100 >= tenths_pp/10, exact."""
    return (new * old_tokens - old * new_tokens) * 1000 >= tenths_pp * new_tokens * old_tokens


def gain_pp_exact(new, new_tokens, old, old_tokens):
    if new_tokens == old_tokens:
        return f"{(new - old) * 100}/{new_tokens}"
    return f"{(new * old_tokens - old * new_tokens) * 100}/{new_tokens * old_tokens}"


def checkpoint_gates(h):
    c, x = h["counts"], h["x64"]
    y, x32, x109 = c["Y72_unseen"], c["X32_unseen"], c["X109_unseen"]
    return {
        "R_y72_ge_80_0": pct_ge(y["segments_exact"], y["segments"], Y72_FLOOR_TENTHS_PCT),
        "A_x64_heldout_seg_ge_resume": x["heldout"]["segments_exact"] >= X64_HELDOUT_SEG_FLOOR,
        "N_x32_ge_40_0": pct_ge(x32["segments_exact"], x32["segments"], X32_FLOOR_TENTHS_PCT),
        "N_x109_ge_75_0": pct_ge(x109["segments_exact"], x109["segments"], X109_FLOOR_TENTHS_PCT),
    }


def evaluate(history):
    """history: {update: {"x64": {"train": counts, "heldout": counts}, "counts": {"Y72_unseen":..., "X32_unseen":..., "X109_unseen":...}}}.
    Update 225 may be present (the restored measurement); it is not gated. Needs all of 250/275/300 for a verdict."""
    if RESUME_UPDATE in history:
        r = history[RESUME_UPDATE]["x64"]["heldout"]
        for k in ("tokens_correct", "tokens", "segments_exact"):
            if r[k] != RESUME_X64_HELDOUT[k]:
                raise ValueError("Y72FIX_RESUME_BASELINE_MISMATCH")
    seen = sorted(u for u in history if u in CHECKPOINTS)
    per = {}
    for u in seen:
        g = checkpoint_gates(history[u])
        y = history[u]["counts"]["Y72_unseen"]
        per[str(u)] = {**g, "y72_segments_exact": f"{y['segments_exact']}/{y['segments']}",
                       "x64_heldout_tokens_correct": history[u]["x64"]["heldout"]["tokens_correct"],
                       "x64_heldout_segments_exact": history[u]["x64"]["heldout"]["segments_exact"]}
    out = {"rule": RULE_ID, "checkpoints_seen": seen, "per_checkpoint": per, "final": FINAL in history,
           "retention_failed_at": [u for u in seen if not per[str(u)]["R_y72_ge_80_0"]],
           "always_stops_at": FINAL, "automatic_followup": None,
           "note": "single seed, single run; the lambda 0.25 control is the observed 225->300 trajectory; not a replication"}
    paired = {}
    for u in seen:
        if u in CONTROL:
            h, ctl = history[u], CONTROL[u]
            paired[str(u)] = {"y72_seg_minus_control": h["counts"]["Y72_unseen"]["segments_exact"] - ctl["y72"],
                              "x32_seg_minus_control": h["counts"]["X32_unseen"]["segments_exact"] - ctl["x32"],
                              "x109_seg_minus_control": h["counts"]["X109_unseen"]["segments_exact"] - ctl["x109"],
                              "x64_heldout_tokens_minus_control": h["x64"]["heldout"]["tokens_correct"] - ctl["x64_heldout"][0],
                              "x64_heldout_seg_minus_control": h["x64"]["heldout"]["segments_exact"] - ctl["x64_heldout"][1],
                              "x64_train_tokens_minus_control": h["x64"]["train"]["tokens_correct"] - ctl["x64_train"][0]}
    out["paired_vs_control_lambda025_nongating"] = paired
    if FINAL not in history or any(u not in history for u in CHECKPOINTS):
        out["verdict"] = "NOT_EVALUATED_BEFORE_300"
        return out
    R = all(per[str(u)]["R_y72_ge_80_0"] for u in CHECKPOINTS)
    A_seg = all(per[str(u)]["A_x64_heldout_seg_ge_resume"] for u in CHECKPOINTS)
    h300 = history[FINAL]["x64"]["heldout"]
    A_gain = gain_ge(h300["tokens_correct"], h300["tokens"], RESUME_X64_HELDOUT["tokens_correct"], RESUME_X64_HELDOUT["tokens"], X64_HELDOUT_GAIN_TENTHS_PP)
    A = bool(A_seg and A_gain)
    N = all(per[str(u)]["N_x32_ge_40_0"] and per[str(u)]["N_x109_ge_75_0"] for u in CHECKPOINTS)
    out["gates"] = {"R_retention": bool(R), "A_acquisition": A, "A_seg_floor_every_checkpoint": bool(A_seg), "A_heldout_gain_225_to_300_ge_10_0_pp": bool(A_gain),
                    "A_heldout_gain_pp_exact": gain_pp_exact(h300["tokens_correct"], h300["tokens"], RESUME_X64_HELDOUT["tokens_correct"], RESUME_X64_HELDOUT["tokens"]),
                    "N_nonregression": bool(N)}
    out["verdict"] = VERDICTS[(bool(R), A, bool(N))]
    out["success_claim_allowed"] = out["verdict"] == "SUCCESS"
    return out
