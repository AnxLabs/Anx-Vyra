"""Pre-registered 200 -> 300 stop/assessment rule, integer arithmetic only.

Source: HANDOFF.md of the 200->300 preparation (AJ-approved scope):

  At 300, stop/declare stuck if held-out X64 token gain 200->300 < 10 absolute
  percentage points OR training token gain 250->300 < 3 points with both
  train/held-out segment-exact 0. Evidence consistent with slow learner: any
  X64 HELD-OUT segment-exact > 0 with same-checkpoint Y72 segment-exact >= 80%.
  Flag retention failure if Y72 < 80% at any checkpoint; no success claim
  then. Report overlapping/inconclusive criteria honestly. ALWAYS stop at 300;
  no threshold relaxation or automatic experiment.

No float and no rounded value is ever compared. Every comparison is an integer
cross-multiplication of exact counts. The rule has NO effect on training: the
run always ends at update 300 and never starts a follow-up.

Ambiguity, reported rather than resolved: "A OR B with both segment-exact 0"
can be read R1 = A or (B and C), or R2 = (A or B) and C, where
A = held-out gain < 10pp, B = train gain 250->300 < 3pp, C = train and
held-out X64 segment-exact are both 0 at 300. Both readings are computed.
When they disagree the verdict is AMBIGUOUS, never silently STUCK/NOT_STUCK.
"""
BASELINE_UPDATE = 200
X64_TOKENS = 1920
# Counts at update 200, taken from x64_predictions_200.json (sum of byte_correct).
BASELINE_X64 = {
    "heldout": {"tokens_correct": 257, "tokens": 1920, "segments_exact": 0, "segments": 240},
    "train": {"tokens_correct": 669, "tokens": 1920, "segments_exact": 0, "segments": 240},
}
HELDOUT_GAIN_PP = 10   # A: gain < 10 percentage points
TRAIN_GAIN_PP = 3      # B: gain < 3 percentage points
Y72_FLOOR_PCT = 80     # retention floor, segment-exact, percent
CHECKPOINTS = (225, 250, 275, 300)
FINAL = 300


def _gain_below(new, old, tokens, pp):
    """(new-old)/tokens*100 < pp  <=>  (new-old)*100 < pp*tokens, exact."""
    return (new - old) * 100 < pp * tokens


def _gain_pp_exact(new, old, tokens):
    return f"{(new - old) * 100}/{tokens}"


def _y72_ok(counts):
    c = counts["Y72_unseen"]
    return c["segments_exact"] * 100 >= Y72_FLOOR_PCT * c["segments"]


def evaluate(history):
    """history: {update: {"x64": {"train": counts, "heldout": counts}, "counts": {"Y72_unseen": counts, ...}}}.

    Update 200 may be present (restored-baseline measurement); otherwise the
    pinned BASELINE_X64 is used. Returns a plain dict of exact integers/booleans.
    """
    base = history.get(BASELINE_UPDATE, {}).get("x64", BASELINE_X64)
    if base["heldout"]["tokens_correct"] != BASELINE_X64["heldout"]["tokens_correct"] or \
            base["train"]["tokens_correct"] != BASELINE_X64["train"]["tokens_correct"]:
        raise ValueError("STOP_RULE_BASELINE_MISMATCH")
    out = {"baseline_200": BASELINE_X64, "per_checkpoint": {}, "final": FINAL in history}
    retention_failure = []
    slow = []
    for u in sorted(k for k in history if k in CHECKPOINTS):
        h = history[u]
        x = h["x64"]
        y_ok = _y72_ok(h["counts"])
        y = h["counts"]["Y72_unseen"]
        if not y_ok:
            retention_failure.append(u)
        held_exact_pos = x["heldout"]["segments_exact"] > 0
        out["per_checkpoint"][str(u)] = {
            "x64_heldout_tokens_correct": x["heldout"]["tokens_correct"],
            "x64_train_tokens_correct": x["train"]["tokens_correct"],
            "x64_heldout_gain_since_200_pp_exact": _gain_pp_exact(x["heldout"]["tokens_correct"], BASELINE_X64["heldout"]["tokens_correct"], X64_TOKENS),
            "x64_train_gain_since_200_pp_exact": _gain_pp_exact(x["train"]["tokens_correct"], BASELINE_X64["train"]["tokens_correct"], X64_TOKENS),
            "x64_heldout_segments_exact": x["heldout"]["segments_exact"],
            "x64_train_segments_exact": x["train"]["segments_exact"],
            "y72_segments_exact": f"{y['segments_exact']}/{y['segments']}",
            "y72_retention_ok_ge_80pct": y_ok,
            "slow_learner_evidence": bool(held_exact_pos and y_ok),
        }
        if held_exact_pos and y_ok:
            slow.append(u)
    out["retention_failure_at"] = retention_failure
    out["retention_failure"] = bool(retention_failure)
    out["success_claim_allowed"] = not retention_failure
    out["slow_learner_evidence_at"] = slow
    if FINAL not in history:
        out["stuck_verdict"] = "NOT_EVALUATED_BEFORE_300"
        return out
    if 250 not in history:
        raise ValueError("STOP_RULE_NEEDS_250")
    x300, x250 = history[FINAL]["x64"], history[250]["x64"]
    A = _gain_below(x300["heldout"]["tokens_correct"], BASELINE_X64["heldout"]["tokens_correct"], X64_TOKENS, HELDOUT_GAIN_PP)
    B = _gain_below(x300["train"]["tokens_correct"], x250["train"]["tokens_correct"], X64_TOKENS, TRAIN_GAIN_PP)
    C = x300["train"]["segments_exact"] == 0 and x300["heldout"]["segments_exact"] == 0
    r1 = A or (B and C)
    r2 = (A or B) and C
    if r1 and r2:
        verdict = "STUCK"
    elif not r1 and not r2:
        verdict = "NOT_STUCK"
    else:
        verdict = "AMBIGUOUS_PRE_REGISTRATION_READINGS_DISAGREE"
    out["terms"] = {
        "A_heldout_gain_200_to_300_below_10pp": A,
        "B_train_gain_250_to_300_below_3pp": B,
        "C_train_and_heldout_segment_exact_both_zero_at_300": C,
        "heldout_gain_200_to_300_pp_exact": _gain_pp_exact(x300["heldout"]["tokens_correct"], BASELINE_X64["heldout"]["tokens_correct"], X64_TOKENS),
        "train_gain_250_to_300_pp_exact": _gain_pp_exact(x300["train"]["tokens_correct"], x250["train"]["tokens_correct"], X64_TOKENS),
        "reading_R1_A_or_(B_and_C)": r1,
        "reading_R2_(A_or_B)_and_C": r2,
    }
    out["stuck_verdict"] = verdict
    out["overlap_note"] = None
    if verdict == "STUCK" and slow:
        out["overlap_note"] = "OVERLAPPING/INCONCLUSIVE: stuck criteria met AND slow-learner evidence at " + ",".join(map(str, slow))
    elif verdict.startswith("AMBIGUOUS"):
        out["overlap_note"] = "Pre-registered wording supports two readings that disagree; neither is selected here."
    out["always_stops_at"] = FINAL
    out["automatic_followup"] = None
    return out
