"""Pre-registered 200 -> 300 stop rule: ONE rule, integer arithmetic only.

The same text is committed as prereg/stop_rule_300.json and prereg/stop_rule_300.md
(a test pins them to this module). It was registered BEFORE any update 201..300 data
from the real (GPU) run existed. FORGE chose this reading of the original wording
("stop/declare stuck if held-out gain < 10 pp OR train gain < 3 pp with both
segment-exact 0"): the exact-0 condition gates BOTH gain clauses.

  STUCK at update 300  iff
      X64 train segment-exact count(300) == 0  AND  X64 held-out segment-exact count(300) == 0
      AND ( heldout_token_acc(300) - heldout_token_acc(200) < 10.0 percentage points
            OR train_token_acc(300) - train_token_acc(250) < 3.0 percentage points )
  SLOW-LEARNER evidence iff, at ANY of 225/250/275/300:
      X64 held-out segment-exact count > 0  AND  Y72 segment-exact (same checkpoint) >= 80.0 %
  RETENTION FAILURE iff Y72 segment-exact < 80.0 % at ANY of 225/250/275/300
      (blocks any success claim)
  otherwise the verdict is INCONCLUSIVE.
  Training ALWAYS stops at 300, whatever the verdict; nothing follows automatically.

token_acc = tokens_correct / tokens * 100 from integer counts (X64: 1920 tokens per split,
240 segments per split); comparisons are exact integer cross-multiplications with a strict `<`
for the gain clauses and `>=` for the Y72 floor. No float and no rounded value is ever compared.
"""
BASELINE_UPDATE = 200
X64_TOKENS = 1920
# Counts at update 200, taken from x64_predictions_200.json (sum of byte_correct).
BASELINE_X64 = {
    "heldout": {"tokens_correct": 257, "tokens": 1920, "segments_exact": 0, "segments": 240},
    "train": {"tokens_correct": 669, "tokens": 1920, "segments_exact": 0, "segments": 240},
}
# Thresholds in tenths of a percentage point, so 10.0 pp == 100 and 3.0 pp == 30.
HELDOUT_GAIN_TENTHS_PP = 100   # held-out gain 200 -> 300 must be  < 10.0 pp for the clause to hold
TRAIN_GAIN_TENTHS_PP = 30      # train gain 250 -> 300 must be  < 3.0 pp for the clause to hold
Y72_FLOOR_TENTHS_PCT = 800     # Y72 segment-exact floor, 80.0 %
CHECKPOINTS = (225, 250, 275, 300)
FINAL = 300
RULE_ID = "vyra-stop-rule-300-v1"
VERDICTS = ("STUCK", "INCONCLUSIVE")


def gain_below(new, new_tokens, old, old_tokens, tenths_pp):
    """(new/new_tokens - old/old_tokens) * 100 < tenths_pp / 10, exact (strict)."""
    return (new * old_tokens - old * new_tokens) * 1000 < tenths_pp * new_tokens * old_tokens


def gain_pp_exact(new, new_tokens, old, old_tokens):
    """The exact gain in percentage points as a fraction string 'num/den'."""
    if new_tokens == old_tokens:
        return f"{(new - old) * 100}/{new_tokens}"
    return f"{(new * old_tokens - old * new_tokens) * 100}/{new_tokens * old_tokens}"


def y72_ok(counts):
    c = counts["Y72_unseen"]
    return c["segments_exact"] * 1000 >= Y72_FLOOR_TENTHS_PCT * c["segments"]


def evaluate(history):
    """history: {update: {"x64": {"train": counts, "heldout": counts}, "counts": {"Y72_unseen": counts, ...}}}.

    Update 200 may be present (restored-baseline measurement, must equal the pinned
    baseline); otherwise the pinned BASELINE_X64 is used. Returns plain ints/bools/strings.
    """
    base = history.get(BASELINE_UPDATE, {}).get("x64", BASELINE_X64)
    for k in ("heldout", "train"):
        if base[k]["tokens_correct"] != BASELINE_X64[k]["tokens_correct"] or base[k]["tokens"] != BASELINE_X64[k]["tokens"]:
            raise ValueError("STOP_RULE_BASELINE_MISMATCH")
    seen = sorted(k for k in history if k in CHECKPOINTS)
    retention_failure_at, slow_at, per = [], [], {}
    for u in seen:
        x, y = history[u]["x64"], history[u]["counts"]["Y72_unseen"]
        ok = y72_ok(history[u]["counts"])
        slow = bool(x["heldout"]["segments_exact"] > 0 and ok)
        if not ok:
            retention_failure_at.append(u)
        if slow:
            slow_at.append(u)
        per[str(u)] = {
            "x64_heldout_tokens_correct": x["heldout"]["tokens_correct"], "x64_train_tokens_correct": x["train"]["tokens_correct"],
            "x64_heldout_segments_exact": x["heldout"]["segments_exact"], "x64_train_segments_exact": x["train"]["segments_exact"],
            "x64_heldout_gain_since_200_pp_exact": gain_pp_exact(x["heldout"]["tokens_correct"], x["heldout"]["tokens"], BASELINE_X64["heldout"]["tokens_correct"], BASELINE_X64["heldout"]["tokens"]),
            "x64_train_gain_since_200_pp_exact": gain_pp_exact(x["train"]["tokens_correct"], x["train"]["tokens"], BASELINE_X64["train"]["tokens_correct"], BASELINE_X64["train"]["tokens"]),
            "y72_segments_exact": f"{y['segments_exact']}/{y['segments']}",
            "y72_ge_80_0_pct": ok,
            "slow_learner_evidence": slow,
        }
    out = {
        "rule": RULE_ID, "baseline_200": BASELINE_X64, "per_checkpoint": per, "final": FINAL in history,
        "retention_failure": bool(retention_failure_at), "retention_failure_at": retention_failure_at,
        "success_claim_allowed": not retention_failure_at,
        "slow_learner_evidence": bool(slow_at), "slow_learner_evidence_at": slow_at,
        "always_stops_at": FINAL, "automatic_followup": None,
    }
    if FINAL not in history:
        out["verdict"] = "NOT_EVALUATED_BEFORE_300"
        return out
    if 250 not in history:
        raise ValueError("STOP_RULE_NEEDS_250")
    x300, x250 = history[FINAL]["x64"], history[250]["x64"]
    exact_zero = x300["train"]["segments_exact"] == 0 and x300["heldout"]["segments_exact"] == 0
    held_below = gain_below(x300["heldout"]["tokens_correct"], x300["heldout"]["tokens"],
                            BASELINE_X64["heldout"]["tokens_correct"], BASELINE_X64["heldout"]["tokens"], HELDOUT_GAIN_TENTHS_PP)
    train_below = gain_below(x300["train"]["tokens_correct"], x300["train"]["tokens"],
                             x250["train"]["tokens_correct"], x250["train"]["tokens"], TRAIN_GAIN_TENTHS_PP)
    stuck = bool(exact_zero and (held_below or train_below))
    out["terms"] = {
        "segment_exact_both_zero_at_300": exact_zero,
        "heldout_gain_200_to_300_below_10_0_pp": held_below,
        "train_gain_250_to_300_below_3_0_pp": train_below,
        "heldout_gain_200_to_300_pp_exact": gain_pp_exact(x300["heldout"]["tokens_correct"], x300["heldout"]["tokens"], BASELINE_X64["heldout"]["tokens_correct"], BASELINE_X64["heldout"]["tokens"]),
        "train_gain_250_to_300_pp_exact": gain_pp_exact(x300["train"]["tokens_correct"], x300["train"]["tokens"], x250["train"]["tokens_correct"], x250["train"]["tokens"]),
    }
    out["verdict"] = "STUCK" if stuck else "INCONCLUSIVE"
    out["notes"] = []
    if out["verdict"] == "STUCK" and slow_at:
        out["notes"].append("STUCK criteria met and slow-learner evidence at " + ",".join(map(str, slow_at)) + "; both reported, neither overrides the other")
    if retention_failure_at:
        out["notes"].append("retention failure at " + ",".join(map(str, retention_failure_at)) + "; no success claim")
    return out
