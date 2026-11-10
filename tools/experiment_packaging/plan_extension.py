"""Prefix-preserving full-X sampling-plan extension for updates 201..300.

The pinned train_X.pt plan has exactly 200 entries (updates 1..200). Updates
201..300 need plan indices 200..299, which do not exist. This module defines
ONE documented, deterministic policy that builds those 100 entries ONLY from
row tensors that already exist inside the pinned train_X.pt. It never
regenerates, repacks, or re-indexes anything, and it never changes, reorders
or re-derives entries 0..199.

Policy (version 1)
------------------
* needles (distances 32, 64 and 109 share one index, exactly as the original):
  concatenated freshly shuffled epochs over range(40) from the NEEDLE stream,
  truncated to 100 entries. 100 updates = 2 full epochs + 20 entries of a
  third. Over 300 updates every row is presented 7 times (20 rows) or 8 times
  (20 rows).
* slot0 / slot1 (general rows): per update, two independent draws WITH
  replacement from range(len(rows_gen)) = 0..305 (GEN stream, order slot0 then
  slot1, mirroring GEN_SEQ[2k], GEN_SEQ[2k+1] of the original builder).
* slot2 (needle-32 corpus rows): one draw WITH replacement from
  range(len(rows32)) = 0..116 (N32 stream).
  The original builder drew from the full corpus pool and packed only the rows
  it happened to draw (306 and 117 rows). Only those packed rows exist, so the
  extension resamples inside them. That is a documented departure from the
  original draw distribution, not a hidden one.

Independent RNG streams
-----------------------
The streams are SHA-256 in counter mode, keyed by a fixed label. They do not
use `random`, `numpy` or `torch`, so they cannot perturb Random(901) (builder
GEN/N32 draws) or Random(9101) (needle epochs), cannot touch any global RNG,
and give the same bytes on every Python version.
"""
import hashlib
import json

PINNED_TRAIN_X_SHA256 = "59b3f636b11bcacbe8df27a1ee9f0fc4d3589e4645d2601b93e641de5f786440"
PREFIX_LEN = 200
EXTENSION_LEN = 100
FULL_LEN = PREFIX_LEN + EXTENSION_LEN
N_NEEDLE_ROWS = 40
DISTANCES = (32, 64, 109)
POLICY_VERSION = "vyra-x-plan-extension-v1"
STREAM_LABELS = {
    "needle": POLICY_VERSION + "/needle-epochs",
    "gen": POLICY_VERSION + "/gen-slots",
    "n32": POLICY_VERSION + "/n32-slot",
}
# Pinned results of the policy above. Any drift (code, labels, inputs) fails closed.
PINNED_PREFIX_PLAN_SHA256 = "c3ac1155e0e21b25133d7e76aa63589a98e53d3e80ee3b0c645daa3fcd365ae7"
PINNED_EXTENSION_SHA256 = "6a866bd9800c5559a14d703fe5d4b3a016c8cf453064d9709917059b8afc698f"
PINNED_FULL_PLAN_SHA256 = "2c59794e261a88216e5d77141b3373073c6e518545dce8a2578f279e95de86aa"


class PlanExtensionError(ValueError):
    pass


class Stream:
    """Deterministic uniform stream: SHA-256(label|counter), rejection sampling."""

    def __init__(self, label):
        self.label = label
        self.counter = 0

    def _word(self):
        digest = hashlib.sha256(f"{self.label}|{self.counter}".encode()).digest()
        self.counter += 1
        return int.from_bytes(digest[:8], "big")

    def randbelow(self, n):
        if n <= 0:
            raise PlanExtensionError("randbelow n<=0")
        limit = (1 << 64) // n * n
        while True:
            word = self._word()
            if word < limit:
                return word % n

    def shuffle(self, items):
        for i in range(len(items) - 1, 0, -1):
            j = self.randbelow(i + 1)
            items[i], items[j] = items[j], items[i]


def canonical_entry(entry):
    needles = entry["needles"]
    return [int(entry["slot0"]), int(entry["slot1"]), int(entry["slot2"])] + [int(needles[d]) for d in DISTANCES]


def plan_sha256(plan):
    text = json.dumps([canonical_entry(e) for e in plan], separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _entry(slot0, slot1, slot2, needle):
    return {"slot0": slot0, "slot1": slot1, "slot2": slot2, "needles": {d: needle for d in DISTANCES}}


def build_extension(n_gen, n32, count=EXTENSION_LEN):
    """The 100 new entries for plan indices 200..299. Pure function of (n_gen, n32, count)."""
    needle, gen, n32s = (Stream(STREAM_LABELS[k]) for k in ("needle", "gen", "n32"))
    order = []
    while len(order) < count:
        epoch = list(range(N_NEEDLE_ROWS))
        needle.shuffle(epoch)
        order.extend(epoch)
    order = order[:count]
    out = []
    for k in range(count):
        s0 = gen.randbelow(n_gen)
        s1 = gen.randbelow(n_gen)
        s2 = n32s.randbelow(n32)
        out.append(_entry(s0, s1, s2, order[k]))
    return out


def assert_prefix_identity(extended, original):
    """Entries 0..len(original)-1 must be identical objects-by-value and hash."""
    if len(extended) < len(original):
        raise PlanExtensionError("PLAN_PREFIX_MISMATCH length")
    for i, entry in enumerate(original):
        if extended[i] != entry or canonical_entry(extended[i]) != canonical_entry(entry):
            raise PlanExtensionError(f"PLAN_PREFIX_MISMATCH entry {i}")
    if plan_sha256(extended[:len(original)]) != plan_sha256(original):
        raise PlanExtensionError("PLAN_PREFIX_MISMATCH hash")


def validate_entries(plan, n_gen, n32):
    for i, e in enumerate(plan):
        if not (0 <= e["slot0"] < n_gen and 0 <= e["slot1"] < n_gen and 0 <= e["slot2"] < n32):
            raise PlanExtensionError(f"PLAN_INDEX_OUT_OF_RANGE entry {i}")
        if sorted(e["needles"]) != sorted(DISTANCES) or not all(0 <= e["needles"][d] < N_NEEDLE_ROWS for d in DISTANCES):
            raise PlanExtensionError(f"PLAN_NEEDLE_OUT_OF_RANGE entry {i}")


def build_extended_plan(SX, steps=FULL_LEN, *, check_pins=True):
    """Return plan[0:200] (untouched) + the 100-entry extension, fail closed.

    Does not mutate SX. `check_pins=False` is for hermetic unit tests on
    synthetic inputs only; the notebook always calls with the default.
    """
    plan = SX["plan"]
    n_gen, n32 = len(SX["rows_gen"]), len(SX["rows32"])
    if len(plan) != PREFIX_LEN:
        raise PlanExtensionError(f"PLAN_PREFIX_MISMATCH pinned plan length {len(plan)} != {PREFIX_LEN}")
    if int(steps) != FULL_LEN:
        raise PlanExtensionError("PLAN_EXTENSION steps must be 300")
    if check_pins and plan_sha256(plan) != PINNED_PREFIX_PLAN_SHA256:
        raise PlanExtensionError("PLAN_PREFIX_MISMATCH pinned prefix hash")
    validate_entries(plan, n_gen, n32)
    extension = build_extension(n_gen, n32)
    validate_entries(extension, n_gen, n32)
    if check_pins and plan_sha256(extension) != PINNED_EXTENSION_SHA256:
        raise PlanExtensionError("PLAN_EXTENSION_HASH mismatch")
    full = [dict(e, needles=dict(e["needles"])) for e in plan] + extension
    assert_prefix_identity(full, plan)
    if check_pins and plan_sha256(full) != PINNED_FULL_PLAN_SHA256:
        raise PlanExtensionError("PLAN_FULL_HASH mismatch")
    return full
