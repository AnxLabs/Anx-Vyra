"""Plan for updates 301..400 of the y72fix2 continuation (prereg/y72fix2_cont400.json): the v1 extension policy, continued.

plan[0:300] is the pinned 300-entry plan of plan_extension (v1), unchanged. Entries 300..399 are entries 100..199 of the SAME v1
SHA-256 counter streams (plan_extension.build_extension(n_gen, n32, count=200)[100:]); build_extension is prefix-stable, so its
first 100 entries are exactly the pinned v1 extension (checked below). No new stream labels, no global RNG, no new data.
Every one of the 40 needle rows is presented 2 times (20 rows) or 3 times (20 rows) in 301..400.
"""
import plan_extension as v1

POLICY_VERSION = "vyra-x-plan-extension-v1-continued-400"
BASE_POLICY_VERSION = "vyra-x-plan-extension-v1"
PREFIX_LEN = 300
CONTINUATION_LEN = 100
FULL_LEN = PREFIX_LEN + CONTINUATION_LEN
PINNED_PREFIX_PLAN_SHA256 = "c3ac1155e0e21b25133d7e76aa63589a98e53d3e80ee3b0c645daa3fcd365ae7"      # plan[0:200] (train_X.pt)
PINNED_PREFIX300_PLAN_SHA256 = "2c59794e261a88216e5d77141b3373073c6e518545dce8a2578f279e95de86aa"   # plan[0:300] (v1 full plan)
PINNED_CONTINUATION_SHA256 = "1cfdd4555fe624363454b886fedafdbd08b04a7c8b67976281f9e70936cb2499"     # plan[300:400]
PINNED_FULL_PLAN_SHA256 = "d9f132342c569d400a737bc7739ff9b87b427858984006437db3fda7f654d943"        # plan[0:400]

PlanExtensionError = v1.PlanExtensionError
plan_sha256 = v1.plan_sha256
canonical_entry = v1.canonical_entry

if (v1.POLICY_VERSION != BASE_POLICY_VERSION or v1.PINNED_PREFIX_PLAN_SHA256 != PINNED_PREFIX_PLAN_SHA256
        or v1.PINNED_FULL_PLAN_SHA256 != PINNED_PREFIX300_PLAN_SHA256 or v1.FULL_LEN != PREFIX_LEN):
    raise PlanExtensionError("PLAN_EXTENSION400 base policy pins differ")


def build_continuation(n_gen, n32):
    """Entries 300..399 and a check that the stream prefix is the pinned v1 extension. Pure function of (n_gen, n32)."""
    ext200 = v1.build_extension(n_gen, n32, count=v1.EXTENSION_LEN + CONTINUATION_LEN)
    if ext200[:v1.EXTENSION_LEN] != v1.build_extension(n_gen, n32):
        raise PlanExtensionError("PLAN_EXTENSION400 stream prefix differs from the v1 extension")
    return ext200[v1.EXTENSION_LEN:]


def build_extended_plan(SX, steps=FULL_LEN, *, check_pins=True):
    """Return the v1 300-entry plan (built and pin-checked by plan_extension) + entries 300..399, fail closed. Does not mutate SX."""
    if int(steps) != FULL_LEN:
        raise PlanExtensionError("PLAN_EXTENSION400 steps must be 400")
    full300 = v1.build_extended_plan(SX, v1.FULL_LEN, check_pins=check_pins)
    n_gen, n32 = len(SX["rows_gen"]), len(SX["rows32"])
    cont = build_continuation(n_gen, n32)
    v1.validate_entries(cont, n_gen, n32)
    if check_pins and plan_sha256(cont) != PINNED_CONTINUATION_SHA256:
        raise PlanExtensionError("PLAN_EXTENSION400_HASH mismatch")
    full = [dict(e, needles=dict(e["needles"])) for e in full300] + cont
    v1.assert_prefix_identity(full, full300)
    if check_pins and (plan_sha256(full[:PREFIX_LEN]) != PINNED_PREFIX300_PLAN_SHA256 or plan_sha256(full) != PINNED_FULL_PLAN_SHA256):
        raise PlanExtensionError("PLAN_FULL400_HASH mismatch")
    return full
