"""Experiment Z data builder - parameterised, derived, self-describing.

Same generation logic as Experiment Y, with every quantity Y hardcoded turned into a parameter and
the exposure arithmetic DERIVED rather than asserted to a constant.

Experiment Y asked one question: does presenting a BAND of needle-to-query distances around each
trained target repair the neighbours (40 and 72) that collapsed in X? Y answered it, and Z reuses
its machinery unchanged. What Z adds is the ability to build several arms from one script, so that
Arm A and Arm B differ by declared parameters and nothing else, and so that each corpus records in
its own payload the parameters that produced it.

THE EXPOSURE INVARIANT - the part that must never be fudged.

Each step presents every target exactly once, at one geometry drawn from that target's jitter band.
So over STEPS steps a target is presented STEPS times, spread evenly over len(JITTER) geometries:

    presentations_per_geometry = STEPS // len(JITTER)

Rows are laid out as len(JITTER) jitter values x N_TR rows, so one full epoch over the
(geometry, row) grid takes len(base) = len(JITTER) * N_TR steps. Every (geometry, row) pair is
presented once per epoch, therefore

    passes = STEPS // len(base)
    presentations_per_geometry = passes * N_TR

Two independent routes to the same number. If they disagree the plan is not balanced and the run is
not what its label says it is, so this module REFUSES TO BUILD rather than rounding to fit.

Y needed PRESENT_PER_GEOM // N_TR == 5 and got it with exactly one epoch (len(base) == 200 // 5).
That is a coincidence of Y's parameter choice, not a law, and asserting it is why the Y script
cannot build any other arm. Z derives passes and refuses to build when the division is inexact.
"""
import hashlib, json, os, random, sys
from datetime import datetime, timezone
from collections import Counter

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_code():
    cands = [os.environ.get("T2_SRC", ""), "bundle",
             os.path.join(HERE, "src"), os.path.join(os.path.dirname(HERE), "src"), HERE]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "run_vyra1_copy.py")):
            return c
    raise SystemExit("cannot locate run_vyra1_copy.py; set T2_SRC")


CODE = _find_code()
sys.path.insert(0, HERE)
sys.path.insert(0, CODE)
import vyra_shim
import run_vyra1_copy as rv

LAB = os.environ.get("VYRA_LAB", "./lab")
OUT = os.environ.get("Z_OUT", os.environ.get("T2_OUT", "./z_data"))
os.makedirs(OUT, exist_ok=True)
os.makedirs(LAB, exist_ok=True)
APPLIED = vyra_shim.patch_library(rv, LAB)
# The shim records where the lab was set up. On the build machine that is an absolute path, and
# baking it into a published corpus makes the corpus hash a function of WHERE it was built: Y's
# published hash could never be reproduced by a reader for exactly this reason, and the path also
# leaks a local directory layout into an artifact. Record the SHAPE of the redirection, not the
# live path. The parameter is echoed as given so a build is still describable.
APPLIED = [a if not a.startswith("LAB->") else "LAB->$VYRA_LAB" for a in APPLIED]

BOS, EOS, BODY, SEQ = 256, 257, 510, 512
MEMB, QRYB = b"MEM:", b"QUERY:"
COPT_MAX = 8
def _env(name, legacy=None, default=None):
    """Read a parameter. Z_<NAME> wins, then the Y_ name Y used, then the default."""
    for key in (f"Z_{name}",) + ((legacy,) if legacy else ()):
        v = os.environ.get(key)
        if v is not None and v.strip() != "":
            return v.strip()
    if default is None:
        raise SystemExit(f"{name} is required (set Z_{name})")
    return default


def _int_list(raw):
    vals = [int(x) for x in raw.split(",") if x.strip()]
    if not vals:
        raise SystemExit(f"empty integer list: {raw!r}")
    return vals


# ---- the approved arm table ------------------------------------------------------------------
# "A flipped default is still a default." A default that happens to build the withdrawn arm is a
# trap, and flipping it just moves the trap. So a Z build must NAME its arm, and the resolved
# numbers must match that arm's approved row exactly or the build refuses. This closes the whole
# class of unset-variable defects at once: right arm with a stale N_TR, right jitter with the old
# GAPS, and so on, all stop being silently buildable.
ARM_TABLE = {
    "A": {"steps": 600, "jitter": [-4, 0, 4], "n_tr": 40,
          "gaps": [40, 44, 48, 52, 72, 76, 80, 84, 88, 92, 96, 121, 128],
          "passes": 5, "geometries": 9},
    "B": {"steps": 200, "jitter": [0], "n_tr": 8,
          "gaps": [40, 44, 48, 52, 72, 76, 80, 84, 88, 92, 96, 121, 128],
          "passes": 25, "geometries": 3},
}
# Arm B carries the 40/72 probes too. The verdict reads the recovery of the two distances X
# collapsed on BOTH arms, and a distance absent from every namespace makes the verdict self-test
# refuse to start - correctly, but it would have surfaced as Arm B failing to launch rather than as
# a package that was built one probe pair short. Neither arm trains 40 or 72, so adding the probes
# creates no band collision on either. gap_mean stays over the original eleven either way.
ARM = _env("ARM", default="")
if not ARM:
    raise SystemExit(
        "Z_ARM is required: name the arm this corpus is for (one of "
        f"{sorted(ARM_TABLE)}). Building without naming the arm is how a stale default produces a "
        "corpus that reports itself as an arm it is not.")
if ARM not in ARM_TABLE:
    raise SystemExit(f"Z_ARM={ARM!r} is not an approved arm; approved: {sorted(ARM_TABLE)}")
_ARM_REQUESTED = ARM
_SPEC = ARM_TABLE[ARM]

STEPS = int(_env("STEPS", default=str(_SPEC["steps"])))
N_TR = int(_env("N_TR", default=str(_SPEC["n_tr"])))   # rows per geometry
N_UN = int(_env("N_UN", default="40"))       # unseen rows per geometry
N_GAP = int(_env("N_GAP", default="20"))     # rows per interpolation probe distance

TARGETS = _int_list(_env("TARGETS", "Y_TARGETS", "32,64,109"))
# Default comes from the named arm, never a global constant. The withdrawn +/-8 band must not be
# reachable by forgetting a variable: it builds 15 geometries and 3 passes while still calling
# itself Arm A.
JITTER = _int_list(_env("JITTER", "Y_JITTER", ",".join(str(j) for j in _SPEC["jitter"])))
GAPS = _int_list(_env("GAPS", "Y_GAPS", ",".join(str(g) for g in _SPEC["gaps"])))
# Band distances this arm is REQUIRED to cover, e.g. "40,72" for the X regression.
# Empty by default: Z's Arm A trains 28..36/60..68/105..113 and Arm B trains only 32/64/109, so
# neither covers 40 or 72, and demanding it would refuse to build the very arms it exists to build.
_REQ_RAW = _env("REQUIRE_BANDS", default="")
REQUIRE_BANDS = _int_list(_REQ_RAW) if _REQ_RAW.strip() else []

if STEPS <= 0 or N_TR <= 0 or N_UN <= 0 or N_GAP <= 0:
    raise SystemExit(
        f"non-positive parameter: STEPS={STEPS} N_TR={N_TR} N_UN={N_UN} N_GAP={N_GAP}")
if not JITTER:
    raise SystemExit("JITTER is empty")

BANDS = {t: [t + j for j in JITTER] for t in TARGETS}
ALL_GEOM = sorted({d for b in BANDS.values() for d in b})
if len(ALL_GEOM) != len(TARGETS) * len(JITTER):
    raise SystemExit(
        f"BANDS_OVERLAP: {len(TARGETS)} targets x {len(JITTER)} jitter values should give "
        f"{len(TARGETS)*len(JITTER)} distinct geometries but the bands overlap into {len(ALL_GEOM)}. "
        f"Jitter {JITTER} is too wide for targets {TARGETS}.")
if set(GAPS) & set(ALL_GEOM):
    raise SystemExit(
        f"GAP_BAND_COLLISION: {sorted(set(GAPS) & set(ALL_GEOM))} are both gap probes and trained "
        f"geometries. A gap probe that collides with a trained distance reports a fake null, "
        f"because the rows were seen during training. Gaps={sorted(GAPS)} geometries={ALL_GEOM}")

# ---- derived exposure. Exact or refuse; never round. ----
PRESENT_PER_TARGET = STEPS                      # one slot per target per step
EPOCH_STEPS = len(JITTER) * N_TR                # steps to present every (geometry, row) once
if STEPS % len(JITTER) != 0:
    raise SystemExit(
        f"EXPOSURE_INEXACT: STEPS={STEPS} is not divisible by len(JITTER)={len(JITTER)}, so the "
        f"jitter bands cannot each receive the same number of presentations")
PASSES = STEPS // EPOCH_STEPS
if STEPS % EPOCH_STEPS != 0:
    raise SystemExit(
        f"EXPOSURE_INEXACT: STEPS={STEPS} % len(base)={EPOCH_STEPS} = {STEPS % EPOCH_STEPS}, so "
        f"the final epoch is partial and some (geometry, row) pairs get fewer passes than others. "
        f"Refusing to round. Choose STEPS as a multiple of {EPOCH_STEPS}.")
PRESENT_PER_GEOM = PRESENT_PER_TARGET // len(JITTER)
if PRESENT_PER_GEOM != PASSES * N_TR:
    raise SystemExit(
        f"EXPOSURE_INCONSISTENT: {PRESENT_PER_GEOM} presentations/geometry from "
        f"STEPS//len(JITTER) but {PASSES} passes x {N_TR} rows = {PASSES * N_TR} from the epoch "
        f"arithmetic")
if PASSES < 1:
    raise SystemExit(
        f"EXPOSURE_INEXACT: PASSES={PASSES}; STEPS={STEPS} is under one full epoch of "
        f"len(base)={EPOCH_STEPS}")

# ---- the named arm must actually be the arm that got built -----------------------------------
_diffs = []
if STEPS != _SPEC["steps"]:
    _diffs.append(f"steps {STEPS} != {_SPEC['steps']}")
if JITTER != _SPEC["jitter"]:
    _diffs.append(f"jitter {JITTER} != {_SPEC['jitter']}")
if N_TR != _SPEC["n_tr"]:
    _diffs.append(f"n_tr {N_TR} != {_SPEC['n_tr']}")
if sorted(GAPS) != sorted(_SPEC["gaps"]):
    _diffs.append(f"gaps {sorted(GAPS)} != {sorted(_SPEC['gaps'])}")
if PASSES != _SPEC["passes"]:
    _diffs.append(f"passes {PASSES} != {_SPEC['passes']}")
if len(ALL_GEOM) != _SPEC["geometries"]:
    _diffs.append(f"geometries {len(ALL_GEOM)} != {_SPEC['geometries']}")
if _diffs:
    # Z_BUILD_DEV lets a throwaway corpus exist for smoke tests. It is recorded as arm "DEV" in the
    # payload so it can never be mistaken for an approved arm, and it is not a way to build A or B
    # with the wrong numbers - those two names stay bound to the table above.
    if _env("BUILD_DEV", default="0") == "1":
        ARM = "DEV"
        print(f"DEV_BUILD_NOT_APPROVED resolved numbers differ from Z_ARM={_ARM_REQUESTED}: "
              f"{'; '.join(_diffs)}. Recorded as arm DEV.", flush=True)
    else:
        raise SystemExit(
            f"Z_ARM={ARM} was named but the resolved build is not that arm: " + "; ".join(_diffs) +
            ". A corpus that reports itself as an arm it is not is worse than a failed build.")
print(f"ARM_TABLE_OK arm={ARM} steps={STEPS} jitter={JITTER} n_tr={N_TR} passes={PASSES} "
      f"geometries={len(ALL_GEOM)} gaps={len(GAPS)}", flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def filler_for(distance):
    return distance - 8


def piece_len(F):
    return 4 + 8 + 1 + F + 6 + 8


def copies_for(F):
    return max(1, min(COPT_MAX, BODY // piece_len(F)))


def geometry(distance):
    F = filler_for(distance)
    c = copies_for(F)
    mem_end = 4 + 8 - 1
    qry_start = 4 + 8 + 1 + F + 6
    return {"distance": distance, "filler": F, "piece_len": piece_len(F),
            "copies_per_row": c, "row_bytes": c * piece_len(F),
            "needle_distance": qry_start - mem_end, "window": 128,
            "inside_window": (qry_start - mem_end) < 128}


def build_needle_body(rng, filler_len, copies):
    body = [rng.randrange(256) for _ in range(BODY)]
    mask = [0] * BODY
    payloads = []
    cursor = 0
    for _ in range(copies):
        payload = bytes(rng.randrange(1, 256) for _ in range(8))
        filler = bytes(rng.randrange(97, 123) for _ in range(filler_len))
        piece = MEMB + payload + b"\n" + filler + QRYB + payload
        body[cursor:cursor + len(piece)] = list(piece)
        at = cursor + len(piece) - 8
        for q in range(at, at + 8):
            mask[q] = 1
        payloads.append(payload)
        cursor += len(piece)
    return body, mask, payloads


B = rv.load_corpus()
bt = B["train_task"]
GEN_IDX = [i for i in range(B["train"].shape[0]) if int(bt[i]) in (0, 1, 2, 3)]
N32_IDX = [i for i in range(B["train"].shape[0]) if int(bt[i]) == 4]
print("corpus train", tuple(B["train"].shape), "task counts", torch.bincount(bt).tolist())
print("general", len(GEN_IDX), "corpus needle32", len(N32_IDX))
assert len(GEN_IDX) >= 500 and len(N32_IDX) >= 100

blob = bytes(int(v) & 0xFF for v in B["train"][:, 1:-1].reshape(-1).tolist())
vblob = bytes(int(v) & 0xFF for v in B["valid"][:, 1:-1].reshape(-1).tolist())

BANNED = set()


def probe_payloads():
    """Every payload in the shipped probe file, so nothing here can collide with a probe.

    FAIL CLOSED. The probe set is the leakage guard: a generated payload that collides with a
    probe payload silently invalidates the unseen/gap comparisons. Returning an empty set when the
    file is absent would leave the build looking clean while the guard was never applied, so a
    missing probe file is a hard error unless the caller explicitly opts out.
    """
    out = []
    p = os.environ.get("Z_PROBE_FILE") or next(
        (os.path.join(base, "probe_Q.pt") for base in (HERE, CODE)
         if os.path.exists(os.path.join(base, "probe_Q.pt"))), None)
    if p is None:
        if os.environ.get("Z_ALLOW_NO_PROBES", "").strip() in ("1", "true", "yes"):
            print("PROBE_GUARD SKIPPED (Z_ALLOW_NO_PROBES set) - leakage check has no probe set")
            return out
        raise SystemExit(
            "PROBE_SET_MISSING: no probe_Q.pt found in "
            f"{HERE} or {CODE}. The probe-collision guard cannot run, and building without it "
            "would produce a corpus whose leakage claims are unverified. Publish probe_Q.pt next "
            "to this script, point Z_PROBE_FILE at it, or set Z_ALLOW_NO_PROBES=1 to build anyway "
            "and accept that result.")
    if not os.path.exists(p):
        raise SystemExit(f"PROBE_SET_MISSING: Z_PROBE_FILE={p} does not exist")
    PQ = torch.load(p, map_location="cpu", weights_only=False)
    for _, v in PQ.items():
        if not (isinstance(v, dict) and "rows" in v):
            continue
        rr = v["rows"].long()
        for i in range(rr.shape[0]):
            bd = bytes(int(x) & 0xFF for x in rr[i, 1:511].tolist())
            j = 0
            while True:
                a = bd.find(MEMB, j)
                if a < 0:
                    break
                out.append(bd[a + 4:a + 12])
                j = a + 4
    return out


PP = probe_payloads()
BANNED |= set(PP)
print("probe payloads", len(PP), "from", os.environ.get("Z_PROBE_FILE", "probe_Q.pt"))

imported = []
for nm in ("train_T2.pt", "train_V.pt", "train_W.pt", "train_X.pt"):
    for base in (HERE, CODE):
        p = os.path.join(base, nm)
        if not os.path.exists(p):
            continue
        T = torch.load(p, map_location="cpu", weights_only=False)
        n = 0
        if nm in ("train_W.pt", "train_X.pt"):
            for _d, d in T["distances"].items():
                for k in ("payloads", "unseen_payloads"):
                    for h in d.get(k, []):
                        BANNED.add(bytes.fromhex(h)); n += 1
        else:
            for k in ("A_payloads", "B_payloads", "U_payloads"):
                for h in T.get(k, []):
                    BANNED.add(bytes.fromhex(h)); n += 1
        imported.append((nm, n))
        break
print("banned payloads imported:", imported, "| total banned", len(BANNED))
for nm in ("train_T2.pt", "train_V.pt", "train_W.pt", "train_X.pt"):
    if not any(i[0] == nm for i in imported):
        raise SystemExit(f"{nm} missing - refusing to build Y without full leakage history")

used = set()


def gen_rows(distance, count, seed, tag):
    F = filler_for(distance)
    rng = random.Random(seed)
    out, tries = [], 0
    nc = copies_for(F)
    while len(out) < count:
        tries += 1
        assert tries < count * 6000, f"leakage avoidance failed for {tag} distance {distance}"
        body, mask, pls = build_needle_body(rng, F, nc)
        if any(p in BANNED or p in used or p in blob or p in vblob for p in pls):
            continue
        assert sum(mask) == 8 * nc
        used.update(pls)
        out.append(([BOS, *body, EOS], mask, pls))
    return out, nc, tries


DATA, GAPDATA = {}, {}
for dist in ALL_GEOM:
    g = geometry(dist)
    assert g["needle_distance"] == dist, (dist, g)
    tr, nc, t_tr = gen_rows(dist, N_TR, 20261500 + dist, "train")
    un, ncu, t_un = gen_rows(dist, N_UN, 20261600 + dist, "unseen")
    assert nc == ncu
    DATA[dist] = {"train": tr, "unseen": un, "copies": nc,
                  "tries_train": t_tr, "tries_unseen": t_un, "geometry": g}
    print(f"D{dist} filler {g['filler']} copies {nc} train {len(tr)} unseen {len(un)} "
          f"(tries {t_tr}/{t_un})")

for dist in GAPS:
    g = geometry(dist)
    assert g["needle_distance"] == dist
    un, nc, t = gen_rows(dist, N_GAP, 20261700 + dist, "gap")
    GAPDATA[dist] = {"rows": un, "copies": nc, "geometry": g, "tries": t}
    print(f"GAP D{dist} filler {g['filler']} copies {nc} rows {len(un)} (tries {t})")

allp = {}
for dist in ALL_GEOM:
    allp[(dist, "train")] = [p for _, _, ps in DATA[dist]["train"] for p in ps]
    allp[(dist, "unseen")] = [p for _, _, ps in DATA[dist]["unseen"] for p in ps]
for dist in GAPS:
    allp[(dist, "gap")] = [p for _, _, ps in GAPDATA[dist]["rows"] for p in ps]
keys = sorted(allp)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        assert not (set(allp[keys[i]]) & set(allp[keys[j]])), f"payload overlap {keys[i]} {keys[j]}"
for k, v in allp.items():
    assert not (set(v) & BANNED), f"{k} collides with banned payloads"
    for p in v:
        assert p not in blob and p not in vblob, f"{k} payload found in corpus"
print("LEAKAGE_OK", len(keys), "sets pairwise disjoint, absent from probes/blobs/prior experiments")

# Band distances the caller REQUIRED this arm to cover. Empty by default; see REQUIRE_BANDS.
for d in REQUIRE_BANDS:
    assert d in ALL_GEOM, f"required band distance {d} is not in any jitter band"
if REQUIRE_BANDS:
    print(f"BAND_REQUIREMENTS_OK {sorted(REQUIRE_BANDS)} are inside training bands")
else:
    print("BAND_REQUIREMENTS none requested (REQUIRE_BANDS empty)")


def stack(rows):
    R = torch.tensor([r[0] for r in rows], dtype=torch.int16)
    M = torch.tensor([r[1] for r in rows], dtype=torch.int16)
    assert R.shape[1] == SEQ and M.shape[1] == BODY
    return R, M


g_rng = random.Random(int(_env("GEN_SEED", "Y_GEN_SEED", "1101")))
GEN_SEQ = [GEN_IDX[g_rng.randrange(len(GEN_IDX))] for _ in range(STEPS * 2)]
N32_SEQ = [N32_IDX[g_rng.randrange(len(N32_IDX))] for _ in range(STEPS)]
gen_ids = sorted(set(GEN_SEQ))
n32_ids = sorted(set(N32_SEQ))
gp = {o: p for p, o in enumerate(gen_ids)}
np_ = {o: p for p, o in enumerate(n32_ids)}


def make_plan(seed):
    """Balanced jitter plan.

    For each target, the (geometry, row) pairs are laid out as len(JITTER) jitter values x N_TR
    rows, repeated PASSES times and shuffled per epoch. Every (geometry, row) pair is therefore
    presented exactly PASSES times. All targets walk the SAME permutation so the mix stays balanced
    step by step, not only in total.
    """
    r = random.Random(seed)
    base = [(g, row) for g in range(len(JITTER)) for row in range(N_TR)]
    assert len(base) == EPOCH_STEPS
    assert STEPS % len(base) == 0
    slots = {t: [] for t in TARGETS}
    while len(slots[TARGETS[0]]) < STEPS:
        e = list(base)
        r.shuffle(e)
        for t in TARGETS:
            slots[t].extend(e)
    for t in TARGETS:
        slots[t] = slots[t][:STEPS]
    # identical jitter/row walk for every target, so the balance is step-wise
    for t in TARGETS[1:]:
        assert slots[t] == slots[TARGETS[0]], "targets must share one balanced walk"
    walk = slots[TARGETS[0]]
    plan = [{"slot0": gp[GEN_SEQ[2 * k]], "slot1": gp[GEN_SEQ[2 * k + 1]],
             "slot2": np_[N32_SEQ[k]],
             "geom": {t: BANDS[t][walk[k][0]] for t in TARGETS},
             "row": {t: walk[k][1] for t in TARGETS}}
            for k in range(STEPS)]
    cnt = Counter((walk[k][0], walk[k][1]) for k in range(STEPS))
    pres = {"min": min(cnt.values()), "max": max(cnt.values()),
            "mean": round(sum(cnt.values()) / len(cnt), 3)}
    geom_cnt = Counter(d for p in plan for d in p["geom"].values())
    return plan, pres, dict(sorted(geom_cnt.items()))


plan, pres, geom_cnt = make_plan(int(_env("PLAN_SEED", "Y_PLAN_SEED", "9201")))
expected_pres = {"min": PASSES, "max": PASSES, "mean": float(PASSES)}
assert pres == expected_pres, f"plan is not balanced: {pres} != {expected_pres}"
assert set(geom_cnt.values()) == {PRESENT_PER_GEOM}, geom_cnt
print("PLAN_OK per (geometry,row) pair", pres, "| presentations per geometry", geom_cnt)

payload = {
    "rows_gen": B["train"][torch.tensor(gen_ids)].clone(),
    "mask_gen": B["train_mask"][torch.tensor(gen_ids)].clone(),
    "gen_ids": gen_ids,
    "rows32": B["train"][torch.tensor(n32_ids)].clone(),
    "mask32": B["train_mask"][torch.tensor(n32_ids)].clone(),
    "n32_ids": n32_ids,
    "plan": plan,
    "targets": TARGETS,
    "jitter": JITTER,
    "bands": {str(t): BANDS[t] for t in TARGETS},
    "geometries": ALL_GEOM,
    "gaps": GAPS,
    "distances": {},
    "gap_probes": {},
    "meta": {"steps": STEPS, "targets": TARGETS, "jitter": JITTER, "bands_geometry": ALL_GEOM,
             "passes": pres, "rows_per_geometry": N_TR, "unseen_per_geometry": N_UN,
             "batch_rows": 2 + len(TARGETS), "presentations_per_geometry": PRESENT_PER_GEOM,
             "all_targets_share_walk": True, "gaps_never_trained": GAPS,
             "shim": APPLIED},
    # Self-describing build record. A corpus is a run artifact, so it states the parameters that
    # produced it instead of leaving an auditor to reconstruct them from a command line that may not
    # have been saved. Derived quantities are recorded next to the inputs they came from.
    "build": {
        "arm": ARM,
        "parameters": {
            "targets": TARGETS, "jitter": JITTER, "n_tr": N_TR, "steps": STEPS,
            "n_unseen": N_UN, "n_gap": N_GAP, "gaps": GAPS,
            "require_bands": REQUIRE_BANDS, "gaps": GAPS,
            "gap_count": len(GAPS),
            "seeds": {"gen": int(_env("GEN_SEED", "Y_GEN_SEED", "1101")),
                      "plan": int(_env("PLAN_SEED", "Y_PLAN_SEED", "9201"))},
        },
        "derived": {
            "passes": PASSES, "epoch_steps": EPOCH_STEPS,
            "presentations_per_target": PRESENT_PER_TARGET,
            "presentations_per_geometry": PRESENT_PER_GEOM,
            "geometry_count": len(ALL_GEOM),
            "target_count": len(TARGETS),
            "jitter_count": len(JITTER),
            "train_rows": len(ALL_GEOM) * N_TR,
            "exact_division": {"steps_mod_jitter": STEPS % len(JITTER),
                               "steps_mod_epoch": STEPS % EPOCH_STEPS},
        },
        "verification": {"plan_passes": pres, "presentations_per_geometry": geom_cnt,
                         "bands_overlap_free": True, "gaps_never_trained": True},
        "script": {"file": os.path.basename(os.path.abspath(__file__)),
                   "sha256": sha(os.path.abspath(__file__))},
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    },
    "authorized": True,
}
for dist in ALL_GEOM:
    RT, MT = stack(DATA[dist]["train"])
    RU, MU = stack(DATA[dist]["unseen"])
    assert int(MT.sum()) == N_TR * 8 * DATA[dist]["copies"]
    assert int(MU.sum()) == N_UN * 8 * DATA[dist]["copies"]
    payload["distances"][dist] = {
        "rows": RT, "mask": MT, "unseen": RU, "unseen_mask": MU,
        "payloads": [p.hex() for p in allp[(dist, "train")]],
        "unseen_payloads": [p.hex() for p in allp[(dist, "unseen")]],
        "copies": DATA[dist]["copies"], "geometry": DATA[dist]["geometry"]}
for dist in GAPS:
    RG, MG = stack(GAPDATA[dist]["rows"])
    assert int(MG.sum()) == N_GAP * 8 * GAPDATA[dist]["copies"]
    payload["gap_probes"][dist] = {
        "rows": RG, "mask": MG,
        "payloads": [p.hex() for p in allp[(dist, "gap")]],
        "copies": GAPDATA[dist]["copies"], "geometry": GAPDATA[dist]["geometry"]}

outp = os.path.join(OUT, _env("OUT_NAME", default="train_Z.pt"))
torch.save(payload, outp)
h = sha(outp)
print("BUILD_DONE", outp, h)
json.dump({os.path.basename(outp): h,
           "bands": {str(t): BANDS[t] for t in TARGETS},
           "geometries": {str(d): payload["distances"][d]["geometry"] for d in ALL_GEOM},
           "gap_probes": {str(d): payload["gap_probes"][d]["geometry"] for d in GAPS},
           "counts": {"rows_per_geometry": N_TR, "unseen_per_geometry": N_UN,
                      "gap_rows": N_GAP, "steps": STEPS},
           "presentations_per_geometry": geom_cnt,
           "build": payload["build"],
           "shim": APPLIED},
          open(os.path.join(OUT, _env("REPORT_NAME", default="z_build_report.json")), "w"),
          indent=1)
