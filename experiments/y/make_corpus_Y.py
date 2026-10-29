"""Experiment Y data builder.

TARGETED FIX for the one regression Experiment X left behind.

X trained three exact distances (32, 64, 109) and matched unseen accuracy on all three, but the
full distance profile showed the model is sharp at the offsets it was trained on: the two probes
that fell were at distance 40 and distance 72, each exactly +8 past a trained distance, while every
trained distance and general validation held flat.

Hypothesis: the collapse is not forgetting and not rehearsal dilution, but the model keying on an
exact needle-to-query offset. If so, presenting a BAND of distances around each target instead of a
single point should repair the neighbours without costing the targets.

Design, changing exactly one variable against X:
  * same start point      - X's step-200 checkpoint (so the damage being repaired is present)
  * same learning rate    - 3e-4, verified by first_update_max_abs
  * same 200 steps        - batch 5, three needle slots, two general replay rows
  * same needle exposure  - 600 presentations over 200 steps, as X
  * ONLY change           - each needle slot draws its filler per step from a jitter band
                            target +/- 8 tokens in steps of 4, i.e. five geometries per target

Bands (needle-to-query distance):
  target  32 -> 24, 28, 32, 36, 40        (filler 16, 20, 24, 28, 32)
  target  64 -> 56, 60, 64, 68, 72        (filler 48, 52, 56, 60, 64)
  target 109 -> 101, 105, 109, 113, 117   (filler 93, 97, 101, 105, 109)

The two distances that collapsed in X (40 and 72) sit exactly on the outer edge of the first two
bands, so they are covered by construction.

Rows: 8 per geometry, 15 geometries = 120 training rows. Each geometry gets 40 presentations over
200 steps = 5 passes per row, matching X's per-row exposure.

INTERPOLATION CHECK - this is what stops the experiment from being self-congratulatory.
Training bands cannot be used to prove the model became smoother, because those distances are
trained. So a separate probe set is built at distances Y NEVER trains:
  44, 48, 52, 76, 80, 84, 88, 92, 96, 121, 128
If those hold up too, the fix generalises the rule. If only the band edges recover, the model has
simply moved its sharp points, and that must be reported as a partial or negative result.
"""
import hashlib, json, os, random, sys
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
OUT = os.environ.get("T2_OUT", "./y_data")
os.makedirs(OUT, exist_ok=True)
os.makedirs(LAB, exist_ok=True)
APPLIED = vyra_shim.patch_library(rv, LAB)

BOS, EOS, BODY, SEQ = 256, 257, 510, 512
MEMB, QRYB = b"MEM:", b"QUERY:"
COPT_MAX = 8
STEPS = 200
N_TR = 8          # rows per geometry
N_UN = 40         # unseen rows per geometry
N_GAP = 20        # rows per interpolation probe distance

TARGETS = [int(x) for x in os.environ.get("Y_TARGETS", "32,64,109").split(",") if x.strip()]
JITTER = [int(x) for x in os.environ.get("Y_JITTER", "-8,-4,0,4,8").split(",") if x.strip()]
GAPS = [int(x) for x in
        os.environ.get("Y_GAPS", "44,48,52,76,80,84,88,92,96,121,128").split(",") if x.strip()]

BANDS = {t: [t + j for j in JITTER] for t in TARGETS}
ALL_GEOM = sorted({d for b in BANDS.values() for d in b})
assert len(ALL_GEOM) == len(TARGETS) * len(JITTER), "bands overlap; jitter too wide"
assert not (set(GAPS) & set(ALL_GEOM)), "interpolation probes must not be trained distances"
PRESENT_PER_TARGET = STEPS                     # one slot per target per step
PRESENT_PER_GEOM = PRESENT_PER_TARGET // len(JITTER)
assert PRESENT_PER_GEOM // N_TR == 5, f"per-row passes {PRESENT_PER_GEOM // N_TR}"


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
    """Every payload in the shipped probe file, so nothing here can collide with a probe."""
    out = []
    p = os.path.join(HERE, "probe_Q.pt")
    if not os.path.exists(p):
        return out
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
print("probe payloads", len(PP))

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

# The two distances that collapsed in X must actually be covered by a band.
for d in (40, 72):
    assert d in ALL_GEOM, f"targeted-fix distance {d} is not in any jitter band"
print("BAND_COVERS_REGRESSION 40 and 72 are inside training bands")


def stack(rows):
    R = torch.tensor([r[0] for r in rows], dtype=torch.int16)
    M = torch.tensor([r[1] for r in rows], dtype=torch.int16)
    assert R.shape[1] == SEQ and M.shape[1] == BODY
    return R, M


g_rng = random.Random(int(os.environ.get("Y_GEN_SEED", "1101")))
GEN_SEQ = [GEN_IDX[g_rng.randrange(len(GEN_IDX))] for _ in range(STEPS * 2)]
N32_SEQ = [N32_IDX[g_rng.randrange(len(N32_IDX))] for _ in range(STEPS)]
gen_ids = sorted(set(GEN_SEQ))
n32_ids = sorted(set(N32_SEQ))
gp = {o: p for p, o in enumerate(gen_ids)}
np_ = {o: p for p, o in enumerate(n32_ids)}


def make_plan(seed):
    """Balanced jitter plan.

    For each target, the (geometry, row) pairs are laid out as 5 jitter values x 8 rows, repeated
    five times and shuffled per epoch. Every (geometry, row) pair is therefore presented exactly
    five times, which is X's per-row exposure exactly. All three targets walk the SAME permutation
    so the mix stays balanced step by step, not only in total.
    """
    r = random.Random(seed)
    base = [(g, row) for g in range(len(JITTER)) for row in range(N_TR)]
    assert len(base) == PRESENT_PER_GEOM
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


plan, pres, geom_cnt = make_plan(int(os.environ.get("Y_PLAN_SEED", "9201")))
assert pres == {"min": 5, "max": 5, "mean": 5.0}, pres
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

outp = os.path.join(OUT, "train_Y.pt")
torch.save(payload, outp)
h = sha(outp)
print("BUILD_DONE", outp, h)
json.dump({"train_Y.pt": h,
           "bands": {str(t): BANDS[t] for t in TARGETS},
           "geometries": {str(d): payload["distances"][d]["geometry"] for d in ALL_GEOM},
           "gap_probes": {str(d): payload["gap_probes"][d]["geometry"] for d in GAPS},
           "counts": {"rows_per_geometry": N_TR, "unseen_per_geometry": N_UN,
                      "gap_rows": N_GAP, "steps": STEPS},
           "presentations_per_geometry": geom_cnt,
           "shim": APPLIED}, open(os.path.join(OUT, "y_build_report.json"), "w"), indent=1)
