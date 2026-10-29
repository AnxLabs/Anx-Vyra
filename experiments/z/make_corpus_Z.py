"""Experiment Z data builder.

Parameterised form of the Experiment Y builder. Y's published script is left
unchanged. This one does not bake a single arm: jitter, rows per geometry,
passes per row, and steps are inputs, and the presentations that follow from
them are derived.

    presentations_per_geometry = steps / len(jitter)
    passes_per_row             = presentations_per_geometry / rows
                               = steps / (len(jitter) * rows)

Each target has one needle slot per step, so the number of targets cancels:
with T targets and J jitter values there are G = T*J geometries, and
steps * T / G = steps / J. A combination whose passes/row is not an integer,
or is not the requested one, is refused with the step count that would make
it exact.

REQUIRE_BANDS (Z_REQUIRE_BANDS / --require-bands) is an explicit list of
distances that must sit inside a training band. It defaults to empty. Y's
40 and 72 coverage check is that experiment's purpose, not a property of
every build, and it is not inferred from len(jitter): a jitter set such as
[-4, 0, 4] has more than one value and still does not cover 40 or 72. When
the list is empty the check is skipped out loud. When it is not, a miss names
the missing distance.

The leakage checks are the same ones Y runs: banned payloads from train_T2 /
train_V / train_W / train_X, the corpus blob and valid-blob substring checks,
pairwise disjointness, and a hard stop when a prior-experiment file is
missing. probe_Q.pt is required. Y returned an empty probe list when that
file was absent, which dropped the ban list and still looked successful.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))

# Filled by main() before run_build(). Importing this module does not build.
TARGETS = []
JITTER = []
GAPS = []
BANDS = {}
ALL_GEOM = []
N_TR = 8
N_UN = 40
N_GAP = 20
STEPS = 200
REQUESTED_PASSES = 5
PRESENT_PER_GEOM = 0
LAB = "./lab"
OUT = "./z_data"
CODE = ""
APPLIED = None


def parse_int_list(text):
    return [int(x) for x in text.split(",") if x.strip()]


def derive_shape(targets, jitter, rows, steps, requested_passes, gaps=None):
    """Pure arithmetic of one build. No corpus, no torch.

    Raises SystemExit when the requested shape is not an integer number of
    passes, or when that number is not the one that was asked for. The message
    includes the step count that would realise the request.
    """
    gaps = list(gaps or [])
    targets = list(targets)
    jitter = list(jitter)
    if not targets or not jitter:
        raise SystemExit("targets and jitter must both be non-empty")
    if rows < 1 or steps < 1 or requested_passes < 1:
        raise SystemExit("rows, steps, and passes/row must be positive integers")
    bands = {t: [t + j for j in jitter] for t in targets}
    geometries = sorted({d for band in bands.values() for d in band})
    if len(geometries) != len(targets) * len(jitter):
        raise SystemExit(
            f"bands overlap; jitter too wide (targets={targets} jitter={jitter} "
            f"geometries={geometries})"
        )
    overlap = sorted(set(gaps) & set(geometries))
    if overlap:
        raise SystemExit(f"interpolation probes must not be trained distances: {overlap}")
    len_base = len(jitter) * rows  # (geometry, row) pairs walked by one target
    need = requested_passes * len_base
    if steps % len(jitter) != 0:
        raise SystemExit(
            f"impossible combination: {steps} steps does not divide evenly across "
            f"{len(jitter)} jitter values, so presentations/geometry is not an integer. "
            f"steps needed for {requested_passes} passes/row with {rows} rows: {need}"
        )
    presentations = steps // len(jitter)
    if steps % len_base != 0:
        raise SystemExit(
            f"impossible combination: {steps} steps does not divide {len_base} "
            f"(geometry, row) pairs evenly (jitter={len(jitter)}, rows={rows}), "
            f"so passes/row is not an integer. presentations/geometry would be "
            f"{presentations}. steps needed for {requested_passes} passes/row: {need}"
        )
    passes = steps // len_base
    if passes != requested_passes:
        raise SystemExit(
            f"impossible combination: derived {passes} passes/row "
            f"({presentations} presentations/geometry, {rows} rows/geometry), "
            f"requested {requested_passes}. "
            f"steps needed for the requested shape: {need}"
        )
    return {
        "targets": targets,
        "jitter": jitter,
        "gaps": gaps,
        "bands": bands,
        "geometries": geometries,
        "n_geometries": len(geometries),
        "rows": rows,
        "steps": steps,
        "requested_passes": requested_passes,
        "len_base": len_base,
        "passes_per_row": passes,
        "presentations_per_geometry": presentations,
        "steps_needed": need,
    }


def coverage_status(geometries, required):
    """Require each listed distance to be inside a band, or say the check is off.

    `required` empty is a skipped check, not a passed one. A miss names the
    distance that was asked for and was not covered.
    """
    required = list(required)
    if not required:
        return "BAND_COVERAGE skipped (REQUIRE_BANDS empty)"
    missing = [d for d in required if d not in set(geometries)]
    if missing:
        names = ", ".join(str(d) for d in missing)
        raise SystemExit(
            f"required band not covered: {names} "
            f"(requested {required}; geometries {list(geometries)})"
        )
    return "BAND_COVERAGE ok: " + ",".join(str(d) for d in required)


def _find_code():
    cands = [os.environ.get("T2_SRC", ""), "bundle",
             os.path.join(HERE, "src"), os.path.join(os.path.dirname(HERE), "src"), HERE,
             os.path.join(os.path.dirname(HERE), "y")]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "run_vyra1_copy.py")):
            return c
    raise SystemExit("cannot locate run_vyra1_copy.py; set T2_SRC")


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


def copies_for(F, body, copt_max):
    return max(1, min(copt_max, body // piece_len(F)))


def geometry(distance, body, copt_max):
    F = filler_for(distance)
    c = copies_for(F, body, copt_max)
    mem_end = 4 + 8 - 1
    qry_start = 4 + 8 + 1 + F + 6
    return {"distance": distance, "filler": F, "piece_len": piece_len(F),
            "copies_per_row": c, "row_bytes": c * piece_len(F),
            "needle_distance": qry_start - mem_end, "window": 128,
            "inside_window": (qry_start - mem_end) < 128}


def probe_payloads(probe_path=None):
    """Every payload in the shipped probe file.

    Missing probe_Q.pt is a hard failure. Returning an empty list here used to
    drop the probe ban silently.
    """
    p = probe_path or os.path.join(HERE, "probe_Q.pt")
    if not os.path.exists(p):
        raise SystemExit(
            f"probe_Q.pt missing at {p} - refusing to build without the probe ban list"
        )
    import torch
    out = []
    PQ = torch.load(p, map_location="cpu", weights_only=False)
    memb = b"MEM:"
    for _, v in PQ.items():
        if not (isinstance(v, dict) and "rows" in v):
            continue
        rr = v["rows"].long()
        for i in range(rr.shape[0]):
            bd = bytes(int(x) & 0xFF for x in rr[i, 1:511].tolist())
            j = 0
            while True:
                a = bd.find(memb, j)
                if a < 0:
                    break
                out.append(bd[a + 4:a + 12])
                j = a + 4
    return out


def apply_shape(shape):
    global TARGETS, JITTER, GAPS, BANDS, ALL_GEOM, N_TR, STEPS, REQUESTED_PASSES, PRESENT_PER_GEOM
    TARGETS = list(shape["targets"])
    JITTER = list(shape["jitter"])
    GAPS = list(shape["gaps"])
    BANDS = {t: list(shape["bands"][t]) for t in TARGETS}
    ALL_GEOM = list(shape["geometries"])
    N_TR = int(shape["rows"])
    STEPS = int(shape["steps"])
    REQUESTED_PASSES = int(shape["requested_passes"])
    PRESENT_PER_GEOM = int(shape["presentations_per_geometry"])
    assert len(ALL_GEOM) == len(TARGETS) * len(JITTER), "bands overlap; jitter too wide"
    assert PRESENT_PER_GEOM // N_TR == REQUESTED_PASSES


def run_build():
    import torch

    global CODE, APPLIED, LAB, OUT
    CODE = _find_code()
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "y"))
    sys.path.insert(0, CODE)
    import vyra_shim
    import run_vyra1_copy as rv

    LAB = os.environ.get("VYRA_LAB", "./lab")
    OUT = os.environ.get("Z_OUT", os.environ.get("T2_OUT", "./z_data"))
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(LAB, exist_ok=True)
    APPLIED = vyra_shim.patch_library(rv, LAB)

    BOS, EOS, BODY, SEQ = 256, 257, 510, 512
    MEMB, QRYB = b"MEM:", b"QUERY:"
    COPT_MAX = 8

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
    PP = probe_payloads()
    BANNED |= set(PP)
    print("probe payloads", len(PP))

    imported = []
    for nm in ("train_T2.pt", "train_V.pt", "train_W.pt", "train_X.pt"):
        for base in (HERE, CODE, os.path.join(os.path.dirname(HERE), "y")):
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
            raise SystemExit(f"{nm} missing - refusing to build Z without full leakage history")

    used = set()

    def gen_rows(distance, count, seed, tag):
        F = filler_for(distance)
        rng = random.Random(seed)
        out, tries = [], 0
        nc = copies_for(F, BODY, COPT_MAX)
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
        g = geometry(dist, BODY, COPT_MAX)
        assert g["needle_distance"] == dist, (dist, g)
        tr, nc, t_tr = gen_rows(dist, N_TR, 20261500 + dist, "train")
        un, ncu, t_un = gen_rows(dist, N_UN, 20261600 + dist, "unseen")
        assert nc == ncu
        DATA[dist] = {"train": tr, "unseen": un, "copies": nc,
                      "tries_train": t_tr, "tries_unseen": t_un, "geometry": g}
        print(f"D{dist} filler {g['filler']} copies {nc} train {len(tr)} unseen {len(un)} "
              f"(tries {t_tr}/{t_un})")

    for dist in GAPS:
        g = geometry(dist, BODY, COPT_MAX)
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

    def stack(rows):
        R = torch.tensor([r[0] for r in rows], dtype=torch.int16)
        M = torch.tensor([r[1] for r in rows], dtype=torch.int16)
        assert R.shape[1] == SEQ and M.shape[1] == BODY
        return R, M

    g_rng = random.Random(int(os.environ.get("Z_GEN_SEED", os.environ.get("Y_GEN_SEED", "1101"))))
    GEN_SEQ = [GEN_IDX[g_rng.randrange(len(GEN_IDX))] for _ in range(STEPS * 2)]
    N32_SEQ = [N32_IDX[g_rng.randrange(len(N32_IDX))] for _ in range(STEPS)]
    gen_ids = sorted(set(GEN_SEQ))
    n32_ids = sorted(set(N32_SEQ))
    gp = {o: p for p, o in enumerate(gen_ids)}
    np_ = {o: p for p, o in enumerate(n32_ids)}

    def make_plan(seed):
        r = random.Random(seed)
        base = [(g, row) for g in range(len(JITTER)) for row in range(N_TR)]
        # len(base) is the (geometry, row) pairs of one target. It equals
        # presentations_per_geometry only for Y's particular numbers. What has
        # to hold for any parameterisation is exact division into the requested
        # pass count.
        passes = STEPS // len(base)
        assert STEPS % len(base) == 0, (
            f"{STEPS} steps does not divide {len(base)} (geometry,row) pairs evenly"
        )
        assert passes == REQUESTED_PASSES, (
            f"derived {passes} passes/row, requested {REQUESTED_PASSES}"
        )
        slots = {t: [] for t in TARGETS}
        while len(slots[TARGETS[0]]) < STEPS:
            e = list(base)
            r.shuffle(e)
            for t in TARGETS:
                slots[t].extend(e)
        for t in TARGETS:
            slots[t] = slots[t][:STEPS]
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
        assert pres == {"min": passes, "max": passes, "mean": float(passes)}, pres
        assert set(geom_cnt.values()) == {PRESENT_PER_GEOM}, geom_cnt
        return plan, pres, dict(sorted(geom_cnt.items()))

    plan, pres, geom_cnt = make_plan(int(os.environ.get("Z_PLAN_SEED", os.environ.get("Y_PLAN_SEED", "9201"))))
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
                 "requested_passes": REQUESTED_PASSES,
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

    outp = os.path.join(OUT, "train_Z.pt")
    torch.save(payload, outp)
    h = sha(outp)
    print("BUILD_DONE", outp, h)
    json.dump({"train_Z.pt": h,
               "bands": {str(t): BANDS[t] for t in TARGETS},
               "geometries": {str(d): payload["distances"][d]["geometry"] for d in ALL_GEOM},
               "gap_probes": {str(d): payload["gap_probes"][d]["geometry"] for d in GAPS},
               "counts": {"rows_per_geometry": N_TR, "unseen_per_geometry": N_UN,
                          "gap_rows": N_GAP, "steps": STEPS,
                          "requested_passes": REQUESTED_PASSES},
               "presentations_per_geometry": geom_cnt,
               "shim": APPLIED}, open(os.path.join(OUT, "z_build_report.json"), "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build a Z corpus from explicit parameters.")
    ap.add_argument("--targets", default=os.environ.get("Z_TARGETS", "32,64,109"))
    ap.add_argument("--jitter", default=os.environ.get("Z_JITTER", "-8,-4,0,4,8"))
    ap.add_argument("--gaps", default=os.environ.get("Z_GAPS", "44,48,52,76,80,84,88,92,96,121,128"))
    ap.add_argument("--rows", type=int, default=int(os.environ.get("Z_ROWS", "8")))
    ap.add_argument("--passes", type=int, default=int(os.environ.get("Z_PASSES", "5")))
    ap.add_argument("--steps", type=int, default=int(os.environ.get("Z_STEPS", "200")))
    # Empty by default. A 40,72 default would make Y's targeted fix the silent
    # shape of every build that forgot the variable.
    ap.add_argument("--require-bands", default=os.environ.get("Z_REQUIRE_BANDS", ""))
    args = ap.parse_args(argv)
    targets = parse_int_list(args.targets)
    jitter = parse_int_list(args.jitter)
    gaps = parse_int_list(args.gaps)
    required = parse_int_list(args.require_bands)
    shape = derive_shape(targets, jitter, args.rows, args.steps, args.passes, gaps)
    print(
        f"DERIVED geometries={shape['n_geometries']} "
        f"presentations_per_geometry={shape['presentations_per_geometry']} "
        f"passes_per_row={shape['passes_per_row']} "
        f"len_base={shape['len_base']} steps={shape['steps']} "
        f"rows={shape['rows']} jitter={shape['jitter']}"
    )
    print(coverage_status(shape["geometries"], required))
    if shape["passes_per_row"] != args.passes:
        raise SystemExit(
            f"derived passes/row {shape['passes_per_row']} != requested {args.passes}"
        )
    apply_shape(shape)
    run_build()


if __name__ == "__main__":
    main()
