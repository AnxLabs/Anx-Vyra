"""[rehearsal2 fresh pool F200: derived verbatim from prereg/y72_heldout_v2/gen_y72_heldout.py; changes: 5 new seed bases 20264600..20265000, 600-set payloads added to the ban set, output names] Generate the larger Y72 held-out set with the EXISTING Experiment-Y generator code and seed policy. CPU only; no model, no scoring.

* Generator code: the functions filler_for / piece_len / copies_for / build_needle_body / gen_rows and their constants are taken VERBATIM
  (by AST, then exec) from experiments/y/make_corpus_Y.py (path and sha256 are recorded; nothing is re-implemented).
* Seed policy (make_corpus_Y.py): seed = base + distance, one base per role: train 20261500, unseen 20261600, gap 20261700 (+ the rehearsal rows' 20262672 = 20261600+72+1000).
  New roles use bases 20263600 / 20263700 / 20263800, i.e. seeds 20263672, 20263772, 20263872 for distance 72, 40 rows each (the same gen_rows(72, 40, seed) call the
  original makes for the 40 unseen rows), 3 blocks -> 120 rows x 5 segments = 600 segments.
* Leakage policy (same as the original): BANNED = probe_Q payloads + all payloads of train_T2/V/W/X, plus the corpus train/valid byte blobs, plus a running `used`
  set; EXTENDED here with every payload of train_Y.pt (all geometries train+unseen, gap probes) and of the 40 rehearsal rows.
usage: gen_y72_heldout.py OUT_DIR
"""
import ast, hashlib, json, os, random, sys
from pathlib import Path
import torch
GEN_SRC = Path("/workspace/sentinel/vyra-public/repo/experiments/y/make_corpus_Y.py")
BUNDLE = Path(os.environ["VYRA_Y_BUNDLE"])   # local Experiment-Y bundle directory (tracked files carry no bundle names)
REHEARSAL = Path("/workspace/forge/vyra-y72fix2-cont400-run-20261112/out/rehearsal_Y72_40.pt")
PKG = "/workspace/forge/vyra-y72fix2-cont400-prep-20261112/repo/tools/experiment_packaging"
DIST = 72
N_ROWS = 40
BASES = {f"fresh_pool_block{i}": 20264600 + 100 * i for i in range(5)}
SET600 = Path("/workspace/forge/vyra-rehearsal2-design-20261112/repo/prereg/y72_heldout_v2/y72_heldout_v2_set.json")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BUNDLE)); sys.path.insert(0, PKG)
os.environ.setdefault("VYRA_LAB", "/tmp/y72heldout_lab")
import vyra_shim
import run_vyra1_copy as rv
os.makedirs(os.environ["VYRA_LAB"], exist_ok=True)
vyra_shim.patch_library(rv, os.environ["VYRA_LAB"])

src = GEN_SRC.read_text(); tree = ast.parse(src)
want_fn = {"filler_for", "piece_len", "copies_for", "build_needle_body", "gen_rows"}
want_const = {"BOS", "EOS", "BODY", "SEQ", "MEMB", "QRYB", "COPT_MAX"}
parts = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in want_fn:
        parts.append(ast.get_source_segment(src, node))
    elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in want_const for t in node.targets):
        parts.append(ast.get_source_segment(src, node))
    elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple) and {getattr(e, "id", None) for e in node.targets[0].elts} & want_const:
        parts.append(ast.get_source_segment(src, node))
extracted = "\n".join(parts)
ns = {"random": random}
# BOS, EOS, BODY, SEQ = ... is a tuple assignment; COPT_MAX / MEMB,QRYB too
exec(extracted, ns)
assert want_fn <= set(ns) and want_const <= set(ns), (want_fn | want_const) - set(ns)

B = rv.load_corpus()
blob = bytes(int(v) & 0xFF for v in B["train"][:, 1:-1].reshape(-1).tolist())
vblob = bytes(int(v) & 0xFF for v in B["valid"][:, 1:-1].reshape(-1).tolist())
MEMB = ns["MEMB"]

def row_payloads(rows):
    out = []
    for i in range(rows.shape[0]):
        bd = bytes(int(x) & 0xFF for x in rows[i, 1:511].tolist()); j = 0
        while True:
            a = bd.find(MEMB, j)
            if a < 0: break
            out.append(bd[a + 4:a + 12]); j = a + 4
    return out

BANNED = set(); prior_rows = {}; ban_src = {}
PQ = torch.load(BUNDLE / "probe_Q.pt", map_location="cpu", weights_only=False)
n = 0
for _, v in PQ.items():
    if isinstance(v, dict) and "rows" in v:
        ps = row_payloads(v["rows"].long()); BANNED |= set(ps); n += len(ps)
ban_src["probe_Q"] = n
for nm in ("train_T2.pt", "train_V.pt", "train_W.pt", "train_X.pt"):
    T = torch.load(BUNDLE / nm, map_location="cpu", weights_only=False); n = 0
    if nm in ("train_W.pt", "train_X.pt"):
        for _d, d in T["distances"].items():
            for k in ("payloads", "unseen_payloads"):
                for h in d.get(k, []): BANNED.add(bytes.fromhex(h)); n += 1
    else:
        for k in ("A_payloads", "B_payloads", "U_payloads"):
            for h in T.get(k, []): BANNED.add(bytes.fromhex(h)); n += 1
    ban_src[nm] = n
SY = torch.load(BUNDLE / "train_Y.pt", map_location="cpu", weights_only=False); n = 0
for d, e in SY["distances"].items():
    for k in ("payloads", "unseen_payloads"):
        for h in e[k]: BANNED.add(bytes.fromhex(h)); n += 1
for d, e in SY["gap_probes"].items():
    for h in e["payloads"]: BANNED.add(bytes.fromhex(h)); n += 1
ban_src["train_Y.pt(all geometries + gaps)"] = n
RH = torch.load(REHEARSAL, map_location="cpu", weights_only=False)
rh_rows = RH["rows"].long() if isinstance(RH, dict) else None
rp = row_payloads(rh_rows); BANNED |= set(rp); ban_src["rehearsal_Y72_40 (decoded from rows)"] = len(rp)
S6 = json.loads(SET600.read_text()); p6 = [bytes.fromhex(h) for h in S6["payloads"]]; assert len(p6) == 600; BANNED |= set(p6); ban_src["y72_heldout_v2_600set"] = len(p6)
print("banned payloads", len(BANNED), ban_src, flush=True)
ns.update({"BANNED": BANNED, "used": set(), "blob": blob, "vblob": vblob})

# 1) faithfulness check: the verbatim extracted gen_rows reproduces the shipped unseen D72 rows exactly (seed 20261600+72)
ref_ns_used = set()
ns["used"] = ref_ns_used
saved_banned = set(BANNED)
# the original's BANNED does not contain train_Y itself; remove train_Y/rehearsal payloads for the reproduction only
orig_banned = set()
for nm, T in (("train_X.pt", torch.load(BUNDLE / "train_X.pt", map_location="cpu", weights_only=False)),):
    pass
ns["BANNED"] = set(saved_banned)
for e in SY["distances"].values():
    for k in ("payloads", "unseen_payloads"):
        for h in e[k]: ns["BANNED"].discard(bytes.fromhex(h))
for e in SY["gap_probes"].values():
    for h in e["payloads"]: ns["BANNED"].discard(bytes.fromhex(h))
for p in rp: ns["BANNED"].discard(p)
for p in p6: ns["BANNED"].discard(p)
# (a payload that is both in an old-experiment ban set and train_Y cannot exist: the original asserted Y does not reuse earlier payloads)
rep, nc, tries = ns["gen_rows"](DIST, N_ROWS, 20261600 + DIST, "repro")
ref = SY["distances"][DIST]["unseen"].long(); refm = SY["distances"][DIST]["unseen_mask"].long()
got = torch.tensor([r[0] for r in rep]); gotm = torch.tensor([r[1] for r in rep])
repro = {"rows_equal": bool(torch.equal(got, ref)), "mask_equal": bool(torch.equal(gotm, refm)), "copies": nc, "tries": tries,
         "seed": 20261600 + DIST, "payloads_equal": [p.hex() for _, _, ps in rep for p in ps] == SY["distances"][DIST]["unseen_payloads"]}
print("REPRO of shipped D72 unseen rows:", repro, flush=True)
assert repro["rows_equal"] and repro["mask_equal"] and repro["payloads_equal"], "generator extraction does not reproduce shipped rows"

# 2) the new blocks with the extended ban set
ns["BANNED"] = saved_banned; ns["used"] = set()
blocks = []
for tag, base in BASES.items():
    seed = base + DIST
    rows, nc, tries = ns["gen_rows"](DIST, N_ROWS, seed, tag)
    blocks.append({"tag": tag, "seed": seed, "base": base, "rows": rows, "copies": nc, "tries": tries})
    print(tag, "seed", seed, "rows", len(rows), "copies", nc, "tries", tries, flush=True)
R = torch.tensor([r[0] for b in blocks for r in b["rows"]], dtype=torch.int16)
M = torch.tensor([r[1] for b in blocks for r in b["rows"]], dtype=torch.int16)
P = [p.hex() for b in blocks for _, _, ps in b["rows"] for p in ps]
assert R.shape == (200, 512) and M.shape == (200, 510) and len(P) == 1000 and len(set(P)) == 1000
assert int(M.sum()) == 200 * 8 * blocks[0]["copies"]
assert not (set(bytes.fromhex(h) for h in P) & saved_banned), "pool payload collides with a banned payload"
geom = SY["distances"][DIST]["geometry"]
payload = {"schema": "vyra-y72-rehearsal2-fresh-pool-F200", "rows": R, "mask": M, "payloads": P, "copies": blocks[0]["copies"], "geometry": geom,
           "block_of_row": [i // N_ROWS for i in range(200)], "seeds": [b["seed"] for b in blocks], "distance": DIST,
           "generator_source": str(GEN_SRC), "generator_source_sha256": hashlib.sha256(src.encode()).hexdigest()}
pt = OUT / "y72_fresh_pool_F200.pt"; torch.save(payload, pt)
pt2 = OUT / "y72_fresh_pool_F200.again.pt"; torch.save(payload, pt2)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
canon = hashlib.sha256(R.numpy().astype("<i2").tobytes() + M.numpy().astype("<i2").tobytes()).hexdigest()
blk = [hashlib.sha256(R[i*40:(i+1)*40].numpy().astype("<i2").tobytes() + M[i*40:(i+1)*40].numpy().astype("<i2").tobytes()).hexdigest() for i in range(5)]
pay_sha = hashlib.sha256("".join(P).encode()).hexdigest()
info = {"count_rows": 200, "count_segments": 1000, "tokens_total": 1000 * 8, "copies_per_row": blocks[0]["copies"], "seeds": payload["seeds"], "bases": BASES,
        "tries": [b["tries"] for b in blocks], "pt_sha256": sha(pt), "pt_bytes": pt.stat().st_size, "pt_save_deterministic": sha(pt) == sha(pt2),
        "canonical_rows_mask_sha256": canon, "block_canonical_sha256": blk, "payload_list_sha256": pay_sha,
        "generator_source": str(GEN_SRC), "generator_source_sha256": payload["generator_source_sha256"], "generator_git_head": "c38958cb2d4ed049725b99cce0ed5bb2ceae6a43",
        "banned_sources_counts": ban_src, "banned_total": len(saved_banned), "reproduction_check": repro, "torch": torch.__version__}
pt2.unlink()
json.dump(info, open(OUT / "fresh_pool_gen_info.json", "w"), indent=1)
print(json.dumps(info, indent=1))
