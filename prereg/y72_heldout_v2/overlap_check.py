"""Independent overlap check of the new Y72 held-out set (seed level + content level). CPU only. Does not import the generator.
usage: overlap_check.py SET.pt OUT.json"""
import json, re, sys, os, hashlib
from pathlib import Path
import torch
S = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
R = S["rows"].long(); M = S["mask"].long()
BUNDLES = [Path("/workspace/vera/packaging-rootfix-20261106/inputs/vyra-y-bundle"), Path("/workspace/vera/packaging-rootfix-20261106/inputs/vyra-x-bundle")]
REHEARSAL = Path("/workspace/forge/vyra-y72fix-run-20261110/out/rehearsal_Y72_40.pt")
out = {"set_rows": int(R.shape[0]), "set_segments": int(R.shape[0]) * int(S["copies"]), "new_seeds": S["seeds"]}

# ---- seed level ------------------------------------------------------------------------------------------------------------------
lit = {}
roots = [Path("/workspace/sentinel/vyra-public"), Path("/workspace/forge/vyra-resume300-prep-20261110/repo"), *BUNDLES]
scan_ext = {".py", ".md", ".json", ".txt", ".sh", ".yml", ".yaml", ".ipynb"}
nfiles = 0
for root in roots:
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in (".git", "__pycache__", "node_modules")]
        for f in fn:
            p = Path(dp) / f
            if p.suffix in scan_ext and p.stat().st_size < 5_000_000:
                nfiles += 1
                try: t = p.read_text(errors="ignore")
                except Exception: continue
                for m in re.finditer(r"(?<!\d)(2026\d{4})(?!\d)", t):
                    lit.setdefault(int(m.group(1)), set()).add(str(p))
# the seeds implied by the original builder's policy base+distance (read from its source, then expanded)
src = Path("/workspace/sentinel/vyra-public/repo/experiments/y/make_corpus_Y.py").read_text()
bases = sorted({int(x) for x in re.findall(r"(?<!\d)(20261[5-7]00)(?!\d)", src)})
TARGETS = [32, 64, 109]; JIT = [-8, -4, 0, 4, 8]; GAPS = [44, 48, 52, 76, 80, 84, 88, 92, 96, 121, 128]
geoms = sorted({t + j for t in TARGETS for j in JIT})
implied = {b + d for b in bases for d in geoms + GAPS}
rh = json.loads(Path("/workspace/forge/vyra-y72fix-run-20261110/out/rehearsal_Y72_40_manifest.json").read_text())
implied.add(int(rh["added_rows_generator_seed"]))
used = set(lit) | implied
out["seed_level"] = {"files_scanned": nfiles, "distinct_2026xxxx_literals_found": len(lit), "implied_builder_seeds": len(implied), "builder_bases_found": bases,
                     "rehearsal_added_rows_seed": int(rh["added_rows_generator_seed"]), "all_used_seeds_count": len(used),
                     "new_seeds_in_used": sorted(set(S["seeds"]) & used), "new_seeds_pairwise_distinct": len(set(S["seeds"])) == len(S["seeds"]),
                     "new_bases_in_used_literals": sorted({b for b in (20263600, 20263700, 20263800) if b in lit}),
                     "pass": not (set(S["seeds"]) & used) and len(set(S["seeds"])) == len(S["seeds"])}

# ---- content level ---------------------------------------------------------------------------------------------------------------
def tens(obj, path=""):
    if isinstance(obj, torch.Tensor):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items(): yield from tens(v, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            if isinstance(v, torch.Tensor) or isinstance(v, (dict, list, tuple)): yield from tens(v, f"{path}[{i}]")
sources = []   # (label, rows tensor [n,512])
for b in BUNDLES:
    for f in sorted(b.glob("*.pt")):
        if f.name.startswith("vyra_") or f.name == "y72_heldout_v2.pt": continue   # weights
        try: o = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e: out.setdefault("load_failures", []).append(f"{f}: {e}"); continue
        for p, t in tens(o):
            if t.dim() == 2 and t.shape[1] == 512 and t.dtype in (torch.int16, torch.int32, torch.int64, torch.uint8):
                sources.append((f"{b.name}/{f.name}{p}", t.long()))
RH = torch.load(REHEARSAL, map_location="cpu", weights_only=False)
sources.append(("rehearsal_Y72_40.pt/rows", RH["rows"].long()))
# the 200 shipped held-out Y72 rows are inside train_Y.pt distances[72]/unseen (already scanned) - listed explicitly for the report
rowbytes = lambda r: bytes(int(x) & 0xFF for x in r[1:511].tolist())
def rowkey(r): return hashlib.sha256(r.to(torch.int64).numpy().tobytes()).digest()
new_keys = [rowkey(R[i]) for i in range(R.shape[0])]
new_set = set(new_keys)
MEMB, QRYB = b"MEM:", b"QUERY:"
def pieces(bd):
    j = 0; res = []
    while True:
        a = bd.find(MEMB, j)
        if a < 0: break
        nl = bd.find(b"\n", a + 12); q = bd.find(QRYB, nl + 1) if nl >= 0 else -1
        if nl == a + 12 and q > nl:
            res.append((bd[a + 4:a + 12], bd[nl + 1:q], bd[q + 6:q + 14]))
        j = a + 4
    return res
new_pieces = [pc for i in range(R.shape[0]) for pc in pieces(rowbytes(R[i]))]
new_pay = {pc[0] for pc in new_pieces}; new_fill = {pc[1] for pc in new_pieces}
assert len(new_pieces) == out["set_segments"] and len(new_pay) == out["set_segments"], (len(new_pieces), len(new_pay))
out["within_new_set"] = {"duplicate_rows": R.shape[0] - len(new_set), "needle_pieces": len(new_pieces), "distinct_payloads": len(new_pay), "distinct_fillers": len(new_fill),
                         "payload_equals_query_answer_all": all(pc[0] == pc[2] for pc in new_pieces)}
old_rows = 0; hit_rows = []; hit_pay = set(); hit_fill = set(); per_src = {}
W = 16
win_new = set()
for i in range(R.shape[0]):
    bd = rowbytes(R[i])
    for k in range(0, len(bd) - W + 1): win_new.add(bd[k:k + W])
win_hits = 0
for label, T in sources:
    n = T.shape[0]; hr = hp = hf = hw = 0
    for i in range(n):
        if rowkey(T[i]) in new_set: hr += 1; hit_rows.append((label, i))
        bd = rowbytes(T[i])
        for pc in pieces(bd):
            if pc[0] in new_pay: hp += 1; hit_pay.add(pc[0])
            if pc[1] in new_fill: hf += 1; hit_fill.add(pc[1])
        for k in range(0, len(bd) - W + 1):
            if bd[k:k + W] in win_new: hw += 1
    old_rows += n; win_hits += hw
    per_src[label] = {"rows": int(n), "exact_row_matches": hr, "payload_matches": hp, "filler_string_matches": hf, "16byte_window_matches": hw}
out["content_level"] = {"old_rows_scanned": old_rows, "sources_scanned": len(sources), "exact_row_matches_total": len(hit_rows), "payload_matches_total": len(hit_pay),
                        "filler_string_matches_total": len(hit_fill), "16byte_window_matches_total": win_hits, "per_source": per_src}
# byte-blob level: corpus train/valid raw bytes (the generator's own leakage check, repeated independently): no new payload occurs anywhere in them
import importlib, sys as _s
cp = torch.load(BUNDLES[0] / "corpus.pt", map_location="cpu", weights_only=False)
blobs = []
for k in ("train", "valid"):
    if k in cp: blobs.append(bytes(int(v) & 0xFF for v in cp[k][:, 1:-1].reshape(-1).tolist()))
out["content_level"]["corpus_blob_payload_occurrences"] = sum(sum(1 for p in new_pay if p in b) for b in blobs)
out["content_level"]["corpus_blob_keys"] = [k for k in ("train", "valid") if k in cp]
# the shipped 200-item Y72 held-out set specifically
SY = torch.load(BUNDLES[0] / "train_Y.pt", map_location="cpu", weights_only=False)
old72 = SY["distances"][72]["unseen"].long()
out["vs_current_200_heldout"] = {"rows": int(old72.shape[0]), "exact_row_matches": sum(rowkey(old72[i]) in new_set for i in range(old72.shape[0])),
                                 "payload_matches": len(new_pay & {bytes.fromhex(h) for h in SY["distances"][72]["unseen_payloads"]}),
                                 "payload_matches_vs_all_train_Y_hex_lists": len(new_pay & {bytes.fromhex(h) for e in SY["distances"].values() for k in ("payloads", "unseen_payloads") for h in e[k]} | new_pay & {bytes.fromhex(h) for e in SY["gap_probes"].values() for h in e["payloads"]})}
out["pass"] = bool(out["seed_level"]["pass"] and out["within_new_set"]["duplicate_rows"] == 0 and out["content_level"]["exact_row_matches_total"] == 0 and out["content_level"]["payload_matches_total"] == 0
                   and out["content_level"]["filler_string_matches_total"] == 0 and out["content_level"]["16byte_window_matches_total"] == 0 and out["content_level"]["corpus_blob_payload_occurrences"] == 0
                   and out["vs_current_200_heldout"]["exact_row_matches"] == 0 and out["vs_current_200_heldout"]["payload_matches"] == 0 and out["vs_current_200_heldout"]["payload_matches_vs_all_train_Y_hex_lists"] == 0)
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "content_level"}, indent=1))
print({k: v for k, v in out["content_level"].items() if k != "per_source"})
