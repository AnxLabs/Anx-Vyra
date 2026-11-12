"""Independent disjointness check of the fresh pool F200 vs the gated 600-set and every other Y72 set (does not import the generator)."""
import json, os, re, sys, torch
from pathlib import Path
sys.path.insert(0, "."); import lab3
P = torch.load("../pool/y72_fresh_pool_F200.pt", map_location="cpu", weights_only=False)
def rb(r): return bytes(int(v) & 0xFF for v in r[1:511].tolist())
F = [rb(r) for r in P["rows"].long()]
sets = {k: [rb(a.long()) for a, _ in lab3.SETS[k]] for k in ("Y600", "Y200", "REP40")}
sets["Y72_train8"] = [rb(r.long()) for r in lab3.Y72["rows"]]
S6 = json.loads(Path(lab3.SET600).read_text())
out = {"pool_rows": len(F), "pool_payloads": len(P["payloads"]), "pool_payloads_distinct": len(set(P["payloads"])),
       "payload_overlap": {"Y600": len(set(P["payloads"]) & set(S6["payloads"])), "Y200_unseen_payloads": len(set(P["payloads"]) & set(lab3.Y72["unseen_payloads"])),
                           "Y72_train_payloads": len(set(P["payloads"]) & set(lab3.Y72["payloads"])), "REP40": len(set(P["payloads"]) & set(lab3.RY["payloads"]) if "payloads" in lab3.RY else -1)}}
REP = torch.load(lab3.REHEARSAL, map_location="cpu", weights_only=False); out["payload_overlap"]["REP40"] = len(set(P["payloads"]) & set(REP["payloads"]))
out["exact_row_matches"] = {k: sum(f in set(v) for f in F) for k, v in sets.items()}
def windows(rows, w=16): s = set(); [s.update(r[i:i + w] for i in range(len(r) - w + 1)) for r in rows]; return s
for w in (16, 32):
    fw = windows(F, w); out[f"shared_{w}byte_windows"] = {k: len(fw & windows(v, w)) for k, v in sets.items()}
    out[f"pool_{w}byte_windows_total"] = len(fw)
# payload-bearing windows: 16-byte windows that contain a pool payload, matched against the 600-set
pay = [bytes.fromhex(h) for h in P["payloads"]]
s6w = windows(sets["Y600"], 8); out["pool_payload_bytes_found_anywhere_in_600set_rows"] = sum(p in s6w for p in pay)
lits = set(); n = 0
for root in ("/workspace/sentinel/vyra-public", "/workspace/forge/vyra-rehearsal2-design-20261112/repo"):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in (".git", "__pycache__", "node_modules")]
        for f in fn:
            p = Path(dp) / f
            if p.suffix in {".py", ".md", ".json", ".txt"} and p.stat().st_size < 5_000_000:
                n += 1; lits |= {int(m) for m in re.findall(r"(?<!\d)(2026\d{4})(?!\d)", p.read_text(errors="ignore"))}
out["seed_check"] = {"files_scanned": n, "pool_seeds": P["seeds"], "pool_seeds_found_as_literals_elsewhere": sorted(set(P["seeds"]) & lits),
                     "known_y72_seeds": [20261672, 20262672, 20263672, 20263772, 20263872]}
json.dump(out, open("overlap_pool.json", "w"), indent=1); print(json.dumps(out, indent=1))
