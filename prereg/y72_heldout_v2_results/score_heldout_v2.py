"""Score existing checkpoints on the pre-registered 600-segment Y72 held-out set (CPU only, read-only on checkpoints).
Fail closed: checkpoint file sha256 == pinned value (taken from the earlier runs' ARTIFACTS files) BEFORE torch.load; set file/canonical sha256 verified first.
Also re-scores the legacy 200-segment set in the same process and asserts it equals the earlier study's counts."""
import os, sys, json, time, hashlib
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "8"
import torch
torch.set_num_threads(8)
import torch.nn.functional as F
REPO = Path("/workspace/forge/vyra-resume300-prep-20261110/repo")
sys.path.insert(0, str(REPO / "tools" / "experiment_packaging"))
import y72_heldout_v2 as g
X = Path("/workspace/vera/packaging-rootfix-20261106/inputs/vyra-x-bundle"); Yb = Path("/workspace/vera/packaging-rootfix-20261106/inputs/vyra-y-bundle")
R25 = Path("/workspace/forge/vyra-resume300-run-20261110/out/replay_out"); R05 = Path("/workspace/forge/vyra-y72fix-run-20261110/out/replay_out")
CK200 = Path("/workspace/vera/expanded40-200u-run-20261107/completed/replay_out/checkpoint_200.pt")
OUTJ = Path("/workspace/forge/vyra-y72heldout-20261111/score/y72_v2_scores.json")
PINS = {  # pinned from the earlier runs' ARTIFACTS.sha256 files (re-checked below against those files)
 "l025_200": ("8542c3405309a5fcf9e419a6e3dd733beab346765fa7bd9cbdd61226dd867e81", CK200),
 "l025_225": ("a84d35bee3323cf60d0f0a09f0eece5ffb4c1e7bddbd196d4c478df15cb1b283", R25 / "checkpoint_225.pt"),
 "l025_250": ("7756a0760cb18dc9b6bcc80e5a77d7818b5a1d33fce16cf64b904a2a39de9163", R25 / "checkpoint_250.pt"),
 "l025_275": ("917f334fe8a519450462f17b1e9ce27ee6c27cb0dc40525ffb8d477eaa913155", R25 / "checkpoint_275.pt"),
 "l025_300": ("efce99e8a4be1dd9f9b86d55b228c30577dd8f2e68594e250d7e95e03b78e9e0", R25 / "checkpoint_300.pt"),
 "l05_250": ("666c0a9e51251c96a768f98c2b38a04e417cd08f831170e99ed0d781b2cacdaf", R05 / "checkpoint_250.pt"),
 "l05_275": ("c92ca96b5656357bfd417277355d38df81c4f9f51868ac2d7f0a31b4e3e6f1bc", R05 / "checkpoint_275.pt"),
 "l05_300": ("c962c8ae5805e84aea7356677e1f529351a29b89b5c26efd938432ec8bab26ab", R05 / "checkpoint_300.pt"),
}
# cross-check the pins against the earlier ARTIFACTS files
art = {}
for f in ("/workspace/forge/vyra-resume300-run-20261110/ARTIFACTS.sha256", "/workspace/forge/vyra-y72fix-run-20261110/ARTIFACTS.sha256"):
    for line in Path(f).read_text().splitlines():
        h, _, p = line.partition("  ")
        if "checkpoint_" in p: art[Path(p).parent.parent.parent.name + "/" + Path(p).name if False else p] = h
assert set(h for h, _ in PINS.values()) - {PINS["l025_200"][0]} <= set(art.values()), "pins not found in earlier ARTIFACTS files"
SETP = REPO / "prereg" / "y72_heldout_v2" / "y72_heldout_v2.pt"
S = g.verify_set(SETP)                                   # fail closed
new_pairs = list(zip(S["rows"], S["mask"]))
sys.path.insert(0, str(X))
import vyra_shim
with vyra_shim.allow_cuda():
    import run_vyra1_copy as rv
import verify_shapes as vs
vyra_shim.install_rope(vs); vyra_shim.bind_dims(vs); vyra_shim.install_attention_mask(vs)
vyra_shim.patch_library(rv, "/tmp/y72v2_lab")
rv.Experts.forward = rv.stable_experts_forward
SY = torch.load(Yb / "train_Y.pt", map_location="cpu", weights_only=False)
old_pairs = list(zip(SY["distances"][72]["unseen"], SY["distances"][72]["unseen_mask"]))
W = torch.load(X / "vyra_W_step0_weights.pt", map_location="cpu", weights_only=False)
prev = json.load(open("/workspace/forge/vyra-noise-clip-study-20261110/diag/y72_tok.json"))

@torch.no_grad()
def per_segment(m, pairs, bs=4):
    m.eval(); tokc = []; nllm = []
    for i in range(0, len(pairs), bs):
        x = torch.stack([p[0] for p in pairs[i:i + bs]]).long(); mm = torch.stack([p[1] for p in pairs[i:i + bs]]).bool()
        lg = m(x)[:, :510].float(); pr = lg.argmax(-1); tg = x[:, 1:511]
        nll = F.cross_entropy(lg.reshape(-1, rv.VOCAB), tg.reshape(-1), reduction="none").view(x.shape[0], -1)
        for r in range(x.shape[0]):
            ok = (pr[r] == tg[r])[mm[r]]; nl = nll[r][mm[r]]
            tokc.append(ok.view(-1, 8).sum(1).tolist()); nllm.append(nl.view(-1, 8).mean(1).tolist())
    return tokc, nllm

out = {"set_file_sha256": g.SET_FILE_SHA256, "set_canonical_sha256": g.SET_CANONICAL_SHA256, "ckpt": {}, "timing": {}}
T0 = time.time(); C0 = time.process_time()
for name, (pin, path) in PINS.items():
    t0 = time.time(); c0 = time.process_time()
    raw = Path(path).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != pin:
        raise SystemExit(f"CHECKPOINT_SHA256_MISMATCH {name} {got} != {pin}")          # fail closed, before torch.load
    del raw
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = rv.build_arm("R"); m.load_state_dict(W["model"], strict=True); m.load_state_dict(ck["model"], strict=True)
    tn, nn_ = per_segment(m, new_pairs); to, no = per_segment(m, old_pairs)
    seg_new = sum(1 for r in tn for v in r if v == 8); seg_old = sum(1 for r in to for v in r if v == 8)
    assert to == prev[name]["tok"], f"legacy-set counts differ from the earlier study for {name}"
    out["ckpt"][name] = {"sha256_verified": pin, "new": {"tok": tn, "nll": nn_, "segment_exact": seg_new, "tokens_correct": sum(sum(r) for r in tn)},
                         "legacy200": {"segment_exact": seg_old, "tokens_correct": sum(sum(r) for r in to)}}
    out["timing"][name] = {"wall_s": round(time.time() - t0, 1), "cpu_process_s": round(time.process_time() - c0, 1)}
    print(name, "sha OK", pin[:8], "new600 seg", seg_new, "tok", out["ckpt"][name]["new"]["tokens_correct"], "| legacy200 seg", seg_old, "(matches earlier study) | wall", out["timing"][name]["wall_s"], "s cpu", out["timing"][name]["cpu_process_s"], "s", flush=True)
out["total"] = {"wall_s": round(time.time() - T0, 1), "cpu_process_s": round(time.process_time() - C0, 1)}
json.dump(out, open(OUTJ, "w"))
print("total", out["total"])
