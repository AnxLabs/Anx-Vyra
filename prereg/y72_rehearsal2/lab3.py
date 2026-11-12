"""CPU lab for the cont400 swing diagnosis (adapted from vyra-y72fix2-design-20261111/probes/lab2.py; originals untouched).
Pinned model classes/data, the 400-entry plan, read-only restore of the downloaded v2 checkpoints, eval + routing helpers."""
import os, sys, json
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ["MKL_CBWR"] = "COMPATIBLE"
import torch
torch.set_num_threads(8)
import torch.nn.functional as F
X = Path(os.environ["VYRA_X_BUNDLE"]); Y = Path(os.environ["VYRA_Y_BUNDLE"])   # local bundle directories (tracked files carry no bundle names)
PKG = "/workspace/forge/vyra-y72fix2-cont400-prep-20261112/repo/tools/experiment_packaging"
RUN = Path("/workspace/forge/vyra-y72fix2-cont400-run-20261112/out/replay_out")
CK300 = Path("/workspace/forge/vyra-y72fix2-cont400-prep-20261112/ckpt300/checkpoint_300.pt")
REHEARSAL = "/workspace/forge/vyra-y72fix2-cont400-run-20261112/out/rehearsal_Y72_40.pt"
SET600 = "/workspace/forge/vyra-y72fix2-cont400-prep-20261112/repo/prereg/y72_heldout_v2/y72_heldout_v2_set.json"
sys.path.insert(0, PKG); sys.path.insert(0, str(X))
import vyra_shim
with vyra_shim.allow_cuda():
    import run_vyra1_copy as rv
import verify_shapes as vs
vyra_shim.install_rope(vs); vyra_shim.bind_dims(vs); vyra_shim.install_attention_mask(vs)
vyra_shim.patch_library(rv, "/tmp/rehearsal2_lab")
_orig_fwd = rv.stable_experts_forward
def _rec_fwd(self, x):
    out = _orig_fwd(self, x)
    logits = F.linear(x, self.router)
    self._chosen = torch.argsort(logits, dim=-1, descending=True, stable=True)[..., :vs.TOP_K].detach()
    return out
rv.Experts.forward = _rec_fwd
import continuation as cont, plan_extension as pe3, plan_extension400 as pe4, rehearsal_artifact, y72_heldout_v2 as h600
SX = torch.load(X / "train_X.pt", map_location="cpu", weights_only=False)
SY = torch.load(Y / "train_Y.pt", map_location="cpu", weights_only=False)
PLAN = pe4.build_extended_plan(SX, 400)
XD = SX["distances"]; Y72 = SY["distances"][72]
RY = rehearsal_artifact.load_for_training(REHEARSAL, set(Y72["unseen_payloads"]))
W = torch.load(X / "vyra_W_step0_weights.pt", map_location="cpu", weights_only=False)
RB = 2
s = h600.verify_text_set(SET600)
_r = torch.frombuffer(bytearray(bytes.fromhex(s["rows_int16le_hex"])), dtype=torch.int16).view(120, 512).clone()
_m = torch.frombuffer(bytearray(bytes.fromhex(s["mask_int16le_hex"])), dtype=torch.int16).view(120, 510).clone()
SETS = {"Y600": list(zip(_r, _m)), "Y200": list(zip(Y72["unseen"], Y72["unseen_mask"])),
        "X64": list(zip(XD[64]["unseen"], XD[64]["unseen_mask"])), "X32": list(zip(XD[32]["unseen"], XD[32]["unseen_mask"])),
        "X109": list(zip(XD[109]["unseen"], XD[109]["unseen_mask"])), "REP40": list(zip(RY["rows"], RY["mask"]))}

def batch_at(s):
    e = PLAN[s]; rows = [SX["rows_gen"][e["slot0"]], SX["rows_gen"][e["slot1"]], SX["rows32"][e["slot2"]]]
    ms = [SX["mask_gen"][e["slot0"]], SX["mask_gen"][e["slot1"]], SX["mask32"][e["slot2"]]]
    for d in (32, 64, 109):
        k = e["needles"][d]; rows.append(XD[d]["rows"][k]); ms.append(XD[d]["mask"][k])
    return torch.stack(rows).long(), torch.stack(ms).long()

def replay_at(s):
    idx = [(s * RB + j) % 40 for j in range(RB)]
    return RY["rows"][idx].long(), RY["mask"][idx].long(), idx

def ce_rows(lg, tok, mask):
    nll = F.cross_entropy(lg[:, :-1].reshape(-1, rv.VOCAB), tok[:, 1:].reshape(-1), reduction="none").view(tok.shape[0], -1)[:, :510]
    w = mask.to(nll.dtype); return (nll * w).sum(1) / w.sum(1).clamp_min(1)

def make_model_opt():
    m = rv.build_arm("R"); m.load_state_dict(W["model"], strict=True)
    rv.LR = 3e-4; opt = rv.make_optimizer(m); return m, opt

def load(u):
    """Restore checkpoint u (300 from the verified copy, 325..400 from the downloaded run outputs), read-only, sampler-checked."""
    m, opt = make_model_opt()
    path = CK300 if u == 300 else RUN / f"checkpoint_{u}.pt"
    ck = cont.load_full_checkpoint(path)
    assert ck["update"] == u
    if u == 300:
        cont.restore_full(m, opt, ck, lib=rv, plan=PLAN[:300], plan_module=pe3, replay_batch=RB, cuda=False)
    else:
        cont.restore_full(m, opt, ck, lib=rv, plan=PLAN, plan_module=pe4, replay_batch=RB, cuda=False)
    return m, opt, ck

def experts_modules(m):
    return [b.ffn for b in m.blocks if isinstance(b.ffn, rv.Experts)]

@torch.no_grad()
def seg_eval(m, pairs, bs=4, routing=False):
    """per-segment exact vector, tokens correct, per-row masked CE, and (optional) per-layer top-1/top-2 expert histograms at scored positions."""
    m.eval(); seg = []; tokc = 0; ces = []; ex = experts_modules(m); hist = [torch.zeros(vs.N_EXPERTS, 2, dtype=torch.long) for _ in ex]
    for i in range(0, len(pairs), bs):
        x = torch.stack([p[0] for p in pairs[i:i + bs]]).long(); mm = torch.stack([p[1] for p in pairs[i:i + bs]]).bool()
        lg = m(x)
        ces += ce_rows(lg, x, mm.long()).tolist()
        lg = lg[:, :510].float(); pr = lg.argmax(-1); tg = x[:, 1:511]
        for r in range(x.shape[0]):
            o = (pr[r] == tg[r])[mm[r]]; f = (o.numel() // 8) * 8; tokc += int(o.sum())
            seg += o[:f].view(-1, 8).all(1).int().tolist()
        if routing:
            for L, e in enumerate(ex):
                ch = e._chosen[:, :510][mm]            # router choice at the position that predicts each scored token
                for slot in range(2):
                    hist[L][:, slot] += torch.bincount(ch[:, slot], minlength=vs.N_EXPERTS)
    return {"seg": seg, "tokens_correct": tokc, "row_ce": ces, "routing": [h.tolist() for h in hist] if routing else None}

def named_groups(m):
    """'expert' = sliced expert tensors (gate/up/down), 'router', 'shared' = everything else."""
    out = {}
    for n, p in m.named_parameters():
        out[n] = "expert" if any(k in n for k in (".gate", ".up", ".down")) and "ffn" in n else ("router" if "router" in n else "shared")
    return out
