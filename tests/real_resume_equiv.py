"""Resume-equivalence on the REAL model classes (pinned run_vyra1_copy), reduced sequence length.

Run as a script by tests/test_resume_equivalence_real.py. Real ToyLM (arm R,
16.4M params, W starting weights), real SliceAdam, real Experts with the stable
forward, the replay-style two-forward update with the selection union and one
global clip. Only the sequence length (96 tokens), batch (3 X rows + 1 replay
row) and the loss (unmasked CE over those tokens) are reduced so CPU time is
small. Exact equality is asserted; no tolerance.
"""
import json, os, random, sys, time
from pathlib import Path

import numpy as np

X = Path(os.environ.get("VYRA_X_BUNDLE", "/workspace/vera/packaging-rootfix-20261106/inputs/vyra-x-bundle"))
PKG = Path(__file__).resolve().parents[1] / "tools" / "experiment_packaging"
sys.path.insert(0, str(PKG)); sys.path.insert(0, str(X))
os.environ["OMP_NUM_THREADS"] = "2"
import torch
torch.set_num_threads(2)
import torch.nn.functional as F
import vyra_shim
with vyra_shim.allow_cuda():
    import run_vyra1_copy as rv
import verify_shapes as vs
vyra_shim.install_rope(vs); vyra_shim.bind_dims(vs); vyra_shim.install_attention_mask(vs)
vyra_shim.patch_library(rv, "/tmp/vyra_real_resume_lab")
rv.Experts.forward = rv.stable_experts_forward
import continuation as cont
import plan_extension as pe

SX = torch.load(X / "train_X.pt", map_location="cpu", weights_only=False)
W = torch.load(X / "vyra_W_step0_weights.pt", map_location="cpu", weights_only=False)
PLAN = pe.build_extended_plan(SX, 300)
SEQ = 96
N, M = 2, 2
START = 200            # real plan indices 200.. exercise the extension
ARGS = dict(replay_lambda=0.25, replay_batch=2, seed=0, clip=1.0)


def batch(s):
    e = PLAN[s]
    rows = [SX["rows_gen"][e["slot0"]], SX["rows_gen"][e["slot1"]], SX["rows32"][e["slot2"]]]
    return torch.stack(rows).long()[:, :SEQ]


def ce(lg, tok):
    return F.cross_entropy(lg[:, :-1].reshape(-1, rv.VOCAB), tok[:, 1:].reshape(-1))


class World:
    def __init__(self, load_w=True):
        self.m = rv.build_arm("R")
        if load_w:
            self.m.load_state_dict(W["model"], strict=True)
        self.opt = rv.make_optimizer(self.m)
        self.losses = []

    def step(self, s):
        m, opt = self.m, self.opt
        rv.LR = 3e-4
        m.train(); m.zero_grad(set_to_none=True)
        tok = batch(s)
        lx = ce(m(tok), tok); sel_a = rv.expert_selection(m)
        rep = batch((s * 2 + 1) % 300)[:1]
        ly = ce(m(rep), rep); sel_b = rv.expert_selection(m)
        selected = {k: sorted(set(sel_a[k]) | set(sel_b[k])) for k in sel_a}
        (lx + 0.25 * ly).backward()
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.grad is not None], 1.0)
        opt.step(selected); m.zero_grad(set_to_none=True)
        self.losses.append((float(lx), float(ly)))

    def run(self, a, n):
        for s in range(a, a + n):
            self.step(s)

    def capture(self, update):
        return cont.capture_full_state(self.m, self.opt, update=update, plan=PLAN, plan_module=pe, lib=rv,
                                       provenance={"test": "real-small-config"}, **ARGS)


def snap(w):
    names = {id(p): n for n, p in w.m.named_parameters()}
    return {"losses": list(w.losses),
            "sd": {k: v.clone() for k, v in w.m.state_dict().items()},
            "opt": {names[k]: {"step": int(v["step"]), "a": v["exp_avg"].clone(), "b": v["exp_avg_sq"].clone(),
                               "e": None if v["expert_steps"] is None else v["expert_steps"].clone()} for k, v in w.opt.state.items()},
            "torch": torch.get_rng_state().clone(), "py": random.getstate(),
            "np": [x if not isinstance(x, np.ndarray) else x.tolist() for x in np.random.get_state()],
            "router": cont.capture_router(w.m)}


def diff(a, b):
    bad = set()
    if a["losses"] != b["losses"]: bad.add("losses")
    if any(not torch.equal(a["sd"][k], b["sd"][k]) for k in a["sd"]): bad.add("params")
    for k in a["opt"]:
        x, y = a["opt"][k], b["opt"][k]
        if x["step"] != y["step"]: bad.add("opt.step")
        if not torch.equal(x["a"], y["a"]): bad.add("opt.exp_avg")
        if not torch.equal(x["b"], y["b"]): bad.add("opt.exp_avg_sq")
        if (x["e"] is None) != (y["e"] is None) or (x["e"] is not None and not torch.equal(x["e"], y["e"])): bad.add("opt.expert_steps")
    if not torch.equal(a["torch"], b["torch"]): bad.add("torch_rng")
    if a["py"] != b["py"]: bad.add("python_rng")
    if a["np"] != b["np"]: bad.add("numpy_rng")
    ra, rb = a["router"], b["router"]
    if set(ra) != set(rb) or any(ra[k]["selected"] != rb[k]["selected"] or not torch.equal(ra[k]["load"], rb[k]["load"]) for k in ra):
        bad.add("router")
    return sorted(bad)


def seed_all(v):
    torch.manual_seed(v); random.seed(v); np.random.seed(v)


def garbage():
    random.seed(222); np.random.seed(333); torch.manual_seed(444)
    for _ in range(3): random.random(); np.random.random(); torch.rand(1)


def fresh_wrong(w_losses):
    w = World(load_w=False)          # random init != W, fresh optimizer
    for p in w.m.parameters():
        st = w.opt.state[id(p)]
        st["exp_avg"].fill_(0.1)
    for mod in w.m.modules():
        if isinstance(mod, rv.Experts):
            mod._load = torch.full((rv.N_EXPERTS,), 7, dtype=torch.long); mod._selected = {0}
    w.m.eval(); garbage()
    w.losses = list(w_losses)
    return w


def restore(w, ck, skip=()):
    for name in ("model", "optimizer", "model_mode", "router", "python_rng", "numpy_rng", "torch_rng"):
        if name not in skip:
            cont.COMPONENTS[name](w.m, w.opt, ck, cuda=False)


out = {"seq": SEQ, "N": N, "M": M, "plan_start": START, "controls": {}}
t0 = time.time()
seed_all(0); A = World(); A.run(START, N + M); straight = snap(A)
seed_all(0); B = World(); B.run(START, N)
state = B.capture(START + N)
tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "real_resume_ck.pt"
out["checkpoint"] = cont.save_full_checkpoint(tmp, state)
ck = cont.load_full_checkpoint(tmp)
out["sampler"] = ck["sampler"]
# positive
C = fresh_wrong(B.losses)
cont.restore_full(C.m, C.opt, ck, lib=rv, plan=PLAN, plan_module=pe, replay_batch=2, cuda=False)
C.run(ck["sampler"]["next_update_index"], M)
out["positive_mismatch"] = diff(straight, snap(C))
out["losses_straight"] = straight["losses"]; out["losses_resumed"] = C.losses
# negative controls
for dropped in ("model", "optimizer", "torch_rng", "python_rng", "numpy_rng", "router"):
    D = fresh_wrong(B.losses); restore(D, ck, skip=(dropped,)); D.run(START + N, M)
    out["controls"][dropped] = diff(straight, snap(D))
for name in ("moments_only", "clocks_only"):
    D = fresh_wrong(B.losses); restore(D, ck)
    for p in D.m.parameters():
        st = D.opt.state[id(p)]
        if name == "moments_only": st["exp_avg"].zero_(); st["exp_avg_sq"].zero_()
        elif st["expert_steps"] is not None: st["expert_steps"].zero_()
    D.run(START + N, M)
    out["controls"][name] = diff(straight, snap(D))
D = fresh_wrong(B.losses); restore(D, ck); D.run(START + N + 1, M)
out["controls"]["sampler_position"] = diff(straight, snap(D))
out["elapsed_s"] = time.time() - t0
print(json.dumps(out))
