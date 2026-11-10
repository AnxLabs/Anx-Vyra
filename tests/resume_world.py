"""Tiny hermetic training world shared by the resume/observer tests.

Real pieces: repository SliceAdam, the repository's Experts with the stable
forward (`_selected`, cumulative `_load`), the replay-style two-forward update
with the expert-selection union and one global clip. The step additionally
draws from the torch, Python and NumPy RNG streams and feeds each draw into the
loss, so a lost RNG stream changes the loss. That coupling is test-only: it is
what makes each negative control able to fail.
"""
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PACKAGING = Path(__file__).resolve().parents[1] / "tools" / "experiment_packaging"
sys.path.insert(0, str(PACKAGING))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_expert_union_replay import _library  # noqa: E402
import continuation as cont  # noqa: E402

LIB = _library()
ARGS = dict(replay_lambda=0.25, replay_batch=2, seed=0, clip=1.0)
N_STEPS_PLAN = 12


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Parameter(torch.randn(8, 4) * 0.5)
        self.e = LIB.Experts(0)
        self.head = torch.nn.Parameter(torch.randn(8, 4) * 0.5)
        self.shared = torch.nn.Parameter(torch.tensor(1.0))
        self.unused = torch.nn.Parameter(torch.tensor(2.0))   # never gets a gradient

    def forward(self, tokens, jitter=None):
        x = self.embed[tokens]
        if jitter is not None:
            x = x * jitter[0] + jitter[1]
        x = x + self.e(x) * self.shared
        return x @ self.head.t()


def make_plan():
    g = random.Random(5)
    return [{"slot0": g.randrange(6), "slot1": g.randrange(6), "slot2": g.randrange(4),
             "needles": {32: i % 5, 64: i % 5, 109: i % 5}} for i in range(N_STEPS_PLAN)]


def plan_module_for(plan):
    import hashlib, json
    def sha(p):
        return hashlib.sha256(json.dumps([[e["slot0"], e["slot1"], e["slot2"]] for e in p]).encode()).hexdigest()
    return SimpleNamespace(plan_sha256=sha, PINNED_PREFIX_PLAN_SHA256=sha(plan[:200]) if len(plan) >= 200 else None,
                           POLICY_VERSION="test-plan")


class World:
    def __init__(self, init_seed):
        torch.manual_seed(init_seed)
        self.model = TinyLM()
        self.opt = LIB.make_optimizer(self.model)
        self.data = torch.randint(0, 8, (N_STEPS_PLAN, 6, 7), generator=torch.Generator().manual_seed(11))
        self.target = torch.randn(N_STEPS_PLAN, 6, 7, 8, generator=torch.Generator().manual_seed(12))
        self.plan = make_plan()
        self.plan_mod = plan_module_for(self.plan)
        self.losses = []

    def step(self, s):
        """One update at plan index s. Draws once from each RNG stream."""
        m, opt = self.model, self.opt
        LIB._namespace["LR"] = 3e-4
        m.train(); m.zero_grad(set_to_none=True)
        jitter = (1.0 + 0.1 * random.random(), 0.05 * float(np.random.random()))
        noise = torch.randn(6, 7, 8) * 0.01
        tokens = self.data[s]
        lx = ((m(tokens, jitter) - self.target[s] - noise) ** 2).mean()
        sel_a = LIB.expert_selection(m)
        ly = ((m(self.data[(s * 2 + 1) % N_STEPS_PLAN][:2], jitter) - self.target[(s * 2 + 1) % N_STEPS_PLAN][:2]) ** 2).mean()
        sel_b = LIB.expert_selection(m)
        selected = {k: sorted(set(sel_a[k]) | set(sel_b[k])) for k in sel_a}
        loss = lx + 0.25 * ly
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.grad is not None], 1.0)
        opt.step(selected)
        m.zero_grad(set_to_none=True)
        self.losses.append((float(lx), float(ly)))

    def run(self, start, n):
        for s in range(start, start + n):
            self.step(s)

    def capture(self, update):
        return cont.capture_full_state(self.model, self.opt, update=update, plan=self.plan, plan_module=self.plan_mod,
                                       lib=LIB, provenance={"test": "tiny"}, **ARGS)


def global_rng_garbage():
    """Simulate a fresh process: every stream at a different, known-wrong state."""
    random.seed(222); np.random.seed(333); torch.manual_seed(444)
    for _ in range(5):
        random.random(); np.random.random(); torch.rand(1)


def snapshot(world):
    """Everything the equivalence is asserted over."""
    m, opt = world.model, world.opt
    names = {id(p): n for n, p in m.named_parameters()}
    return {
        "losses": list(world.losses),
        "params": {n: p.detach().clone() for n, p in m.named_parameters()},
        "opt": {names[k]: {"step": int(v["step"]), "exp_avg": v["exp_avg"].clone(), "exp_avg_sq": v["exp_avg_sq"].clone(),
                           "expert_steps": None if v["expert_steps"] is None else v["expert_steps"].clone()}
                for k, v in opt.state.items()},
        "torch_rng": torch.get_rng_state().clone(),
        "python_rng": random.getstate(),
        "numpy_rng": tuple(x if not isinstance(x, np.ndarray) else x.tolist() for x in np.random.get_state()),
        "router": cont.capture_router(m),
        "mode": m.training,
    }


def diff(a, b):
    """List of component names that differ (exact equality)."""
    bad = []
    if a["losses"] != b["losses"]:
        bad.append("losses")
    if any(not torch.equal(a["params"][k], b["params"][k]) for k in a["params"]):
        bad.append("params")
    for k in a["opt"]:
        x, y = a["opt"][k], b["opt"][k]
        if x["step"] != y["step"]:
            bad.append("opt.step")
        for key in ("exp_avg", "exp_avg_sq"):
            if not torch.equal(x[key], y[key]):
                bad.append("opt." + key)
        if (x["expert_steps"] is None) != (y["expert_steps"] is None) or (
                x["expert_steps"] is not None and not torch.equal(x["expert_steps"], y["expert_steps"])):
            bad.append("opt.expert_steps")
    if not torch.equal(a["torch_rng"], b["torch_rng"]):
        bad.append("torch_rng")
    if a["python_rng"] != b["python_rng"]:
        bad.append("python_rng")
    if a["numpy_rng"] != b["numpy_rng"]:
        bad.append("numpy_rng")
    ra, rb = a["router"], b["router"]
    if set(ra) != set(rb) or any(ra[k]["selected"] != rb[k]["selected"] or (ra[k]["load"] is None) != (rb[k]["load"] is None) or
                                 (ra[k]["load"] is not None and not torch.equal(ra[k]["load"], rb[k]["load"])) for k in ra):
        bad.append("router")
    if a["mode"] != b["mode"]:
        bad.append("mode")
    return sorted(set(bad))
