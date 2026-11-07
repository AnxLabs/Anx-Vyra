"""One CPU regression for the multi-forward replay caller.

The original caller keeps a nonzero gradient on experts used only by the first
forward and then does not update them. The repaired caller updates those
experts once. Experts._selected stays the last forward, and SliceAdam is the
repository optimizer, not a substitute. This is not a training run.
"""
import ast
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

PACKAGING = Path(__file__).resolve().parents[1] / "tools" / "experiment_packaging"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGING))

from expert_union import apply_expert_union  # noqa: E402

GREEN_DELTAS = {
    "e.gate": [2.2292137145996094e-05, 1.8775463104248047e-05, 1.7940998077392578e-05, 0.0],
    "e.up": [2.0712614059448242e-05, 1.8388032913208008e-05, 1.7762184143066406e-05, 0.0],
    "e.down": [1.8209218978881836e-05, 1.7762184143066406e-05, 1.7642974853515625e-05, 0.0],
}
GREEN_CLOCKS = [1, 1, 1, 0]
RED_CLOCKS = [0, 1, 1, 0]

CALLER = """
import time
STEPS = 1
for s in range(STEPS):
    if time.time()-T0>SOFT: stop(f'time budget at {arm} step {s}')
    ts=time.time(); rv.LR=3e-4; m.train(); m.zero_grad(set_to_none=True)
    tok,msk=batch_at(s); tok=tok.to('cuda'); msk=msk.to('cuda'); lx=ce(m(tok),tok,msk); loss=lx; ly=None
    if lam: yt,ym,_=replay_at(s); yt=yt.to('cuda'); ym=ym.to('cuda'); ly=ce(m(yt),yt,ym); loss=lx+lam*ly
    loss.backward()
    if not math.isfinite(float(loss)) or not rv.finite_grads(m): stop(f'nonfinite loss/grad {arm} {s}')
    torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.grad is not None],1.0)
    opt.step(rv.expert_selection(m)); m.zero_grad(set_to_none=True)
    if not rv.finite_params(m): stop(f'nonfinite params {arm} {s}')
"""


def _load(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def _library():
    namespace = {
        "torch": torch,
        "F": F,
        "D": 4,
        "N_EXPERTS": 4,
        "TOP_K": 2,
        "EXPERT_M": 3,
        "INIT_STD": 0.02,
        "B1": 0.9,
        "B2": 0.999,
        "EPS": 1e-8,
        "LR": 3e-4,
        "ToyLM": torch.nn.Module,
    }
    _load(
        REPO / "anx_vyra" / "model.py",
        ["init_normal", "Experts", "stable_experts_forward", "expert_selection"],
        namespace,
    )
    _load(REPO / "anx_vyra" / "optim.py", ["SliceAdam", "make_optimizer"], namespace)
    namespace["Experts"].forward = namespace["stable_experts_forward"]
    library = types.SimpleNamespace(**namespace)

    def finite_params(model):
        return all(torch.isfinite(param).all() for param in model.parameters())

    def finite_grads(model):
        for param in model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                return False
        return True

    library.finite_params = finite_params
    library.finite_grads = finite_grads
    library._namespace = namespace
    return library


def _caller_body(source):
    tree = ast.parse(source)
    loop = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "s"
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range"
        and any(isinstance(arg, ast.Name) and arg.id == "STEPS" for arg in node.iter.args)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "opt"
            and child.func.attr == "step"
            for child in ast.walk(node)
        )
    )
    body = []
    active = False
    for node in loop.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Tuple)
            and any(isinstance(elt, ast.Name) and elt.id == "tok" for elt in target.elts)
            for target in node.targets
        ):
            active = True
        if active:
            body.append(node)
        if (
            active
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "opt"
            and node.value.func.attr == "step"
        ):
            break
    assert active and body[-1].value.func.attr == "step"

    class _CPU(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == "cuda":
                return ast.copy_location(ast.Constant("cpu"), node)
            return node

    return ast.fix_missing_locations(_CPU().visit(ast.Module(body=body, type_ignores=[])))


def _execute(source, library):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.e = library.Experts(0)
            self.shared = torch.nn.Parameter(torch.tensor(1.0))
            self.unused = torch.nn.Parameter(torch.tensor(2.0))
            self.outs = []
            with torch.no_grad():
                self.e.router.copy_(torch.tensor([
                    [4.0, 0, 0, 0],
                    [0, 4.0, 0, 0],
                    [1.0, 1.0, 0, 0],
                    [-8.0, -8, 0, 0],
                ]))
                self.e.gate.fill_(0.3)
                self.e.up.fill_(0.4)
                self.e.down.fill_(0.5)
                self.e.up[2].fill_(0.6)

        def forward(self, tokens):
            out = self.e(tokens) * self.shared
            out.retain_grad()
            self.outs.append(out)
            return out

    model = Tiny()
    library._namespace["LR"] = 3e-4
    opt = library.make_optimizer(model)
    for param in model.parameters():
        state = opt.state[id(param)]
        state["exp_avg"].fill_(0.17)
        state["exp_avg_sq"].fill_(0.23)
        if state["expert_steps"] is None:
            state["step"] = 3
        else:
            state["expert_steps"].fill_(3)
    before = {
        id(param): {
            "param": param.detach().clone(),
            "step": opt.state[id(param)]["step"],
            "moment": opt.state[id(param)]["exp_avg"].clone(),
            "second": opt.state[id(param)]["exp_avg_sq"].clone(),
            "expert_steps": None if opt.state[id(param)]["expert_steps"] is None
            else opt.state[id(param)]["expert_steps"].clone(),
        }
        for param in model.parameters()
    }
    raw = {}
    for param in model.parameters():
        param.register_hook(lambda grad, param=param: raw.__setitem__(id(param), grad.detach().clone()))
    primary = torch.tensor([[[1.0, 0, 0, 0], [1.0, 0, 0, 0]]], requires_grad=True)
    replay = torch.tensor([[[0.0, 1, 0, 0], [0.0, 1, 0, 0]]], requires_grad=True)
    mask = torch.tensor([[1.0, 0.0]])

    def ce(logits, _tokens, token_mask):
        return 100 * ((logits.square().sum(-1)) * token_mask).sum() / token_mask.sum()

    def stop(message):
        raise AssertionError(message)

    env = dict(
        m=model, opt=opt, rv=library, torch=torch, math=math, ce=ce,
        batch_at=lambda _step: (primary, mask),
        replay_at=lambda _step: (replay, mask, [0]),
        stop=stop, s=0, lam=0.25, arm="regression",
    )
    exec(compile(_caller_body(source), "replay-caller", "exec"), env)
    assert raw[id(model.e.router)].abs().sum() > 0, "TOP_K2 router must have nonzero gradient"
    assert model.e._selected == {1, 2}, "second forward must replace the recorded selection"
    assert primary.grad is not None and replay.grad is not None
    for values in (primary, replay):
        assert torch.equal(values.grad[:, 1], torch.zeros_like(values.grad[:, 1]))
    for out in model.outs:
        assert torch.equal(out.grad[:, 1], torch.zeros_like(out.grad[:, 1]))
    raw_norm = torch.sqrt(sum(value.square().sum() for value in raw.values()))
    coef = min(1.0, float(1.0 / (raw_norm + 1e-6)))
    assert coef < 1
    clipped_norm = torch.sqrt(sum(
        param.grad.square().sum() for param in model.parameters() if param.grad is not None
    ))
    assert float(clipped_norm) <= 1.000001
    changes = {}
    clocks = {}
    max_error = 0.0
    for name, param in model.named_parameters():
        saved = before[id(param)]
        state = opt.state[id(param)]
        expert_steps = state["expert_steps"]
        if param.grad is None:
            assert torch.equal(param, saved["param"])
            assert torch.equal(state["exp_avg"], saved["moment"])
            assert torch.equal(state["exp_avg_sq"], saved["second"])
            assert state["step"] == saved["step"]
            continue
        assert torch.allclose(param.grad, raw[id(param)] * coef, atol=1e-7, rtol=1e-5)
        if expert_steps is not None:
            assert param.grad[0].abs().sum() > 0 and param.grad[1].abs().sum() > 0
            assert torch.equal(param.grad[3:], torch.zeros_like(param.grad[3:]))
            assert torch.equal(param[3:], saved["param"][3:])
            assert torch.equal(state["exp_avg"][3:], saved["moment"][3:])
            assert torch.equal(state["exp_avg_sq"][3:], saved["second"][3:])
            assert torch.equal(expert_steps[3:], saved["expert_steps"][3:])
            assert state["step"] == saved["step"]
            changes[name] = [(param[i] - saved["param"][i]).abs().max().item() for i in range(4)]
            clocks[name] = (expert_steps - saved["expert_steps"]).tolist()
            indices = [i for i in (0, 1, 2) if int(expert_steps[i]) == int(saved["expert_steps"][i]) + 1]
        else:
            indices = [None]
            assert state["step"] == saved["step"] + 1
        for index in indices:
            grad = param.grad if index is None else param.grad[index]
            moment = saved["moment"] if index is None else saved["moment"][index]
            second = saved["second"] if index is None else saved["second"][index]
            previous = saved["param"] if index is None else saved["param"][index]
            expected_moment = 0.9 * moment + 0.1 * grad
            expected_second = 0.999 * second + 0.001 * grad.square()
            expected = previous - 3e-4 * (expected_moment / (1 - 0.9 ** 4)) / (
                (expected_second / (1 - 0.999 ** 4)).sqrt() + 1e-8
            )
            actual_moment = state["exp_avg"] if index is None else state["exp_avg"][index]
            actual_second = state["exp_avg_sq"] if index is None else state["exp_avg_sq"][index]
            actual = param if index is None else param[index]
            assert torch.allclose(actual_moment, expected_moment, atol=1e-7)
            assert torch.allclose(actual_second, expected_second, atol=1e-7)
            max_error = max(max_error, float((actual - expected).abs().max()))
            assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    assert abs(float(model.shared - before[id(model.shared)]["param"])) > 0
    report = {
        "per_expert_parameter_max_delta": changes,
        "per_expert_clock_increment": clocks,
        "raw_global_norm": float(raw_norm),
        "clipped_global_norm": float(clipped_norm),
        "clip_factor": coef,
        "max_Adam_reference_delta_error": max_error,
        "last_forward_selection": sorted(model.e._selected),
    }
    passed = all(
        values[0] > 0 and values[1] > 0 and values[2] > 0 and values[3:] == [0.0]
        for values in changes.values()
    ) and all(values == GREEN_CLOCKS for values in clocks.values())
    report["pass"] = passed
    if not passed:
        error = AssertionError(
            "UNION_REGRESSION: nonzero X-only gradients were dropped by last-forward selection"
        )
        error.report = report
        raise error
    return report


def test_replay_union_updates_first_only_experts_once():
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    library = _library()
    try:
        _execute(CALLER, library)
    except AssertionError as exc:
        assert "UNION_REGRESSION" in str(exc)
        red = exc.report
    else:
        raise AssertionError("original caller updated the first-only experts")
    assert red["last_forward_selection"] == [1, 2]
    assert all(values == RED_CLOCKS for values in red["per_expert_clock_increment"].values())
    assert all(values[0] == 0.0 and values[1] > 0 and values[3] == 0.0
               for values in red["per_expert_parameter_max_delta"].values())
    green = _execute(apply_expert_union(CALLER), library)
    assert green["pass"] is True
    assert green["last_forward_selection"] == [1, 2]
    assert green["per_expert_parameter_max_delta"] == GREEN_DELTAS
    assert all(values == GREEN_CLOCKS for values in green["per_expert_clock_increment"].values())
    assert green["raw_global_norm"] == 25.591171264648438
    assert green["clip_factor"] == 0.03907597437500954
    assert green["max_Adam_reference_delta_error"] == 0.0
    assert abs(green["clipped_global_norm"] - 1.0) < 1e-6
