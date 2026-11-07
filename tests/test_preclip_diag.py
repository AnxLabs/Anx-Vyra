"""CPU checks that sparse pre-clip reads do not change the update.

The analytic graph checks norms and cosine. The arm-R check is two synthetic
forwards, one combined backward, and one SliceAdam step, with diagnostics on
and off. It is not a training run and it does not use the rehearsal rows.
"""
import math
import random
import sys
from pathlib import Path

import pytest
import torch

PACKAGING = Path(__file__).resolve().parents[1] / "tools" / "experiment_packaging"
sys.path.insert(0, str(PACKAGING))

from anx_vyra.model import build_arm, expert_selection
from anx_vyra.optim import make_optimizer
import preclip_diag


class _Opt:
    def __init__(self, param):
        self.state = {
            id(param): {
                "step": 3,
                "expert_steps": torch.tensor([4, 5]),
                "exp_avg": torch.zeros(2),
                "exp_avg_sq": torch.ones(2),
            }
        }
        self.calls = 0

    def step(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("diagnostic helper called optimizer.step")


def test_separate_norms_match_analytic_values_and_do_not_step():
    weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    unused = torch.nn.Parameter(torch.tensor([5.0]))
    unused.grad = torch.tensor([7.0])
    opt = _Opt(weight)
    logits = weight[0] * torch.ones(6, 8, 5)
    tokens = torch.zeros(6, 8, dtype=torch.long)
    mask = torch.ones(6, 6)
    loss_x = (weight * torch.tensor([1.0, 1.0])).sum()
    loss_y = (weight * torch.tensor([1.0, 2.0])).sum()
    first = {id(weight): [0]}
    second = {id(weight): [1]}
    passed = {id(weight): [0, 1]}
    rng = random.getstate(), torch.get_rng_state().clone()
    holder = torch.nn.Module()
    holder.weight = weight
    holder.unused = unused
    holder._selected = {3}
    holder._load = torch.zeros(4)
    record = preclip_diag.begin_separate(
        holder, opt, logits, loss_x, loss_y, tokens, mask, first, second, passed, 0.25
    )
    assert opt.calls == 0
    assert math.isclose(record["gx_norm"], math.sqrt(2.0), rel_tol=0, abs_tol=1e-12)
    assert math.isclose(record["gy_norm"], math.sqrt(5.0), rel_tol=0, abs_tol=1e-12)
    assert math.isclose(record["gy_weighted_norm"], 0.25 * math.sqrt(5.0), rel_tol=0, abs_tol=1e-12)
    assert math.isclose(record["cosine_x_y"], 3.0 / math.sqrt(10.0), rel_tol=0, abs_tol=1e-12)
    assert record["cosine_x64_y"] is not None
    assert record["x64_scope"].startswith("one sampled")
    assert record["selection_passed"][0]["experts"] == [0, 1]
    assert weight.grad is None
    assert torch.equal(unused.grad, torch.tensor([7.0]))
    assert holder._selected == {3}
    assert random.getstate() == rng[0]
    assert torch.equal(torch.get_rng_state(), rng[1])
    finished = preclip_diag.finish_combined(torch.tensor(2.5), 1)
    assert finished["combined_preclip_norm"] == 2.5
    assert math.isclose(finished["clip_multiplier"], 1.0 / 2.500001, rel_tol=0, abs_tol=1e-12)
    zero = (weight * 0).sum()
    zero_record = preclip_diag.begin_separate(
        holder, opt, None, loss_x, zero, tokens, mask, first, None, first, 0.25
    )
    assert zero_record["gy_norm"] == 0.0
    assert zero_record["cosine_x_y"] is None
    assert zero_record["cosine_x64_y"] is None
    preclip_diag.finish_combined(torch.tensor(0.25), 75)
    assert opt.calls == 0


def _run_arm(diagnostics):
    torch.manual_seed(0)
    random.seed(0)
    model = build_arm("R")
    opt = make_optimizer(model)
    for state in opt.state.values():
        if state["expert_steps"] is None:
            state["step"] = 2
            state["exp_avg"].fill_(0.01)
        else:
            state["expert_steps"].fill_(3)
            state["exp_avg"].fill_(0.01)
    tokens_x = torch.arange(6 * 16).reshape(6, 16) % 40
    tokens_y = (torch.arange(6 * 16).reshape(6, 16) + 3) % 40
    mask = torch.ones(6, 14)
    model.train()
    logits_x = model(tokens_x)
    first = expert_selection(model)
    logits_y = model(tokens_y)
    second = expert_selection(model)
    passed = {
        key: sorted(set(first[key]) | set(value))
        for key, value in second.items()
    }
    names = {id(param): name for name, param in model.named_parameters()}
    passed_named = tuple(sorted((names[key], tuple(value)) for key, value in passed.items()))
    loss_x = logits_x.float().square().mean()
    loss_y = logits_y.float().square().mean()
    if diagnostics:
        preclip_diag.begin_separate(
            model, opt, logits_x, loss_x, loss_y, tokens_x, mask, first, second, passed, 0.25
        )
    (loss_x + 0.25 * loss_y).backward()
    grad = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    clip_inputs = [param for param in model.parameters() if param.grad is not None]
    total = torch.nn.utils.clip_grad_norm_(clip_inputs, 1.0)
    record = None
    if diagnostics:
        record = preclip_diag.finish_combined(total, 1)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    clocks = {
        ident: None if state["expert_steps"] is None else state["expert_steps"].clone()
        for ident, state in opt.state.items()
    }
    opt.step(passed)
    delta = {
        name: (param.detach() - before[name]).abs().max().item()
        for name, param in model.named_parameters()
    }
    after_clocks = {
        ident: None if state["expert_steps"] is None else state["expert_steps"].clone()
        for ident, state in opt.state.items()
    }
    return grad, float(total), delta, after_clocks, clocks, passed_named, record, torch.get_rng_state().clone()


def test_diagnostics_do_not_change_gradients_union_or_one_step():
    off = _run_arm(False)
    on = _run_arm(True)
    for name, grad in off[0].items():
        assert torch.allclose(grad, on[0][name], rtol=1e-5, atol=1e-6)
    assert math.isclose(off[1], on[1], rel_tol=0, abs_tol=1e-6)
    value, multiplier = preclip_diag.clip_multiplier(torch.tensor(off[1]))
    assert on[6]["combined_preclip_norm"] == pytest.approx(value)
    assert on[6]["clip_multiplier"] == pytest.approx(multiplier)
    assert on[6]["selection_passed"]
    for name, delta in off[2].items():
        assert math.isclose(delta, on[2][name], rel_tol=0, abs_tol=1e-8)
    def clock_deltas(before, after):
        rows = []
        for ident, prev in before.items():
            if prev is None:
                rows.append(None)
            else:
                rows.append((after[ident] - prev).tolist())
        return rows
    assert clock_deltas(off[4], off[3]) == clock_deltas(on[4], on[3])
    assert off[5] == on[5]
    assert torch.equal(off[7], on[7])


def test_rehearsal_readout_restores_mode_rng_and_routes():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self._selected = {1}
            self._load = torch.zeros(4)

        def forward(self, tokens):
            self._selected = {7}
            self._load = self._load + 1
            torch.rand(1)
            return torch.zeros(tokens.shape[0], tokens.shape[1], 6)

    model = Tiny()
    model.train()
    rows = torch.zeros(40, 8, dtype=torch.int16)
    mask = torch.ones(40, 6)
    rng = (random.getstate(), torch.get_rng_state().clone())
    report = preclip_diag.rehearsal_readout(model, rows, mask)
    assert model.training
    assert model._selected == {1}
    assert torch.equal(model._load, torch.zeros(4))
    assert random.getstate() == rng[0]
    assert torch.equal(torch.get_rng_state(), rng[1])
    assert report["original8"]["rows"] == 8
    assert report["full40"]["rows"] == 40
    assert report["original8"]["token_denominator"] > 0
    assert "row_mean" in report["original8"]["ce_definition"]
