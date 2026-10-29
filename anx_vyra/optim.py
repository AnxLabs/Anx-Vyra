"""SliceAdam: Adam that only steps the expert slices that were actually used.

In a top-2-of-8 mixture, most experts are idle on any given token. Vanilla Adam still updates every
expert slice (its gradient is exactly zero, but the moment and variance buffers decay and the
parameter moves), which both wastes work and quietly drags unused experts toward the origin.

SliceAdam instead keeps a separate step count per expert slice and only touches slices that were
selected in the forward pass that produced the current gradients. That means the number of steps
recorded for an expert equals the number of times it was actually used, so bias correction is
per-expert and correct.

Gotcha worth stating plainly, because it cost real runs: this optimizer's learning rate is read
from the module-level `LR` constant at the moment of the update, not captured from a constructor
argument. Set the module constant and assert it before every step, or you will silently train at
whatever rate happens to be in the global.
"""
from __future__ import annotations

import torch

# ---------------------------------------------------------------------------------------------
# THE TRAP: _apply() below reads this module-level constant, not an instance attribute.
# A trainer that forgets to set it trains at this default while believing otherwise.
# ---------------------------------------------------------------------------------------------
LR = 3e-4
B1 = 0.9
B2 = 0.999
EPS = 1e-8


class SliceAdam:
    """Adam with per-expert step counts; expert slices step only when they were selected."""

    def __init__(self, params: list[torch.nn.Parameter], lr: float | None = None) -> None:
        global LR
        if lr is not None:
            LR = float(lr)          # keep the module contract: the global is the source of truth
        self.params = list(params)
        self.state: dict[int, dict] = {}
        for param in self.params:
            self.state[id(param)] = {
                "step": 0,
                "exp_avg": torch.zeros_like(param),
                "exp_avg_sq": torch.zeros_like(param),
                "expert_steps": None,
            }

    @property
    def lr(self) -> float:
        return float(LR)

    def mark_expert(self, param: torch.nn.Parameter) -> None:
        """Declare a parameter whose first dimension indexes experts."""
        self.state[id(param)]["expert_steps"] = torch.zeros(param.shape[0], dtype=torch.long)

    def step(self, selected: dict[int, list[int]] | None = None) -> None:
        selected = selected or {}
        with torch.no_grad():
            for param in self.params:
                if param.grad is None:
                    continue
                state = self.state[id(param)]
                if state["expert_steps"] is None:
                    self._apply(param, param.grad, state, None)
                else:
                    for expert in selected.get(id(param), []):
                        self._apply(param, param.grad, state, expert)

    def _apply(self, param, grad, state, expert: int | None) -> None:
        if expert is None:
            state["step"] += 1
            step = state["step"]
            moment, second = state["exp_avg"], state["exp_avg_sq"]
            piece, gradient = param, grad
        else:
            state["expert_steps"][expert] += 1
            step = int(state["expert_steps"][expert])
            moment, second = state["exp_avg"][expert], state["exp_avg_sq"][expert]
            piece, gradient = param[expert], grad[expert]
        moment.mul_(B1).add_(gradient, alpha=1 - B1)
        second.mul_(B2).addcmul_(gradient, gradient, value=1 - B2)
        bias1 = 1 - B1**step
        bias2 = 1 - B2**step
        piece.addcdiv_(moment / bias1, (second / bias2).sqrt().add(EPS), value=-float(LR))


def make_optimizer(model: torch.nn.Module, lr: float | None = None) -> SliceAdam:
    """Build a SliceAdam for `model`, marking sparse-expert parameters as sliced."""
    optimizer = SliceAdam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for module in model.modules():
        if module.__class__.__name__ == "Experts":
            for name in ("gate", "up", "down"):
                optimizer.mark_expert(getattr(module, name))
    return optimizer


__all__ = ["SliceAdam", "make_optimizer", "LR", "B1", "B2", "EPS"]
