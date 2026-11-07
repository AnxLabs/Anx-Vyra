"""Observational counters for one replay update. No forward, backward, or step.

The caller passes the selections it already computed and the model after the
existing backward, clip, and SliceAdam step. This module does not import the
pinned training library and does not change parameters, gradients, or optimizer
state.

Per expert slice, four facts are kept apart:

- selection: the expert index was in that forward's selection, and whether
  the dict actually passed to SliceAdam contained it
- nonzero gradient: after the combined backward and the one clip, some element
  of that slice's gradient was nonzero
- state delta: SliceAdam's per-expert clock increased. That is not proof the
  stored values changed
- bitsum: the int64 sum of the float32 bit patterns of the whole slice changed,
  which means at least one stored value changed. An unchanged sum can still
  hide two writes that cancel in that sum

Every expert-sliced parameter and every expert index is counted on every
completed update. The bitsum and gradient reductions stay on device and are
copied once in summary(). Nothing here calls cuda.synchronize or .item()
per update.
"""
import torch


def _expert_params(model, opt):
    params = []
    for param in model.parameters():
        state = opt.state.get(id(param))
        if state is None or state.get("expert_steps") is None:
            continue
        if param.dtype != torch.float32:
            raise RuntimeError("REPLAY_TELEMETRY_DTYPE")
        if state["expert_steps"].shape[0] != param.shape[0]:
            raise RuntimeError("REPLAY_TELEMETRY_CLOCK_SHAPE")
        params.append(param)
    return params


def _bitsum(param):
    flat = param.detach().reshape(param.shape[0], -1)
    return flat.view(torch.int32).to(torch.int64).sum(dim=1)


def _nonzero_grad(param):
    grad = param.grad
    width = param.shape[0]
    if grad is None:
        return torch.zeros(width, dtype=torch.bool, device=param.device)
    return grad.detach().reshape(width, -1).ne(0).any(dim=1)


_CATEGORIES = (
    "first_only",
    "second_only",
    "overlap",
    "unselected",
    "single_selected",
    "single_unselected",
)


def _blank():
    return {
        "slots": 0,
        "passed_to_optimizer": 0,
        "nonzero_grad": 0,
        "state_delta_0": 0,
        "state_delta_1": 0,
        "state_delta_other": 0,
        "bitsum_changed": 0,
        "state_step_bitsum_unchanged": 0,
        "no_state_step_bitsum_changed": 0,
        "nonzero_grad_no_state_step": 0,
        "nonzero_grad_state_step_bitsum_changed": 0,
    }


class ReplayTelemetry:
    """Aggregated per-slice counters for a bounded number of updates."""

    def __init__(self, model, opt, max_steps):
        self.params = _expert_params(model, opt)
        if not self.params:
            raise RuntimeError("REPLAY_TELEMETRY_NO_EXPERT_PARAMS")
        device = self.params[0].device
        self.n_slots = sum(param.shape[0] for param in self.params)
        self.max_steps = int(max_steps)
        self.grad_nonzero = torch.zeros(self.max_steps, self.n_slots, dtype=torch.bool, device=device)
        self.bitsum_changed = torch.zeros(self.max_steps, self.n_slots, dtype=torch.bool, device=device)
        self.in_first = torch.zeros(self.max_steps, self.n_slots, dtype=torch.bool)
        self.in_second = torch.zeros(self.max_steps, self.n_slots, dtype=torch.bool)
        self.in_passed = torch.zeros(self.max_steps, self.n_slots, dtype=torch.bool)
        self.second_ran = torch.zeros(self.max_steps, dtype=torch.bool)
        self.state_delta = torch.zeros(self.max_steps, self.n_slots, dtype=torch.int16)
        self._step = 0
        self._open = False
        self._clocks = {}
        self._bits = {}

    def _same_model(self, model, opt):
        found = [id(param) for param in _expert_params(model, opt)]
        if found != [id(param) for param in self.params]:
            raise RuntimeError("REPLAY_TELEMETRY_MODEL")

    def begin_update(self, model, opt):
        """Snapshot clocks and slice bit-sums before the existing step."""
        if self._open or self._step >= self.max_steps:
            raise RuntimeError("REPLAY_TELEMETRY_BUFFER")
        self._same_model(model, opt)
        self._clocks = {}
        self._bits = {}
        for param in self.params:
            state = opt.state[id(param)]
            self._clocks[id(param)] = state["expert_steps"].detach().to("cpu").clone()
            self._bits[id(param)] = _bitsum(param)
        self._open = True

    def note_grads(self, model):
        """Record post-clip nonzero gradients. Call this after the one clip."""
        if not self._open:
            raise RuntimeError("REPLAY_TELEMETRY_ORDER")
        offset = 0
        for param in self.params:
            width = param.shape[0]
            self.grad_nonzero[self._step, offset:offset + width] = _nonzero_grad(param)
            offset += width

    def finish_update(self, model, opt, first, second, passed):
        """Record the step's clock delta, write signature, and selections."""
        if not self._open:
            raise RuntimeError("REPLAY_TELEMETRY_ORDER")
        self._same_model(model, opt)
        row = self._step
        offset = 0
        self.second_ran[row] = second is not None
        for param in self.params:
            width = param.shape[0]
            ident = id(param)
            changed = _bitsum(param).ne(self._bits[ident])
            self.bitsum_changed[row, offset:offset + width] = changed
            after = opt.state[ident]["expert_steps"].detach().to("cpu")
            delta = (after - self._clocks[ident]).to(torch.int16)
            self.state_delta[row, offset:offset + width] = delta
            first_set = set(first.get(ident, ()))
            second_set = set() if second is None else set(second.get(ident, ()))
            passed_set = set(passed.get(ident, ()))
            for expert in range(width):
                slot = offset + expert
                self.in_first[row, slot] = expert in first_set
                self.in_second[row, slot] = expert in second_set
                self.in_passed[row, slot] = expert in passed_set
            offset += width
        self._open = False
        self._step += 1
        self._clocks = {}
        self._bits = {}

    def summary(self):
        """Fold device-side write and gradient bits into integer counters once."""
        if self._open:
            raise RuntimeError("REPLAY_TELEMETRY_ORDER")
        done = self._step
        grad = self.grad_nonzero[:done].to("cpu").tolist()
        wrote = self.bitsum_changed[:done].to("cpu").tolist()
        first = self.in_first[:done].tolist()
        second = self.in_second[:done].tolist()
        passed = self.in_passed[:done].tolist()
        delta = self.state_delta[:done].tolist()
        ran = self.second_ran[:done].tolist()
        categories = {name: _blank() for name in _CATEGORIES}
        mismatches = 0
        for row in range(done):
            for slot in range(self.n_slots):
                in_first = bool(first[row][slot])
                in_second = bool(second[row][slot])
                in_passed = bool(passed[row][slot])
                if ran[row]:
                    if in_first and in_second:
                        name = "overlap"
                    elif in_first:
                        name = "first_only"
                    elif in_second:
                        name = "second_only"
                    else:
                        name = "unselected"
                    expected = in_first or in_second
                else:
                    name = "single_selected" if in_first else "single_unselected"
                    expected = in_first
                if in_passed != expected:
                    mismatches += 1
                bucket = categories[name]
                change = int(delta[row][slot])
                nonzero = bool(grad[row][slot])
                bit_changed = bool(wrote[row][slot])
                bucket["slots"] += 1
                bucket["passed_to_optimizer"] += int(in_passed)
                bucket["nonzero_grad"] += int(nonzero)
                if change == 0:
                    bucket["state_delta_0"] += 1
                elif change == 1:
                    bucket["state_delta_1"] += 1
                else:
                    bucket["state_delta_other"] += 1
                bucket["bitsum_changed"] += int(bit_changed)
                if change >= 1 and not bit_changed:
                    bucket["state_step_bitsum_unchanged"] += 1
                if change == 0 and bit_changed:
                    bucket["no_state_step_bitsum_changed"] += 1
                if nonzero and change == 0:
                    bucket["nonzero_grad_no_state_step"] += 1
                if nonzero and change >= 1 and bit_changed:
                    bucket["nonzero_grad_state_step_bitsum_changed"] += 1
        return {
            "updates": done,
            "expert_parameters": len(self.params),
            "slots_per_update": self.n_slots,
            "union_mismatches": mismatches,
            "categories": categories,
            "coverage": (
                "every expert-sliced parameter and every expert index on every "
                "completed update; not a sample of experts or steps"
            ),
            "semantics": {
                "selection": "expert index was in that forward's expert_selection for this parameter",
                "passed_to_optimizer": "expert index was in the dict passed to SliceAdam.step",
                "nonzero_grad": "after the combined backward and the one clip, some element of that slice gradient was nonzero",
                "state_delta": "increase in SliceAdam expert_steps for that index; not proof the parameter values changed",
                "bitsum_changed": "int64 sum of the float32 bit patterns of the whole slice changed, so at least one stored value changed; an unchanged sum can hide cancelling writes",
                "flush": "gradient and bitsum bits stay on device until summary(); selection and state_delta are counted without a device sync",
                "abort": "summary() is recorded at the end of the arm; an earlier stop does not flush gradient or bitsum cross-tabs",
            },
        }
