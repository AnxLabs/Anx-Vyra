"""CPU resume-equivalence regression with negative controls.

N+M steps straight through versus N steps, save a COMPLETE checkpoint with the
production capture/save code, simulate a fresh process (every RNG stream, the
model weights, the optimizer and the router metadata set to wrong values), restore with
the production restore code, run M more steps, and assert EXACT equality (torch.equal / ==;
no tolerance: same ops, same order, same thread count on CPU) of losses,
parameters, optimizer moments and clocks, every RNG stream, router metadata,
mode and sampler position.

Negative controls restore everything EXCEPT one component, using the same
production component functions. The process was really replaced, so skipping a
component leaves a genuinely wrong value in place; each control must produce a
mismatch in the component(s) it drops.
"""
import random

import numpy as np
import pytest
import torch

from resume_world import LIB, World, diff, global_rng_garbage, snapshot
import continuation as cont

N, M = 4, 5


def _straight():
    global_rng_garbage_seed()
    w = World(7)
    w.run(0, N + M)
    return snapshot(w)


def global_rng_garbage_seed():
    torch.manual_seed(0); random.seed(0); np.random.seed(0)


def _checkpoint(tmp_path):
    global_rng_garbage_seed()
    w = World(7)
    w.run(0, N)
    # a sampler/plan position of N: next plan index to run is N
    state = w.capture(N)
    info = cont.save_full_checkpoint(tmp_path / "ck.pt", state)
    return w, info


def _fresh_world_with_wrong_state():
    w = World(999)                       # different weights
    for p in w.model.parameters():       # and different optimizer moments/clocks
        st = w.opt.state[id(p)]
        st["exp_avg"].fill_(0.123); st["exp_avg_sq"].fill_(0.456)
        if st["expert_steps"] is None:
            st["step"] = 77
        else:
            st["expert_steps"].fill_(9)
    w.model._load_probe = None
    w.model.e._load = torch.full((4,), 99, dtype=torch.long)       # wrong router metadata
    w.model.e._selected = {3}
    w.model.eval()                                                  # wrong mode
    global_rng_garbage()                                            # wrong RNG, all streams
    return w


def _restore(w, ck, skip=()):
    comps = dict(cont.COMPONENTS)
    for name in skip:
        comps.pop(name)
    cfg = ck["optimizer_config"]
    assert cfg["class"] == "SliceAdam"
    for name in ("model", "optimizer", "model_mode", "router", "python_rng", "numpy_rng", "torch_rng"):
        if name in comps:
            comps[name](w.model, w.opt, ck, cuda=False)


def test_resume_equivalence_is_exact(tmp_path):
    straight = _straight()
    _w, info = _checkpoint(tmp_path)
    ck = cont.load_full_checkpoint(tmp_path / "ck.pt")
    w = _fresh_world_with_wrong_state()
    w.losses = list(_w.losses)                       # loss history of the first N steps
    cont.restore_full(w.model, w.opt, ck, lib=LIB, plan=w.plan, plan_module=w.plan_mod, replay_batch=2, cuda=False)
    assert ck["sampler"]["next_update_index"] == N
    w.run(ck["sampler"]["next_update_index"], M)
    assert diff(straight, snapshot(w)) == []


def test_checkpoint_is_complete_and_reloadable(tmp_path):
    _w, info = _checkpoint(tmp_path)
    ck = cont.load_full_checkpoint(tmp_path / "ck.pt")
    for key in cont.V2_REQUIRED:
        assert key in ck
    assert ck["schema"] == cont.V2_SCHEMA and ck["update"] == N
    assert set(ck["optimizer_config"]) >= {"class", "B1", "B2", "EPS", "LR", "grad_clip_max_norm", "replay_lambda", "replay_batch", "seed"}
    assert ck["python_rng"]["internal"].numel() == 625 and ck["numpy_rng"]["keys"].numel() == 624
    assert ck["router"]["e"]["load"].sum() > 0 and ck["router"]["e"]["selected"]
    assert ck["sampler"]["next_update_index"] == N and ck["sampler"]["replay_next_idx"] == [(N * 2) % 40, (N * 2 + 1) % 40]
    assert info["checkpoint_bytes"] > 0 and len(info["checkpoint_sha256"]) == 64


CONTROLS = {
    # dropped component -> components that MUST differ at the end
    "model": {"params"},
    "optimizer": {"opt.exp_avg", "opt.step"},
    "torch_rng": {"torch_rng", "losses"},
    "python_rng": {"python_rng", "losses"},
    "numpy_rng": {"numpy_rng", "losses"},
    "router": {"router"},
}


@pytest.mark.parametrize("dropped", sorted(CONTROLS))
def test_negative_control_dropping_state_breaks_equivalence(tmp_path, dropped):
    straight = _straight()
    _w, _ = _checkpoint(tmp_path)
    ck = cont.load_full_checkpoint(tmp_path / "ck.pt")
    w = _fresh_world_with_wrong_state()
    w.losses = list(_w.losses)
    _restore(w, ck, skip=(dropped,))
    w.run(N, M)
    bad = set(diff(straight, snapshot(w)))
    assert bad, f"dropping {dropped} did not change anything: the control cannot fail"
    assert CONTROLS[dropped] <= bad, (dropped, bad)
    print(f"NEGATIVE_CONTROL dropped={dropped} -> mismatch in {sorted(bad)}")


def test_negative_control_optimizer_moments_only(tmp_path):
    straight = _straight()
    _w, _ = _checkpoint(tmp_path)
    ck = cont.load_full_checkpoint(tmp_path / "ck.pt")
    w = _fresh_world_with_wrong_state()
    w.losses = list(_w.losses)
    _restore(w, ck)
    for p in w.model.parameters():          # restore everything, then lose only the moments
        st = w.opt.state[id(p)]
        st["exp_avg"].zero_(); st["exp_avg_sq"].zero_()
    w.run(N, M)
    bad = set(diff(straight, snapshot(w)))
    assert {"opt.exp_avg", "opt.exp_avg_sq", "params"} <= bad
    print(f"NEGATIVE_CONTROL dropped=optimizer_moments -> mismatch in {sorted(bad)}")


def test_negative_control_expert_clocks_only(tmp_path):
    straight = _straight()
    _w, _ = _checkpoint(tmp_path)
    ck = cont.load_full_checkpoint(tmp_path / "ck.pt")
    w = _fresh_world_with_wrong_state()
    w.losses = list(_w.losses)
    _restore(w, ck)
    for p in w.model.parameters():          # restore everything, then lose only the per-expert clocks
        st = w.opt.state[id(p)]
        if st["expert_steps"] is not None:
            st["expert_steps"].zero_()
    w.run(N, M)
    bad = set(diff(straight, snapshot(w)))
    assert {"opt.expert_steps", "params"} <= bad
    print(f"NEGATIVE_CONTROL dropped=expert_clocks -> mismatch in {sorted(bad)}")


def test_negative_control_wrong_sampler_position(tmp_path):
    straight = _straight()
    _w, _ = _checkpoint(tmp_path)
    ck = cont.load_full_checkpoint(tmp_path / "ck.pt")
    w = _fresh_world_with_wrong_state()
    w.losses = list(_w.losses)
    _restore(w, ck)
    w.run(N + 1, M)                          # resume one plan index late
    bad = set(diff(straight, snapshot(w)))
    assert {"losses", "params"} <= bad
    print(f"NEGATIVE_CONTROL dropped=sampler_position -> mismatch in {sorted(bad)}")


def test_negative_control_corrupted_checkpoint_bit(tmp_path):
    straight = _straight()
    _w, _ = _checkpoint(tmp_path)
    ck = cont.load_full_checkpoint(tmp_path / "ck.pt")
    key = next(k for k, v in ck["optimizer"].items() if v["expert_steps"] is None and v["exp_avg"].numel() > 1)
    ck["optimizer"][key]["exp_avg"].view(-1)[0] += 1e-6
    w = _fresh_world_with_wrong_state()
    w.losses = list(_w.losses)
    _restore(w, ck)
    w.run(N, M)
    assert diff(straight, snapshot(w)), "a corrupted moment must not be equivalent"


def test_restore_refuses_mismatched_config_or_sampler(tmp_path):
    _w, _ = _checkpoint(tmp_path)
    ck = cont.load_full_checkpoint(tmp_path / "ck.pt")
    w = _fresh_world_with_wrong_state()
    ck2 = dict(ck); ck2["optimizer_config"] = dict(ck["optimizer_config"], LR=1e-3)
    with pytest.raises(cont.ContinuationGateError, match="OPTIMIZER_CONFIG_MISMATCH"):
        cont.restore_full(w.model, w.opt, ck2, lib=LIB, plan=w.plan, plan_module=w.plan_mod, replay_batch=2, cuda=False)
    ck3 = dict(ck); ck3["sampler"] = dict(ck["sampler"], next_update_index=N + 1)
    with pytest.raises(cont.ContinuationGateError, match="SAMPLER_MISMATCH"):
        cont.restore_full(w.model, w.opt, ck3, lib=LIB, plan=w.plan, plan_module=w.plan_mod, replay_batch=2, cuda=False)
