"""Gradient-combination schemes (specification only): global == production path, per-task bounds, PCGrad orthogonality, read-only task gradients."""
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "experiment_packaging"))
import clip_scheme as cs  # noqa: E402


def world(seed=0, conflict=False):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4))
    x, y = torch.randn(5, 6) * 3, torch.randint(0, 4, (5,))
    xr, yr = torch.randn(2, 6) * 3, torch.randint(0, 4, (2,))
    if conflict:
        yr = (net(xr).argmax(-1) + 1) % 4
    f = torch.nn.functional.cross_entropy
    return net, list(net.parameters()), f(net(x), y), f(net(xr), yr)


def flat(gs):
    return torch.cat([g.reshape(-1) for g in gs])


def test_global_scheme_equals_the_production_backward_plus_clip():
    for seed in range(4):
        net, params, lx, ly = world(seed)
        gx, gy = cs.task_gradients(params, lx, ly)
        out, rec = cs.combine(params, gx, gy, 0.25, "global")
        (lx + 0.25 * ly).backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        ref = torch.cat([p.grad.reshape(-1) for p in params])
        assert float((flat(out) - ref).norm() / ref.norm()) < 1e-5
        assert rec["update_norm"] <= 1.0 + 1e-6


def test_task_gradients_do_not_touch_grad_state():
    net, params, lx, ly = world(1)
    cs.task_gradients(params, lx, ly)
    assert all(p.grad is None for p in params)
    cs.task_gradients(params, lx, None)
    assert all(p.grad is None for p in params)


def test_pertask_clip_bounds_each_task_and_the_sum():
    for seed in range(4):
        net, params, lx, ly = world(seed)
        gx, gy = cs.task_gradients(params, lx, ly)
        out, rec = cs.combine(params, gx, gy, 0.25, "pertask", tau=0.5)
        nx = rec["alpha_x"] * rec["norm_gx"]; ny = rec["alpha_y"] * rec["norm_gy"]
        assert nx <= 0.5 + 1e-6 and ny <= 0.5 + 1e-6 and flat(out).norm() <= 1.0 + 1e-6 and rec["update_norm"] <= 1.0 + 1e-6
        assert abs(float(flat(out).norm()) - rec["update_norm"]) < 1e-5


def test_pertask_leaves_a_small_task_untouched_and_clips_a_large_one():
    p, q, r = 0.04, 100.0, 0.0     # ||gx||=0.2 (below tau), lambda*||gy||=2.5 (above tau)
    ax, ay = cs.coefficients(p, q, r, 0.25, "pertask", 0.5)
    assert ax == 1.0 and abs(ay * 10.0 - 0.5) < 1e-12      # applied Y contribution norm = alpha_y*||gy|| = tau
    assert cs.coefficients(4.0, 0.0, 0.0, 0.25, "pertask", 0.5) == (0.25, 0.25)    # ||gx||=2 -> 0.25 ; gy=0 -> unclipped (alpha_y = lambda)


def test_pcgrad_projects_conflicting_tasks_apart_and_is_global_when_not_conflicting():
    for seed in range(200):      # deterministic search for a fixture whose two task gradients conflict
        net, params, lx, ly = world(seed)
        gx, gy = cs.task_gradients(params, lx, ly)
        r = float((flat(gx) * flat(gy)).sum())
        if r < 0:
            break
    assert r < 0, "fixture must conflict"
    p, q = float((flat(gx) ** 2).sum()), float((flat(gy) ** 2).sum())
    lam = 0.25
    ax, ay = cs.coefficients(p, q, r, lam, "pcgrad")
    n_before = math.sqrt(p + 2 * lam * r + lam * lam * q)
    c = min(1.0, 1.0 / math.sqrt((1 - lam * r / p) ** 2 * p + 2 * (1 - lam * r / p) * (lam - r / q) * r + (lam - r / q) ** 2 * q))
    # reconstruct the two projected tasks and check the defining property: a' _|_ b and b' _|_ a
    a_proj = flat(gx) - (lam * r / (lam * lam * q)) * lam * flat(gy)
    b_proj = lam * flat(gy) - (lam * r / p) * flat(gx)
    assert abs(float((a_proj * (lam * flat(gy))).sum())) < 1e-4 * math.sqrt(p) * lam * math.sqrt(q) + 1e-6
    assert abs(float((b_proj * flat(gx)).sum())) < 1e-4 * math.sqrt(p) * lam * math.sqrt(q) + 1e-6
    assert abs(ax - c * (1 - lam * r / p)) < 1e-9 and abs(ay - c * (lam - r / q)) < 1e-9 and n_before > 0
    net, params, lx, ly = world(3)
    gx, gy = cs.task_gradients(params, lx, ly)
    p, q, r = (float((flat(gx) ** 2).sum()), float((flat(gy) ** 2).sum()), float((flat(gx) * flat(gy)).sum()))
    if r >= 0:
        assert cs.coefficients(p, q, r, 0.25, "pcgrad") == cs.coefficients(p, q, r, 0.25, "global")


def test_no_replay_means_x_only_global_clip_and_unknown_scheme_fails_closed():
    net, params, lx, _ = world(2)
    gx, _ = cs.task_gradients(params, lx, None)
    out, rec = cs.combine(params, gx, None, 0.25, "pertask")
    assert rec["lambda"] == 0.0 and rec["update_norm"] <= 0.5 + 1e-6
    with pytest.raises(ValueError, match="UNKNOWN_SCHEME"):
        cs.coefficients(1.0, 1.0, 0.0, 0.25, "median")
    with pytest.raises(ValueError, match="NEGATIVE_NORM_SQUARED"):
        cs.coefficients(-1.0, 1.0, 0.0, 0.25, "global")


def test_assign_sets_grads_with_parameter_dtype_and_none_for_unused():
    net, params, lx, ly = world(0)
    gx, gy = cs.task_gradients(params, lx, ly)
    out, _ = cs.combine(params, gx, gy, 0.25, "pertask")
    cs.assign(params, out)
    assert all(p.grad is not None and p.grad.dtype == p.dtype for p in params)
