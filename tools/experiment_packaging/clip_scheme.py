"""Gradient-combination schemes for the shared-clip study: global clip (as run), per-task clip, PCGrad-then-global.

SPECIFICATION ONLY: not wired into any built notebook. Pure functions over lists of gradient tensors; no state, no RNG, no I/O.
`task_gradients` takes the two task losses from the graph the training step already built (X loss and replay loss) and returns
their per-parameter gradients without touching `.grad` (same read-only discipline as preclip_diag.begin_separate).
`combine` returns the final (already clipped) per-parameter gradients plus a record of the scalar coefficients, which are the only
thing that differs between schemes: u = alpha_x * gx + alpha_y * gy.
  global : alpha = c*(1, lambda), c = min(1, 1/||gx + lambda*gy||)             == loss.backward(); clip_grad_norm_(1.0)
  pertask: alpha = (min(1, tau/||gx||), lambda*min(1, tau/(lambda*||gy||)))     each task clipped to tau; ||u|| <= 2*tau
  pcgrad : if <gx, lambda*gy> < 0 project each task off the other, then c = min(1, 1/||.||)
"""
import math

import torch

SCHEMES = ("global", "pertask", "pcgrad")


def task_gradients(params, loss_x, loss_y):
    gx = torch.autograd.grad(loss_x, params, retain_graph=True, allow_unused=True)
    gy = None if loss_y is None else torch.autograd.grad(loss_y, params, retain_graph=True, allow_unused=True)
    return gx, gy


def _dot(a, b):
    return float(sum((x.double() * y.double()).sum() for x, y in zip(a, b) if x is not None and y is not None))


def coefficients(p, q, r, lam, scheme, tau=0.5):
    """p=<gx,gx>, q=<gy,gy>, r=<gx,gy> (unweighted gradients). Returns (alpha_x, alpha_y)."""
    if scheme not in SCHEMES:
        raise ValueError("UNKNOWN_SCHEME " + str(scheme))
    if not (p >= 0 and q >= 0):
        raise ValueError("NEGATIVE_NORM_SQUARED")
    nrm = lambda ax, ay: math.sqrt(max(ax * ax * p + 2 * ax * ay * r + ay * ay * q, 0.0))
    if scheme == "global":
        n = nrm(1.0, lam)
        c = 1.0 if n <= 1.0 else 1.0 / n
        return c, c * lam
    if scheme == "pertask":
        nx, ny = math.sqrt(p), lam * math.sqrt(q)
        return (1.0 if nx <= tau else tau / nx), lam * (1.0 if ny <= tau else tau / ny)
    # pcgrad
    ax, ay = 1.0, lam
    if lam * r < 0 and p > 0 and q > 0:
        ax, ay = 1.0 - lam * r / p, lam - r / q
    n = nrm(ax, ay)
    c = 1.0 if n <= 1.0 else 1.0 / n
    return c * ax, c * ay


def combine(params, gx, gy, lam, scheme, tau=0.5):
    """Final gradients (list aligned with params, None-safe) and the record. gy None means no replay (lam ignored)."""
    if gy is None:
        gy = tuple(None for _ in gx)
        lam_eff = 0.0
    else:
        lam_eff = lam
    p, q, r = _dot(gx, gx), _dot(gy, gy), _dot(gx, gy)
    ax, ay = coefficients(p, q, r, lam_eff, scheme, tau)
    out = []
    for g, h in zip(gx, gy):
        if g is None and h is None:
            out.append(None)
        else:
            v = torch.zeros_like(g if g is not None else h)
            if g is not None:
                v = v + ax * g
            if h is not None:
                v = v + ay * h
            out.append(v)
    nu = math.sqrt(max(ax * ax * p + 2 * ax * ay * r + ay * ay * q, 0.0))
    rec = {"scheme": scheme, "lambda": lam_eff, "tau": tau, "alpha_x": ax, "alpha_y": ay, "norm_gx": math.sqrt(p), "norm_gy": math.sqrt(q),
           "cos_gx_gy": (r / math.sqrt(p * q)) if p > 0 and q > 0 else None, "update_norm": nu}
    return out, rec


def assign(params, grads):
    for prm, g in zip(params, grads):
        prm.grad = None if g is None else g.to(prm.dtype)
