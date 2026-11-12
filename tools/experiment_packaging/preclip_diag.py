"""Sparse pre-clip gradient reads. No forward, no backward, no optimizer step.

Separate objective gradients come from torch.autograd.grad on losses the
caller already computed. They are not written into .grad. The combined norm
is the value returned by the caller's existing clip_grad_norm_. Reductions
use float64. A zero norm makes cosine null, not zero. Scalar conversion
synchronizes that one value; GPU time for these four updates is unmeasured.
"""
import random

import torch

DIAGNOSTIC_UPDATES = (1, 75, 125, 175, 200)
X64_ROW = 4
# Needle rows in the six-row X batch (batch_at: gen, gen, slot32, then d=32,64,109).
TASK_ROWS = (("X32", 3), ("X64", 4), ("X109", 5))
CLIP_EPSILON = 1e-6
_PENDING = None

CE_DEFINITION = (
    "row_mean_masked_ce is the mean over rows of each row's masked mean NLL; "
    "token_mean_masked_ce is the masked NLL pooled over tokens in the rows"
)


def wanted(step_zero_based):
    return (int(step_zero_based) + 1) in DIAGNOSTIC_UPDATES


def clip_multiplier(total_norm):
    """Match torch.nn.utils.clip_grad_norm_ for max_norm 1: min(1, 1/(norm+1e-6))."""
    value = float(total_norm.detach().to(dtype=torch.float64)) if torch.is_tensor(total_norm) else float(total_norm)
    return value, min(1.0, 1.0 / (value + CLIP_EPSILON))


def _trainable(model):
    return [param for param in model.parameters() if param.requires_grad]


def _backward_hooks(model):
    for module in model.modules():
        for name in ("_backward_hooks", "_backward_pre_hooks"):
            if getattr(module, name, None):
                return True
    for param in model.parameters():
        if getattr(param, "_backward_hooks", None):
            return True
    return False


def _rng():
    cuda = None
    if torch.cuda.is_available():
        cuda = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return (random.getstate(), torch.get_rng_state().clone(), cuda)


def _restore_rng(state):
    random.setstate(state[0])
    torch.set_rng_state(state[1])
    if state[2] is not None:
        torch.cuda.set_rng_state_all(state[2])


def _grad_sig(param):
    grad = param.grad
    if grad is None:
        return None
    return (grad.data_ptr(), grad._version, tuple(grad.shape), str(grad.dtype))


def _route_sig(model):
    found = []
    for module in model.modules():
        if not hasattr(module, "_selected") and not hasattr(module, "_load"):
            continue
        selected = getattr(module, "_selected", None)
        if isinstance(selected, set):
            selected = tuple(sorted(selected))
        load = getattr(module, "_load", None)
        if torch.is_tensor(load):
            load = (load.data_ptr(), load._version, load.detach().cpu().tolist())
        found.append((id(module), selected, load))
    return tuple(found)


def _capture_routes(model):
    saved = []
    for module in model.modules():
        if not hasattr(module, "_selected") and not hasattr(module, "_load"):
            continue
        selected = getattr(module, "_selected", None)
        load = getattr(module, "_load", None)
        saved.append((
            module,
            None if selected is None else set(selected) if isinstance(selected, set) else selected,
            None if load is None else load.detach().clone(),
        ))
    return saved


def _restore_routes(saved):
    for module, selected, load in saved:
        if selected is not None:
            module._selected = set(selected) if isinstance(selected, set) else selected
        if load is not None:
            module._load = load


def _opt_sig(opt):
    if opt is None:
        return None
    rows = []
    for ident, state in opt.state.items():
        clocks = state.get("expert_steps")
        moment = state.get("exp_avg")
        second = state.get("exp_avg_sq")
        rows.append((
            ident,
            state.get("step"),
            None if clocks is None else (clocks.data_ptr(), clocks._version, clocks.detach().cpu().tolist()),
            None if moment is None else (moment.data_ptr(), moment._version),
            None if second is None else (second.data_ptr(), second._version),
        ))
    return tuple(rows)


def _snapshot(model, opt):
    return {
        "versions": tuple(param._version for param in model.parameters()),
        "grads": tuple(_grad_sig(param) for param in model.parameters()),
        "routes": _route_sig(model),
        "opt": _opt_sig(opt),
    }


def _assert_unchanged(model, opt, snap):
    if _snapshot(model, opt) != snap:
        raise RuntimeError("REPLAY_DIAG_STATE_CHANGED")


def _pack(grads):
    parts = []
    for grad in grads:
        if grad is None:
            continue
        parts.append(grad.detach().reshape(-1).to(dtype=torch.float64))
    if not parts:
        return None
    return torch.cat(parts)


def _norm(vector):
    if vector is None or vector.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(vector))


def _cosine(left, right):
    if left is None or right is None:
        return None
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm) == 0.0 or float(right_norm) == 0.0:
        return None
    return float(torch.dot(left, right) / (left_norm * right_norm))


def _selection(model, selection):
    if selection is None:
        return None
    names = {id(param): name for name, param in model.named_parameters()}
    rows = []
    for ident, experts in selection.items():
        rows.append({"parameter": names.get(ident, str(ident)), "experts": list(experts)})
    rows.sort(key=lambda row: row["parameter"])
    return rows


def row_masked_mean_ce(logits, tokens, mask, row):
    """One row of the training CE, using that row's already computed logits."""
    if row < 0 or row >= logits.shape[0]:
        return None
    vocab = logits.shape[-1]
    tokens = tokens.long()
    nll = torch.nn.functional.cross_entropy(
        logits[row:row + 1, :-1].reshape(-1, vocab),
        tokens[row:row + 1, 1:].reshape(-1),
        reduction="none",
    ).view(1, -1)[:, : mask.shape[-1]]
    weight = mask[row:row + 1].to(dtype=nll.dtype)
    return ((nll * weight).sum(1) / weight.sum(1).clamp_min(1)).mean()


def begin_separate(model, opt, logits_x, loss_x, loss_y, tokens_x, mask_x, first, second, passed, weight):
    """Read gx, gy, and the single X64 row from the graph the caller already built."""
    global _PENDING
    if _PENDING is not None:
        raise RuntimeError("REPLAY_DIAG_ORDER")
    if _backward_hooks(model):
        raise RuntimeError("REPLAY_DIAG_BACKWARD_HOOK")
    params = _trainable(model)
    rng = _rng()
    snap = _snapshot(model, opt)
    try:
        gx = torch.autograd.grad(loss_x, params, retain_graph=True, create_graph=False, allow_unused=True)
        gy = None if loss_y is None else torch.autograd.grad(
            loss_y, params, retain_graph=True, create_graph=False, allow_unused=True
        )
        packed_x = _pack(gx)
        packed_y = None if gy is None else _pack(gy)
        x64_norm = None
        x64_cosine = None
        x64_reason = None
        if logits_x is None or logits_x.shape[0] <= X64_ROW:
            x64_reason = "X forward batch has no row 4; X64 cosine omitted"
        else:
            loss_x64 = row_masked_mean_ce(logits_x, tokens_x, mask_x, X64_ROW)
            g64 = torch.autograd.grad(loss_x64, params, retain_graph=True, create_graph=False, allow_unused=True)
            packed_64 = _pack(g64)
            x64_norm = _norm(packed_64)
            x64_cosine = _cosine(packed_64, packed_y)
            del g64, packed_64, loss_x64
        gy_norm = 0.0 if packed_y is None else _norm(packed_y)
        weighted = None if packed_y is None else _norm(packed_y * float(weight))
        per_task = {}
        for task, row in TASK_ROWS:
            if logits_x is None or logits_x.shape[0] <= row:
                per_task[task] = {"row_index": row, "omitted": "X batch has no such row"}
                continue
            loss_t = row_masked_mean_ce(logits_x, tokens_x, mask_x, row)
            g_t = torch.autograd.grad(loss_t, params, retain_graph=True, create_graph=False, allow_unused=True)
            packed_t = _pack(g_t)
            norm_t = _norm(packed_t)
            per_task[task] = {
                "row_index": row,
                "row_ce": float(loss_t.detach()),
                "grad_norm": norm_t,
                "cosine_vs_y": _cosine(packed_t, packed_y),
                "cosine_vs_x_total": _cosine(packed_t, packed_x),
                "weighted_y_over_task_norm": None if weighted is None or norm_t == 0.0 else weighted / norm_t,
            }
            del g_t, packed_t, loss_t
        record = {
            "gx_norm": _norm(packed_x),
            "gy_norm": gy_norm,
            "gy_weighted_norm": weighted,
            "cosine_x_y": _cosine(packed_x, packed_y),
            "gx64_norm": x64_norm,
            "cosine_x64_y": x64_cosine,
            "x64_row_index": X64_ROW,
            "per_task_rows": per_task,
            "per_task_scope": "one sampled needle row per task from the same X forward; row grad is of that row's mean CE, unweighted by the 1/6 batch mean",
            "x64_scope": x64_reason or "one sampled X row, index 4, from the same X forward; not the X64 distribution",
            "weight": float(weight),
            "norm_dtype": "float64",
            "selection_first": _selection(model, first),
            "selection_second": _selection(model, second),
            "selection_passed": _selection(model, passed),
            "device_sync": "scalar conversion of these norms on this update only",
            "gpu_runtime_cost": "unmeasured",
        }
        del gx, gy, packed_x, packed_y
        _assert_unchanged(model, opt, snap)
    finally:
        _restore_rng(rng)
    _PENDING = record
    return record


def finish_combined(total_norm, update_one_indexed):
    """Attach the existing clip return. Does not clip again."""
    global _PENDING
    if _PENDING is None:
        raise RuntimeError("REPLAY_DIAG_ORDER")
    value, multiplier = clip_multiplier(total_norm)
    record = dict(_PENDING)
    _PENDING = None
    record.update({
        "update": int(update_one_indexed),
        "combined_preclip_norm": value,
        "clip_multiplier": multiplier,
        "clip_epsilon": CLIP_EPSILON,
        "clip_semantics": (
            "one global norm of every parameter gradient included in the existing "
            "clip_grad_norm_ call; weighted objective norms are not an update share"
        ),
    })
    return record


def _metrics(model, rows, mask):
    device = next(model.parameters()).device
    tokens = rows.to(device=device, dtype=torch.long)
    logits = model(tokens)
    vocab = logits.shape[-1]
    nll = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, vocab),
        tokens[:, 1:].reshape(-1),
        reduction="none",
    ).view(rows.shape[0], -1)[:, : mask.shape[-1]]
    weight = mask.to(device=device, dtype=nll.dtype)
    token_denom = float(weight.sum())
    row_denom = int(rows.shape[0])
    row_mean = float(((nll * weight).sum(1) / weight.sum(1).clamp_min(1)).mean())
    token_mean = float((nll * weight).sum() / weight.sum().clamp_min(1))
    pred = logits[:, : mask.shape[-1]].argmax(-1)
    target = rows.to(device)[:, 1:1 + mask.shape[-1]]
    correct = (pred == target) & weight.bool()
    tokens = int(weight.sum())
    segments = 0
    segment_hits = 0
    flagged = weight.bool()
    for row in range(rows.shape[0]):
        hits = correct[row][flagged[row]]
        usable = (hits.numel() // 8) * 8
        if usable:
            segments += usable // 8
            segment_hits += int(hits[:usable].view(-1, 8).all(1).sum())
    return {
        "rows": row_denom,
        "token_acc": (tokens and correct.sum().item() / tokens),
        "token_denominator": tokens,
        "segment_exact": (segments and segment_hits / segments),
        "segment_denominator": segments,
        "row_mean_masked_ce": row_mean,
        "token_mean_masked_ce": token_mean,
        "ce_definition": CE_DEFINITION,
    }


def rehearsal_readout(model, rows, mask):
    """no_grad metrics for the original 8 rows and all rehearsal rows.

    Restores mode, RNG, and non-parameter route metadata. Does not enter the
    loss or the optimizer.
    """
    mode = model.training
    rng = _rng()
    routes = _capture_routes(model)
    try:
        model.eval()
        with torch.no_grad():
            original = _metrics(model, rows[:8], mask[:8])
            full = _metrics(model, rows, mask)
    finally:
        _restore_rng(rng)
        _restore_routes(routes)
        model.train(mode)
    return {
        "original8": original,
        "full40": full,
        "scope": "observational held-in rehearsal rows; heldout eval is unchanged and separate",
        "gpu_runtime_cost": "unmeasured",
    }
