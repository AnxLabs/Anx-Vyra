"""Update-200 -> 300 continuation: verified restore, disclosed init, full-state
checkpoints and a state-invariant observer guard.

Nothing here trains, steps an optimizer, or touches Kaggle. The notebook calls
these functions; the CPU regression imports the very same functions.

Restoration contract
--------------------
checkpoint_200.pt (schema v1) holds ONLY: update, model, optimizer (name keyed
SliceAdam state), torch_rng, cuda_rng. Everything else is NOT in it and cannot
be recovered. `apply_disclosed_init` sets those pieces to the fixed values in
DISCLOSURE and the run records exactly that. Checkpoints written by
`save_full_checkpoint` (schema v2) hold every piece, and `restore_full`
restores them exactly.
"""
import hashlib
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

CKPT200_SHA256 = "8542c3405309a5fcf9e419a6e3dd733beab346765fa7bd9cbdd61226dd867e81"
CKPT200_BYTES = 197225718
CKPT200_DATASET_ID = "anjorosimoaj/vyra-expanded40-200u-checkpoint"
CKPT200_MOUNT = "/kaggle/input/datasets/anjorosimoaj/vyra-expanded40-200u-checkpoint"
CKPT200_PATH = CKPT200_MOUNT + "/checkpoint_200.pt"
RESUME_UPDATE = 200
FINAL_UPDATE = 300
CHECKPOINT_UPDATES = (225, 250, 275, 300)
V1_KEYS = ("update", "model", "optimizer", "torch_rng", "cuda_rng")
OPT_STATE_KEYS = ("exp_avg", "exp_avg_sq", "expert_steps", "step")
V2_SCHEMA = "vyra-continuation-checkpoint-v2"
V2_REQUIRED = (
    "schema", "update", "model", "optimizer", "optimizer_config", "torch_rng", "cuda_rng",
    "python_rng", "numpy_rng", "router", "model_training", "sampler", "provenance",
)
PYTHON_SEED_INIT = 0
NUMPY_SEED_INIT = 0

DISCLOSURE = {
    "approved_by": "AJ, FORGE chat, Nov 7 2026 ~1:02 PM: accept the disclosed non-restored state",
    "restored_exactly_from_checkpoint_200": [
        "model parameters (90 tensors, strict)",
        "SliceAdam per-parameter step, exp_avg, exp_avg_sq, expert_steps (90 states, strict)",
        "torch CPU RNG state",
        "torch CUDA RNG states (all saved devices; device count must match)",
    ],
    "not_in_checkpoint_200_and_initialized": {
        "python_rng": {
            "was": "state of random after random.seed(0) plus any draws during updates 1..200; not saved",
            "now": f"random.seed({PYTHON_SEED_INIT})",
            "exact_only_if": "the training path made zero Python random draws in updates 1..200 (source audit found none; not proven from the checkpoint)",
        },
        "numpy_rng": {
            "was": "legacy global numpy RNG, never seeded by the trainer, process-entropy seeded; unrecoverable",
            "now": f"numpy.random.seed({NUMPY_SEED_INIT})",
            "exact_only_if": "never; no numpy randomness found in the audited training path, so the value is expected to be inert, not proven",
        },
        "router__selected": {
            "was": "set of experts chosen by the last forward of update 200; overwritten by every forward",
            "now": "attribute absent (as in a fresh model); the first forward of update 201 sets it before any read",
        },
        "router__load": {
            "was": "cumulative per-layer expert-load counts over all forwards of updates 1..200 (train and eval); diagnostic only, never read for routing or optimization",
            "now": "zeros(N_EXPERTS, int64) per Experts layer; counts therefore cover updates 201.. only",
        },
        "sampler_position": {
            "was": "not saved",
            "now": "reconstructed: next plan index 200 (update 201); replay rows (200*RB+j)%40 = [0, 1]",
        },
        "optimizer_config": {
            "was": "not saved",
            "now": "constants of the pinned library: B1/B2/EPS from run_vyra1_copy, LR re-set to 3e-4 before every update (constant, no schedule), grad clip 1.0, replay lambda 0.25, RB 2, seed 0",
        },
        "model_mode": {
            "was": "not saved",
            "now": "train() at the start of every update, as in the original loop; eval() only inside observers, restored afterward",
        },
    },
    "new_checkpoints_225_250_275_300_capture": list(V2_REQUIRED),
}


class ContinuationGateError(RuntimeError):
    """Fail-closed refusal. The message starts with a stable CODE."""


class ObserverStateChanged(RuntimeError):
    pass


def _fail(code, detail=""):
    raise ContinuationGateError(code + (": " + str(detail) if detail != "" else ""))


# ---------------------------------------------------------------- hashing
def tensor_bytes(tensor):
    t = tensor.detach().cpu().contiguous().reshape(-1)
    return t.view(torch.uint8).numpy().tobytes() if t.numel() else b""


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def state_digest(model, opt):
    """Per-component sha256 of params, grads, and every optimizer tensor/clock."""
    names = {id(p): n for n, p in model.named_parameters()}
    h_params, h_grads, h_opt = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    for name, p in model.named_parameters():
        h_params.update(name.encode()); h_params.update(tensor_bytes(p))
        h_grads.update(name.encode())
        h_grads.update(b"none" if p.grad is None else tensor_bytes(p.grad))
    if opt is not None:
        for ident, state in opt.state.items():
            h_opt.update(names.get(ident, str(ident)).encode())
            h_opt.update(repr(int(state["step"])).encode())
            for key in ("exp_avg", "exp_avg_sq", "expert_steps"):
                v = state.get(key)
                h_opt.update(b"none" if v is None else tensor_bytes(v))
    return {"params": h_params.hexdigest(), "grads": h_grads.hexdigest(), "optimizer": h_opt.hexdigest()}


# ---------------------------------------------------------------- RNG / router capture
def _cuda_states():
    return [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None


def capture_python_rng():
    version, internal, gauss = random.getstate()
    return {"version": int(version), "internal": torch.tensor(internal, dtype=torch.int64), "gauss_next": gauss}


def restore_python_rng(rec):
    random.setstate((int(rec["version"]), tuple(int(x) for x in rec["internal"].tolist()), rec["gauss_next"]))


def capture_numpy_rng():
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {"bit_generator": str(name), "keys": torch.from_numpy(keys.astype(np.int64)), "pos": int(pos),
            "has_gauss": int(has_gauss), "cached_gaussian": float(cached)}


def restore_numpy_rng(rec):
    if rec["bit_generator"] != "MT19937":
        _fail("NUMPY_RNG_KIND", rec["bit_generator"])
    np.random.set_state(("MT19937", rec["keys"].numpy().astype(np.uint32), int(rec["pos"]),
                         int(rec["has_gauss"]), float(rec["cached_gaussian"])))


def capture_torch_rng():
    return torch.get_rng_state().clone(), _cuda_states()


def restore_torch_rng(cpu, cuda):
    torch.set_rng_state(cpu.clone())
    if cuda is not None:
        torch.cuda.set_rng_state_all([s.clone() for s in cuda])


def capture_router(model):
    out = {}
    for name, module in model.named_modules():
        if not hasattr(module, "_selected") and not hasattr(module, "_load"):
            continue
        selected = getattr(module, "_selected", None)
        load = getattr(module, "_load", None)
        out[name] = {
            "selected": None if selected is None else sorted(int(i) for i in selected),
            "load": None if load is None else load.detach().cpu().clone(),
        }
    return out


def restore_router(model, rec):
    modules = dict(model.named_modules())
    for name in [n for n, m in modules.items() if (hasattr(m, "_selected") or hasattr(m, "_load")) and n not in rec]:
        for attr in ("_selected", "_load"):
            if hasattr(modules[name], attr):
                delattr(modules[name], attr)
    for name, entry in rec.items():
        if name not in modules:
            _fail("ROUTER_MODULE_MISSING", name)
        m = modules[name]
        if entry["selected"] is None:
            if hasattr(m, "_selected"):
                delattr(m, "_selected")
        else:
            m._selected = set(int(i) for i in entry["selected"])
        if entry["load"] is None:
            if hasattr(m, "_load"):
                delattr(m, "_load")
        else:
            m._load = entry["load"].clone()


def module_modes(model):
    return tuple(bool(m.training) for m in model.modules())


def set_module_modes(model, modes):
    for m, flag in zip(model.modules(), modes):
        m.training = flag


# ---------------------------------------------------------------- observer guard
def _full_snapshot(model, opt):
    cpu, cuda = capture_torch_rng()
    py = capture_python_rng()
    npr = capture_numpy_rng()
    router = capture_router(model)
    return {
        "digest": state_digest(model, opt),
        "py": (py["version"], tuple(py["internal"].tolist()), py["gauss_next"]),
        "np": (npr["bit_generator"], tuple(npr["keys"].tolist()), npr["pos"], npr["has_gauss"], npr["cached_gaussian"]),
        "torch_cpu": tensor_bytes(cpu),
        "torch_cuda": None if cuda is None else [tensor_bytes(s) for s in cuda],
        "router": {k: (v["selected"], None if v["load"] is None else (tuple(v["load"].tolist()), v["load"].dtype)) for k, v in router.items()},
        "modes": module_modes(model),
        "_restore": (py, npr, cpu, cuda, router, module_modes(model)),
    }


class observe:
    """Context manager for observers (eval, readouts, X64 dump, checkpoint save).

    On exit it RESTORES RNG (python, numpy, torch cpu/cuda), route metadata and
    module modes, then verifies that parameters, gradients, optimizer state,
    RNG, routes and modes are bit-identical to entry. It cannot undo a write
    to params/grads/optimizer; that raises ObserverStateChanged, which stops
    the run (fail closed).
    """

    def __init__(self, model, opt):
        self.model, self.opt = model, opt

    def __enter__(self):
        self.before = _full_snapshot(self.model, self.opt)
        return self

    def __exit__(self, exc_type, exc, tb):
        py, npr, cpu, cuda, router, modes = self.before["_restore"]
        restore_python_rng(py); restore_numpy_rng(npr); restore_torch_rng(cpu, cuda)
        restore_router(self.model, router); set_module_modes(self.model, modes)
        if exc_type is not None:
            return False
        after = _full_snapshot(self.model, self.opt)
        changed = [k for k in after if k != "_restore" and after[k] != self.before[k]]
        if "digest" in changed:
            changed = [k for k in changed if k != "digest"] + [
                "digest." + k for k in after["digest"] if after["digest"][k] != self.before["digest"][k]]
        if changed:
            raise ObserverStateChanged("OBSERVER_CHANGED_STATE " + ",".join(changed))
        return False


# ---------------------------------------------------------------- verified load (schema v1)
def read_verified_bytes(path, expected_sha=CKPT200_SHA256, expected_bytes=CKPT200_BYTES):
    """SHA-256 gate. No glob, no sibling lookup, no retry. Runs BEFORE torch.load."""
    p = Path(path)
    if not p.exists():
        _fail("CKPT_MISSING", p)
    if p.is_symlink() or not p.is_file():
        _fail("CKPT_NOT_REGULAR_FILE", p)
    raw = p.read_bytes()
    if expected_bytes is not None and len(raw) != expected_bytes:
        _fail("CKPT_SIZE", f"{len(raw)} != {expected_bytes}")
    got = sha256_bytes(raw)
    if got != expected_sha:
        _fail("CKPT_SHA256", f"{got} != {expected_sha}")
    return raw


def _named_params(model):
    return dict(model.named_parameters())


def validate_v1_schema(ck, model, opt, *, expected_update=RESUME_UPDATE, cuda_required=True):
    if not isinstance(ck, dict) or tuple(sorted(ck)) != tuple(sorted(V1_KEYS)):
        _fail("CKPT_SCHEMA", f"top-level keys {sorted(ck) if isinstance(ck, dict) else type(ck)}")
    if type(ck["update"]) is not int or ck["update"] != expected_update:
        _fail("CKPT_UPDATE", f"{ck['update']!r} != {expected_update}")
    if type(opt).__name__ != "SliceAdam":
        _fail("OPTIMIZER_NAME", f"live optimizer is {type(opt).__name__}, need SliceAdam")
    sd = model.state_dict()
    if set(ck["model"]) != set(sd):
        _fail("CKPT_SCHEMA", "model tensor names differ")
    for name, t in sd.items():
        c = ck["model"][name]
        if not torch.is_tensor(c) or c.shape != t.shape or c.dtype != t.dtype:
            _fail("CKPT_SCHEMA", f"model tensor {name} shape/dtype")
        if c.is_floating_point() and not bool(torch.isfinite(c).all()):
            _fail("CKPT_SCHEMA", f"model tensor {name} non-finite")
    params = _named_params(model)
    if not isinstance(ck["optimizer"], dict) or set(ck["optimizer"]) != set(params):
        _fail("OPTIMIZER_NAME", "checkpoint optimizer state names differ from model parameter names")
    for name, p in params.items():
        s, live = ck["optimizer"][name], opt.state[id(p)]
        if not isinstance(s, dict) or tuple(sorted(s)) != OPT_STATE_KEYS:
            _fail("CKPT_SCHEMA", f"optimizer keys for {name}")
        for key in ("exp_avg", "exp_avg_sq"):
            if not torch.is_tensor(s[key]) or s[key].shape != p.shape or s[key].dtype != p.dtype:
                _fail("CKPT_SCHEMA", f"optimizer {name}.{key} shape/dtype")
            if not bool(torch.isfinite(s[key]).all()):
                _fail("CKPT_SCHEMA", f"optimizer {name}.{key} non-finite")
        if type(s["step"]) is not int or s["step"] < 0:
            _fail("CKPT_SCHEMA", f"optimizer {name}.step")
        if (s["expert_steps"] is None) != (live["expert_steps"] is None):
            _fail("OPTIMIZER_NAME", f"{name} expert slicing differs from the live optimizer")
        if s["expert_steps"] is not None:
            e = s["expert_steps"]
            if not torch.is_tensor(e) or e.dtype != torch.long or tuple(e.shape) != (p.shape[0],):
                _fail("CKPT_SCHEMA", f"optimizer {name}.expert_steps")
            if int(e.max()) > expected_update or int(e.min()) < 0:
                _fail("CKPT_SCHEMA", f"optimizer {name}.expert_steps range")
        elif s["step"] != expected_update:
            _fail("CKPT_SCHEMA", f"optimizer {name}.step {s['step']} != update {expected_update}")
    t = ck["torch_rng"]
    if not torch.is_tensor(t) or t.dtype != torch.uint8 or t.shape != torch.get_rng_state().shape:
        _fail("CKPT_SCHEMA", "torch_rng")
    c = ck["cuda_rng"]
    if cuda_required:
        if not isinstance(c, list) or not c or not all(torch.is_tensor(x) and x.dtype == torch.uint8 for x in c):
            _fail("CKPT_SCHEMA", "cuda_rng")


def check_device(model, ck, *, device, cpu_smoke=False, torch_mod=torch):
    """Device gate: must be CUDA with the saved device count, unless the explicit CPU smoke flag is set."""
    on = {p.device.type for p in model.parameters()}
    if cpu_smoke:
        if device != "cpu" or on != {"cpu"}:
            _fail("DEVICE_MISMATCH", f"cpu smoke but device={device} params on {sorted(on)}")
        return {"mode": "cpu_smoke", "cuda_rng_restored": False}
    if device != "cuda" or on != {"cuda"}:
        _fail("DEVICE_MISMATCH", f"need cuda, device={device} params on {sorted(on)}")
    if not torch_mod.cuda.is_available():
        _fail("DEVICE_MISMATCH", "cuda unavailable")
    saved = ck["cuda_rng"]
    if saved is None or torch_mod.cuda.device_count() != len(saved):
        _fail("DEVICE_MISMATCH", f"visible cuda devices {torch_mod.cuda.device_count()} != saved cuda_rng states {None if saved is None else len(saved)}")
    return {"mode": "cuda", "device_count": len(saved), "cuda_rng_restored": True}


def check_plan_prefix(plan, *, plan_module, resume_update=RESUME_UPDATE):
    if len(plan) < resume_update + 1:
        _fail("PLAN_PREFIX_MISMATCH", f"plan length {len(plan)} cannot serve update {resume_update + 1}")
    if plan_module.plan_sha256(plan[:resume_update]) != plan_module.PINNED_PREFIX_PLAN_SHA256:
        _fail("PLAN_PREFIX_MISMATCH", "plan[0:200] hash differs from the pinned train_X.pt plan")


def apply_disclosed_init(model, experts_cls, *, python_seed=PYTHON_SEED_INIT, numpy_seed=NUMPY_SEED_INIT):
    """The disclosed initialization of everything checkpoint_200 does not hold."""
    random.seed(python_seed)
    np.random.seed(numpy_seed)
    n_layers = 0
    for module in model.modules():
        if isinstance(module, experts_cls):
            if hasattr(module, "_selected"):
                delattr(module, "_selected")
            module._load = torch.zeros(module.gate.shape[0], dtype=torch.long)
            n_layers += 1
    return {"python_seed": python_seed, "numpy_seed": numpy_seed, "router_layers_reset": n_layers}


def restore_v1(model, opt, ck, *, device, cuda_restore):
    """Strict model + optimizer restore and torch RNG from a validated v1 checkpoint."""
    model.load_state_dict(ck["model"], strict=True)
    names = {n: p for n, p in model.named_parameters()}
    with torch.no_grad():
        for name, p in names.items():
            s, live = ck["optimizer"][name], opt.state[id(p)]
            live["step"] = int(s["step"])
            live["exp_avg"] = s["exp_avg"].clone().to(p.device)
            live["exp_avg_sq"] = s["exp_avg_sq"].clone().to(p.device)
            live["expert_steps"] = None if s["expert_steps"] is None else s["expert_steps"].clone()
    torch.set_rng_state(ck["torch_rng"].clone())
    if cuda_restore:
        torch.cuda.set_rng_state_all([s.clone() for s in ck["cuda_rng"]])


def load_checkpoint200(path, *, model, opt, plan, plan_module, device, cpu_smoke=False,
                       expected_sha=CKPT200_SHA256, expected_bytes=CKPT200_BYTES, experts_cls=None,
                       torch_mod=torch):
    """Everything fail-closed, in this order. Returns an audit record.

    1 sha256/size of the exact path (BEFORE torch.load) 2 torch.load(weights_only)
    of the verified bytes, never the path 3 schema, update, optimizer names
    4 plan prefix 5 device 6 restore + disclosed init.
    """
    raw = read_verified_bytes(path, expected_sha, expected_bytes)
    ck = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    validate_v1_schema(ck, model, opt, cuda_required=not cpu_smoke)
    check_plan_prefix(plan, plan_module=plan_module)
    dev = check_device(model, ck, device=device, cpu_smoke=cpu_smoke, torch_mod=torch_mod)
    restore_v1(model, opt, ck, device=device, cuda_restore=dev["cuda_rng_restored"])
    init = apply_disclosed_init(model, experts_cls)
    return {"path": str(path), "sha256": expected_sha, "bytes": len(raw), "update": ck["update"],
            "device": dev, "disclosed_init": init, "disclosure": DISCLOSURE}


# ---------------------------------------------------------------- schema v2 (full state)
def sampler_record(update, plan, plan_module, replay_batch, replay_rows=40):
    nxt = int(update)
    return {
        "next_update_index": nxt,
        "next_update_number": nxt + 1,
        "replay_rule": f"idx=(s*{replay_batch}+j)%{replay_rows}",
        "replay_next_idx": [(nxt * replay_batch + j) % replay_rows for j in range(replay_batch)],
        "plan_length": len(plan),
        "plan_prefix200_sha256": plan_module.plan_sha256(plan[:RESUME_UPDATE]),
        "plan_full_sha256": plan_module.plan_sha256(plan),
        "plan_extension_policy": plan_module.POLICY_VERSION,
    }


def optimizer_config(lib, *, replay_lambda, replay_batch, seed, clip, opt):
    return {"class": type(opt).__name__, "B1": float(lib.B1), "B2": float(lib.B2), "EPS": float(lib.EPS),
            "LR": float(lib.LR), "grad_clip_max_norm": float(clip), "replay_lambda": float(replay_lambda),
            "replay_batch": int(replay_batch), "seed": int(seed)}


def capture_full_state(model, opt, *, update, plan, plan_module, lib, replay_lambda, replay_batch, seed, clip, provenance):
    names = {id(p): n for n, p in model.named_parameters()}
    cpu, cuda = capture_torch_rng()
    return {
        "schema": V2_SCHEMA,
        "update": int(update),
        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer": {names[k]: {"step": int(v["step"]), "exp_avg": v["exp_avg"].detach().cpu().clone(),
                                 "exp_avg_sq": v["exp_avg_sq"].detach().cpu().clone(),
                                 "expert_steps": None if v["expert_steps"] is None else v["expert_steps"].detach().cpu().clone()}
                      for k, v in opt.state.items()},
        "optimizer_config": optimizer_config(lib, replay_lambda=replay_lambda, replay_batch=replay_batch, seed=seed, clip=clip, opt=opt),
        "torch_rng": cpu, "cuda_rng": cuda,
        "python_rng": capture_python_rng(), "numpy_rng": capture_numpy_rng(),
        "router": capture_router(model),
        "model_training": module_modes(model)[0],
        "sampler": sampler_record(update, plan, plan_module, replay_batch),
        "provenance": provenance,
    }


def save_full_checkpoint(path, state):
    missing = [k for k in V2_REQUIRED if k not in state]
    if missing:
        _fail("CKPT_INCOMPLETE", missing)
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, str(tmp))
    tmp.replace(path)
    raw = path.read_bytes()
    return {"checkpoint_file": path.name, "checkpoint_bytes": len(raw), "checkpoint_sha256": sha256_bytes(raw)}


def load_full_checkpoint(path_or_bytes):
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else str(path_or_bytes)
    ck = torch.load(src, map_location="cpu", weights_only=True)
    if ck.get("schema") != V2_SCHEMA or any(k not in ck for k in V2_REQUIRED):
        _fail("CKPT_SCHEMA", "not a complete v2 checkpoint")
    return ck


# Components in restore order. A caller (and the negative controls) can compose
# them; restore_full runs all of them.
def _r_model(model, opt, ck, **_):
    model.load_state_dict(ck["model"], strict=True)


def _r_optimizer(model, opt, ck, **_):
    names = dict(model.named_parameters())
    if set(ck["optimizer"]) != set(names):
        _fail("OPTIMIZER_NAME", "names differ")
    for name, p in names.items():
        s, live = ck["optimizer"][name], opt.state[id(p)]
        live["step"] = int(s["step"])
        live["exp_avg"] = s["exp_avg"].clone().to(p.device)
        live["exp_avg_sq"] = s["exp_avg_sq"].clone().to(p.device)
        live["expert_steps"] = None if s["expert_steps"] is None else s["expert_steps"].clone()


def _r_torch(model, opt, ck, *, cuda=True, **_):
    restore_torch_rng(ck["torch_rng"], ck["cuda_rng"] if cuda else None)


def _r_python(model, opt, ck, **_):
    restore_python_rng(ck["python_rng"])


def _r_numpy(model, opt, ck, **_):
    restore_numpy_rng(ck["numpy_rng"])


def _r_router(model, opt, ck, **_):
    restore_router(model, ck["router"])


def _r_mode(model, opt, ck, **_):
    model.train(bool(ck["model_training"]))


COMPONENTS = {"model": _r_model, "optimizer": _r_optimizer, "torch_rng": _r_torch, "python_rng": _r_python,
              "numpy_rng": _r_numpy, "router": _r_router, "model_mode": _r_mode}


def restore_full(model, opt, ck, *, lib, plan, plan_module, replay_batch, cuda=True):
    cfg = ck["optimizer_config"]
    live = optimizer_config(lib, replay_lambda=cfg["replay_lambda"], replay_batch=replay_batch, seed=cfg["seed"],
                            clip=cfg["grad_clip_max_norm"], opt=opt)
    if live != cfg:
        _fail("OPTIMIZER_CONFIG_MISMATCH", f"{live} != {cfg}")
    if ck["sampler"] != sampler_record(ck["update"], plan, plan_module, replay_batch):
        _fail("SAMPLER_MISMATCH", "sampler position/plan hash differs")
    for name in ("model", "optimizer", "model_mode", "router", "python_rng", "numpy_rng", "torch_rng"):
        COMPONENTS[name](model, opt, ck, cuda=cuda)
    return {"update": ck["update"], "components": list(COMPONENTS)}


# ---------------------------------------------------------------- integer-count eval + X64 dump
@torch.no_grad()
def eval_counts(model, pairs, device, *, nonfinite=None):
    """Exact integer counts (same scoring as the notebook's ev())."""
    model.eval()
    ok = n = seg = segs = 0
    for i in range(0, len(pairs), 4):
        x = torch.stack([p[0] for p in pairs[i:i + 4]]).long().to(device)
        mm = torch.stack([p[1] for p in pairs[i:i + 4]]).bool().to(device)
        lg = model(x)[:, :510].float()
        if not bool(torch.isfinite(lg).all()):
            if nonfinite:
                nonfinite("nonfinite eval")
            _fail("EVAL_NONFINITE")
        pr = lg.argmax(-1)
        tg = x[:, 1:511]
        for r in range(x.shape[0]):
            o = (pr[r] == tg[r])[mm[r]]
            ok += int(o.sum()); n += o.numel()
            f = (o.numel() // 8) * 8
            if f:
                seg += int(o[:f].view(-1, 8).all(1).sum()); segs += f // 8
    return {"tokens_correct": ok, "tokens": n, "segments_exact": seg, "segments": segs}


# Pinned measurements of checkpoint_200 from the completed 200-update run
# (result.json / x64_predictions_200.json / CHECKPOINT_VERIFICATION).
BASELINE_200_MODEL_STATE_SHA256 = "35bd2f486abf383965b93ffe08ff5db38e0873a6874bb039c3a188c30b2a0b34"
BASELINE_200_EVAL = {
    "Y72_unseen": {"token_acc": 0.9756, "record_exact": 0.825, "tokens": 1600, "records": 200},
    "X32_unseen": {"token_acc": 0.9285, "record_exact": 0.5406, "tokens": 2560, "records": 320},
    "X64_unseen": {"token_acc": 0.1339, "record_exact": 0.0, "tokens": 1920, "records": 240},
    "X109_unseen": {"token_acc": 0.9458, "record_exact": 0.6667, "tokens": 960, "records": 120},
}


def observe_counts(model, *, XD, HO, EV, plan, update, device, x64_ck):
    """Teacher-forced X64 readouts + exact integer counts. Call inside `observe`."""
    model.eval()
    with torch.no_grad():
        first = x64_ck.train_x64_rows(plan, FINAL_UPDATE)
        idx = sorted(first)
        train = x64_ck._readout(model, [XD[64]["rows"][k] for k in idx], [XD[64]["mask"][k] for k in idx], device)
        for k, row in zip(idx, train["rows"]):
            row["train_index"] = k
            row["first_presented_update"] = first[k]
            row["presented_by_this_update"] = first[k] <= int(update)
        held = x64_ck._readout(model, list(XD[64]["unseen"]), list(XD[64]["unseen_mask"]), device)
    counts = {k: eval_counts(model, HO[k], device) for k in EV}
    x64 = {
        "train": {"tokens_correct": sum(r["byte_correct"] for r in train["rows"]), "tokens": train["bytes"],
                  "segments_exact": sum(r["segments_exact"] for r in train["rows"]), "segments": train["segments"]},
        "heldout": {"tokens_correct": sum(r["byte_correct"] for r in held["rows"]), "tokens": held["bytes"],
                    "segments_exact": sum(r["segments_exact"] for r in held["rows"]), "segments": held["segments"]},
    }
    return train, held, counts, x64


def check_baseline_restoration(model, x64, eval_rounded, *, strict, x64_ck, baseline_x64):
    """After restore, the model must reproduce the recorded update-200 measurements.

    The parameter sha256 is hardware independent and always fatal. The eval
    comparisons depend on the device's argmax and are fatal only when `strict`
    (the real T4 run); the CPU smoke records them.
    """
    sha = x64_ck._state_sha(model)
    rec = {"model_state_sha256": sha, "model_state_matches": sha == BASELINE_200_MODEL_STATE_SHA256}
    if not rec["model_state_matches"]:
        _fail("RESTORE_CHECK", f"model_state_sha256 {sha} != {BASELINE_200_MODEL_STATE_SHA256}")
    rec["x64_counts_match"] = all(
        x64[k]["tokens_correct"] == baseline_x64[k]["tokens_correct"] and x64[k]["segments_exact"] == baseline_x64[k]["segments_exact"]
        for k in ("train", "heldout"))
    rec["eval_rounded_match"] = eval_rounded == BASELINE_200_EVAL
    if strict and not (rec["x64_counts_match"] and rec["eval_rounded_match"]):
        _fail("RESTORE_CHECK", f"restored model does not reproduce recorded update-200 evals: {rec}")
    return rec


def dump_full(model, opt, update, *, XD, HO, EV, plan, plan_module, out_dir, diag, x64_ck, lib, replay_lambda,
              replay_batch, seed, clip, provenance, device):
    """Predictions, retention counts and the complete v2 checkpoint at one update.

    Observer part runs inside `observe`, so RNG/routes/mode are restored and
    params/grads/optimizer are verified unchanged before the checkpoint is
    captured. IO failures are recorded; observer violations raise.
    """
    t0 = time.time()
    out_dir = Path(out_dir)
    record = {"update": int(update)}
    with observe(model, opt):
        train, held, counts, x64 = observe_counts(model, XD=XD, HO=HO, EV=EV, plan=plan, update=update, device=device, x64_ck=x64_ck)
    record.update({"x64": x64, "counts": counts})
    try:
        pred = out_dir / f"x64_predictions_{int(update)}.json"
        pred.write_text(json.dumps({"update": int(update), "train": train, "heldout": held, "counts": record,
                                    "scope": "teacher-forced argmax over masked answer positions, same as ev()"}))
        record["predictions_file"] = pred.name
        state = capture_full_state(model, opt, update=update, plan=plan, plan_module=plan_module, lib=lib,
                                   replay_lambda=replay_lambda, replay_batch=replay_batch, seed=seed, clip=clip,
                                   provenance=provenance)
        record.update(save_full_checkpoint(out_dir / f"checkpoint_{int(update)}.pt", state))
    except Exception as exc:  # IO failures are recorded; the run itself continues
        record["checkpoint_error"] = repr(exc)
    record["dump_s"] = time.time() - t0
    return record
