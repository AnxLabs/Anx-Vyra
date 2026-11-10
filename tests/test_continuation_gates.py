"""Fail-closed gates of load_checkpoint200, on a synthetic v1 checkpoint of the same schema.

Each gate is exercised separately. The SHA-256 gate must run before torch.load
(proved by making torch.load explode), there is no discovery fallback, and the
disclosed initialization is applied exactly as documented.
"""
import ast
import hashlib
import io
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from resume_world import LIB, World, plan_module_for
import continuation as cont

PACKAGING = Path(cont.__file__).parent


def _v1(world, update=200, cuda=True):
    names = {id(p): n for n, p in world.model.named_parameters()}
    return {
        "update": update,
        "model": {k: v.detach().clone() for k, v in world.model.state_dict().items()},
        "optimizer": {names[k]: {"step": update if v["expert_steps"] is None else 0, "exp_avg": v["exp_avg"].clone(),
                                 "exp_avg_sq": v["exp_avg_sq"].clone(),
                                 "expert_steps": None if v["expert_steps"] is None else torch.full_like(v["expert_steps"], 3)}
                      for k, v in world.opt.state.items()},
        "torch_rng": torch.get_rng_state().clone(),
        "cuda_rng": [torch.zeros(16, dtype=torch.uint8), torch.zeros(16, dtype=torch.uint8)] if cuda else None,
    }


def _write(tmp_path, ck, name="checkpoint_200.pt"):
    p = tmp_path / name
    torch.save(ck, p)
    raw = p.read_bytes()
    return p, hashlib.sha256(raw).hexdigest(), len(raw)


def _plan200():
    import resume_world
    w = World(0)
    plan = [dict(w.plan[i % 12], slot0=(i * 7) % 6) for i in range(300)]
    return plan, plan_module_for(plan)


def _load(path, sha, nbytes, *, model=None, opt=None, device="cpu", cpu_smoke=True, plan=None, plan_mod=None, w=None):
    w = w or World(1)
    if plan is None:
        plan, plan_mod = _plan200()
    return cont.load_checkpoint200(path, model=model or w.model, opt=opt or w.opt, plan=plan, plan_module=plan_mod,
                                   device=device, cpu_smoke=cpu_smoke, expected_sha=sha, expected_bytes=nbytes,
                                   experts_cls=LIB.Experts)


def test_happy_path_restores_exactly_and_applies_the_documented_init(tmp_path):
    src = World(5); src.run(0, 3)
    ck = _v1(src)
    p, sha, n = _write(tmp_path, ck)
    w = World(1)
    w.model.e._selected = {1}
    rec = _load(p, sha, n, w=w)
    for name, t in w.model.state_dict().items():
        assert torch.equal(t, ck["model"][name])
    names = {id(q): k for k, q in w.model.named_parameters()}
    for ident, st in w.opt.state.items():
        s = ck["optimizer"][names[ident]]
        assert st["step"] == s["step"] and torch.equal(st["exp_avg"], s["exp_avg"]) and torch.equal(st["exp_avg_sq"], s["exp_avg_sq"])
        assert (st["expert_steps"] is None) == (s["expert_steps"] is None)
    assert torch.equal(torch.get_rng_state(), ck["torch_rng"])
    # disclosed init: exactly what DISCLOSURE says
    rec_state = rec["disclosed_init"]
    assert rec_state == {"python_seed": 0, "numpy_seed": 0, "router_layers_reset": 1}
    assert not hasattr(w.model.e, "_selected")
    assert torch.equal(w.model.e._load, torch.zeros(4, dtype=torch.long)) and w.model.e._load.dtype == torch.long
    assert rec["sha256"] == sha and rec["update"] == 200 and rec["device"]["mode"] == "cpu_smoke"
    assert "python_rng" in rec["disclosure"]["not_in_checkpoint_200_and_initialized"]


def test_python_and_numpy_init_state_equals_seed_state(tmp_path):
    src = World(5); ck = _v1(src); p, sha, n = _write(tmp_path, ck)
    random.seed(12345); np.random.seed(54321)
    _load(p, sha, n)
    a = random.getstate(); b = np.random.get_state()
    random.seed(0); np.random.seed(0)
    assert a == random.getstate()
    assert np.array_equal(b[1], np.random.get_state()[1]) and b[2:] == np.random.get_state()[2:]


def test_missing_file_wrong_path_and_directory_fail_closed(tmp_path):
    src = World(5); p, sha, n = _write(tmp_path, _v1(src))
    with pytest.raises(cont.ContinuationGateError, match="CKPT_MISSING"):
        _load(tmp_path / "absent.pt", sha, n)
    with pytest.raises(cont.ContinuationGateError, match="CKPT_MISSING"):
        _load(tmp_path / "sub" / "checkpoint_200.pt", sha, n)          # right name, wrong directory
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "checkpoint_200.pt").write_bytes(p.read_bytes())
    with pytest.raises(cont.ContinuationGateError, match="CKPT_MISSING"):
        _load(tmp_path / "other" / "ckpt.pt", sha, n)                  # a sibling holds a valid copy: still no fallback
    with pytest.raises(cont.ContinuationGateError, match="CKPT_NOT_REGULAR_FILE"):
        _load(tmp_path / "other", sha, n)
    link = tmp_path / "link.pt"; link.symlink_to(p)
    with pytest.raises(cont.ContinuationGateError, match="CKPT_NOT_REGULAR_FILE"):
        _load(link, sha, n)


def test_sha256_gate_runs_before_torch_load(tmp_path, monkeypatch):
    src = World(5); p, sha, n = _write(tmp_path, _v1(src))
    raw = bytearray(p.read_bytes()); raw[len(raw) // 2] ^= 1
    bad = tmp_path / "bad.pt"; bad.write_bytes(bytes(raw))

    def boom(*a, **k):
        raise AssertionError("torch.load was called before the SHA-256 gate")
    monkeypatch.setattr(torch, "load", boom)
    with pytest.raises(cont.ContinuationGateError, match="CKPT_SHA256"):
        _load(bad, sha, n)
    with pytest.raises(cont.ContinuationGateError, match="CKPT_SIZE"):
        _load(p, sha, n + 1)


def test_source_has_no_discovery_retry_or_restart():
    text = (PACKAGING / "continuation.py").read_text()
    tree = ast.parse(text)
    banned_attrs = {"glob", "rglob", "iterdir", "listdir", "scandir", "walk"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in banned_attrs, node.attr
    # the loader reads one explicit path exactly once
    assert text.count("p.read_bytes()") == 1


@pytest.mark.parametrize("mutate,code", [
    (lambda ck: ck.pop("cuda_rng") and None, "CKPT_SCHEMA"),
    (lambda ck: ck.__setitem__("extra", 1), "CKPT_SCHEMA"),
    (lambda ck: ck.__setitem__("update", 201), "CKPT_UPDATE"),
    (lambda ck: ck.__setitem__("update", 199), "CKPT_UPDATE"),
    (lambda ck: ck["model"].pop(next(iter(ck["model"]))), "CKPT_SCHEMA"),
    (lambda ck: ck["model"].__setitem__("embed", ck["model"]["embed"][:3].clone()), "CKPT_SCHEMA"),
    (lambda ck: ck["model"]["embed"].view(-1).__setitem__(0, float("nan")), "CKPT_SCHEMA"),
    (lambda ck: ck["optimizer"].__setitem__("renamed", ck["optimizer"].pop("embed")), "OPTIMIZER_NAME"),
    (lambda ck: ck["optimizer"]["embed"].pop("exp_avg"), "CKPT_SCHEMA"),
    (lambda ck: ck["optimizer"]["embed"].__setitem__("exp_avg", ck["optimizer"]["embed"]["exp_avg"].double()), "CKPT_SCHEMA"),
    (lambda ck: ck["optimizer"]["e.gate"].__setitem__("expert_steps", None), "OPTIMIZER_NAME"),
    (lambda ck: ck["optimizer"]["embed"].__setitem__("step", 7), "CKPT_SCHEMA"),
    (lambda ck: ck.__setitem__("torch_rng", torch.zeros(3, dtype=torch.uint8)), "CKPT_SCHEMA"),
    (lambda ck: ck.__setitem__("cuda_rng", None), "CKPT_SCHEMA"),
])
def test_schema_update_and_optimizer_name_gates(tmp_path, mutate, code):
    src = World(5)
    ck = _v1(src)
    mutate(ck)
    # SHA is computed over the mutated file, so only the schema-level gate can refuse it
    p, sha, n = _write(tmp_path, ck)
    w = World(1)
    with pytest.raises(cont.ContinuationGateError, match=code):
        # cuda_required: use the real (non-smoke) schema rule so cuda_rng gates are active
        cont.load_checkpoint200(p, model=w.model, opt=w.opt, plan=_plan200()[0], plan_module=_plan200()[1], device="cpu",
                                cpu_smoke=False, expected_sha=sha, expected_bytes=n, experts_cls=LIB.Experts)


def test_optimizer_class_name_gate(tmp_path):
    src = World(5); p, sha, n = _write(tmp_path, _v1(src))
    w = World(1)

    class NotSliceAdam:
        state = w.opt.state
    with pytest.raises(cont.ContinuationGateError, match="OPTIMIZER_NAME"):
        cont.load_checkpoint200(p, model=w.model, opt=NotSliceAdam(), plan=_plan200()[0], plan_module=_plan200()[1],
                                device="cpu", cpu_smoke=True, expected_sha=sha, expected_bytes=n, experts_cls=LIB.Experts)


def test_plan_prefix_gate(tmp_path):
    src = World(5); p, sha, n = _write(tmp_path, _v1(src))
    plan, mod = _plan200()
    tampered = [dict(e) for e in plan]; tampered[3] = dict(tampered[3], slot0=(tampered[3]["slot0"] + 1) % 6)
    with pytest.raises(cont.ContinuationGateError, match="PLAN_PREFIX_MISMATCH"):
        _load(p, sha, n, plan=tampered, plan_mod=mod)
    with pytest.raises(cont.ContinuationGateError, match="PLAN_PREFIX_MISMATCH"):
        _load(p, sha, n, plan=plan[:200], plan_mod=mod)      # no entry for update 201
    with pytest.raises(cont.ContinuationGateError, match="PLAN_PREFIX_MISMATCH"):
        _load(p, sha, n, plan=plan[:199], plan_mod=mod)


def test_device_gate(tmp_path):
    src = World(5); p, sha, n = _write(tmp_path, _v1(src))
    # real (non-smoke) mode on this CPU box: params are on cpu, device 'cpu', no CUDA -> refused
    with pytest.raises(cont.ContinuationGateError, match="DEVICE_MISMATCH"):
        _load(p, sha, n, device="cpu", cpu_smoke=False)
    with pytest.raises(cont.ContinuationGateError, match="DEVICE_MISMATCH"):
        _load(p, sha, n, device="cuda", cpu_smoke=False)
    # smoke flag but asking for cuda: refused
    with pytest.raises(cont.ContinuationGateError, match="DEVICE_MISMATCH"):
        _load(p, sha, n, device="cuda", cpu_smoke=True)
    # device-count rule with a faked cuda module: saved 2 states, 1 visible device
    w = World(1)
    ck = _v1(src)
    fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1))
    fake_params = SimpleNamespace(parameters=lambda: [SimpleNamespace(device=SimpleNamespace(type="cuda"))])
    with pytest.raises(cont.ContinuationGateError, match="DEVICE_MISMATCH"):
        cont.check_device(fake_params, ck, device="cuda", torch_mod=fake)
    fake2 = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2))
    assert cont.check_device(fake_params, ck, device="cuda", torch_mod=fake2)["device_count"] == 2


def test_invalid_pickle_content_is_refused_by_weights_only(tmp_path):
    class Evil:
        def __reduce__(self):
            return (eval, ("1+1",))
    p = tmp_path / "evil.pt"
    torch.save({"update": 200, "x": Evil()}, p)
    raw = p.read_bytes()
    with pytest.raises(Exception):
        _load(p, hashlib.sha256(raw).hexdigest(), len(raw))
