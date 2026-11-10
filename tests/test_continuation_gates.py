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
    ck = _v1(src)
    assert cont.check_device(_params(), ck, device="cuda", torch_mod=_cuda(2))["device_count"] == 2   # accepted shape


def _cuda(n, names=None, available=True):
    names = names if names is not None else ["Tesla T4"] * n
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available, device_count=lambda: n,
                                                get_device_name=lambda i: names[i]))


def _params(*devs):
    devs = devs or (("cuda", 0),)
    return SimpleNamespace(parameters=lambda: [SimpleNamespace(device=SimpleNamespace(type=t, index=i)) for t, i in devs])


def test_accelerator_constants_match_the_original_run():
    assert cont.MACHINE_SHAPE == "NvidiaTeslaT4" and cont.REQUIRED_CUDA_DEVICES == 2
    assert cont.REQUIRED_GPU_NAME == "Tesla T4" and cont.TRAINING_CUDA_DEVICE == 0
    assert "accelerator" in cont.DISCLOSURE and "cuda:0 only" in cont.DISCLOSURE["accelerator"]["training_device"]


def test_device_gate_accepts_exactly_two_t4_and_records_what_it_saw():
    src = World(5)
    ck = _v1(src)
    rec = cont.check_device(_params(), ck, device="cuda", torch_mod=_cuda(2))
    assert rec == {"mode": "cuda", "device_count": 2, "device_names": ["Tesla T4", "Tesla T4"], "training_device": "cuda:0",
                   "cuda_rng_restored": True, "saved_cuda_states_identical": True}
    ck["cuda_rng"][1] = torch.ones(16, dtype=torch.uint8)
    assert cont.check_device(_params(), ck, device="cuda", torch_mod=_cuda(2))["saved_cuda_states_identical"] is False
    assert cont.check_device(_params(("cuda", None)), ck, device="cuda", torch_mod=_cuda(2))["training_device"] == "cuda:0"


@pytest.mark.parametrize("label, params, fake, nsaved", [
    ("one visible device", _params(), _cuda(1), 2),
    ("three visible devices", _params(), _cuda(3), 2),
    ("zero visible devices", _params(), _cuda(0), 2),
    ("cuda unavailable", _params(), _cuda(2, available=False), 2),
    ("a P100 among the devices", _params(), _cuda(2, ["Tesla T4", "Tesla P100-PCIE-16GB"]), 2),
    ("both not T4", _params(), _cuda(2, ["NVIDIA L4", "NVIDIA L4"]), 2),
    ("params on cuda:1", _params(("cuda", 1)), _cuda(2), 2),
    ("params split over cuda:0 and cuda:1", _params(("cuda", 0), ("cuda", 1)), _cuda(2), 2),
    ("params on cpu", _params(("cpu", None)), _cuda(2), 2),
    ("params mixed cpu and cuda", _params(("cuda", 0), ("cpu", None)), _cuda(2), 2),
    ("saved states != visible (1 saved, 2 visible)", _params(), _cuda(2), 1),
    ("saved states != visible (3 saved, 3 visible, but 2 required)", _params(), _cuda(3), 3),
])
def test_device_gate_negative_cases(label, params, fake, nsaved):
    ck = _v1(World(5))
    ck["cuda_rng"] = [torch.zeros(16, dtype=torch.uint8) for _ in range(nsaved)]
    with pytest.raises(cont.ContinuationGateError, match="DEVICE_MISMATCH"):
        cont.check_device(params, ck, device="cuda", torch_mod=fake)


def test_device_gate_missing_cuda_rng_and_wrong_device_string():
    ck = _v1(World(5), cuda=False)
    with pytest.raises(cont.ContinuationGateError, match="DEVICE_MISMATCH"):
        cont.check_device(_params(), ck, device="cuda", torch_mod=_cuda(2))
    with pytest.raises(cont.ContinuationGateError, match="DEVICE_MISMATCH"):
        cont.check_device(_params(), _v1(World(5)), device="cpu", torch_mod=_cuda(2))


# ---- restoration check at update 200 ---------------------------------------------------------------------
def _derived(name):
    e = cont.BASELINE_200_EVAL[name]
    t = [k for k in range(e["tokens"] + 1) if round(k / e["tokens"], 4) == e["token_acc"]]
    r = [k for k in range(e["records"] + 1) if round(k / e["records"], 4) == e["record_exact"]]
    return t, r


def test_baseline_counts_are_the_unique_integers_behind_the_recorded_evals():
    for name, want in cont.BASELINE_200_COUNTS.items():
        t, r = _derived(name)
        assert t == [want["tokens_correct"]] and r == [want["segments_exact"]], name
        assert want["tokens"] == cont.BASELINE_200_EVAL[name]["tokens"] and want["segments"] == cont.BASELINE_200_EVAL[name]["records"]
    import stop_rule
    assert cont.BASELINE_200_COUNTS["X64_unseen"]["tokens_correct"] == stop_rule.BASELINE_X64["heldout"]["tokens_correct"] == 257
    assert stop_rule.BASELINE_X64["train"]["tokens_correct"] == 669 and cont.RESTORATION_TOLERANCE_TOKENS == 0


class _X(SimpleNamespace):
    pass


def _xck(sha=None):
    return _X(_state_sha=lambda m: sha or cont.BASELINE_200_MODEL_STATE_SHA256)


def _counts():
    return {k: dict(v) for k, v in cont.BASELINE_200_COUNTS.items()}


def _x64():
    import stop_rule
    return {k: dict(v) for k, v in stop_rule.BASELINE_X64.items()}


def _check(sha=None, counts=None, x64=None, strict=True, rounded=None):
    import stop_rule
    return cont.check_baseline_restoration(None, x64 or _x64(), counts or _counts(), rounded or cont.BASELINE_200_EVAL,
                                           strict=strict, x64_ck=_xck(sha), baseline_x64=stop_rule.BASELINE_X64)


def test_restoration_check_passes_on_exact_counts():
    rec = _check()
    assert rec["model_state_matches"] and rec["x64_counts_match"] and rec["eval_counts_match"] and rec["eval_rounded_match"]
    assert rec["tolerance_tokens"] == 0


def test_restoration_check_param_hash_is_always_fatal_even_non_strict():
    for strict in (True, False):
        with pytest.raises(cont.ContinuationGateError, match="RESTORE_CHECK"):
            _check(sha="0" * 64, strict=strict)


@pytest.mark.parametrize("mutate", [
    lambda c, x: x["heldout"].__setitem__("tokens_correct", 258),
    lambda c, x: x["heldout"].__setitem__("tokens_correct", 256),
    lambda c, x: x["train"].__setitem__("tokens_correct", 670),
    lambda c, x: x["train"].__setitem__("tokens_correct", 668),
    lambda c, x: x["heldout"].__setitem__("segments_exact", 1),
    lambda c, x: x["train"].__setitem__("segments_exact", 1),
    lambda c, x: c["Y72_unseen"].__setitem__("tokens_correct", 1562),
    lambda c, x: c["Y72_unseen"].__setitem__("segments_exact", 164),
    lambda c, x: c["X32_unseen"].__setitem__("tokens_correct", 2376),
    lambda c, x: c["X109_unseen"].__setitem__("segments_exact", 81),
    lambda c, x: c["X64_unseen"].__setitem__("tokens_correct", 258),
    lambda c, x: c.pop("X109_unseen"),
])
def test_restoration_check_is_exact_one_off_counts_stop_a_strict_run(mutate):
    c, x = _counts(), _x64()
    mutate(c, x)
    with pytest.raises(cont.ContinuationGateError, match="RESTORE_CHECK"):
        _check(counts=c, x64=x, strict=True)
    rec = _check(counts=c, x64=x, strict=False)         # CPU smoke records, never stops
    assert not (rec["x64_counts_match"] and rec["eval_counts_match"])
    assert rec["x64_counts_diff"] or rec["eval_counts_diff"]


def test_invalid_pickle_content_is_refused_by_weights_only(tmp_path):
    class Evil:
        def __reduce__(self):
            return (eval, ("1+1",))
    p = tmp_path / "evil.pt"
    torch.save({"update": 200, "x": Evil()}, p)
    raw = p.read_bytes()
    with pytest.raises(Exception):
        _load(p, hashlib.sha256(raw).hexdigest(), len(raw))


def test_committed_restoration_prereg_matches_the_code():
    import json
    import stop_rule
    reg = json.loads((Path(__file__).resolve().parents[1] / "prereg" / "restoration_check_200.json").read_text())
    exact = reg["gpu_recomputed_counts_must_equal_exactly"]
    assert exact["x64_heldout"] == stop_rule.BASELINE_X64["heldout"] and exact["x64_train"] == stop_rule.BASELINE_X64["train"]
    for k, v in exact["retention_counts"].items():
        assert v == cont.BASELINE_200_COUNTS[k]
    assert reg["always_fatal_hardware_independent"]["model_state_sha256"] == cont.BASELINE_200_MODEL_STATE_SHA256
    assert reg["always_fatal_hardware_independent"]["also"][-1].endswith("c3ac1155e0e21b25133d7e76aa63589a98e53d3e80ee3b0c645daa3fcd365ae7")
    assert exact["tolerance"].startswith("0 tokens") and cont.RESTORATION_TOLERANCE_TOKENS == 0


def test_restore_v1_applies_every_saved_cuda_state_when_the_gate_says_so(monkeypatch):
    w = World(1)
    ck = _v1(World(5))
    ck["model"] = {k: v.clone() for k, v in w.model.state_dict().items()}
    ck["optimizer"] = {n: {"step": 0, "exp_avg": torch.zeros_like(p), "exp_avg_sq": torch.zeros_like(p), "expert_steps": None}
                       for n, p in w.model.named_parameters()}
    seen = {}
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: seen.setdefault("states", [s.clone() for s in states]))
    cont.restore_v1(w.model, w.opt, ck, device="cpu", cuda_restore=True)
    assert len(seen["states"]) == 2 and all(torch.equal(a, b) for a, b in zip(seen["states"], ck["cuda_rng"]))
    seen.clear()
    cont.restore_v1(w.model, w.opt, ck, device="cpu", cuda_restore=False)
    assert "states" not in seen
