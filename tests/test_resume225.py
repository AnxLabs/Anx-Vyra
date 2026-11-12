"""Fail-closed resume from checkpoint_225 and the replay-weight change (lambda 0.25 -> 0.5), tiny hermetic world.

The real checkpoint is pinned by SHA-256 in resume225.py; these tests use a synthetic v2 checkpoint of the same schema
(pinned constants monkeypatched to its values) to exercise every gate branch, plus the resume-equivalence controls for
the one changed variable.
"""
import ast
import hashlib
import sys
from pathlib import Path

import pytest
import torch

from resume_world import ARGS, LIB, N_STEPS_PLAN, World, diff, global_rng_garbage, plan_module_for, snapshot
import continuation as cont
import resume225 as r
import y72fix_gates as g

PACKAGING = Path(cont.__file__).parent
N, M = 4, 5


def _plan():
    w = World(0)
    plan = [dict(w.plan[i % N_STEPS_PLAN], slot0=(i * 7) % 6) for i in range(300)]
    return plan, plan_module_for(plan)


def _make(tmp_path, monkeypatch, *, update=225, lam=0.25, mutate=None, name="checkpoint_225.pt"):
    plan, pm = _plan()
    w = World(5)
    w.run(0, N)
    state = cont.capture_full_state(w.model, w.opt, update=update, plan=plan, plan_module=pm, lib=LIB,
                                    provenance={"test": "tiny"}, **dict(ARGS, replay_lambda=lam))
    cfg = {k: v for k, v in state["optimizer_config"].items() if k != "replay_lambda"}   # the pinned, unmutated config
    p = tmp_path / name
    if mutate:
        mutate(state)
        torch.save(state, p)                         # bypass save_full_checkpoint's completeness refusal on purpose
        raw = p.read_bytes()
        info = {"checkpoint_sha256": hashlib.sha256(raw).hexdigest(), "checkpoint_bytes": len(raw)}
    else:
        info = cont.save_full_checkpoint(p, state)
    monkeypatch.setattr(r, "SAVED_CONFIG_EXCEPT_LAMBDA", cfg)
    monkeypatch.setattr(r, "PINNED_PREFIX_PLAN_SHA256", pm.plan_sha256(plan[:200]))
    monkeypatch.setattr(r, "PINNED_FULL_PLAN_SHA256", pm.plan_sha256(plan))
    return w, p, info["checkpoint_sha256"], info["checkpoint_bytes"], plan, pm


def _load(p, sha, nbytes, plan, pm, *, world=None, new_world_seed=1):
    wn = world or World(new_world_seed)
    rec = r.load_checkpoint225(p, model=wn.model, opt=wn.opt, plan=plan, plan_module=pm, lib=LIB, device="cpu",
                               cpu_smoke=True, expected_sha=sha, expected_bytes=nbytes)
    return wn, rec


def test_pinned_constants_are_consistent_with_the_prereg_and_the_gates():
    assert len(r.CKPT225_SHA256) == 64 and r.CKPT225_BYTES == 197237331 and len(r.MODEL_STATE_SHA256_225) == 64
    assert r.NEW_LAMBDA == g.NEW_LAMBDA == 0.5 and r.SAVED_LAMBDA == g.SAVED_LAMBDA == 0.25 and r.RESUME_UPDATE == g.RESUME_UPDATE == 225
    assert r.X64_225["heldout"] == g.RESUME_X64_HELDOUT and r.COUNTS_225["X64_unseen"] == g.RESUME_X64_HELDOUT
    assert r.CKPT225_PATH == r.CKPT225_MOUNT + "/checkpoint_225.pt" and r.CKPT225_MOUNT.endswith(r.CKPT225_DATASET_ID)
    assert r.CKPT225_DATASET_ID != cont.CKPT200_DATASET_ID
    for name, floor in (("Y72_unseen", 160), ("X32_unseen", 128), ("X109_unseen", 90)):
        assert r.COUNTS_225[name]["segments_exact"] >= floor        # the resume point itself is above every floor
    assert r.COUNTS_225["Y72_unseen"]["segments_exact"] == 168 and r.COUNTS_225["Y72_unseen"]["segments"] == 200
    assert r.PINNED_PREFIX_PLAN_SHA256 == "c3ac1155e0e21b25133d7e76aa63589a98e53d3e80ee3b0c645daa3fcd365ae7"


def test_happy_path_restores_every_component_exactly_and_records_the_one_change(tmp_path, monkeypatch):
    w, p, sha, n, plan, pm = _make(tmp_path, monkeypatch)
    want = snapshot(w)
    global_rng_garbage()
    wn, rec = _load(p, sha, n, plan, pm)
    assert rec["update"] == 225 and rec["saved_lambda"] == 0.25 and rec["new_lambda"] == 0.5
    assert rec["restored_components"] == list(cont.COMPONENTS) and rec["device"]["mode"] == "cpu_smoke"
    got = snapshot(wn)
    got["losses"] = want["losses"]
    assert diff(want, got) == []


def test_source_and_paths_have_no_discovery_retry_or_second_checkpoint():
    text = (PACKAGING / "resume225.py").read_text()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"glob", "rglob", "iterdir", "listdir", "scandir", "walk", "read_bytes"}
    assert text.count("cont.read_verified_bytes(") == 1 and "checkpoint_200" not in text.replace("checkpoint_200 ", "")
    assert text.index("cont.read_verified_bytes(") < text.index("cont.load_full_checkpoint(") < text.index("    validate_v2_resume(ck, cuda_required")
    assert text.index("    validate_v2_resume(ck, cuda_required") < text.index("cont.restore_full(")


def test_missing_wrong_path_directory_and_symlink_fail_closed(tmp_path, monkeypatch):
    w, p, sha, n, plan, pm = _make(tmp_path, monkeypatch)
    with pytest.raises(cont.ContinuationGateError, match="CKPT_MISSING"):
        _load(tmp_path / "absent.pt", sha, n, plan, pm)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "checkpoint_225.pt").write_bytes(p.read_bytes())
    with pytest.raises(cont.ContinuationGateError, match="CKPT_MISSING"):
        _load(tmp_path / "other" / "ckpt.pt", sha, n, plan, pm)        # a sibling holds a valid copy: still no fallback
    with pytest.raises(cont.ContinuationGateError, match="CKPT_NOT_REGULAR_FILE"):
        _load(tmp_path / "other", sha, n, plan, pm)
    link = tmp_path / "link.pt"; link.symlink_to(p)
    with pytest.raises(cont.ContinuationGateError, match="CKPT_NOT_REGULAR_FILE"):
        _load(link, sha, n, plan, pm)


def test_sha256_is_checked_before_torch_load(tmp_path, monkeypatch):
    w, p, sha, n, plan, pm = _make(tmp_path, monkeypatch)
    raw = bytearray(p.read_bytes()); raw[len(raw) // 2] ^= 1
    bad = tmp_path / "bad.pt"; bad.write_bytes(bytes(raw))

    def boom(*a, **k):
        raise AssertionError("torch.load before the SHA-256 gate")
    monkeypatch.setattr(torch, "load", boom)
    with pytest.raises(cont.ContinuationGateError, match="CKPT_SHA256"):
        _load(bad, sha, n, plan, pm)
    with pytest.raises(cont.ContinuationGateError, match="CKPT_SIZE"):
        _load(p, sha, n + 1, plan, pm)
    with pytest.raises(cont.ContinuationGateError, match="CKPT_SHA256"):
        _load(p, hashlib.sha256(b"x").hexdigest(), n, plan, pm)            # the file is intact but is not the pinned one


def test_the_checkpoint_200_file_can_not_be_mistaken_for_225(tmp_path, monkeypatch):
    w, p, sha, n, plan, pm = _make(tmp_path, monkeypatch)
    q = tmp_path / "checkpoint_200.pt"; q.write_bytes(b"\x00" * 64)
    with pytest.raises(cont.ContinuationGateError, match="CKPT_SIZE"):
        _load(q, sha, n, plan, pm)


@pytest.mark.parametrize("label, kwargs, code", [
    ("update 224", dict(update=224), "CKPT_UPDATE"),
    ("update 226", dict(update=226), "CKPT_UPDATE"),
    ("update 200", dict(update=200), "CKPT_UPDATE"),
    ("saved lambda 0.5 (not the control)", dict(lam=0.5), "OPTIMIZER_CONFIG_MISMATCH"),
    ("saved lambda 0.3", dict(lam=0.3), "OPTIMIZER_CONFIG_MISMATCH"),
])
def test_update_and_saved_lambda_gates(tmp_path, monkeypatch, label, kwargs, code):
    w, p, sha, n, plan, pm = _make(tmp_path, monkeypatch, **kwargs)
    with pytest.raises(cont.ContinuationGateError, match=code):
        _load(p, sha, n, plan, pm)


@pytest.mark.parametrize("mutate, code", [
    (lambda s: s.pop("cuda_rng") and None, "CKPT_SCHEMA"),
    (lambda s: s.__setitem__("schema", "vyra-continuation-checkpoint-v1"), "CKPT_SCHEMA"),
    (lambda s: s["optimizer_config"].__setitem__("LR", 1e-3), "OPTIMIZER_CONFIG_MISMATCH"),
    (lambda s: s["optimizer_config"].__setitem__("grad_clip_max_norm", 2.0), "OPTIMIZER_CONFIG_MISMATCH"),
    (lambda s: s["optimizer_config"].__setitem__("replay_batch", 3), "OPTIMIZER_CONFIG_MISMATCH"),
    (lambda s: s["optimizer_config"].__setitem__("seed", 1), "OPTIMIZER_CONFIG_MISMATCH"),
    (lambda s: s["optimizer_config"].__setitem__("class", "AdamW"), "OPTIMIZER_CONFIG_MISMATCH"),
    (lambda s: s["sampler"].__setitem__("next_update_index", 226), "SAMPLER_MISMATCH"),
    (lambda s: s["sampler"].__setitem__("plan_full_sha256", "0" * 64), "PLAN_PREFIX_MISMATCH"),
    (lambda s: s["sampler"].__setitem__("plan_prefix200_sha256", "0" * 64), "PLAN_PREFIX_MISMATCH"),
])
def test_schema_config_sampler_and_plan_gates(tmp_path, monkeypatch, mutate, code):
    w, p, sha, n, plan, pm = _make(tmp_path, monkeypatch, mutate=mutate)
    with pytest.raises(cont.ContinuationGateError, match=code):
        _load(p, sha, n, plan, pm)


def test_live_plan_must_be_the_pinned_plan(tmp_path, monkeypatch):
    w, p, sha, n, plan, pm = _make(tmp_path, monkeypatch)
    plan2 = [dict(e) for e in plan]; plan2[250] = dict(plan2[250], slot0=(plan2[250]["slot0"] + 1) % 6)    # extension differs, prefix intact
    with pytest.raises(cont.ContinuationGateError, match="PLAN_PREFIX_MISMATCH"):
        _load(p, sha, n, plan2, pm)
    plan3 = [dict(e) for e in plan]; plan3[10] = dict(plan3[10], slot0=(plan3[10]["slot0"] + 1) % 6)       # prefix differs
    with pytest.raises(cont.ContinuationGateError, match="PLAN_PREFIX_MISMATCH"):
        _load(p, sha, n, plan3, pm)


def test_new_lambda_must_be_exactly_the_preregistered_value(tmp_path, monkeypatch):
    w, p, sha, n, plan, pm = _make(tmp_path, monkeypatch)
    ck = cont.load_full_checkpoint(p)
    r.validate_v2_resume(ck, new_lambda=0.5, cuda_required=False)
    for bad in (0.25, 0.4999999, 0.5000001, 1.0, 0, 0.75):
        with pytest.raises(cont.ContinuationGateError, match="LAMBDA_NOT_PREREGISTERED"):
            r.validate_v2_resume(ck, new_lambda=bad, cuda_required=False)


def test_cuda_states_are_required_outside_cpu_smoke(tmp_path, monkeypatch):
    w, p, sha, n, plan, pm = _make(tmp_path, monkeypatch)
    ck = cont.load_full_checkpoint(p)
    assert ck["cuda_rng"] is None                                           # CPU box: no CUDA state saved
    with pytest.raises(cont.ContinuationGateError, match="CKPT_SCHEMA"):
        r.validate_v2_resume(ck, cuda_required=True)
    ck["cuda_rng"] = [torch.zeros(16, dtype=torch.uint8)]
    with pytest.raises(cont.ContinuationGateError, match="CKPT_SCHEMA"):
        r.validate_v2_resume(ck, cuda_required=True)                         # one device state, two required
    ck["cuda_rng"] = [torch.zeros(16, dtype=torch.uint8)] * 2
    r.validate_v2_resume(ck, cuda_required=True)


def test_real_mode_on_a_cpu_box_is_refused_by_the_device_gate(tmp_path, monkeypatch):
    w, p, sha, n, plan, pm = _make(tmp_path, monkeypatch)
    wn = World(1)
    with pytest.raises(cont.ContinuationGateError, match="CKPT_SCHEMA|DEVICE_MISMATCH"):
        r.load_checkpoint225(p, model=wn.model, opt=wn.opt, plan=plan, plan_module=pm, lib=LIB, device="cuda", cpu_smoke=False,
                             expected_sha=sha, expected_bytes=n)


# ---- restoration check at 225 --------------------------------------------------------------------------------
class _X:
    def __init__(self, sha):
        self._sha = sha

    def _state_sha(self, model):
        return self._sha


def _counts():
    return {k: dict(v) for k, v in r.COUNTS_225.items()}


def _x64():
    return {k: dict(v) for k, v in r.X64_225.items()}


def _check(sha=None, counts=None, x64=None, strict=True):
    return r.check_resume_restoration(None, x64 or _x64(), counts or _counts(), strict=strict, x64_ck=_X(sha or r.MODEL_STATE_SHA256_225))


def test_restoration_passes_on_the_recorded_integers():
    rec = _check()
    assert rec["model_state_matches"] and rec["x64_counts_match"] and rec["eval_counts_match"] and rec["tolerance"] == 0


def test_model_state_hash_is_always_fatal():
    for strict in (True, False):
        with pytest.raises(cont.ContinuationGateError, match="RESTORE_CHECK"):
            _check(sha="0" * 64, strict=strict)


@pytest.mark.parametrize("mutate", [
    lambda c, x: x["heldout"].__setitem__("tokens_correct", 1419),
    lambda c, x: x["heldout"].__setitem__("tokens_correct", 1417),
    lambda c, x: x["heldout"].__setitem__("segments_exact", 12),
    lambda c, x: x["heldout"].__setitem__("segments_exact", 10),
    lambda c, x: x["train"].__setitem__("tokens_correct", 1563),
    lambda c, x: x["train"].__setitem__("segments_exact", 27),
    lambda c, x: c["Y72_unseen"].__setitem__("segments_exact", 169),
    lambda c, x: c["Y72_unseen"].__setitem__("segments_exact", 167),
    lambda c, x: c["Y72_unseen"].__setitem__("tokens_correct", 1564),
    lambda c, x: c["X32_unseen"].__setitem__("segments_exact", 161),
    lambda c, x: c["X32_unseen"].__setitem__("tokens_correct", 2335),
    lambda c, x: c["X109_unseen"].__setitem__("segments_exact", 101),
    lambda c, x: c["X109_unseen"].__setitem__("tokens_correct", 935),
    lambda c, x: c["X64_unseen"].__setitem__("tokens_correct", 1419),
    lambda c, x: c.pop("X109_unseen"),
])
def test_restoration_is_exact_one_off_counts_stop_a_strict_run(mutate):
    c, x = _counts(), _x64()
    mutate(c, x)
    with pytest.raises(cont.ContinuationGateError, match="RESTORE_CHECK"):
        _check(counts=c, x64=x, strict=True)
    rec = _check(counts=c, x64=x, strict=False)                  # CPU smoke records, never stops
    assert not (rec["x64_counts_match"] and rec["eval_counts_match"]) and (rec["x64_counts_diff"] or rec["eval_counts_diff"])


# ---- resume equivalence with the one changed variable -------------------------------------------------------
def _ref_switched(lam_after):
    """Straight run: N steps at lambda 0.25, then M steps at lam_after, no interruption."""
    global_rng_garbage()
    w = World(7)
    w.run(0, N)
    w.lam = lam_after
    w.run(N, M)
    return w, snapshot(w)


def _checkpoint_at_n(tmp_path):
    global_rng_garbage()
    w = World(7)
    w.run(0, N)
    info = cont.save_full_checkpoint(tmp_path / "ck.pt", w.capture(N))
    return w, info


def _resume(tmp_path, lam):
    w0, _ = _checkpoint_at_n(tmp_path)
    ck = cont.load_full_checkpoint(tmp_path / "ck.pt")
    w = World(999)
    for p in w.model.parameters():
        st = w.opt.state[id(p)]
        st["exp_avg"].fill_(0.1); st["exp_avg_sq"].fill_(0.2)
        if st["expert_steps"] is None:
            st["step"] = 55
        else:
            st["expert_steps"].fill_(8)
    global_rng_garbage()
    cont.restore_full(w.model, w.opt, ck, lib=LIB, plan=w.plan, plan_module=w.plan_mod, replay_batch=2, cuda=False)
    w.losses = list(w0.losses)
    w.lam = lam
    w.run(ck["sampler"]["next_update_index"], M)
    return w


def test_resume_with_lambda_change_equals_a_straight_run_that_switches_lambda(tmp_path):
    _w, ref = _ref_switched(0.5)
    got = snapshot(_resume(tmp_path, 0.5))
    assert diff(ref, got) == []


def test_negative_control_resuming_without_the_lambda_change_differs(tmp_path):
    """If lambda were not actually applied after the resume, the equivalence with the switched run must break."""
    _w, ref = _ref_switched(0.5)
    bad = diff(ref, snapshot(_resume(tmp_path, 0.25)))
    assert "losses" in bad and "params" in bad and "opt.exp_avg" in bad


def test_lambda_changes_nothing_about_the_restored_state_or_the_first_forward(tmp_path):
    """At the resume point the two lambdas are indistinguishable; the first update's lx and ly are identical, and
    only the combined gradient (hence parameters after the step) differs."""
    a, b = _resume(tmp_path, 0.25), _resume(tmp_path, 0.5)
    la, lb = a.losses[N], b.losses[N]
    assert la == lb                                                       # same (lx, ly) at the first update after the resume
    assert not torch.equal(a.model.embed, b.model.embed) or not torch.equal(a.model.head, b.model.head)
    ck_a = cont.capture_full_state(*_two(a), update=N, plan=a.plan, plan_module=a.plan_mod, lib=LIB, provenance={}, **dict(ARGS))
    assert ck_a["optimizer_config"]["replay_lambda"] == 0.25


def _two(w):
    return (w.model, w.opt)


def test_checkpoints_written_after_the_change_record_lambda_0_5(tmp_path):
    w = _resume(tmp_path, 0.5)
    state = cont.capture_full_state(w.model, w.opt, update=N + M, plan=w.plan, plan_module=w.plan_mod, lib=LIB, provenance={}, **dict(ARGS, replay_lambda=0.5))
    assert state["optimizer_config"]["replay_lambda"] == 0.5 and state["sampler"]["next_update_index"] == N + M
    assert state["optimizer_config"] != dict(state["optimizer_config"], replay_lambda=0.25)
