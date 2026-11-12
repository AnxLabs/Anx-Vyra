"""y72fix2 continuation 300 -> 400: gates, resume gate, plan pins, runtime glue, generator edits and the built notebook (env-gated).

Pinned to prereg/y72fix2_cont400.json (registered before any of this code existed).
"""
import ast
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "experiment_packaging"))

import continuation as cont  # noqa: E402
import continue225 as c5  # noqa: E402
import continue225_y72fix2 as c2  # noqa: E402
import continue300_y72fix2 as c4  # noqa: E402
import plan_extension as pe  # noqa: E402
import plan_extension400 as pe4  # noqa: E402
import resume300_y72fix2 as r3  # noqa: E402
import y72fix2_cont400_gates as gates  # noqa: E402
import y72fix2_cont400_runtime as rt  # noqa: E402

PREREG = ROOT / "prereg" / "y72fix2_cont400.json"
PREREG_SHA256 = "f71193df06d7fa5be41d3e6409d3f9cc65db53459e6f2e606c0bebd184868c41"
P = json.loads(PREREG.read_text())
G = P["gates"]


def test_prereg_file_is_the_registered_one():
    assert hashlib.sha256(PREREG.read_bytes()).hexdigest() == PREREG_SHA256
    import build_y72fix2_cont400 as b
    assert b.PREREG_SHA256 == PREREG_SHA256 and b.PREREG_FILE == PREREG


def test_gate_module_matches_the_registration():
    assert gates.RULE_ID == P["rule_id"]
    assert list(gates.CHECKPOINTS) == P["gate_checkpoints"] if "gate_checkpoints" in P else list(gates.CHECKPOINTS) == P["design"]["gate_checkpoints"]
    assert list(gates.OBSERVATION_ONLY) == [P["design"]["observation_only_checkpoint"]]
    assert 3 * gates.Y600_FLOOR_TENTHS_PCT * 600 // 1000 == G["R_hold"]["sum3"] == 1440
    assert gates.Y600_GUARD_TENTHS_PCT * 600 // 1000 == G["R_hold"]["each_checkpoint"] == 432
    assert gates.X64_HELDOUT_SEG_FLOOR == G["A_progress"]["x64_seg_each"] == 44
    assert gates.X64_HELDOUT_GAIN_TENTHS_PP == G["A_progress"]["gain_tenths_pp"] == 30
    assert gates.RESUME_X64_HELDOUT == G["A_progress"]["resume_x64_heldout"]
    assert 320 * 400 // 1000 == G["N_nonregression"]["x32_min"] and 120 * 750 // 1000 == G["N_nonregression"]["x109_min"]
    assert gates.TASK_CLIP_TENTHS * 10 == round(P["design"]["task_clip"] * 100) and gates.REPLAY_LAMBDA_NUM_DEN == (1, 4)
    for (r, a, n), v in gates.VERDICTS.items():
        assert v == ("SUCCESS" if r and a and n else "FAIL_" + "_".join(k for k, ok in (("HOLD", r), ("PROGRESS", a), ("NONREGRESSION", n)) if not ok))
    for key in ("SUCCESS", "FAIL_HOLD", "FAIL_PROGRESS", "FAIL_NONREGRESSION", "NOT_EVALUATED_BEFORE_400"):
        assert key in P["verdict_table"]


def _h(y=(480, 480, 480), seg=(44, 44, 44), tok=1723, x32=128, x109=90):
    return {u: {"y600": y[i], "x64_seg": seg[i], "x64_tokens": tok if u == 400 else 1700, "x32": x32, "x109": x109} for i, u in enumerate((350, 375, 400))}


def test_boundaries_are_exactly_the_registered_ones():
    assert gates.evaluate(_h())["verdict"] == "SUCCESS"
    assert gates.evaluate(_h(y=(480, 480, 479)))["verdict"] == "FAIL_HOLD"            # sum 1439
    assert gates.evaluate(_h(y=(432, 504, 504)))["verdict"] == "SUCCESS"              # guard 432, sum 1440
    assert gates.evaluate(_h(y=(431, 505, 505)))["verdict"] == "FAIL_HOLD"            # guard 431
    assert gates.evaluate(_h(tok=1722))["verdict"] == "FAIL_PROGRESS"
    assert gates.evaluate(_h(seg=(43, 60, 60)))["verdict"] == "FAIL_PROGRESS"
    assert gates.evaluate(_h(seg=(60, 60, 43)))["verdict"] == "FAIL_PROGRESS"
    assert gates.evaluate(_h(x32=127))["verdict"] == "FAIL_NONREGRESSION"
    assert gates.evaluate(_h(x109=89))["verdict"] == "FAIL_NONREGRESSION"
    assert gates.evaluate(_h(y=(400, 400, 400), tok=1600, x109=10))["verdict"] == "FAIL_HOLD_PROGRESS_NONREGRESSION"
    for k, table in G["R_hold"]["boundary"].items():
        for v, want in table.items():
            ys = (int(v) // 3, int(v) // 3, int(v) - 2 * (int(v) // 3)) if k == "sum3" else (int(v), 600, 600)
            assert gates.evaluate(_h(y=ys))["R_hold"] is want
    for v, want in G["A_progress"]["boundary"]["tokens_at_400"].items():
        assert gates.evaluate(_h(tok=int(v)))["A_progress"] is want
    for v, want in G["A_progress"]["boundary"]["seg"].items():
        assert gates.evaluate(_h(seg=(int(v),) * 3))["A_progress"] is want


def test_gate_is_integer_only_and_needs_every_gate_checkpoint():
    h = _h()
    for u in (350, 375, 400):
        part = {k: v for k, v in h.items() if k != u}
        assert gates.evaluate(part)["verdict"] == "NOT_EVALUATED_BEFORE_400"
    assert gates.evaluate({**h, 325: h[350]})["verdict"] == "SUCCESS"          # 325 is observation only
    for bad in (480.0, True, -1, 601):
        hh = _h(); hh[375]["y600"] = bad
        with pytest.raises(ValueError, match="COUNT_RANGE"):
            gates.evaluate(hh)
    src = (ROOT / "tools" / "experiment_packaging" / "y72fix2_cont400_gates.py").read_text()
    tree = ast.parse(src)
    assert not any(isinstance(n, (ast.Div, ast.Pow)) for n in ast.walk(tree))                       # no true division
    assert not any(isinstance(n, ast.Constant) and isinstance(n.value, float) for n in ast.walk(tree))
    assert not any(isinstance(n, ast.Call) and getattr(n.func, "id", None) in ("float", "round") for n in ast.walk(tree))


def test_exhaustive_token_threshold_matches_cross_multiplication():
    for t in range(1600, 1800):
        assert gates.evaluate(_h(tok=t))["A_token_gain_ge_3pp"] == ((t - 1665) * 1000 >= 30 * 1920)
        assert ((t - 1665) * 1000 >= 30 * 1920) == (t >= 1723)


def _rec(y, seg, tok, x32=229, x109=109, ysegs=600):
    return {"x64": {"heldout": {"tokens_correct": tok, "tokens": 1920, "segments_exact": seg, "segments": 240}},
            "counts": {rt.KEY: {"segments_exact": y, "segments": ysegs}, "X32_unseen": {"segments_exact": x32, "segments": 320},
                       "X109_unseen": {"segments_exact": x109, "segments": 120}}}


def test_history_feeds_the_gate_and_refuses_bad_denominators():
    H = {300: _rec(496, 44, 1665), 325: _rec(400, 1, 1000), 350: _rec(490, 50, 1700), 375: _rec(485, 55, 1720), 400: _rec(480, 60, 1740)}
    h = rt.history(H)
    assert sorted(h) == [350, 375, 400] and h[400]["x64_tokens"] == 1740
    assert gates.evaluate(h)["verdict"] == "SUCCESS" and gates.evaluate(h)["sum3"] == 1455
    with pytest.raises(ValueError, match="DENOMINATOR_MISMATCH"):
        rt.history({400: _rec(480, 60, 1740, ysegs=200)})
    assert rt.KEY == "Y72v2_600"


def test_resume_module_matches_the_registration():
    d = P["design"]["resume_from"]
    assert (r3.CKPT300_SHA256, r3.CKPT300_BYTES, r3.MODEL_STATE_SHA256_300) == (d["sha256"], d["bytes"], d["model_state_sha256"])
    assert r3.SAVED_CONFIG_EXCEPT_LAMBDA == {k: v for k, v in d["saved_optimizer_config"].items() if k != "replay_lambda"}
    assert r3.SAVED_LAMBDA == r3.NEW_LAMBDA == d["saved_optimizer_config"]["replay_lambda"] == P["design"]["replay_lambda"] == 0.25
    for k, v in d["saved_sampler"].items():
        key = {"plan_extension_policy": "plan_extension_policy"}.get(k, k)
        assert r3.SAVED_SAMPLER[key] == v
    rc = P["design"]["exact_restoration_check"]
    assert r3.COUNTS_300 == rc["counts_300"] and r3.X64_300 == rc["x64_300"]
    assert r3.CHECKPOINT_UPDATES == (325, 350, 375, 400) and r3.RESUME_UPDATE == 300 and r3.FINAL_UPDATE == 400
    assert r3.CKPT300_PATH.endswith("/checkpoint_300.pt") and "@@SLUG_CKPT300@@" in r3.CKPT300_MOUNT
    # earlier asset modules are untouched (old notebooks keep their hashes)
    assert cont.CHECKPOINT_UPDATES == (225, 250, 275, 300) and cont.FINAL_UPDATE == 300 and pe.FULL_LEN == 300


def _ck(lam=0.25, **over):
    cfg = dict(r3.SAVED_CONFIG_EXCEPT_LAMBDA, replay_lambda=lam)
    ck = {k: None for k in cont.V2_REQUIRED}
    ck.update({"schema": cont.V2_SCHEMA, "update": 300, "optimizer_config": cfg, "cuda_rng": [1, 2], "sampler": dict(r3.SAVED_SAMPLER)})
    ck.update(over)
    return ck


def test_resume_gate_fails_closed():
    r3.validate_v2_resume(_ck())
    with pytest.raises(cont.ContinuationGateError, match="LAMBDA_CHANGED"):
        r3.validate_v2_resume(_ck(), new_lambda=0.5)
    with pytest.raises(cont.ContinuationGateError, match="OPTIMIZER_CONFIG_MISMATCH"):
        r3.validate_v2_resume(_ck(lam=0.5))
    with pytest.raises(cont.ContinuationGateError, match="CKPT_UPDATE"):
        r3.validate_v2_resume(_ck(update=225))
    with pytest.raises(cont.ContinuationGateError, match="CKPT_SCHEMA"):
        r3.validate_v2_resume(_ck(cuda_rng=[1]))
    r3.validate_v2_resume(_ck(cuda_rng=None), cuda_required=False)
    for k, v in (("plan_full_sha256", "0" * 64), ("plan_length", 400), ("next_update_index", 299), ("plan_extension_policy", pe4.POLICY_VERSION)):
        with pytest.raises(cont.ContinuationGateError, match="SAMPLER_MISMATCH"):
            r3.validate_v2_resume(_ck(sampler=dict(r3.SAVED_SAMPLER, **{k: v})))
    bad_cfg = dict(r3.SAVED_CONFIG_EXCEPT_LAMBDA, replay_lambda=0.25, LR=0.001)
    with pytest.raises(cont.ContinuationGateError, match="OPTIMIZER_CONFIG_MISMATCH"):
        r3.validate_v2_resume(_ck(optimizer_config=bad_cfg))


def test_checkpoint_bytes_are_verified_before_any_load(tmp_path):
    f = tmp_path / "checkpoint_300.pt"
    f.write_bytes(b"not the checkpoint")
    with pytest.raises(cont.ContinuationGateError):
        r3.load_checkpoint300(f, model=None, opt=None, plan=[], lib=None, device="cpu")
    with pytest.raises(cont.ContinuationGateError):
        r3.load_checkpoint300(tmp_path / "absent.pt", model=None, opt=None, plan=[], lib=None, device="cpu")


class _X64:
    def __init__(self, sha):
        self.sha = sha

    def _state_sha(self, model):
        return self.sha


def test_restoration_check_is_exact():
    x64 = json.loads(json.dumps(r3.X64_300))
    counts = json.loads(json.dumps(r3.COUNTS_300))
    rec = r3.check_resume_restoration(None, x64, counts, strict=True, x64_ck=_X64(r3.MODEL_STATE_SHA256_300))
    assert rec["model_state_matches"] and rec["x64_counts_match"] and rec["eval_counts_match"] and rec["tolerance"] == 0
    with pytest.raises(cont.ContinuationGateError, match="RESTORE_CHECK"):
        r3.check_resume_restoration(None, x64, counts, strict=False, x64_ck=_X64("0" * 64))      # model hash always fatal
    counts["Y72v2_600"]["segments_exact"] = 495
    with pytest.raises(cont.ContinuationGateError, match="RESTORE_CHECK"):
        r3.check_resume_restoration(None, x64, counts, strict=True, x64_ck=_X64(r3.MODEL_STATE_SHA256_300))
    rec = r3.check_resume_restoration(None, x64, counts, strict=False, x64_ck=_X64(r3.MODEL_STATE_SHA256_300))
    assert not rec["eval_counts_match"] and rec["eval_counts_diff"] == {"Y72v2_600": {"segments_exact": [495, 496]}}
    counts = json.loads(json.dumps(r3.COUNTS_300)); counts.pop("Y72v2_600")
    with pytest.raises(cont.ContinuationGateError, match="RESTORE_CHECK"):
        r3.check_resume_restoration(None, x64, counts, strict=True, x64_ck=_X64(r3.MODEL_STATE_SHA256_300))


def test_plan_continuation_pins_and_prefix_stability():
    pp = P["design"]["plan_extension"]
    assert pe4.PINNED_CONTINUATION_SHA256 == pp["plan400_extension_300_399_sha256"]
    assert pe4.PINNED_FULL_PLAN_SHA256 == pp["plan400_full_sha256"]
    assert pe4.PINNED_PREFIX300_PLAN_SHA256 == pp["plan400_prefix300_sha256"] == pe.PINNED_FULL_PLAN_SHA256
    assert pe4.PINNED_PREFIX_PLAN_SHA256 == pe.PINNED_PREFIX_PLAN_SHA256
    n_gen, n32 = pp["inputs"]["n_gen"], pp["inputs"]["n32"]
    c = pe4.build_continuation(n_gen, n32)                      # pure function of (n_gen, n32): checks the registered pin without data
    assert len(c) == 100 and pe4.plan_sha256(c) == pe4.PINNED_CONTINUATION_SHA256
    assert pe.plan_sha256(pe.build_extension(n_gen, n32)) == pe.PINNED_EXTENSION_SHA256
    counts = sorted(sum(1 for e in c if e["needles"][64] == k) for k in range(40))
    assert counts == [2] * 20 + [3] * 20
    with pytest.raises(pe.PlanExtensionError, match="steps must be 400"):
        pe4.build_extended_plan({"plan": [], "rows_gen": [], "rows32": []}, 300)


def test_plan_continuation_on_synthetic_inputs_keeps_the_300_prefix():
    import random
    rnd = random.Random(5)
    plan = [{"slot0": rnd.randrange(306), "slot1": rnd.randrange(306), "slot2": rnd.randrange(117),
             "needles": {d: k % 40 for d in (32, 64, 109)}} for k in range(200)]
    SX = {"plan": plan, "rows_gen": [0] * 306, "rows32": [0] * 117}
    full = pe4.build_extended_plan(SX, 400, check_pins=False)
    assert len(full) == 400 and full[:300] == pe.build_extended_plan(SX, 300, check_pins=False)
    assert pe4.plan_sha256(full[300:]) == pe4.PINNED_CONTINUATION_SHA256 and SX["plan"] == plan
    with pytest.raises(pe.PlanExtensionError):
        pe4.build_extended_plan(SX, 400)                         # synthetic prefix fails the real pins


def _synthetic():
    return "\n".join([
        c5.HEAD_300, c5.LOAD_300.rstrip("\n"), c5.GATE_300.rstrip("\n"), c5.RESTORE_300, c5.PROV_300, c5.LAM_300.rstrip("\n"),
        c4.PXLOAD_OLD.rstrip("\n"), "  try:", c4.PLAN_OLD.rstrip("\n"), "  " + c4.PLANREC_OLD + "'length':len(SX['plan'])}; save()",
        "  HO['nova_diag_5000_5005']=diag", "  EV=['Y72_unseen','X32_unseen','X64_unseen','X109_unseen']",
        "      for s in range(START, STEPS):", c2.BACKWARD_OLD.rstrip("\n"), "          opt.step(selected)",
        "          " + c5.STEPS_300.rstrip("\n"), "          if step in EVAL_AT:", "              " + c4.CKUPD_OLD,
        "                  _rec=cont.dump_full(m, opt, step, XD=XD, HO=HO, EV=EV, " + c4.DUMP_OLD + ", seed=0, clip=1.0)",
        "      " + c5.EVAL_300.rstrip("\n"), c5.FINAL_300.rstrip("\n"), ""])


@pytest.fixture
def identity_300(monkeypatch):
    monkeypatch.setattr(c2, "apply_continue300", lambda s: s)


def test_generator_edits_are_exact(identity_300):
    out = c4.apply_cont400(_synthetic())
    assert "STEPS=400; START=300; RB=2; EVAL_AT={325,350,375,400}" in out and "/checkpoint_300.pt'" in out
    assert "r300.load_checkpoint300(CKPT_PATH, model=m, opt=opt, plan=SX['plan'], lib=rv" in out
    assert "plan_extension400.build_extended_plan(SX, STEPS)" in out and "plan_module=plan_extension400, out_dir=OUT" in out
    assert "if step in r300.CHECKPOINT_UPDATES:" in out and "clip=1.0" in out
    assert "gates.evaluate(c4rt.history(HIST))" in out and "y72fix2_cont400_verdict" in out
    assert "(('treatment',r300.NEW_LAMBDA),)" in out and "assert lam==0.25 and r300.NEW_LAMBDA==0.25" in out
    assert "clip_scheme.combine(_pp,_gx,_gy,lam,'pertask',0.5)" in out and out.count("opt.step(selected)") == 1
    assert "r300.check_resume_restoration(m, _bx, _bc, strict=not CPU_SMOKE" in out and "'resumed_from_sha256':r300.CKPT300_SHA256" in out
    for banned in c4.BANNED:
        assert banned not in out
    assert out.index("plan_extension400=") < out.index("plan_extension400.build_extended_plan(")


def test_generator_refuses_missing_duplicate_and_reapplied_sources(identity_300, monkeypatch):
    with pytest.raises(ValueError, match="PATTERN_MISSING"):
        c4.apply_cont400(c4.apply_cont400(_synthetic()))
    good = c2.apply_y72fix2(_synthetic())              # the y72fix2 treatment text the cont400 edits apply to
    monkeypatch.setattr(c2, "apply_y72fix2", lambda s: s)
    c4.apply_cont400(good)
    for label, old, _new in c4.SWAPS:
        with pytest.raises(ValueError, match=f"PATTERN_MISSING {label}"):
            c4.apply_cont400(good.replace(old, "", 1))
        with pytest.raises(ValueError, match=f"PATTERN_MISSING {label}"):
            c4.apply_cont400(good + "\n" + old)
    with pytest.raises(ValueError, match="RESIDUE"):
        c4.apply_cont400(good + "\nx='resume225'\n")
    assert [s[0] for s in c4.SWAPS] == ["head", "pxload", "plan", "planrec", "load", "gate", "restore", "prov", "lam", "ckupd", "dump", "eval", "final"]


NB = os.environ.get("VYRA_Y72FIX2_CONT400_NOTEBOOK")


@pytest.mark.skipif(not NB, reason="built notebook path not provided")
def test_built_notebook_assets_and_hashes():
    code = "".join(json.loads(Path(NB).read_text())["cells"][0]["source"])
    node = next(n for n in ast.parse(code).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "assets")
    assets = ast.literal_eval(node.value)
    for name in ("y72fix2_cont400_gates.py", "y72fix2_cont400_runtime.py", "resume300_y72fix2.py", "plan_extension400.py", "plan_extension.py",
                 "y72fix2_runtime.py", "clip_scheme.py", "y72_heldout_v2.py", "y72_heldout_v2_set.json", "prereg_y72fix2_cont400.json"):
        assert name in assets
    for name in ("resume225.py", "resume225_y72fix2.py", "y72fix2_gates.py", "prereg_y72fix2.json"):
        assert name not in assets
    assert hashlib.sha256(assets["prereg_y72fix2_cont400.json"].encode()).hexdigest() == PREREG_SHA256
    t = assets["treatment.py"]
    assert "STEPS=400; START=300" in t and "clip_grad_norm_" not in t and "clip_scheme.combine(_pp,_gx,_gy,lam,'pertask',0.5)" in t
    for banned in c4.BANNED:
        assert banned not in t
    assert "DIAGNOSTIC_UPDATES = ()" in assets["preclip_diag.py"]
    assert "@@" not in code
    lock = json.loads(assets["dependency_lock.json"])
    assert lock["datasets"]["C"]["files"] == {"checkpoint_300.pt": r3.CKPT300_SHA256} and lock["datasets"]["C"]["bytes"] == r3.CKPT300_BYTES
    for k, v in lock["assets"].items():
        assert hashlib.sha256(assets[k].encode()).hexdigest() == v
    meta = json.loads((Path(NB).parent / "kernel-metadata.json").read_text())
    assert meta["is_private"] is True and meta["enable_internet"] is False and meta["machine_shape"] == "NvidiaTeslaT4"
    assert len(meta["dataset_sources"]) == 3
