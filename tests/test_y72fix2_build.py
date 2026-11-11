"""Y72 retention fix 2: generator edits, resume/lambda gate, runtime glue, and the built notebook (env-gated). Pinned to prereg/y72fix2.json."""
import ast
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "experiment_packaging"))

import continue225_y72fix2 as c2  # noqa: E402
import continuation as cont  # noqa: E402
import resume225 as r225  # noqa: E402
import resume225_y72fix2 as r2  # noqa: E402
import y72fix2_gates as gates  # noqa: E402
import y72fix2_runtime as rt  # noqa: E402

P = json.loads((ROOT / "prereg" / "y72fix2.json").read_text())


def _synthetic():
    return "\n".join([
        c2.c5.HEAD_300, c2.c5.LOAD_300.rstrip("\n"), c2.c5.GATE_300.rstrip("\n"), c2.c5.RESTORE_300, c2.c5.PROV_300, c2.c5.LAM_300.rstrip("\n"),
        "  HO['nova_diag_5000_5005']=diag", "  EV=['Y72_unseen','X32_unseen','X64_unseen','X109_unseen']",
        "      for s in range(START, STEPS):", c2.BACKWARD_OLD.rstrip("\n"), "          opt.step(selected)",
        "          " + c2.c5.STEPS_300.rstrip("\n"), "      " + c2.c5.EVAL_300.rstrip("\n"), c2.c5.FINAL_300.rstrip("\n"), ""])


@pytest.fixture
def identity_300(monkeypatch):
    monkeypatch.setattr(c2, "apply_continue300", lambda s: s)


def test_generator_edits_are_exact(identity_300):
    out = c2.apply_y72fix2(_synthetic())
    assert "START=225" in out and "EVAL_AT={250,275,300}" in out and "checkpoint_225.pt" in out
    assert "(('treatment',r225.NEW_LAMBDA),)" in out and "assert lam==0.25" in out
    assert "clip_scheme.combine(_pp,_gx,_gy,lam,'pertask',0.5)" in out and "clip_grad_norm_" not in out
    assert "gates.evaluate(y2rt.history(HIST))" in out and "y72fix2_verdict" in out and "y72fix_verdict" not in out
    assert out.count("A.setdefault('task_clip',[])") == 1 and out.count("A.setdefault('preclip_norm',[]).append(float(_preclip_norm))") == 1
    assert out.index("HO[y2rt.KEY]=") < out.index("EV=[") and "y2rt.KEY]" in out
    assert out.index("r225.load_checkpoint225(") < out.index("for s in range(START, STEPS)")
    assert out.index("clip_scheme.assign(") < out.index("opt.step(selected)")
    for banned in ("load_checkpoint200", "stop_rule", "CKPT200", "checkpoint_200.pt", "gates.NEW_LAMBDA"):
        assert banned not in out


def test_generator_refuses_missing_duplicate_and_reapplied_sources(identity_300):
    good = _synthetic()
    out = c2.apply_y72fix2(good)
    with pytest.raises(ValueError, match="PATTERN_MISSING"):
        c2.apply_y72fix2(out)
    for label, old, _new in c2.SWAPS + (("ho", c2.HO_OLD, ""), ("ev", c2.EV_OLD, "")):
        with pytest.raises(ValueError, match=f"PATTERN_MISSING {label}"):
            c2.apply_y72fix2(good.replace(old, "", 1))
        with pytest.raises(ValueError, match=f"PATTERN_MISSING {label}"):
            c2.apply_y72fix2(good + old)
    with pytest.raises(ValueError, match="RESIDUE"):
        c2.apply_y72fix2(good + "\nx='stop_rule'\n")


def test_swap_labels_and_diag_patch():
    assert [s[0] for s in c2.SWAPS] == ["head", "load", "gate", "restore", "prov", "lam", "steps", "eval", "final", "backward"]
    text = "x\nDIAGNOSTIC_UPDATES = (1, 75, 125, 175, 200)\ny\n"
    assert "DIAGNOSTIC_UPDATES = ()" in c2.patch_preclip_diag(text)
    with pytest.raises(ValueError):
        c2.patch_preclip_diag("nothing")


def _ck(lam=0.25):
    cfg = dict(r225.SAVED_CONFIG_EXCEPT_LAMBDA, replay_lambda=lam)
    ck = {k: None for k in cont.V2_REQUIRED}
    ck.update({"schema": cont.V2_SCHEMA, "update": 225, "optimizer_config": cfg, "cuda_rng": [1, 2],
               "sampler": {"plan_prefix200_sha256": r225.PINNED_PREFIX_PLAN_SHA256, "plan_full_sha256": r225.PINNED_FULL_PLAN_SHA256, "next_update_index": 225}})
    return ck


def test_resume_requires_unchanged_lambda():
    r2.validate_v2_resume(_ck())
    with pytest.raises(cont.ContinuationGateError, match="LAMBDA_CHANGED"):
        r2.validate_v2_resume(_ck(), new_lambda=0.5)
    with pytest.raises(cont.ContinuationGateError, match="OPTIMIZER_CONFIG_MISMATCH"):
        r2.validate_v2_resume(_ck(lam=0.5))
    with pytest.raises(cont.ContinuationGateError, match="CKPT_UPDATE"):
        r2.validate_v2_resume(_ck(), expected_update=200)
    assert r2.CKPT225_SHA256 == P["design"]["resume_from"]["sha256"] and r2.CKPT225_BYTES == P["design"]["resume_from"]["bytes"]
    assert r2.MODEL_STATE_SHA256_225 == P["design"]["resume_from"]["model_state_sha256"] and r2.NEW_LAMBDA == P["design"]["replay_lambda"]


def _rec(y, seg, tok, x32=200, x109=100, ysegs=600):
    return {"x64": {"heldout": {"tokens_correct": tok, "tokens": 1920, "segments_exact": seg, "segments": 240}},
            "counts": {rt.KEY: {"segments_exact": y, "segments": ysegs}, "X32_unseen": {"segments_exact": x32, "segments": 320}, "X109_unseen": {"segments_exact": x109, "segments": 120}}}


def test_history_feeds_the_gate_and_refuses_bad_denominators():
    H = {225: _rec(490, 11, 1418), 250: _rec(528, 13, 1526), 275: _rec(519, 28, 1570), 300: _rec(528, 27, 1624, 213, 114)}
    h = rt.history(H)
    assert sorted(h) == [250, 275, 300] and h[300]["x64_tokens"] == 1624
    assert gates.evaluate(h)["verdict"] == "SUCCESS"
    with pytest.raises(ValueError, match="DENOMINATOR_MISMATCH"):
        rt.history({300: _rec(528, 27, 1624, ysegs=200)})


def test_preclip_norm_matches_direct_norm():
    import torch
    g = torch.Generator().manual_seed(3)
    gx, gy = torch.randn(50, generator=g), torch.randn(50, generator=g)
    rec = {"norm_gx": float(gx.norm()), "norm_gy": float(gy.norm()), "cos_gx_gy": float(gx @ gy / (gx.norm() * gy.norm()))}
    assert math.isclose(rt.preclip_norm(rec, 0.25), float((gx + 0.25 * gy).norm()), rel_tol=1e-6)


def test_registered_set_loads_with_verified_hashes():
    pairs = rt.load_y600(str(ROOT / "prereg" / "y72_heldout_v2" / "y72_heldout_v2_set.json"))
    assert len(pairs) == 120 and tuple(pairs[0][0].shape) == (512,) and tuple(pairs[0][1].shape) == (510,)
    bad = ROOT / "prereg" / "y72_heldout_v2" / "y72_heldout_v2_set.json"
    s = json.loads(bad.read_text())
    s["rows_int16le_hex"] = "00" + s["rows_int16le_hex"][2:] if s["rows_int16le_hex"][:2] != "00" else "01" + s["rows_int16le_hex"][2:]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.json"
        p.write_text(json.dumps(s))
        with pytest.raises(Exception):
            rt.load_y600(str(p))


NB = os.environ.get("VYRA_Y72FIX2_NOTEBOOK")


@pytest.mark.skipif(not NB, reason="built notebook path not provided")
def test_built_notebook_assets_and_hashes():
    code = "".join(json.loads(Path(NB).read_text())["cells"][0]["source"])
    node = next(n for n in ast.parse(code).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "assets")
    assets = ast.literal_eval(node.value)
    for name in ("y72fix2_gates.py", "y72fix2_runtime.py", "resume225_y72fix2.py", "clip_scheme.py", "y72_heldout_v2.py", "y72_heldout_v2_set.json", "prereg_y72fix2.json"):
        assert name in assets
    assert hashlib.sha256(assets["prereg_y72fix2.json"].encode()).hexdigest() == "8eaace70888b8ec6c02537354a6ec98118327505f1a482deea01221f8475a89a"
    assert "clip_grad_norm_" not in assets["treatment.py"] and "clip_scheme.combine(_pp,_gx,_gy,lam,'pertask',0.5)" in assets["treatment.py"]
    assert "DIAGNOSTIC_UPDATES = ()" in assets["preclip_diag.py"]
    lock = json.loads(assets["dependency_lock.json"])
    assert lock["datasets"]["C"]["files"]["checkpoint_225.pt"] == r225.CKPT225_SHA256
    for k, v in lock["assets"].items():
        assert hashlib.sha256(assets[k].encode()).hexdigest() == v
