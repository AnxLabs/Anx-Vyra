"""The 225 -> 300 replay-weight generator and the built notebook (env-gated)."""
import ast
import json
import os
import sys
from pathlib import Path

import pytest

PACKAGING = Path(__file__).resolve().parents[1] / "tools" / "experiment_packaging"
sys.path.insert(0, str(PACKAGING))

import continue225 as c5  # noqa: E402
import resume225 as r225  # noqa: E402
import y72fix_gates as gates  # noqa: E402


def _synthetic():
    """Contains every old snippet exactly once, in the order the real treatment has them."""
    return "\n".join([
        c5.HEAD_300, c5.LOAD_300.rstrip("\n"), c5.GATE_300.rstrip("\n"), c5.RESTORE_300, c5.PROV_300,
        c5.LAM_300.rstrip("\n"),
        "      for s in range(START, STEPS):", "          opt.step(selected)",
        "          " + c5.STEPS_300.rstrip("\n"), "      " + c5.EVAL_300.rstrip("\n"), c5.FINAL_300.rstrip("\n"), ""])


@pytest.fixture
def identity_300(monkeypatch):
    monkeypatch.setattr(c5, "apply_continue300", lambda s: s)


def test_generator_edits_are_exact(identity_300):
    out = c5.apply_continue225(_synthetic())
    assert "START=225" in out and "EVAL_AT={250,275,300}" in out and "checkpoint_225.pt" in out
    assert "(('treatment',gates.NEW_LAMBDA),)" in out and "assert lam==0.5" in out
    assert "r225.load_checkpoint225(" in out and "r225.check_resume_restoration(" in out
    assert "gates.evaluate(HIST)" in out and "y72fix_verdict" in out
    assert out.count("A.setdefault('preclip_norm',[]).append(float(_preclip_norm))") == 1
    assert out.index("r225.load_checkpoint225(") < out.index("for s in range(START, STEPS)")
    for banned in ("load_checkpoint200", "stop_rule", "CKPT200", "checkpoint_200.pt"):
        assert banned not in out


def test_generator_refuses_missing_duplicate_and_reapplied_sources(identity_300):
    good = _synthetic()
    out = c5.apply_continue225(good)
    with pytest.raises(ValueError, match="CONTINUE225_PATTERN_MISSING"):
        c5.apply_continue225(out)                                            # already applied
    for label, old, _new in c5.SWAPS:
        with pytest.raises(ValueError, match=f"CONTINUE225_PATTERN_MISSING {label}"):
            c5.apply_continue225(good.replace(old, "", 1))
        with pytest.raises(ValueError, match=f"CONTINUE225_PATTERN_MISSING {label}"):
            c5.apply_continue225(good + old)                                 # count 2
    with pytest.raises(ValueError, match="CONTINUE225_RESIDUE"):
        c5.apply_continue225(good + "\nx='stop_rule'\n")


def test_swaps_change_only_what_the_docstring_says():
    labels = [s[0] for s in c5.SWAPS]
    assert labels == ["head", "load", "gate", "restore", "prov", "lam", "steps", "eval", "final"]
    assert c5.HEAD_225.count("START=225") == 1 and "STEPS=300" in c5.HEAD_225 and "RB=2" in c5.HEAD_225 and "HARD=600; SOFT=598" in c5.HEAD_225
    assert c5.HEAD_225.replace("START=225", "START=200").replace("EVAL_AT={250,275,300}", "EVAL_AT={225,250,275,300}").replace(
        "vyra-resume300-ckpt225/checkpoint_225.pt", "vyra-expanded40-200u-checkpoint/checkpoint_200.pt") == c5.HEAD_300
    assert c5.DIAG_NEW == "DIAGNOSTIC_UPDATES = (250, 275, 300)"
    assert c5.HEAD_225.split("CKPT_PATH='")[1].split("'")[0] == r225.CKPT225_PATH


def test_diag_patch_is_exact():
    text = "a\n" + c5.DIAG_OLD + "\nb\n"
    assert c5.DIAG_NEW in c5.patch_preclip_diag(text)
    with pytest.raises(ValueError):
        c5.patch_preclip_diag("nothing")
    with pytest.raises(ValueError):
        c5.patch_preclip_diag(text + c5.DIAG_OLD)


@pytest.mark.skipif(not os.environ.get("VYRA_Y72FIX_NOTEBOOK"), reason="set VYRA_Y72FIX_NOTEBOOK to a built replay.ipynb")
def test_built_y72fix_notebook():
    nb = Path(os.environ["VYRA_Y72FIX_NOTEBOOK"])
    code = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assets = ast.literal_eval(next(n for n in ast.parse(code).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "assets").value)
    for name in ("continuation.py", "plan_extension.py", "resume225.py", "y72fix_gates.py", "x64_checkpoint.py", "dependency_lock.json"):
        assert name in assets
    assert "stop_rule.py" not in assets
    t = assets["treatment.py"]
    assert "STEPS=300; START=225" in t and "EVAL_AT={250,275,300}" in t and "for s in range(START, STEPS)" in t
    for line in ("rv.LR=3e-4", "(('treatment',gates.NEW_LAMBDA),)", "clip_grad_norm_(_clip_params,1.0)", "%40 for j in range(RB)", "opt.step(selected)",
                 "r225.load_checkpoint225(CKPT_PATH", "r225.check_resume_restoration(m, _bx, _bc, strict=not CPU_SMOKE"):
        assert line in t
    assert "0.25" not in t.split("for arm,lam in")[1].split("\n")[0]                        # lambda only comes from the pinned module
    assert "load_checkpoint200" not in t and "checkpoint_200" not in t and "stop_rule" not in t
    assert t.index("r225.load_checkpoint225(") < t.index("r225.check_resume_restoration(") < t.index("for s in range(START, STEPS)")
    assert "random.seed(0)" not in t and "manual_seed(0)" not in t
    assert assets["resume225.py"] == (PACKAGING / "resume225.py").read_text() and assets["y72fix_gates.py"] == (PACKAGING / "y72fix_gates.py").read_text()
    assert "DIAGNOSTIC_UPDATES = (250, 275, 300)" in assets["preclip_diag.py"]
    lock = json.loads(assets["dependency_lock.json"])
    assert lock["datasets"]["C"]["files"] == {"checkpoint_225.pt": r225.CKPT225_SHA256}
    assert lock["datasets"]["C"]["mount"] == r225.CKPT225_MOUNT and lock["datasets"]["C"]["bytes"] == r225.CKPT225_BYTES
    assert set(lock["datasets"]) == {"X", "Y", "C"}
    assert "launch_gpu('/kaggle/working/timeout_status.json',cap=600,grace=2)" in code        # 600 s trainer limit unchanged
    assert code.rindex("verify_lock()") < code.index("launch_gpu('/kaggle/working/timeout_status.json'")
    meta = json.loads((nb.parent / "kernel-metadata.json").read_text())
    assert meta["is_private"] is True and meta["enable_internet"] is False and meta["enable_gpu"] is True and meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["id"] == "anjorosimoaj/vyra-y72fix-lambda05-packaged" and meta["id"] != "anjorosimoaj/vyra-resume300-packaged"
    assert meta["dataset_sources"] == ["anjorosimoaj/vyra-x-bundle", "anjorosimoaj/vyra-y-bundle", r225.CKPT225_DATASET_ID]
    report = json.loads((nb.parent.parent / "BUILD_REPORT.json").read_text())
    assert report["only_variable"] == {"name": "replay_lambda", "from": 0.25, "to": 0.5} and report["updates"] == {"from": 225, "to": 300}
    assert report["checkpoint_updates"] == [250, 275, 300] and report["no_upload"] and report["no_submission"]
    assert gates.NEW_LAMBDA == 0.5
