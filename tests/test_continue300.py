"""The 200 -> 300 generator and the built continuation notebook."""
import ast
import json
import os
import sys
from pathlib import Path

import pytest

PACKAGING = Path(__file__).resolve().parents[1] / "tools" / "experiment_packaging"
sys.path.insert(0, str(PACKAGING))

import continue300 as c3  # noqa: E402
import continuation as cont  # noqa: E402
import kaggle_config as K  # noqa: E402


def _fixture():
    return "\n".join([
        "import os",
        c3.HEAD,
        c3.PLAN_CHECK.rstrip("\n"),
        c3.BASE_A.rstrip("\n"),
        c3.BASE_B.rstrip("\n"),
        K.expand("  _cspec=_ilu.spec_from_file_location('x64_checkpoint','@@KROOT@@/working/x64_checkpoint.py')"),
        c3.X64_LOADER.rstrip("\n"),
        "  for arm,lam in (('treatment',0.25),):",
        "      " + c3.ARM_INIT,
        "      " + c3.ARM_REC + "'loss':[]}; R['arms'][arm]=A",
        c3.LOOP.rstrip("\n"),
        "          opt.step(selected)",
        "          if step in EVAL_AT:",
        c3.EVAL_LINES.rstrip("\n"),
        c3.VERDICT.rstrip("\n"),
        "",
    ])


def test_generator_edits_are_exact_and_leave_training_untouched():
    out = c3.apply_continue300(_fixture())
    assert "STEPS=300; START=200; RB=2; EVAL_AT={225,250,275,300}" in out
    assert "for s in range(START, STEPS)" in out
    assert "random.seed(0)" not in out and "torch.manual_seed(0)" not in out and "manual_seed_all" not in out
    assert "x64_checkpoint.dump(" not in out and "cont.dump_full(" in out and "cont.observe(m, opt)" in out
    assert out.count("x64_checkpoint=_cmod") == 1                       # loader moved before the baseline, not duplicated
    assert out.index("cont.load_checkpoint200(") < out.index("cont.observe_counts(") < out.index("for s in range(START, STEPS)")
    assert out.index("plan_extension.build_extended_plan") < out.index("if len(SX['plan'])<STEPS")
    assert out.index("check_baseline_restoration") < out.index("for s in range(START, STEPS)")
    assert "tel=ReplayTelemetry(m, opt, STEPS-START)" in out
    assert "R['stop_rule_final']=stop_rule.evaluate(HIST)" in out
    assert "strict=not CPU_SMOKE" in out and "CPU_SMOKE=os.environ.get('VYRA_CONT_CPU_SMOKE')=='1'" in out
    assert K.expand("'@@KROOT@@/input/datasets/@@KUSER@@/@@SLUG_CKPT200@@/checkpoint_200.pt'") in out
    assert "glob" not in out and "iterdir" not in out


def test_generator_refuses_unknown_or_already_applied_sources():
    out = c3.apply_continue300(_fixture())
    with pytest.raises(ValueError, match="CONTINUE300_ALREADY_APPLIED"):
        c3.apply_continue300(out)
    with pytest.raises(ValueError, match="CONTINUE300_PATTERN_MISSING"):
        c3.apply_continue300(_fixture().replace(c3.LOOP, "      for s in range(STEPS):\n"))
    with pytest.raises(ValueError, match="CONTINUE300_PATTERN_MISSING"):
        c3.apply_continue300("STEPS=200\n")


def test_constants_are_the_approved_ones():
    assert cont.CKPT200_SHA256 == "8542c3405309a5fcf9e419a6e3dd733beab346765fa7bd9cbdd61226dd867e81"
    assert cont.CKPT200_BYTES == 197225718 and cont.RESUME_UPDATE == 200 and cont.FINAL_UPDATE == 300
    assert cont.CHECKPOINT_UPDATES == (225, 250, 275, 300)
    assert K.expand(cont.CKPT200_PATH) == K.expand("@@KROOT@@/input/datasets/@@KUSER@@/@@SLUG_CKPT200@@/checkpoint_200.pt")
    assert cont.BASELINE_200_MODEL_STATE_SHA256 == "35bd2f486abf383965b93ffe08ff5db38e0873a6874bb039c3a188c30b2a0b34"
    d = cont.DISCLOSURE["not_in_checkpoint_200_and_initialized"]
    assert set(d) == {"python_rng", "numpy_rng", "router__selected", "router__load", "sampler_position", "optimizer_config", "model_mode"}
    assert set(cont.DISCLOSURE["new_checkpoints_225_250_275_300_capture"]) >= {
        "python_rng", "numpy_rng", "torch_rng", "cuda_rng", "router", "optimizer_config", "sampler", "model_training"}


@pytest.mark.skipif(not os.environ.get("VYRA_CONT_NOTEBOOK"), reason="set VYRA_CONT_NOTEBOOK to a built continuation replay.ipynb")
def test_built_continuation_notebook():
    nb = Path(os.environ["VYRA_CONT_NOTEBOOK"])
    code = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assets = ast.literal_eval(next(n for n in ast.parse(code).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "assets").value)
    for name in ("continuation.py", "plan_extension.py", "stop_rule.py", "x64_checkpoint.py", "dependency_lock.json"):
        assert name in assets
    t = assets["treatment.py"]
    assert "STEPS=300; START=200" in t and "for s in range(START, STEPS)" in t
    for line in ("rv.LR=3e-4", "(('treatment',0.25),)", "clip_grad_norm_(_clip_params,1.0)", "%40 for j in range(RB)", "opt.step(selected)"):
        assert line in t
    assert "random.seed(0)" not in t and "manual_seed(0)" not in t
    assert "DIAGNOSTIC_UPDATES = (225, 250, 275, 300)" in assets["preclip_diag.py"]
    lock = json.loads(assets["dependency_lock.json"])
    assert lock["datasets"]["C"]["files"] == {"checkpoint_200.pt": cont.CKPT200_SHA256}
    assert lock["datasets"]["C"]["mount"] == K.expand(cont.CKPT200_MOUNT)
    assert K.expand("launch_gpu('@@KROOT@@/working/timeout_status.json',cap=600,grace=2)") in code        # 600 s trainer limit unchanged
    assert code.rindex("verify_lock()") < code.index(K.expand("launch_gpu('@@KROOT@@/working/timeout_status.json'"))
    meta = json.loads((nb.parent / "kernel-metadata.json").read_text())
    assert meta["is_private"] is True and meta["enable_internet"] is False and meta["enable_gpu"] is True
    assert meta["machine_shape"] == cont.MACHINE_SHAPE == "NvidiaTeslaT4"
    orig = Path("/workspace/vera/expanded40-200u-run-20261107/actual_submitted/kernel-metadata.json")
    if orig.is_file():                                   # the accelerator request equals the original 200-update run's
        assert json.loads(orig.read_text())["machine_shape"] == meta["machine_shape"]
    assert "cont.check_baseline_restoration(m, _bx, _bc, R['baseline']" in assets["treatment.py"]
    assert "AMBIGUOUS" not in assets["stop_rule.py"] and "INCONCLUSIVE" in assets["stop_rule.py"]
    assert meta["dataset_sources"] == [K.expand("@@KUSER@@/@@SLUG_X_BUNDLE@@"), K.expand("@@KUSER@@/@@SLUG_Y_BUNDLE@@"), K.expand("@@KUSER@@/@@SLUG_CKPT200@@")]
