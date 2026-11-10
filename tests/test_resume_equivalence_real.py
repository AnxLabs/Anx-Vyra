"""Resume equivalence on the real pinned model classes (reduced sequence length).

Runs tests/real_resume_equiv.py in a subprocess (it imports the pinned,
byte-checked training library and its shim, which must not leak into this
interpreter). Needs the private X bundle; skipped without it.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

X = Path(os.environ.get("VYRA_X_BUNDLE", "/workspace/vera/packaging-rootfix-20261106/inputs/vyra-x-bundle"))


@pytest.mark.skipif(not (X / "train_X.pt").is_file(), reason="private X bundle not available (set VYRA_X_BUNDLE)")
def test_real_model_resume_equivalence_and_negative_controls(tmp_path):
    env = dict(os.environ, TMPDIR=str(tmp_path), VYRA_X_BUNDLE=str(X), CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run([sys.executable, str(Path(__file__).with_name("real_resume_equiv.py"))],
                          capture_output=True, text=True, env=env, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    print("REAL_MODEL_RESULT", json.dumps({k: res[k] for k in ("seq", "N", "M", "plan_start", "positive_mismatch", "controls", "elapsed_s")}))
    assert res["positive_mismatch"] == []
    assert res["losses_straight"] == res["losses_resumed"]
    assert res["sampler"]["next_update_index"] == res["plan_start"] + res["N"]
    # the one variable of the replay-weight experiment, on the real model classes
    assert res["lambda_positive_mismatch"] == []
    assert res["lambda_losses"]["straight_switched"] == res["lambda_losses"]["resumed_0_5"]
    assert {"losses", "params"} <= set(res["lambda_negative_control_mismatch"])      # resuming at 0.25 cannot equal the 0.5 run
    assert res["lambda_losses"]["straight_switched"][:res["N"]] == res["lambda_losses"]["resumed_0_25"][:res["N"]]
    expect = {
        "model": {"params"}, "optimizer": {"opt.exp_avg", "opt.step"}, "moments_only": {"opt.exp_avg", "params"},
        "clocks_only": {"opt.expert_steps", "params"}, "sampler_position": {"losses", "params"},
        "torch_rng": {"torch_rng"}, "python_rng": {"python_rng"}, "numpy_rng": {"numpy_rng"}, "router": {"router"},
    }
    for dropped, must in expect.items():
        got = set(res["controls"][dropped])
        assert got, f"control {dropped} cannot fail"
        assert must <= got, (dropped, got)
