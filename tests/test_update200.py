"""200-update extension: generator edits only steps/eval/observation, and the
checkpoint + X64 dump never changes model, grads, optimizer, RNG or mode."""
import os
import random
import sys
from pathlib import Path

import pytest
import torch

PACKAGING = Path(__file__).resolve().parents[1] / "tools" / "experiment_packaging"
sys.path.insert(0, str(PACKAGING))

from build import prepare_treatment  # noqa: E402
from test_packaging_fail_closed import MINIMAL_TREATMENT  # noqa: E402
from update200 import apply_update200  # noqa: E402
import preclip_diag  # noqa: E402
import x64_checkpoint  # noqa: E402


def test_generator_changes_only_steps_eval_and_observation():
    patched, _ = prepare_treatment(MINIMAL_TREATMENT, "b")
    assert "STEPS=200; RB=2; EVAL_AT={75,125,175,200}" in patched
    assert "STEPS=175" not in patched
    assert patched.index("x64_checkpoint=_cmod") < patched.index("x64_checkpoint.wanted(step)")
    with pytest.raises(ValueError, match="UPDATE200_ALREADY_APPLIED"):
        apply_update200(patched)
    # Removing the three update200 edits gives back the 175-step treatment byte for byte.
    unpatched = "\n".join(l for l in patched.splitlines() if "x64_checkpoint" not in l and "_cspec" not in l)
    unpatched = unpatched.replace("STEPS=200; RB=2; EVAL_AT={75,125,175,200}", "STEPS=175; RB=2; EVAL_AT={75,125,175}")
    import build as _b
    orig = _b.apply_update200
    try:
        _b.apply_update200 = lambda s: s
        before, _ = prepare_treatment(MINIMAL_TREATMENT, "b")
    finally:
        _b.apply_update200 = orig
    assert unpatched == before.rstrip("\n")


def test_sparse_diagnostics_and_checkpoints_cover_175_and_200():
    assert preclip_diag.DIAGNOSTIC_UPDATES == (1, 75, 125, 175, 200)
    assert preclip_diag.wanted(199) and not preclip_diag.wanted(198)
    assert x64_checkpoint.CHECKPOINT_UPDATES == (175, 200)
    assert x64_checkpoint.wanted(175) and x64_checkpoint.wanted(200) and not x64_checkpoint.wanted(125)


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(6))
        self._selected = {1}
        self._load = torch.zeros(4)

    def forward(self, tokens):
        self._selected = {9}
        self._load = self._load + 1
        torch.rand(3)
        out = torch.zeros(tokens.shape[0], tokens.shape[1], 6)
        return out + self.weight


class _Opt:
    def __init__(self, model):
        self.state = {id(p): {"step": 3, "exp_avg": torch.ones_like(p)} for p in model.parameters()}

    def step(self, *_a, **_k):
        raise AssertionError("dump stepped the optimizer")


def _xd():
    g = torch.Generator().manual_seed(1)
    rows = torch.randint(0, 6, (5, 512), generator=g)
    mask = torch.zeros(5, 510); mask[:, 100:116] = 1
    unseen = torch.randint(0, 6, (40, 512), generator=g)
    umask = torch.zeros(40, 510); umask[:, 200:248] = 1
    return {64: {"rows": rows, "mask": mask, "unseen": unseen, "unseen_mask": umask}}


def test_dump_restores_state_and_writes_outputs(tmp_path):
    model = _Tiny(); model.train()
    model.weight.grad = torch.full((6,), 0.5)
    opt = _Opt(model)
    plan = [{"needles": {64: k % 3}} for k in range(10)]
    w = model.weight.detach().clone(); v = model.weight._version
    rng = (random.getstate(), torch.get_rng_state().clone())
    rec = x64_checkpoint.dump(model, opt, 4, _xd(), plan, 10, tmp_path, preclip_diag)
    assert model.training and model._selected == {1} and torch.equal(model._load, torch.zeros(4))
    assert random.getstate() == rng[0] and torch.equal(torch.get_rng_state(), rng[1])
    assert torch.equal(model.weight, w) and model.weight._version == v
    assert torch.equal(model.weight.grad, torch.full((6,), 0.5))
    assert opt.state[id(model.weight)]["step"] == 3
    assert rec["train_rows"] == 3 and rec["heldout_rows"] == 40
    assert "checkpoint_error" not in rec and rec["checkpoint_bytes"] > 0
    ck = torch.load(tmp_path / "checkpoint_4.pt", weights_only=False)
    assert set(ck["optimizer"]) == {"weight"} and ck["update"] == 4
    assert (tmp_path / "x64_predictions_4.json").is_file()


def test_checkpoint_failure_is_recorded_not_raised(tmp_path):
    model = _Tiny()

    class Broken:
        @property
        def state(self):
            raise RuntimeError("boom")

    rec = x64_checkpoint.dump(model, Broken(), 2, _xd(), [{"needles": {64: 0}}] * 2, 2, tmp_path, preclip_diag)
    assert "boom" in rec["checkpoint_error"]


@pytest.mark.skipif(not os.environ.get("VYRA_PACKAGED_NOTEBOOK"), reason="set VYRA_PACKAGED_NOTEBOOK to a built replay.ipynb")
def test_packaged_notebook_is_200_updates_with_checkpoint_asset():
    import ast, json
    code = "".join(json.loads(Path(os.environ["VYRA_PACKAGED_NOTEBOOK"]).read_text())["cells"][0]["source"])
    assets = ast.literal_eval(next(n for n in ast.parse(code).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "assets").value)
    t = assets["treatment.py"]
    assert "STEPS=200; RB=2; EVAL_AT={75,125,175,200}" in t
    assert "x64_checkpoint.py" in assets and "(1, 75, 125, 175, 200)" in assets["preclip_diag.py"]
    for line in ("rv.LR=3e-4; opt=rv.make_optimizer(m)", "torch.manual_seed(0)", "(('treatment',0.25),)", "clip_grad_norm_(_clip_params,1.0)", "%40 for j in range(RB)"):
        assert line in t
