"""Observers must not change grads, optimizer state, RNG, routes or mode.

`observe` restores RNG/routes/mode and verifies params, grads, optimizer, RNG,
routes and mode are bit-identical on exit; a write to params/grads/optimizer
raises. dump_full (X64 readouts, retention counts, complete checkpoint) is run
with live grads and an unfinished update to show it leaves everything alone.
"""
import random

import numpy as np
import pytest
import torch

from resume_world import LIB, World
import continuation as cont
import x64_checkpoint
import preclip_diag


def _digest_all(w):
    return (cont.state_digest(w.model, w.opt), random.getstate(), np.random.get_state()[1].tolist(),
            torch.get_rng_state().clone().tolist(), cont.capture_router(w.model)["e"]["selected"],
            cont.capture_router(w.model)["e"]["load"].tolist(), cont.module_modes(w.model))


def _prepared(with_grads):
    w = World(3); w.run(0, 3)
    if with_grads:
        w.model.train()
        loss = (w.model(w.data[3]) ** 2).mean(); loss.backward()
    return w


def test_observer_that_moves_rng_modes_and_routes_is_restored_exactly():
    w = _prepared(True)
    before = _digest_all(w)
    with cont.observe(w.model, w.opt):
        w.model.eval()
        w.model(w.data[4]); w.model(w.data[5])         # changes _selected and cumulative _load
        random.random(); np.random.random(); torch.rand(3)
    after = _digest_all(w)
    assert after == before
    assert w.model.training is True


@pytest.mark.parametrize("what", ["param", "grad", "moment", "clock", "step"])
def test_observer_that_writes_training_state_is_detected(what):
    w = _prepared(True)
    with pytest.raises(cont.ObserverStateChanged) as err:
        with cont.observe(w.model, w.opt):
            if what == "param":
                with torch.no_grad():
                    w.model.head[0, 0] += 1e-7
            elif what == "grad":
                w.model.head.grad.view(-1)[0] += 1e-7
            elif what == "moment":
                w.opt.state[id(w.model.head)]["exp_avg"].view(-1)[0] += 1e-7
            elif what == "clock":
                w.opt.state[id(w.model.e.gate)]["expert_steps"][0] += 1
            elif what == "step":
                w.opt.state[id(w.model.head)]["step"] += 1
    expected = {"param": "digest.params", "grad": "digest.grads", "moment": "digest.optimizer",
                "clock": "digest.optimizer", "step": "digest.optimizer"}[what]
    assert expected in str(err.value)


def test_observer_that_clears_grads_is_detected_and_exception_path_still_restores():
    w = _prepared(True)
    with pytest.raises(cont.ObserverStateChanged):
        with cont.observe(w.model, w.opt):
            w.model.zero_grad(set_to_none=True)
    w2 = _prepared(False)
    before = _digest_all(w2)
    with pytest.raises(ZeroDivisionError):
        with cont.observe(w2.model, w2.opt):
            w2.model.eval(); w2.model(w2.data[4]); random.random()
            1 / 0
    assert _digest_all(w2) == before


def _eval_inputs():
    g = torch.Generator().manual_seed(2)
    def rows(n):
        r = torch.randint(0, 8, (n, 512), generator=g)
        m = torch.zeros(n, 510); m[:, 100:148] = 1
        return r, m
    XD = {64: {}}
    r, m = rows(5); XD[64]["rows"], XD[64]["mask"] = r, m
    u, um = rows(40); XD[64]["unseen"], XD[64]["unseen_mask"] = u, um
    HO = {}
    for k in ("Y72_unseen", "X32_unseen", "X64_unseen", "X109_unseen"):
        rr, mm = rows(8); HO[k] = list(zip(rr, mm))
    return XD, HO


class _SeqLM(torch.nn.Module):
    """Tiny LM over (B,512) tokens with the repo's Experts so routes are real."""
    def __init__(self, w):
        super().__init__()
        self.embed, self.e, self.head, self.shared, self.unused = w.embed, w.e, w.head, w.shared, w.unused

    def forward(self, tokens):
        x = self.embed[tokens]
        x = x + self.e(x) * self.shared
        return x @ self.head.t()


def test_dump_full_leaves_everything_alone_and_writes_a_complete_checkpoint(tmp_path):
    w = World(3); w.run(0, 3)
    seq = _SeqLM(w.model)
    # the dump runs on a model whose forward accepts (B,512) ids; reuse w.model's parameters and optimizer state
    seq.opt = w.opt
    seq.zero_grad(set_to_none=True)
    seq.train()
    loss = (seq(torch.randint(0, 8, (2, 16))) ** 2).mean(); loss.backward()     # live grads, update not finished
    XD, HO = _eval_inputs()
    plan = [{"slot0": 0, "slot1": 0, "slot2": 0, "needles": {64: k % 5}} for k in range(300)]
    before = (cont.state_digest(seq, w.opt), random.getstate(), np.random.get_state()[1].tolist(), torch.get_rng_state().clone().tolist(),
              cont.capture_router(seq)["e"]["load"].tolist(), cont.capture_router(seq)["e"]["selected"], cont.module_modes(seq))
    rec = cont.dump_full(seq, w.opt, 225, XD=XD, HO=HO, EV=list(HO), plan=plan, plan_module=w.plan_mod, out_dir=tmp_path,
                         diag=preclip_diag, x64_ck=x64_checkpoint, lib=LIB, replay_lambda=0.25, replay_batch=2, seed=0, clip=1.0,
                         provenance={"t": 1}, device="cpu")
    after = (cont.state_digest(seq, w.opt), random.getstate(), np.random.get_state()[1].tolist(), torch.get_rng_state().clone().tolist(),
             cont.capture_router(seq)["e"]["load"].tolist(), cont.capture_router(seq)["e"]["selected"], cont.module_modes(seq))
    assert before == after and seq.training is True
    assert "checkpoint_error" not in rec, rec.get("checkpoint_error")
    assert rec["checkpoint_bytes"] > 0
    assert rec["x64"]["train"]["tokens"] == 5 * 48 and rec["x64"]["heldout"]["segments"] == 40 * 6
    assert set(rec["counts"]) == set(HO) and all(v["tokens"] == 8 * 48 for v in rec["counts"].values())
    assert (tmp_path / "x64_predictions_225.json").is_file() and (tmp_path / "checkpoint_225.pt").is_file()
    ck = cont.load_full_checkpoint(tmp_path / "checkpoint_225.pt")
    for key in cont.V2_REQUIRED:
        assert key in ck
    assert ck["update"] == 225 and ck["sampler"]["next_update_index"] == 225 and ck["model_training"] is True
    assert ck["router"]["e"]["load"].tolist() == before[4] and ck["router"]["e"]["selected"] == sorted(before[5])
    # integer counts agree with an independent recount on the same model
    with cont.observe(seq, w.opt):
        again = cont.eval_counts(seq, HO["Y72_unseen"], "cpu")
    assert again == rec["counts"]["Y72_unseen"]


def test_sparse_diagnostics_asset_covers_exactly_225_250_275_300():
    from continue300 import patch_preclip_diag
    text = patch_preclip_diag((cont.Path(cont.__file__).parent / "preclip_diag.py").read_text())
    assert "DIAGNOSTIC_UPDATES = (225, 250, 275, 300)" in text and "(1, 75, 125, 175, 200)" not in text
    with pytest.raises(ValueError, match="DIAG_PATTERN_MISSING"):
        patch_preclip_diag(text)
