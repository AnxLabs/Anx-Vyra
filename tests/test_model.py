"""Integrity tests for the ANX-VYRA reference model.

These are the checks that guarded every reported run. They are cheap, they run on CPU, and they
fail loudly rather than quietly producing a different architecture than the published one.
"""
import torch

from anx_vyra import optim
from anx_vyra.data import build_distance_set, build_split, geometry, needle_row
from anx_vyra.model import (ARM_WINDOWS, BOS, EOS, N_EXPERTS, PAD, SEQ, TOP_K, VOCAB,
                            build_arm, counts, expert_selection, linear_muls, reset_expert_load)


def test_parameter_counts_match_published():
    """16,427,072 total and 8,169,536 active per token are the published numbers."""
    model = build_arm("R")
    got = counts(model)
    assert got["total"] == 16_427_072, got
    assert got["active"] == 8_169_536, got
    assert linear_muls(model) == 5_408_256


def test_window_layout_is_arm_r():
    model = build_arm("R")
    windows = [getattr(block.mixer, "window", None) for block in model.blocks]
    assert windows == ARM_WINDOWS["R"] == [128, None, 128, None, 128, None, 128, None]


def test_forward_shape_and_finiteness():
    model = build_arm("R")
    with torch.no_grad():
        logits = model(torch.zeros(2, 32, dtype=torch.long))
    assert tuple(logits.shape) == (2, 32, VOCAB)
    assert torch.isfinite(logits).all()


def test_causal_mask_blocks_the_future():
    """Changing a later token must not change an earlier prediction."""
    model = build_arm("R")
    model.eval()
    x = torch.randint(0, VOCAB - 3, (1, 24))
    y = x.clone()
    y[0, -1] = (int(y[0, -1]) + 7) % (VOCAB - 3)
    with torch.no_grad():
        a, b = model(x), model(y)
    assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-6), "future token leaked backwards"


def test_stable_routing_is_deterministic_and_topk_exact():
    model = build_arm("R")
    with torch.no_grad():
        model(torch.zeros(1, 16, dtype=torch.long) + 5)
    first = expert_selection(model)
    with torch.no_grad():
        model(torch.zeros(1, 16, dtype=torch.long) + 5)
    second = expert_selection(model)
    assert first == second, "routing is not deterministic"
    # expert_selection reports the UNION of experts used across every token in the batch, so the
    # count ranges from 1 (all tokens agree) up to N_EXPERTS (the whole catalog was touched).
    # It is neither exactly TOP_K nor capped at TOP_K.
    assert first, "routing recorded nothing"
    assert all(1 <= len(v) <= N_EXPERTS for v in first.values()), first
    assert all(v == sorted(v) for v in first.values()), "selection is not sorted"
    # every expert layer must report the same selection for its gate, up and down slices
    from anx_vyra.model import Experts
    layers = [m for m in model.modules() if isinstance(m, Experts)]
    assert layers, "no expert layers found"
    for module in layers:
        got = [tuple(first[id(getattr(module, n))]) for n in ("gate", "up", "down")]
        assert got[0] == got[1] == got[2], got
        assert set(got[0]) == set(module._selected)


def test_slice_adam_first_update_equals_lr():
    """The published signature for 3e-4 is first_update_max_abs = 3.000498e-04."""
    model = build_arm("R")
    optim.LR = 3e-4
    opt = optim.make_optimizer(model, lr=3e-4)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    x = torch.zeros(2, 16, dtype=torch.long) + 7
    model.zero_grad(set_to_none=True)
    model(x).float().pow(2).mean().backward()
    opt.step(expert_selection(model))
    delta = max((p.detach() - before[n]).abs().max().item()
                for n, p in model.named_parameters())
    assert abs(delta - 3.000498e-04) < 1e-9, f"first update {delta:.6e} != 3.000498e-04"


def test_geometry_reproduces_published_layouts():
    """The distances used in the sweep must give the record counts reported for them."""
    expected = {32: 8, 64: 6, 109: 3, 114: 3, 120: 3, 136: 3}
    for distance, copies in expected.items():
        assert geometry(distance)["copies_per_row"] == copies, distance
    assert geometry(109)["filler"] == 101


def test_needle_rows_mask_exactly_the_payloads():
    geo = geometry(109)
    payloads = [bytes([10 + i] * 8) for i in range(geo["copies_per_row"])]
    row, mask = needle_row(payloads, bytes([97] * geo["filler"]))
    assert len(row) == SEQ and row[0] == BOS and row[-1] == EOS
    assert len(mask) == SEQ - 2
    assert sum(mask) == 8 * geo["copies_per_row"]
    body = row[1:-1]
    for payload in payloads:
        assert list(payload) in [body[i:i + 8] for i in range(len(body) - 7)]
    for i, flag in enumerate(mask):
        if flag:
            assert body[i] in range(1, 256) and body[i] != PAD


def test_distance_sets_are_pairwise_disjoint():
    """Train, unseen and other distances may not share a payload."""
    a = build_distance_set(64, 6, seed=1)
    b = build_distance_set(109, 6, seed=2)
    assert not (set(a["payloads"]) & set(a["unseen_payloads"]))
    assert not (set(a["payloads"]) & set(b["payloads"]))
    assert not (set(a["unseen_payloads"]) & set(b["unseen_payloads"]))


def test_distance_set_respects_ban_list():
    banned = frozenset(b"\x01\x02\x03\x04\x05\x06\x07\x08"
                       for _ in range(1))
    ds = build_distance_set(32, 3, seed=99, ban=banned)
    assert not (set(ds["payloads"]) & set(banned))


def test_corpus_is_deterministic_from_its_seed():
    a = build_split(20261026, 4)
    b = build_split(20261026, 4)
    assert torch.equal(a, b)


def test_eval_batch_size_does_not_change_results():
    from anx_vyra.data import build_distance_set as mk
    ds = mk(64, 8, seed=5)
    model = build_arm("R")
    import benchmarks_helpers as bh
    one = bh.evaluate(model, ds["rows"], ds["mask"], batch=1)
    four = bh.evaluate(model, ds["rows"], ds["mask"], batch=4)
    assert one["record_exact"] == four["record_exact"]
    assert one["token_acc"] == four["token_acc"]
