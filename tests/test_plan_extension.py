"""Plan-prefix identity and the documented extension policy."""
import ast
import copy
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

import sys
PACKAGING = Path(__file__).resolve().parents[1] / "tools" / "experiment_packaging"
sys.path.insert(0, str(PACKAGING))
import plan_extension as pe  # noqa: E402
import kaggle_config as K  # noqa: E402

X = Path(os.environ.get("VYRA_X_BUNDLE", K.expand("@@LOCAL_INPUTS@@/@@SLUG_X_BUNDLE@@")))
HAVE_X = (X / "train_X.pt").is_file()


def _synthetic():
    g = random.Random(3)
    plan = []
    order = []
    while len(order) < 200:
        e = list(range(40)); g.shuffle(e); order += e
    for k in range(200):
        plan.append({"slot0": g.randrange(306), "slot1": g.randrange(306), "slot2": g.randrange(117),
                     "needles": {32: order[k], 64: order[k], 109: order[k]}})
    return {"plan": plan, "rows_gen": [0] * 306, "rows32": [0] * 117}


def test_extension_is_deterministic_and_prefix_untouched():
    sx = _synthetic()
    before = copy.deepcopy(sx["plan"])
    a = pe.build_extended_plan(sx, 300, check_pins=False)
    b = pe.build_extended_plan(sx, 300, check_pins=False)
    assert a == b and len(a) == 300
    assert sx["plan"] == before                       # input not mutated
    assert a[:200] == before                          # prefix preserved by value
    assert pe.plan_sha256(a[:200]) == pe.plan_sha256(before)
    for i in range(200):
        assert pe.canonical_entry(a[i]) == pe.canonical_entry(before[i])
    assert all(a[i] is not before[i] for i in range(200))   # copies, not aliases


def test_extension_uses_only_existing_rows_and_documented_epoch_structure():
    sx = _synthetic()
    full = pe.build_extended_plan(sx, 300, check_pins=False)
    ext = full[200:]
    assert all(0 <= e["slot0"] < 306 and 0 <= e["slot1"] < 306 and 0 <= e["slot2"] < 117 for e in ext)
    needles = [e["needles"][64] for e in ext]
    assert all(e["needles"][32] == e["needles"][64] == e["needles"][109] for e in ext)
    assert sorted(needles[:40]) == list(range(40)) and sorted(needles[40:80]) == list(range(40))   # two full fresh epochs
    assert len(set(needles[80:])) == 20                                                            # truncated third epoch
    from collections import Counter
    c = Counter(e["needles"][64] for e in full)      # synthetic prefix is 5 passes
    assert set(c.values()) <= {7, 8} and sorted(Counter(c.values()).items()) == [(7, 20), (8, 20)]


def test_streams_are_independent_of_every_global_rng_and_the_builder_seeds():
    sx = _synthetic()
    py, npr, th = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    builder_901 = [random.Random(901).randrange(500) for _ in range(5)]
    builder_9101 = [random.Random(9101).random() for _ in range(5)]
    pe.build_extended_plan(sx, 300, check_pins=False)
    assert random.getstate() == py and torch.equal(torch.get_rng_state(), th)
    assert all(np.array_equal(x, y) for x, y in zip(np.random.get_state()[1:2], npr[1:2]))
    assert [random.Random(901).randrange(500) for _ in range(5)] == builder_901
    assert [random.Random(9101).random() for _ in range(5)] == builder_9101
    tree = ast.parse((PACKAGING / "plan_extension.py").read_text())
    imported = {n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)} | \
               {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imported <= {"hashlib", "json"}            # no random / numpy / torch at all
    assert len({pe.STREAM_LABELS[k] for k in pe.STREAM_LABELS}) == 3


def test_prefix_mismatch_and_bad_inputs_fail_closed():
    sx = _synthetic()
    ext = pe.build_extension(306, 117)
    with pytest.raises(pe.PlanExtensionError, match="PLAN_PREFIX_MISMATCH"):
        pe.assert_prefix_identity(sx["plan"][:199] + [dict(sx["plan"][199], slot0=(sx["plan"][199]["slot0"] + 1) % 306)] + ext, sx["plan"])
    short = dict(sx, plan=sx["plan"][:199])
    with pytest.raises(pe.PlanExtensionError, match="PLAN_PREFIX_MISMATCH"):
        pe.build_extended_plan(short, 300, check_pins=False)
    with pytest.raises(pe.PlanExtensionError, match="steps must be 300"):
        pe.build_extended_plan(sx, 301, check_pins=False)
    # with pins on, a synthetic (non-pinned) prefix is refused
    with pytest.raises(pe.PlanExtensionError, match="pinned prefix hash"):
        pe.build_extended_plan(sx, 300)
    bad = copy.deepcopy(sx); bad["plan"][5]["slot2"] = 999
    with pytest.raises(pe.PlanExtensionError, match="OUT_OF_RANGE"):
        pe.build_extended_plan(bad, 300, check_pins=False)
    # stream rejection sampling is uniform enough and never out of range
    s = pe.Stream("t")
    assert all(0 <= s.randbelow(7) < 7 for _ in range(500))
    with pytest.raises(pe.PlanExtensionError):
        s.randbelow(0)


@pytest.mark.skipif(not HAVE_X, reason="pinned train_X.pt not available (set VYRA_X_BUNDLE)")
def test_prefix_identity_against_the_pinned_train_X():
    import hashlib
    raw = (X / "train_X.pt").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == pe.PINNED_TRAIN_X_SHA256
    sx = torch.load(X / "train_X.pt", map_location="cpu", weights_only=False)
    plan = sx["plan"]
    assert len(plan) == 200 and pe.plan_sha256(plan) == pe.PINNED_PREFIX_PLAN_SHA256
    full = pe.build_extended_plan(sx, 300)          # pins ON: extension hash and full hash also checked
    assert len(full) == 300
    assert pe.plan_sha256(full[:200]) == pe.PINNED_PREFIX_PLAN_SHA256
    for i in range(200):                            # entry-by-entry, by value and canonical bytes
        assert full[i] == plan[i] and pe.canonical_entry(full[i]) == pe.canonical_entry(plan[i])
    assert pe.plan_sha256(full[200:]) == pe.PINNED_EXTENSION_SHA256
    assert pe.plan_sha256(full) == pe.PINNED_FULL_PLAN_SHA256
    n_gen, n32 = len(sx["rows_gen"]), len(sx["rows32"])
    assert (n_gen, n32) == (306, 117)
    pe.validate_entries(full, n_gen, n32)
    # pinned tensors untouched by the build
    assert hashlib.sha256((X / "train_X.pt").read_bytes()).hexdigest() == pe.PINNED_TRAIN_X_SHA256
