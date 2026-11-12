"""Y72 rehearsal2 (fresh once-only pool F200): gates, pool pins and schedule, manipulation checks, sampler relabel, generator edits,
builder pins and the built notebook (env-gated). Pinned to prereg/y72_rehearsal2.json (committed at fbc9638 before any of this code)."""
import ast
import hashlib
import itertools
import json
import os
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "experiment_packaging"))
sys.path.insert(0, str(ROOT / "tests"))

import continuation as cont  # noqa: E402
import continue225_y72fix2 as c2  # noqa: E402
import continue300_y72fix2 as c4  # noqa: E402
import continue_y72_rehearsal2 as cr  # noqa: E402
import resume300_y72fix2 as r3  # noqa: E402
import y72fix2_cont400_gates as g400  # noqa: E402
import y72_rehearsal2_gates as gates  # noqa: E402
import y72_rehearsal2_pool as pool  # noqa: E402
import y72_rehearsal2_runtime as rt  # noqa: E402

PREREG = ROOT / "prereg" / "y72_rehearsal2.json"
PROSE = ROOT / "prereg" / "y72_rehearsal2.md"
PREREG_SHA256 = "b02c4510ea2072b715a324715506b97db2f66fbe2d356f637c04d38abc577cab"
PROSE_SHA256 = "4354a1fb233df5c583d6be3037b57a50fe9860f8b34f06936593184210f15d40"
POOL_FILE = ROOT / "prereg" / "y72_rehearsal2" / "y72_fresh_pool_F200_set.json"
P = json.loads(PREREG.read_text())
G = P["gates"]


def test_prereg_files_are_the_registered_ones():
    assert hashlib.sha256(PREREG.read_bytes()).hexdigest() == PREREG_SHA256
    assert hashlib.sha256(PROSE.read_bytes()).hexdigest() == PROSE_SHA256
    import build_y72_rehearsal2 as b
    assert b.PREREG_SHA256 == PREREG_SHA256 and b.PREREG_FILE == PREREG and b.PROSE_SHA256 == PROSE_SHA256 and b.POOL_FILE == POOL_FILE


def test_gate_module_matches_the_registration():
    assert gates.RULE_ID == P["rule_id"] == pool.RULE_ID == "vyra-y72-rehearsal2-fresh-v1"
    assert list(gates.CHECKPOINTS) == P["design"]["gate_checkpoints"] == [350, 375, 400]
    assert list(gates.OBSERVATION_ONLY) == [P["design"]["observation_only_checkpoint"]]
    assert 3 * gates.Y600_FLOOR_TENTHS_PCT * 600 // 1000 == G["R_hold"]["sum3"] == 1440
    assert gates.Y600_GUARD_TENTHS_PCT * 600 // 1000 == G["R_hold"]["each_checkpoint"] == 432
    assert gates.X64_HELDOUT_SEG_FLOOR == G["A_progress"]["x64_seg_each"] == 44
    assert gates.RESUME_X64_HELDOUT["tokens_correct"] == 1665 and G["A_progress"]["tokens_at_400_min"] == 1723
    assert 320 * 400 // 1000 == G["N_nonregression"]["x32_min"] and 120 * 750 // 1000 == G["N_nonregression"]["x109_min"]
    assert gates.TASK_CLIP_TENTHS * 10 == round(P["design"]["task_clip"] * 100) and gates.REPLAY_LAMBDA_NUM_DEN == (1, 4)
    assert P["design"]["replay_lambda"] == 0.25 and P["design"]["replay_rows_per_update"] == pool.ROWS_PER_UPDATE == 2
    for key in ("SUCCESS", "FAIL_HOLD", "FAIL_PROGRESS", "FAIL_NONREGRESSION", "NOT_EVALUATED_BEFORE_400"):
        assert key in P["verdict_table"]


def _h(y=(480, 480, 480), seg=(44, 44, 44), tok=1723, x32=128, x109=90):
    return {u: {"y600": y[i], "x64_seg": seg[i], "x64_tokens": tok if u == 400 else 1700, "x32": x32, "x109": x109} for i, u in enumerate((350, 375, 400))}


def test_boundaries_are_exactly_the_registered_ones():
    assert gates.evaluate(_h())["verdict"] == "SUCCESS"
    assert gates.evaluate(_h(y=(480, 480, 479)))["verdict"] == "FAIL_HOLD"
    assert gates.evaluate(_h(y=(432, 504, 504)))["verdict"] == "SUCCESS"
    assert gates.evaluate(_h(y=(431, 505, 505)))["verdict"] == "FAIL_HOLD"
    assert gates.evaluate(_h(tok=1722))["verdict"] == "FAIL_PROGRESS"
    assert gates.evaluate(_h(seg=(43, 60, 60)))["verdict"] == "FAIL_PROGRESS"
    assert gates.evaluate(_h(x32=127))["verdict"] == "FAIL_NONREGRESSION"
    assert gates.evaluate(_h(x109=89))["verdict"] == "FAIL_NONREGRESSION"
    assert gates.evaluate(_h(y=(504, 396, 573), seg=(46, 124, 82), tok=1658))["verdict"] == "FAIL_HOLD_PROGRESS"   # the control's scores
    for k, table in G["R_hold"]["boundary"].items():
        for v, want in table.items():
            ys = (int(v) // 3, int(v) // 3, int(v) - 2 * (int(v) // 3)) if k == "sum3" else (int(v), 600, 600)
            assert gates.evaluate(_h(y=ys))["R_hold"] is want
    for v, want in G["A_progress"]["boundary"]["tokens_at_400"].items():
        assert gates.evaluate(_h(tok=int(v)))["A_progress"] is want
    for v, want in G["A_progress"]["boundary"]["seg"].items():
        assert gates.evaluate(_h(seg=(int(v), 60, 60)))["A_progress"] is want
    assert gates.evaluate({350: _h()[350]})["verdict"] == "NOT_EVALUATED_BEFORE_400"


def test_gates_are_the_cont400_gates_unchanged():
    rnd = random.Random(7)
    for _ in range(3000):
        h = {u: {"y600": rnd.randint(380, 600), "x64_seg": rnd.randint(30, 140), "x64_tokens": rnd.randint(1600, 1800),
                 "x32": rnd.randint(110, 320), "x109": rnd.randint(80, 120)} for u in (350, 375, 400)}
        a, b = gates.evaluate(h), g400.evaluate(h)
        assert a.pop("rule") == "vyra-y72-rehearsal2-fresh-v1" and b.pop("rule") == "vyra-y72fix2-cont400-v1" and a == b
    with pytest.raises(ValueError):
        gates.evaluate(_h(y=(480.0, 480, 480)))


def test_history_feeds_the_gate_and_refuses_bad_denominators():
    def rec(y):
        return {"counts": {"Y72v2_600": {"segments": 600, "segments_exact": y}, "X32_unseen": {"segments": 320, "segments_exact": 200},
                           "X109_unseen": {"segments": 120, "segments_exact": 100}},
                "x64": {"heldout": {"segments": 240, "segments_exact": 50, "tokens": 1920, "tokens_correct": 1730}}}
    H = {325: rec(1), 350: rec(500), 375: rec(500), 400: rec(500)}
    assert set(rt.history(H)) == {350, 375, 400} and gates.evaluate(rt.history(H))["verdict"] == "SUCCESS"
    H[400]["x64"]["heldout"]["tokens"] = 1919
    with pytest.raises(ValueError, match="DENOMINATOR"):
        rt.history(H)


def test_pool_pins_equal_the_registration():
    f = P["design"]["fresh_pool_F200"]
    assert pool.CANONICAL_SHA256 == f["canonical_rows_mask_sha256"] and pool.PAYLOAD_LIST_SHA256 == f["payload_list_sha256"]
    assert list(pool.SEEDS) == f["seeds"] and pool.POOL_ROWS == f["rows"] == 200 and pool.COPIES == f["copies_per_row"]
    assert pool.TEXT_SHA256 == P["design_evidence"]["pool_text_form_sha256"] == hashlib.sha256(POOL_FILE.read_bytes()).hexdigest()
    assert f["text_form"] == "prereg/y72_rehearsal2/" + pool.POOL_ASSET
    d = pool.load_pool(POOL_FILE)
    assert tuple(d["rows"].shape) == (200, 512) and tuple(d["mask"].shape) == (200, 510) and int(d["mask"].sum()) == 200 * 40


def test_pool_is_payload_disjoint_from_the_600_set_and_the_200_set():
    import y72_heldout_v2 as h600
    s6 = h600.verify_text_set(ROOT / "prereg" / "y72_heldout_v2" / "y72_heldout_v2_set.json")
    pp = json.loads(POOL_FILE.read_text())["payloads"]
    assert not set(pp) & set(s6["payloads"])
    rows = bytes.fromhex(json.loads(POOL_FILE.read_text())["rows_int16le_hex"])
    assert rows not in bytes.fromhex(s6["rows_int16le_hex"])


def test_wrong_pools_are_refused(tmp_path):
    with pytest.raises(pool.PoolError, match="POOL_MISSING"):
        pool.verify_text_set(tmp_path / "absent.json")
    with pytest.raises(pool.PoolError, match="POOL_TEXT_SHA256"):
        pool.verify_text_set(ROOT / "prereg" / "y72_heldout_v2" / "y72_heldout_v2_set.json")
    d = json.loads(POOL_FILE.read_text())
    h = d["rows_int16le_hex"]
    d["rows_int16le_hex"] = h[:1200] + ("41" if h[1200:1202] != "41" else "42") + h[1202:]
    bad = tmp_path / pool.POOL_ASSET
    bad.write_text(json.dumps(d, indent=1))
    with pytest.raises(pool.PoolError, match="POOL_TEXT_SHA256"):
        pool.verify_text_set(bad)
    monkey = pool.TEXT_SHA256
    try:                                   # even with the file pin bypassed, the canonical hash refuses the changed rows
        pool.TEXT_SHA256 = hashlib.sha256(bad.read_bytes()).hexdigest()
        with pytest.raises(pool.PoolError, match="POOL_CANONICAL"):
            pool.verify_text_set(bad)
    finally:
        pool.TEXT_SHA256 = monkey


def test_schedule_presents_every_pool_row_exactly_once():
    sch = pool.schedule()
    assert len(sch) == 100 and sch[0] == (301, [0, 1]) and sch[-1] == (400, [198, 199])
    flat = [i for _, ix in sch for i in ix]
    assert sorted(flat) == list(range(200)) and len(set(flat)) == 200
    assert P["design"]["fresh_pool_F200"]["order"].startswith("pool row index = 2*(u-301)+j")
    for s in (299, 400, -1, True, 300.0):
        with pytest.raises(pool.PoolError):
            pool.indices(s)


def _rec(n, pull=0.5, loss=0.1, first=301):
    tc = [{"alpha_y": pull / 2.0, "norm_gy": 2.0} for _ in range(n)]
    ls = [[0.3, loss] for _ in range(n)]
    rows = [pool.indices(first - 1 + i) for i in range(n)]
    return tc, ls, rows


def test_manipulation_checks_are_nongating_and_registered():
    mc = P["manipulation_checks_nongating"]
    assert "0.10" in mc["M1_no_starved_window"] and "0.005" in mc["M2_not_memorized"] and "exactly once" in mc["M3_presentation_once"]
    assert pool.M1_MIN_PULL == 0.10 and pool.M2_MIN_LOSS == 0.005 and pool.WINDOWS == ((301, 325), (326, 350), (351, 375), (376, 400))
    ok = pool.manipulation_checks(*_rec(100))
    assert ok["nongating"] and ok["complete_run"] and ok["M1_no_starved_window"] and ok["M2_not_memorized"] and ok["M3_presentation_once"]
    tc, ls, rows = _rec(100)
    for i in range(50, 75):
        tc[i] = {"alpha_y": 0.25, "norm_gy": 0.2}      # pull 0.05 in 351-375
        ls[i] = [0.3, 0.004]
    bad = pool.manipulation_checks(tc, ls, rows)
    assert bad["M1_no_starved_window"] is False and bad["M2_not_memorized"] is False and bad["M3_presentation_once"] is True
    tc, ls, rows = _rec(100)
    rows[10] = rows[9]
    assert pool.manipulation_checks(tc, ls, rows)["M3_presentation_once"] is False
    part = pool.manipulation_checks(*_rec(2))
    assert part["complete_run"] is False and part["M1_no_starved_window"] is None and part["M3_presentation_once"] is True
    with pytest.raises(pool.PoolError):
        pool.manipulation_checks(tc[:3], ls, rows)


def test_sampler_record_is_relabelled_only_after_300():
    class PM:
        POLICY_VERSION = "p"
        @staticmethod
        def plan_sha256(x):
            return "h%d" % len(x)
    plan = list(range(400))
    w = pool.wrap_sampler_record(cont.sampler_record)
    assert w(300, plan[:300], PM, 2) == cont.sampler_record(300, plan[:300], PM, 2)      # the saved checkpoint_300 record still matches
    r = w(325, plan, PM, 2)
    assert r["replay_rule"] == pool.REPLAY_RULE and r["replay_next_idx"] == [50, 51] and r["replay_source"]["canonical_sha256"] == pool.CANONICAL_SHA256
    assert w(400, plan, PM, 2)["replay_next_idx"] == []
    assert {k: v for k, v in r.items() if k not in ("replay_rule", "replay_next_idx", "replay_source")} == \
        {k: v for k, v in cont.sampler_record(325, plan, PM, 2).items() if k not in ("replay_rule", "replay_next_idx")}


import test_y72fix2_cont400 as t400  # noqa: E402


def _synthetic():
    base = t400._synthetic()
    base = base.replace("      for s in range(START, STEPS):\n", "      for s in range(START, STEPS):\n          " + cr.CALL_OLD + "\n", 1)
    return cr.POOLLOAD_OLD + cr.REPLAY_OLD + base


@pytest.fixture
def identity_300(monkeypatch):
    monkeypatch.setattr(c2, "apply_continue300", lambda s: s)


def test_generator_edits_are_exact(identity_300):
    out = cr.apply_rehearsal2(_synthetic())
    assert "r2pool=_rilu.module_from_spec" in out and "r2pool.load_pool(" in out and "y72_fresh_pool_F200_set.json" in out
    assert "idx=r2pool.indices(s)" in out and "if s<START:" in out and "A.setdefault('replay_rows',[])" in out
    assert "cont.sampler_record=r2pool.wrap_sampler_record(cont.sampler_record)" in out
    assert "gates=_cn_load('y72_rehearsal2_gates')" in out and "c4rt=_cn_load('y72_rehearsal2_runtime')" in out
    assert "R['y72_rehearsal2_verdict']=gates.evaluate(c4rt.history(HIST))" in out and "r2pool.manipulation_checks(" in out
    assert "STEPS=400; START=300; RB=2; EVAL_AT={325,350,375,400}" in out and "clip_scheme.combine(_pp,_gx,_gy,lam,'pertask',0.5)" in out
    assert "assert lam==0.25 and r300.NEW_LAMBDA==0.25" in out and out.count("opt.step(selected)") == 1
    for banned in cr.BANNED:
        assert banned not in out
    assert out.index("cont.sampler_record=r2pool") > out.index("r300.load_checkpoint300(")


def test_generator_refuses_missing_duplicate_and_reapplied_sources(identity_300, monkeypatch):
    with pytest.raises(ValueError, match="PATTERN_MISSING"):
        cr.apply_rehearsal2(cr.apply_rehearsal2(_synthetic()))
    good = c4.apply_cont400(_synthetic())
    monkeypatch.setattr(c4, "apply_cont400", lambda s: s)
    cr.apply_rehearsal2(good)
    for label, old, _new in cr.SWAPS:
        with pytest.raises(ValueError, match=f"REHEARSAL2_PATTERN_MISSING {label}"):
            cr.apply_rehearsal2(good.replace(old, "", 1))
        with pytest.raises(ValueError, match=f"REHEARSAL2_PATTERN_MISSING {label}"):
            cr.apply_rehearsal2(good + "\n" + old)
    with pytest.raises(ValueError, match="RESIDUE"):
        cr.apply_rehearsal2(good + "\nx='y72fix2_cont400_gates'\n")
    assert [s[0] for s in cr.SWAPS] == ["poolload", "replay", "call", "load", "gate", "prov", "eval", "final"]


def test_builder_refuses_a_changed_pool_or_prereg(tmp_path, monkeypatch):
    import build_y72_rehearsal2 as b
    b.check_pool(POOL_FILE)
    bad = tmp_path / pool.POOL_ASSET
    bad.write_text(POOL_FILE.read_text().replace('"copies": 5', '"copies": 5 ', 1))
    with pytest.raises(ValueError, match="POOL_CHANGED"):
        b.check_pool(bad)
    with pytest.raises(ValueError, match="POOL_CHANGED"):
        b.check_pool(tmp_path / "absent.json")
    monkeypatch.setattr(b.r2pool, "CANONICAL_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="POOL_PIN_NOT_REGISTERED"):
        b.check_pool(POOL_FILE)
    assert "y72_rehearsal2_pool.py" in b.EXTRA_ASSETS and "y72_rehearsal2_gates.py" in b.EXTRA_ASSETS
    assert "y72fix2_cont400_gates.py" not in b.EXTRA_ASSETS


NB = os.environ.get("VYRA_Y72_REHEARSAL2_NOTEBOOK")


@pytest.mark.skipif(not NB, reason="built notebook path not provided")
def test_built_notebook_assets_and_hashes():
    code = "".join(json.loads(Path(NB).read_text())["cells"][0]["source"])
    node = next(n for n in ast.parse(code).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "assets")
    assets = ast.literal_eval(node.value)
    for name in ("y72_rehearsal2_gates.py", "y72_rehearsal2_runtime.py", "y72_rehearsal2_pool.py", "y72_fresh_pool_F200_set.json",
                 "resume300_y72fix2.py", "plan_extension400.py", "y72fix2_runtime.py", "clip_scheme.py", "y72_heldout_v2_set.json",
                 "prereg_y72_rehearsal2.json"):
        assert name in assets
    for name in ("y72fix2_cont400_gates.py", "y72fix2_cont400_runtime.py", "prereg_y72fix2_cont400.json", "resume225_y72fix2.py"):
        assert name not in assets
    assert hashlib.sha256(assets["prereg_y72_rehearsal2.json"].encode()).hexdigest() == PREREG_SHA256
    assert hashlib.sha256(assets["y72_fresh_pool_F200_set.json"].encode()).hexdigest() == pool.TEXT_SHA256
    t = assets["treatment.py"]
    assert "STEPS=400; START=300" in t and "idx=r2pool.indices(s)" in t and "clip_scheme.combine(_pp,_gx,_gy,lam,'pertask',0.5)" in t
    for banned in cr.BANNED:
        assert banned not in t
    assert "@@" not in code
    lock = json.loads(assets["dependency_lock.json"])
    assert lock["datasets"]["C"]["files"] == {"checkpoint_300.pt": r3.CKPT300_SHA256} and lock["datasets"]["C"]["bytes"] == r3.CKPT300_BYTES
    for k, v in lock["assets"].items():
        assert hashlib.sha256(assets[k].encode()).hexdigest() == v
    meta = json.loads((Path(NB).parent / "kernel-metadata.json").read_text())
    assert meta["is_private"] is True and meta["enable_internet"] is False and meta["machine_shape"] == "NvidiaTeslaT4"
    assert len(meta["dataset_sources"]) == 3
