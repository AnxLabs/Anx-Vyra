"""Pre-registered gates of the replay-weight experiment: every branch, exact boundaries, pinned to prereg/y72fix_lambda05.json."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "experiment_packaging"))
import y72fix_gates as g  # noqa: E402

PREREG = json.loads((ROOT / "prereg" / "y72fix_lambda05.json").read_text())


def c(tc, se, tokens=1920, segs=240):
    return {"tokens_correct": tc, "tokens": tokens, "segments_exact": se, "segments": segs}


def counts(y=170, x32=200, x109=100):
    return {"Y72_unseen": {"tokens_correct": 1560, "tokens": 1600, "segments_exact": y, "segments": 200},
            "X32_unseen": {"tokens_correct": 2400, "tokens": 2560, "segments_exact": x32, "segments": 320},
            "X109_unseen": {"tokens_correct": 940, "tokens": 960, "segments_exact": x109, "segments": 120}}


def hist(y=(170, 170, 170), x32=(200, 200, 200), x109=(100, 100, 100), hseg=(20, 30, 40), htok300=1700, baseline=True):
    h = {}
    if baseline:
        h[225] = {"x64": {"train": c(1562, 28), "heldout": c(1418, 11)}, "counts": counts(168, 160, 102)}
    toks = (1500, 1600, htok300)
    for i, u in enumerate(g.CHECKPOINTS):
        h[u] = {"x64": {"train": c(1700, 80), "heldout": c(toks[i], hseg[i])}, "counts": counts(y[i], x32[i], x109[i])}
    return h


def test_module_constants_match_the_committed_prereg():
    p = PREREG
    assert p["rule_id"] == g.RULE_ID == "vyra-y72fix-lambda05-v1"
    assert p["design"]["control_value"] == g.SAVED_LAMBDA == 0.25 and p["design"]["treatment_value"] == g.NEW_LAMBDA == 0.5
    assert p["design"]["checkpoints"] == list(g.CHECKPOINTS) == [250, 275, 300]
    assert p["design"]["resume_from"]["update"] == g.RESUME_UPDATE == 225
    assert "160/200" in p["gates"]["R_retention"]["text"] and g.Y72_FLOOR_TENTHS_PCT == 800 and 800 * 200 == 160 * 1000
    assert "128/320" in p["gates"]["N_nonregression"]["text"] and g.X32_FLOOR_TENTHS_PCT == 400 and 400 * 320 == 128 * 1000
    assert "90/120" in p["gates"]["N_nonregression"]["text"] and g.X109_FLOOR_TENTHS_PCT == 750 and 750 * 120 == 90 * 1000
    assert ">= 11" in p["gates"]["A_acquisition"]["text"] and g.X64_HELDOUT_SEG_FLOOR == 11
    assert "1610" in p["gates"]["A_acquisition"]["text"] and g.X64_HELDOUT_GAIN_TENTHS_PP == 100 and 100 * 1920 == 192 * 1000
    for u, row in g.CONTROL.items():
        pc = p["control_trajectory_lambda_0_25"][str(u)]
        assert (row["y72"], row["x32"], row["x109"]) == (pc["y72_seg"], pc["x32_seg"], pc["x109_seg"])
        assert row["x64_heldout"] == (pc["x64_heldout_tokens"], pc["x64_heldout_seg"])
    table = {k: v for k, v in p["verdict_table"].items() if k != "NOT_EVALUATED_BEFORE_300"}
    assert set(table) == set(g.VERDICTS.values())
    assert table["SUCCESS"] == "R and A and N"
    for (R, A, N), name in g.VERDICTS.items():
        if name != "SUCCESS":
            assert table[name] == ", ".join(("" if v else "not ") + n for v, n in ((R, "R"), (A, "A"), (N, "N")))
    assert p["design"]["resume_from"]["sha256"] == "a84d35bee3323cf60d0f0a09f0eece5ffb4c1e7bddbd196d4c478df15cb1b283"


def test_no_float_arithmetic_in_the_gate_module():
    import ast
    tree = ast.parse((ROOT / "tools" / "experiment_packaging" / "y72fix_gates.py").read_text())
    floats = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
    assert floats == [0.5, 0.25] or sorted(floats) == [0.25, 0.5]                   # only the two lambda constants
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)]
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not calls & {"float", "round", "int"}


# ---- gate boundaries --------------------------------------------------------------------------------------
@pytest.mark.parametrize("y, ok", [(160, True), (159, False), (161, True), (200, True), (0, False)])
def test_retention_boundary_at_every_checkpoint(y, ok):
    for pos in range(3):
        ys = [170, 170, 170]; ys[pos] = y
        out = g.evaluate(hist(y=tuple(ys)))
        assert out["per_checkpoint"][str(g.CHECKPOINTS[pos])]["R_y72_ge_80_0"] is ok
        assert out["gates"]["R_retention"] is ok
        assert out["retention_failed_at"] == ([] if ok else [g.CHECKPOINTS[pos]])


@pytest.mark.parametrize("hseg, ok", [((11, 11, 11), True), ((10, 30, 40), False), ((20, 10, 40), False), ((20, 30, 10), False), ((12, 11, 11), True)])
def test_heldout_segment_floor_at_every_checkpoint(hseg, ok):
    out = g.evaluate(hist(hseg=hseg))
    assert out["gates"]["A_seg_floor_every_checkpoint"] is ok and out["gates"]["A_heldout_gain_225_to_300_ge_10_0_pp"] is True
    assert out["gates"]["A_acquisition"] is ok


@pytest.mark.parametrize("tok, ok", [(1610, True), (1609, False), (1611, True), (1418, False), (1920, True)])
def test_heldout_token_gain_boundary_at_300(tok, ok):
    out = g.evaluate(hist(htok300=tok))
    assert out["gates"]["A_heldout_gain_225_to_300_ge_10_0_pp"] is ok and out["gates"]["A_acquisition"] is ok
    assert out["gates"]["A_heldout_gain_pp_exact"] == f"{(tok - 1418) * 100}/1920"


@pytest.mark.parametrize("v, ok", [(128, True), (127, False), (129, True)])
def test_x32_boundary(v, ok):
    for pos in range(3):
        xs = [200, 200, 200]; xs[pos] = v
        out = g.evaluate(hist(x32=tuple(xs)))
        assert out["per_checkpoint"][str(g.CHECKPOINTS[pos])]["N_x32_ge_40_0"] is ok and out["gates"]["N_nonregression"] is ok


@pytest.mark.parametrize("v, ok", [(90, True), (89, False), (91, True)])
def test_x109_boundary(v, ok):
    for pos in range(3):
        xs = [100, 100, 100]; xs[pos] = v
        out = g.evaluate(hist(x109=tuple(xs)))
        assert out["per_checkpoint"][str(g.CHECKPOINTS[pos])]["N_x109_ge_75_0"] is ok and out["gates"]["N_nonregression"] is ok


def test_primitives_are_exact_integers():
    assert g.pct_ge(160, 200, 800) and not g.pct_ge(159, 200, 800)
    assert g.pct_ge(128, 320, 400) and not g.pct_ge(127, 320, 400)
    assert g.pct_ge(90, 120, 750) and not g.pct_ge(89, 120, 750)
    assert g.gain_ge(1610, 1920, 1418, 1920, 100) and not g.gain_ge(1609, 1920, 1418, 1920, 100)
    assert g.gain_ge(110, 100, 100, 100, 100) and not g.gain_ge(109, 100, 100, 100, 100)       # 10.0 pp exactly vs 9.0 pp
    assert g.gain_ge(21, 200, 100, 1000, 5) and not g.gain_ge(20, 200, 100, 1000, 5)          # different denominators: 10.5 vs 10.0 pp, thresh 0.5pp
    assert g.gain_pp_exact(1610, 1920, 1418, 1920) == "19200/1920"


# ---- verdict table: all eight combinations ------------------------------------------------------------------
BAD = {"R": dict(y=(170, 159, 170)), "A": dict(htok300=1609), "N": dict(x109=(100, 100, 89))}


@pytest.mark.parametrize("R, A, N", [(r, a, n) for r in (True, False) for a in (True, False) for n in (True, False)])
def test_verdict_table_all_eight_branches(R, A, N):
    kw = {}
    for name, ok in (("R", R), ("A", A), ("N", N)):
        if not ok:
            kw.update(BAD[name])
    out = g.evaluate(hist(**kw))
    assert (out["gates"]["R_retention"], out["gates"]["A_acquisition"], out["gates"]["N_nonregression"]) == (R, A, N)
    assert out["verdict"] == g.VERDICTS[(R, A, N)]
    assert out["success_claim_allowed"] is (R and A and N)
    parts = [n for n, ok in (("RETENTION", R), ("ACQUISITION", A), ("NONREGRESSION", N)) if not ok]
    assert out["verdict"] == ("SUCCESS" if not parts else "FAIL_" + "_".join(parts))
    assert out["always_stops_at"] == 300 and out["automatic_followup"] is None and out["final"] is True


def test_partial_histories_never_give_a_verdict_but_report_retention_early():
    h = hist(y=(150, 170, 170))
    del h[300]
    out = g.evaluate(h)
    assert out["verdict"] == "NOT_EVALUATED_BEFORE_300" and out["final"] is False and "gates" not in out
    assert out["retention_failed_at"] == [250] and out["checkpoints_seen"] == [250, 275]
    del h[275]
    assert g.evaluate(h)["verdict"] == "NOT_EVALUATED_BEFORE_300"
    assert g.evaluate({225: hist()[225]})["verdict"] == "NOT_EVALUATED_BEFORE_300"
    h = hist(); del h[250]
    assert g.evaluate(h)["verdict"] == "NOT_EVALUATED_BEFORE_300"        # 275 and 300 present, 250 missing: still no verdict


def test_baseline_at_225_must_match_the_pinned_resume_counts():
    h = hist()
    h[225]["x64"]["heldout"]["tokens_correct"] = 1419
    with pytest.raises(ValueError, match="Y72FIX_RESUME_BASELINE_MISMATCH"):
        g.evaluate(h)
    out = g.evaluate(hist(baseline=False))
    assert out["verdict"] == "SUCCESS"


def test_paired_control_differences_are_exact_integers_and_nongating():
    out = g.evaluate(hist(y=(160, 165, 170), x32=(210, 230, 270), x109=(99, 100, 100), hseg=(30, 40, 90), htok300=1700))
    p = out["paired_vs_control_lambda025_nongating"]
    assert p["250"]["y72_seg_minus_control"] == 160 - 137 and p["300"]["y72_seg_minus_control"] == 170 - 153
    assert p["275"]["x32_seg_minus_control"] == 230 - 235 and p["300"]["x64_heldout_tokens_minus_control"] == 1700 - 1687
    assert p["250"]["x109_seg_minus_control"] == 0 and all(isinstance(v, int) for d in p.values() for v in d.values())
    assert out["verdict"] == "SUCCESS"                                   # comparison to the control never changes the verdict
    worse = g.evaluate(hist(x32=(130, 130, 130)))
    assert worse["verdict"] == "SUCCESS"                                  # far below the control, still above the pre-registered floor


def test_the_observed_control_trajectory_itself_would_have_passed_A_and_N_and_failed_R():
    """The lambda 0.25 run (observed): Y72 137/137/153 -> R fails; X64 held-out 29/46/92 and 1687 tokens pass A; X32/X109 pass N."""
    h = {225: {"x64": {"train": c(1562, 28), "heldout": c(1418, 11)}, "counts": counts(168, 160, 102)}}
    for u in g.CHECKPOINTS:
        k = g.CONTROL[u]
        h[u] = {"x64": {"train": c(*k["x64_train"][:1], k["x64_train"][1]), "heldout": c(k["x64_heldout"][0], k["x64_heldout"][1])},
                "counts": counts(k["y72"], k["x32"], k["x109"])}
    out = g.evaluate(h)
    assert out["gates"]["R_retention"] is False and out["gates"]["A_acquisition"] is True and out["gates"]["N_nonregression"] is True
    assert out["verdict"] == "FAIL_RETENTION" and out["retention_failed_at"] == [250, 275, 300]
