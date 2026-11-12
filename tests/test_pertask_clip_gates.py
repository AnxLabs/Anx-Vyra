"""Pre-registered per-task-clip gates: exact boundaries, every verdict branch, pinned to prereg/pertask_clip_v1.json (specification only)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "experiment_packaging"))
import pertask_clip_gates as g  # noqa: E402
import y72fix_gates as old  # noqa: E402

PREREG = json.loads((ROOT / "prereg" / "pertask_clip_v1.json").read_text())


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
    assert p["rule_id"] == g.RULE_ID == "vyra-pertask-clip-v1"
    assert p["design"]["task_clip_tau"] == g.TASK_CLIP == 0.5 and p["design"]["replay_lambda"] == g.REPLAY_LAMBDA == 0.25
    assert p["gates"]["R_retention"]["threshold"] == {"sum3": g.Y72_SUM3_FLOOR, "per_checkpoint_tenths_pct": g.Y72_GUARD_TENTHS_PCT}
    assert g.Y72_SUM3_FLOOR == 3 * 160 and g.Y72_GUARD_TENTHS_PCT * 200 == 144 * 1000
    assert p["gates"]["A_acquisition"]["threshold"] == {"seg": g.X64_HELDOUT_SEG_FLOOR, "gain_tenths_pp": g.X64_HELDOUT_GAIN_TENTHS_PP}
    assert g.X64_HELDOUT_GAIN_TENTHS_PP * 1920 == 134400 and (1553 - 1418) * 1000 >= 134400 > (1552 - 1418) * 1000
    assert p["design"]["resume_from"]["sha256"] == "a84d35bee3323cf60d0f0a09f0eece5ffb4c1e7bddbd196d4c478df15cb1b283"
    table = {k: v for k, v in p["verdict_table"].items() if k != "NOT_EVALUATED_BEFORE_300"}
    assert set(table) == set(g.VERDICTS.values()) and table["SUCCESS"] == "R and A and N"
    for (R, A, N), name in g.VERDICTS.items():
        if name != "SUCCESS":
            assert table[name] == ", ".join(("" if v else "not ") + n for v, n in ((R, "R"), (A, "A"), (N, "N")))
    assert p["applied_to_observed_runs_for_reference"]["lambda_0_25_global"]["sum3"] == sum(g.CONTROL[u]["y72"] for u in g.CHECKPOINTS)
    assert p["applied_to_observed_runs_for_reference"]["lambda_0_5_global"]["sum3"] == sum(g.LAMBDA05.values())


def test_no_float_arithmetic_in_the_gate_module():
    import ast
    tree = ast.parse((ROOT / "tools" / "experiment_packaging" / "pertask_clip_gates.py").read_text())
    floats = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
    assert floats == [0.25, 0.5], floats   # only the two documentation constants REPLAY_LAMBDA and TASK_CLIP, never compared
    assert not any(isinstance(n, ast.Div) for n in ast.walk(tree)), 'true division is never used in the gate module'


def test_sum3_boundary_480_pass_479_fail_and_only_the_sum_matters():
    assert g.evaluate(hist(y=(160, 160, 160)))["gates"]["R_retention"] is True
    assert g.evaluate(hist(y=(160, 160, 159)))["gates"]["R_retention"] is False
    # 'every checkpoint >= 160' would fail this, the sum rule passes it: 152 + 168 + 160 = 480
    r = g.evaluate(hist(y=(152, 168, 160)))
    assert r["gates"]["R_retention"] is True and old.pct_ge(152, 200, 800) is False
    assert g.evaluate(hist(y=(152, 168, 159)))["gates"]["R_retention"] is False


def test_collapse_guard_144_pass_143_fail_even_when_the_sum_passes():
    assert g.evaluate(hist(y=(144, 176, 160)))["gates"]["R_retention"] is True
    r = g.evaluate(hist(y=(143, 177, 160)))      # sum 480 but one checkpoint below 144
    assert r["gates"]["R_sum3_ge_480"] is True and r["gates"]["R_every_checkpoint_ge_144"] is False and r["gates"]["R_retention"] is False


def test_acquisition_boundaries():
    assert g.evaluate(hist(htok300=1553))["gates"]["A_acquisition"] is True
    r = g.evaluate(hist(htok300=1552))
    assert r["gates"]["A_acquisition"] is False and r["gates"]["A_heldout_gain_225_to_300_ge_7_0_pp"] is False
    assert g.evaluate(hist(hseg=(11, 30, 40)))["gates"]["A_acquisition"] is True
    assert g.evaluate(hist(hseg=(10, 30, 40)))["gates"]["A_acquisition"] is False
    assert g.evaluate(hist(hseg=(20, 30, 10)))["gates"]["A_seg_floor_every_checkpoint"] is False


def test_nonregression_boundaries():
    assert g.evaluate(hist(x32=(128, 200, 200)))["gates"]["N_nonregression"] is True
    assert g.evaluate(hist(x32=(127, 200, 200)))["gates"]["N_nonregression"] is False
    assert g.evaluate(hist(x109=(100, 90, 100)))["gates"]["N_nonregression"] is True
    assert g.evaluate(hist(x109=(100, 100, 89)))["gates"]["N_nonregression"] is False


@pytest.mark.parametrize("R,A,N", [(a, b, d) for a in (True, False) for b in (True, False) for d in (True, False)])
def test_every_verdict_branch(R, A, N):
    h = hist(y=(160,) * 3 if R else (140,) * 3, htok300=1700 if A else 1500, x32=(200,) * 3 if N else (100,) * 3)
    assert g.evaluate(h)["verdict"] == g.VERDICTS[(R, A, N)]


def test_incomplete_history_is_not_evaluated_and_baseline_mismatch_raises():
    h = hist(); del h[300]
    r = g.evaluate(h)
    assert r["verdict"] == "NOT_EVALUATED_BEFORE_300" and "gates" not in r
    bad = hist(); bad[225]["x64"]["heldout"]["tokens_correct"] = 1417
    with pytest.raises(ValueError, match="PERTASK_CLIP_RESUME_BASELINE_MISMATCH"):
        g.evaluate(bad)


def test_observed_runs_fail_retention_under_the_sum_rule_and_the_old_rule_agrees():
    for y in ((137, 137, 153), (152, 147, 152)):
        r = g.evaluate(hist(y=y))
        assert r["gates"]["R_retention"] is False and r["verdict"].startswith("FAIL_RETENTION")
        assert not all(old.pct_ge(v, 200, 800) for v in y)
