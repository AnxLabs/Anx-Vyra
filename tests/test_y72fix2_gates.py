"""Y72 retention fix 2: integer gate boundaries, pinned to prereg/y72fix2.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "experiment_packaging"))
import y72fix2_gates as g  # noqa: E402

P = json.loads((ROOT / "prereg" / "y72fix2.json").read_text())


def hist(y=(480, 480, 480), x64=(13, 28, 41), tok=1553, x32=(200, 200, 200), x109=(100, 100, 100)):
    return {u: {"y600": y[i], "x64_seg": x64[i], "x64_tokens": tok if u == 300 else 1500, "x32": x32[i], "x109": x109[i]} for i, u in enumerate((250, 275, 300))}


def test_prereg_pins_module_constants():
    assert P["rule_id"] == g.RULE_ID
    gt = P["gates"]
    assert gt["R_retention"]["sum3"] == 1440 and gt["R_retention"]["each_checkpoint"] == 432
    assert gt["A_acquisition"]["x64_seg_each"] == 11 and gt["A_acquisition"]["tokens_at_300_min"] == 1553
    assert gt["N_nonregression"]["x32_min"] == 128 and gt["N_nonregression"]["x109_min"] == 90
    assert P["design"]["task_clip"] == 0.5 and P["design"]["replay_lambda"] == 0.25
    assert P["design"]["resume_from"]["sha256"] == "a84d35bee3323cf60d0f0a09f0eece5ffb4c1e7bddbd196d4c478df15cb1b283"


def test_boundaries():
    assert g.evaluate(hist())["verdict"] == "SUCCESS"
    assert g.evaluate(hist(y=(480, 480, 479)))["verdict"] == "FAIL_RETENTION"           # sum 1439
    assert g.evaluate(hist(y=(431, 500, 509)))["R_retention"] is False                  # sum 1440 but 431 < 432
    assert g.evaluate(hist(y=(432, 504, 504)))["R_retention"] is True
    assert g.evaluate(hist(tok=1552))["verdict"] == "FAIL_ACQUISITION"
    assert g.evaluate(hist(x64=(10, 28, 41)))["verdict"] == "FAIL_ACQUISITION"
    assert g.evaluate(hist(x32=(127, 200, 200)))["verdict"] == "FAIL_NONREGRESSION"
    assert g.evaluate(hist(x109=(100, 89, 100)))["verdict"] == "FAIL_NONREGRESSION"
    assert g.evaluate({250: hist()[250]})["verdict"] == "NOT_EVALUATED_BEFORE_300"


def test_observed_cpu_probe_trajectories_pass_and_controls_fail():
    assert g.evaluate(hist(y=(528, 519, 528), x64=(13, 28, 27), tok=1624, x32=(0 + 213,) * 3, x109=(114,) * 3))["verdict"] == "SUCCESS"
    assert g.evaluate(hist(y=(363, 390, 454), x64=(29, 46, 92), tok=1687))["R_retention"] is False        # GPU lambda 0.25 control, sum 1207
    assert g.evaluate(hist(y=(450, 421, 420), x64=(17, 29, 42), tok=1618))["R_retention"] is False        # GPU lambda 0.5, sum 1291
