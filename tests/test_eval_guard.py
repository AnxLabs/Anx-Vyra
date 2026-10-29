"""Budget projection and rule-5 threshold. No torch, no bundle."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "experiments" / "z" / "eval_guard.py"
    spec = importlib.util.spec_from_file_location("eval_guard", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def g():
    return _load()


def _at_first_inloop(g, per_step, block=30.3, fixed=38.1, steps=600, every=200, budget=600.0,
                     non_skippable=None):
    """Projection at step 200, before that in-loop eval.

    Baseline has run (one block). The final eval and the metric gate have not,
    so both non-skippable blocks are still ahead. In-loop checks left: 200, 400, 600.
    """
    if non_skippable is None:
        non_skippable = g.NON_SKIPPABLE_BLOCKS
    done_steps = every
    elapsed = done_steps * per_step + block
    remaining_steps = steps - done_steps
    remaining_evals = len(g.in_loop_steps(steps, every))
    return g.decide(
        elapsed, remaining_steps, per_step, remaining_evals, block, fixed, budget,
        non_skippable=non_skippable,
    )


def test_six_block_schedule(g):
    assert g.EVAL_EVERY == 200
    assert g.NON_SKIPPABLE_BLOCKS == 2
    assert g.in_loop_steps(600, 200) == [200, 400, 600]
    # baseline + 3 in-loop + final + gate
    assert g.scheduled_block_count(600, 200) == 6


def test_reference_arithmetic(g):
    # 6 * 30.3 + 600 * 0.465 + 38.1 = 498.9, under the cap, so the guard does not fire.
    # 408.0 is that same projection with the three in-loop blocks removed.
    action, proj, proj_skip = _at_first_inloop(g, 0.465)
    assert action == "run"
    assert proj == pytest.approx(498.9, abs=0.05)
    assert proj_skip == pytest.approx(408.0, abs=0.05)
    assert proj <= 600 * 0.92


def test_two_missing_blocks_are_the_hole(g):
    """Omitting the two non-skippable blocks understates the finish by 60.6s.

    At 0.65 s/step the shortened projection stays under 0.92 * 600 while the
    real finish is 609.9s. Counting the two blocks makes the guard skip the
    in-loop evals and land at 519s. At 0.70 s/step the corrected projection
    skips and lands at 549s.
    """
    silent, short, _ = _at_first_inloop(g, 0.650, non_skippable=0)
    action, full, skipped = _at_first_inloop(g, 0.650, non_skippable=2)
    assert full - short == pytest.approx(2 * 30.3, abs=1e-6)
    assert silent == "run"
    assert full == pytest.approx(609.9, abs=0.05)
    assert action == "skip"
    assert skipped == pytest.approx(519.0, abs=0.05)

    action, full, skipped = _at_first_inloop(g, 0.700, non_skippable=2)
    assert full == pytest.approx(639.9, abs=0.05)
    assert action == "skip"
    assert skipped == pytest.approx(549.0, abs=0.05)
    assert skipped < 600


def test_skipping_cannot_save_a_run_whose_training_blows_the_budget(g):
    # Above ~0.785 s/step, three remaining blocks + training + fixed overhead exceed 600s.
    action, _proj, skipped = _at_first_inloop(g, 0.80)
    assert action == "halt"
    assert skipped > 600
    # Just under that boundary, skipping still fits, so the outcome is skip rather than halt.
    action_low, _proj_low, skipped_low = _at_first_inloop(g, 0.784)
    assert action_low == "skip"
    assert skipped_low < 600


def test_first_block_tolerance(g):
    # Y measured 30.2s, 0.33% from 30.3. The edges of ±25% are 22.725 and 37.875.
    assert g.assert_first_block(30.2) == pytest.approx((30.2 - 30.3) / 30.3)
    assert g.assert_first_block(22.73) < 0
    assert g.assert_first_block(37.87) > 0
    with pytest.raises(SystemExit, match="EVAL_BLOCK_MODEL_MISMATCH"):
        g.assert_first_block(20.0)
    with pytest.raises(SystemExit, match="EVAL_BLOCK_MODEL_MISMATCH"):
        g.assert_first_block(40.0)


def test_rule5_is_one_pass_early_and_does_not_say_five(g):
    # Y's hard-coded 175 is not this number. One pass early on 200 steps and 5 passes is 160.
    assert g.rule5_threshold_step(200, 5) == 160
    assert g.rule5_threshold_step(200, 5) != 175
    assert g.passes_at_step(160, 200, 5) == pytest.approx(4.0)
    assert g.passes_at_step(175, 200, 5) == pytest.approx(4.375)
    message = g.rule5_message("jitter_3e4", 160, 200, 5)
    assert "at 5 passes" not in message
    assert "4.00 passes" in message
    assert "requested 5" in message
    assert "threshold step 160" in message
    # 600 steps, 5 passes: the threshold moves. It is not 175.
    assert g.rule5_threshold_step(600, 5) == 480
