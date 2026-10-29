"""Tests for Experiment Z exposure arithmetic and the adaptive eval budget guard.

These encode the invariants that the Y builder asserted only for Y and that the trainer
previously got wrong. Every one of them is a defect that was live in this branch.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def derive(targets, jitter, n_tr, steps):
    """The builder's exposure derivation, mirrored so it can be tested without torch."""
    all_geom = sorted({t + j for t in targets for j in jitter})
    assert len(all_geom) == len(targets) * len(jitter), "bands overlap"
    epoch_steps = len(jitter) * n_tr
    if steps % len(jitter) != 0:
        raise ValueError("EXPOSURE_INEXACT: steps not divisible by len(jitter)")
    passes = steps // epoch_steps
    if steps % epoch_steps != 0:
        raise ValueError("EXPOSURE_INEXACT: steps not a whole number of epochs")
    pres_geom = steps // len(jitter)
    if pres_geom != passes * n_tr:
        raise ValueError("EXPOSURE_INCONSISTENT")
    return {"geometries": len(all_geom), "epoch_steps": epoch_steps, "passes": passes,
            "presentations_per_geometry": pres_geom}


# ---- the two approved arms -------------------------------------------------------------

def test_arm_a_is_exact():
    d = derive([32, 64, 109], [-4, 0, 4], 40, 600)
    assert d["passes"] == 5
    assert d["epoch_steps"] == 120
    assert 600 % 120 == 0
    assert d["geometries"] == 9
    assert d["presentations_per_geometry"] == 200


def test_arm_b_is_exact():
    d = derive([32, 64, 109], [0], 8, 200)
    assert d["passes"] == 25
    assert d["epoch_steps"] == 8
    assert 200 % 8 == 0
    assert d["geometries"] == 3
    assert d["presentations_per_geometry"] == 200


def test_arm_a_and_b_deliver_the_same_presentations_per_geometry():
    """A and B are matched on per-geometry exposure even though they differ in passes."""
    a = derive([32, 64, 109], [-4, 0, 4], 40, 600)
    b = derive([32, 64, 109], [0], 8, 200)
    assert a["presentations_per_geometry"] == b["presentations_per_geometry"] == 200
    assert a["passes"] != b["passes"]


# ---- the Y-only assertion must not come back -------------------------------------------

def test_y_parameters_would_have_failed_a_hardcoded_five_pass_assert():
    """Y asserted PRESENT_PER_GEOM // N_TR == 5. Arm A is 200 // 40 = 5, Arm B is 200 // 8 = 25,
    so that assertion refuses Arm B. This test exists so nobody restores it."""
    b = derive([32, 64, 109], [0], 8, 200)
    assert b["presentations_per_geometry"] // 8 == 25 != 5


@pytest.mark.parametrize("steps,epoch", [(100, 120), (601, 120), (250, 8), (201, 8)])
def test_inexact_parameterisations_refuse_to_build(steps, epoch):
    """A builder that quietly rounds to make a run fit is the defect. It must raise."""
    n_tr = epoch // 3 if epoch == 120 else 8
    jitter = [-4, 0, 4] if epoch == 120 else [0]
    if steps % epoch != 0:
        with pytest.raises(ValueError):
            derive([32, 64, 109], jitter, n_tr, steps)


# ---- the adaptive eval budget guard ----------------------------------------------------

def project(step, steps, eval_every, per_block, pure_s, deadline=520.0, frac=0.94):
    """Mirror of the trainer's guard.

    Blocks-left MUST come from the step position, not from how many evals have run, or the guard
    stops guarding after its own first skip. Elapsed time MUST include the eval blocks already
    spent, because the real guard measures (monotonic - T0) and evaluations are inside that; a
    mirror that counts only pure training understates the projection and silently passes runs
    that would blow the cap.
    """
    block_no = step // eval_every            # this block, 1-based
    in_loop = steps // eval_every
    after = in_loop - block_no               # blocks still to come after this one
    evals_already_run = block_no - 1         # every earlier in-loop block has completed
    # The two non-skippable blocks that run BEFORE training (baseline, metric gate) are inside
    # (monotonic - T0) in the real guard. Omitting them here is what made this mirror disagree
    # with the code; they are real seconds on the clock and must be projected.
    startup_blocks = 2
    elapsed = step * pure_s + (evals_already_run + startup_blocks) * per_block
    return (elapsed
            + per_block                      # this block
            + (steps - step) * pure_s        # remaining training
            + after * per_block              # remaining in-loop blocks
            + per_block)                     # the final measurement


def test_guard_projection_does_not_collapse_as_the_run_proceeds():
    """Regression: the first implementation derived blocks-remaining from how many evals had RUN,
    so after its own first skip the projection lost the remaining blocks and the guard stopped
    guarding. A valid estimator projects a stable total at a constant rate: it must not shrink
    toward 'almost done' just because the run is nearer its end.
    """
    # 600 steps of pure training + 3 in-loop blocks + 3 non-skippable blocks
    true_total = 600 * 0.62 + 6 * 30.4
    proj_200 = project(200, 600, 200, 30.4, 0.62)
    proj_400 = project(400, 600, 200, 30.4, 0.62)
    proj_600 = project(600, 600, 200, 30.4, 0.62)
    for p in (proj_200, proj_400, proj_600):
        assert p == pytest.approx(true_total, abs=1.0), (
            "projection must track the true total; a value well under it means blocks were "
            "dropped from the count and the guard would go silent")
    assert abs(proj_200 - proj_600) < 1.0


def test_guard_is_silent_when_the_run_fits():
    """Y's measured 0.464 s/step and 30.4 s/block: Arm A fits with margin, so nothing is skipped."""
    limit = 520.0 * 0.94
    for step in (200, 400, 600):
        assert project(step, 600, 200, 30.4, 0.464) <= limit


def test_guard_fires_when_the_run_would_blow_the_cap():
    limit = 520.0 * 0.94
    assert project(400, 600, 200, 30.4, 0.62) > limit


def test_non_skippable_blocks_is_three():
    """baseline + metric gate + final. Two undercounts by one whole eval block."""
    import re
    src = open(os.path.join(ROOT, "experiments", "y", "y_train.py")).read()
    assert re.search(r"^NON_SKIPPABLE_BLOCKS = 3$", src, re.M), \
        "NON_SKIPPABLE_BLOCKS must be 3: baseline, metric gate and final all run unconditionally"


def test_total_block_count_matches_the_arms():
    def total(steps, eval_every, non_skip=3):
        return steps // eval_every + non_skip
    assert total(600, 200) == 6   # Arm A: 3 in-loop + 3
    assert total(200, 200) == 4   # Arm B: 1 in-loop + 3


# ---- RULE5 must be derived, not encoded as a step ---------------------------------------

def test_rule5_step_175_is_wrong_for_every_non_y_arm():
    """y_train.py used 'step + 1 >= 175' to mean '5 passes', which is only true for Y's
    35-step epoch. This pins the correct comparison so the constant cannot return."""
    def passes_at(step, epoch):
        return step // epoch
    assert passes_at(175, 35) == 5        # Y: the coincidence
    assert passes_at(175, 120) == 1       # Arm A: 1 pass, not 5
    assert passes_at(175, 8) == 21        # Arm B: 21 passes, not 5


def test_trainer_derives_epoch_steps_and_rule5_passes():
    src = open(os.path.join(ROOT, "experiments", "y", "y_train.py")).read()
    assert "EPOCH_STEPS = _epoch_steps(SY)" in src
    assert "RULE5_PASSES" in src
    # the literal comparison, not any mention of the old value in prose
    assert "(step + 1) >= 175" not in src, \
        "the hardcoded 5-passes-as-step-175 encoding must be gone"


def test_trainer_resolves_steps_from_the_corpus():
    src = open(os.path.join(ROOT, "experiments", "y", "y_train.py")).read()
    assert "STEPS = _resolve_steps(SY)" in src
    assert 'os.environ.get("Y_STEPS", "200")' not in src, \
        "a blind 200-step default truncates any corpus whose plan is longer"
