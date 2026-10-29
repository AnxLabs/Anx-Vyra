"""Budget projection and rule-5 threshold for the Z runner.

Y's eval cadence defaults to every 25 steps. That is not used here. EVAL_EVERY
is pinned at 200 so a 600-step run has three in-loop checks (200, 400, 600).

The projection counts two blocks that always run and are not part of the
in-loop cadence: the final eval and the metric gate.

    NON_SKIPPABLE_BLOCKS = 2

The 0.92 factor is extra margin on top of those blocks. It is not a substitute
for them. Dropping the two blocks understates the finish by 2 * block_seconds
(60.6s at the 30.3s reference), which is larger than 8% of a 600s budget, so
the guard stays silent on runs that still overrun.

Skipping the remaining in-loop evals is not always enough. If the projection
with those evals removed still exceeds the full budget, the run halts instead
of training into a result that will be thrown away.
"""

from __future__ import annotations

EVAL_EVERY = 200
NON_SKIPPABLE_BLOCKS = 2  # final eval + metric gate; never fold this into the 0.92 margin
BUDGET_MARGIN = 0.92
REF_BLOCK_S = 30.3
BLOCK_TOLERANCE = 0.25
DEFAULT_BUDGET_S = 600.0


def in_loop_steps(steps: int, eval_every: int = EVAL_EVERY) -> list[int]:
    if steps < 1 or eval_every < 1:
        raise ValueError("steps and EVAL_EVERY must be positive")
    return list(range(eval_every, steps + 1, eval_every))


def scheduled_block_count(steps: int, eval_every: int = EVAL_EVERY) -> int:
    """Baseline, plus each in-loop check, plus the two non-skippable blocks."""
    return 1 + len(in_loop_steps(steps, eval_every)) + NON_SKIPPABLE_BLOCKS


def project_finish(
    elapsed: float,
    remaining_steps: int,
    per_step: float,
    remaining_evals: int,
    last_block: float,
    fixed_measured: float,
    non_skippable: int = NON_SKIPPABLE_BLOCKS,
) -> float:
    """Predicted wall-clock finish.

    `elapsed` is training time so far plus eval blocks already finished. It does
    not include `fixed_measured` (startup and other overhead measured outside
    the step and eval timers). `remaining_evals` counts in-loop checks still
    ahead, including the one about to run. `non_skippable` is the number of
    final/gate blocks not yet paid for; the default is both of them.
    """
    return (
        elapsed
        + remaining_steps * per_step
        + remaining_evals * last_block
        + non_skippable * last_block
        + fixed_measured
    )


def decide(
    elapsed: float,
    remaining_steps: int,
    per_step: float,
    remaining_evals: int,
    last_block: float,
    fixed_measured: float,
    budget: float,
    non_skippable: int = NON_SKIPPABLE_BLOCKS,
) -> tuple[str, float, float]:
    """Return ("run" | "skip" | "halt", full projection, projection with in-loop evals dropped).

    "skip" means the remaining in-loop evals can be dropped and the run still
    finishes under the full budget. "halt" means it cannot: even with no
    further in-loop evals the projection exceeds the budget, so continuing
    only spends the wall clock.
    """
    proj = project_finish(
        elapsed, remaining_steps, per_step, remaining_evals, last_block, fixed_measured, non_skippable
    )
    proj_skip = project_finish(
        elapsed, remaining_steps, per_step, 0, last_block, fixed_measured, non_skippable
    )
    if proj <= budget * BUDGET_MARGIN:
        return "run", proj, proj_skip
    if proj_skip > budget:
        return "halt", proj, proj_skip
    return "skip", proj, proj_skip


def first_block_deviation(measured: float, reference: float = REF_BLOCK_S) -> float:
    if reference == 0:
        raise ValueError("reference block time is 0")
    return (measured - reference) / reference


def assert_first_block(
    measured: float,
    reference: float = REF_BLOCK_S,
    tolerance: float = BLOCK_TOLERANCE,
) -> float:
    """Refuse when the first measured block is not the block the budget model assumes.

    A miss larger than 25% means later projections are not about this run.
    """
    dev = first_block_deviation(measured, reference)
    if abs(dev) > tolerance:
        raise SystemExit(
            f"EVAL_BLOCK_MODEL_MISMATCH measured {measured:.2f}s "
            f"deviation {dev:.2%} from {reference}s (tolerance ±{tolerance:.0%}). "
            f"The budget projection is untrustworthy; refusing to continue."
        )
    return dev


def rule5_threshold_step(steps: int, requested_passes: int) -> int:
    """One pass early, derived from the requested pass target.

    Y's trainer halted on `(step + 1) >= 175` and then printed "at 5 passes".
    On the 200-step, 5-pass run that step is 4.38 passes, not 5, and 175 does
    not change when the pass target changes (one pass early there is step 160).
    The rule's intent is a late "own rows are still zero" stop during the final
    pass, not a fixed step and not a claim that the full pass target has been
    reached. One pass early is that intent expressed in the requested passes.
    The halt text must print passes_at_step of the firing step.
    """
    if steps < 1 or requested_passes < 1:
        raise ValueError("steps and requested passes must be positive")
    if requested_passes == 1:
        return 0
    return int(round(steps * (requested_passes - 1) / requested_passes))


def passes_at_step(step_1based: int, steps: int, requested_passes: int) -> float:
    if steps < 1:
        raise ValueError("steps must be positive")
    return requested_passes * step_1based / steps


def rule5_message(arm: str, step_1based: int, steps: int, requested_passes: int) -> str:
    got = passes_at_step(step_1based, steps, requested_passes)
    threshold = rule5_threshold_step(steps, requested_passes)
    return (
        f"RULE 5 {arm} at {step_1based}: own rows still 0.0 at {got:.2f} passes "
        f"(requested {requested_passes}, threshold step {threshold})"
    )
