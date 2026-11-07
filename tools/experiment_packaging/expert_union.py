"""Per-update expert-selection capture for the multi-forward replay caller.

expert_selection reports only the last forward. SliceAdam steps only the dict
it is given. The replay trainer runs the primary forward, then a second
forward, then one backward, one global clip, and one step. The second forward
replaces Experts._selected, so slices used only by the first forward keep a
gradient and receive no update.

The repair captures the first selection immediately, unions the second
selection when the replay term runs, and passes that dict to the existing
step. When the replay term is off, the captured first selection is the only
forward, which is the single-forward behavior. Nothing here accumulates
selections inside Experts or across updates.
"""

BUGGY_LOSS_LINE = "loss=lx; ly=None\n"
FIXED_LOSS_LINE = "loss=lx; ly=None; selected=rv.expert_selection(m)\n"

BUGGY_REPLAY_LINE = "loss=lx+lam*ly\n"
FIXED_REPLAY_LINE = (
    "loss=lx+lam*ly; selected={key: sorted(set(selected[key]) | set(value)) "
    "for key,value in rv.expert_selection(m).items()}\n"
)

BUGGY_STEP = "opt.step(rv.expert_selection(m))"
FIXED_STEP = "opt.step(selected)"

_PAIRS = (
    (BUGGY_LOSS_LINE, FIXED_LOSS_LINE),
    (BUGGY_REPLAY_LINE, FIXED_REPLAY_LINE),
    (BUGGY_STEP, FIXED_STEP),
)


def apply_expert_union(source: str) -> str:
    """Return source with the three confirmed caller edits, or raise.

    The match is exact. A caller that does not contain each buggy snippet
    once is refused, including a caller that already has the repair.
    """
    if all(source.count(buggy) == 0 and source.count(fixed) == 1 for buggy, fixed in _PAIRS):
        raise ValueError("EXPERT_UNION_ALREADY_APPLIED")
    for buggy, _fixed in _PAIRS:
        found = source.count(buggy)
        if found != 1:
            raise ValueError(f"EXPERT_UNION_PATTERN_MISSING count={found} snippet={buggy!r}")
    out = source
    for buggy, fixed in _PAIRS:
        out = out.replace(buggy, fixed, 1)
    if "opt.step(rv.expert_selection(" in out or out.count(FIXED_STEP) != 1:
        raise ValueError("EXPERT_UNION_RESULT_INVALID")
    return out
