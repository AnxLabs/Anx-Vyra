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


def _indent_at(source, needle):
    if source.count(needle) != 1:
        raise ValueError(f"REPLAY_TELEMETRY_PATTERN_MISSING count={source.count(needle)} snippet={needle!r}")
    index = source.index(needle)
    line_start = source.rfind("\n", 0, index) + 1
    indent = source[line_start:index]
    if indent.strip():
        raise ValueError("REPLAY_TELEMETRY_INDENT")
    return indent, line_start, index


def _insert_before(source, needle, statement):
    indent, line_start, _index = _indent_at(source, needle)
    return source[:line_start] + indent + statement + "\n" + source[line_start:]


def _insert_after(source, needle, statement):
    indent, _line_start, index = _indent_at(source, needle)
    end = index + len(needle)
    return source[:end] + indent + statement + "\n" + source[end:]


_TEL_LOSS = FIXED_LOSS_LINE.replace(
    "selected=rv.expert_selection(m)\n",
    "selected=rv.expert_selection(m); tel_first=selected; tel_second=None\n",
)
_TEL_REPLAY = (
    "loss=lx+lam*ly; tel_second=rv.expert_selection(m); selected={key: sorted(set(selected[key]) | set(value)) "
    "for key,value in tel_second.items()}\n"
)
_STEP_TAIL = "; m.zero_grad(set_to_none=True)"
_TEL_FINISH = "; tel.finish_update(m, opt, tel_first, tel_second, selected)"
_BACKWARD = "loss.backward()\n"
_CLIP = "torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.grad is not None],1.0)\n"
_ARM_OPT = "opt=rv.make_optimizer(m); torch.cuda.reset_peak_memory_stats()"
_ARM_FOR = "for arm,lam in (('treatment',0.25),):"
_ARM_NOTE = (
    "A['replay_extra_note']='treatment adds 2-row replay CE+backward each update; "
    "equal updates != equal compute'"
)


def apply_replay_telemetry(source, *, require_arm=True):
    """Insert observational counters into the already repaired caller.

    The union passed to SliceAdam is unchanged. The second expert_selection
    call is stored and reused, not repeated. Arm setup and the final summary
    are required for the embedded treatment and optional for the tiny regression.
    """
    if "tel.begin_update(m, opt)" in source:
        raise ValueError("REPLAY_TELEMETRY_ALREADY_APPLIED")
    step_line = FIXED_STEP + _STEP_TAIL
    for needle in (FIXED_LOSS_LINE, FIXED_REPLAY_LINE, step_line, _BACKWARD, _CLIP):
        if source.count(needle) != 1:
            raise ValueError(f"REPLAY_TELEMETRY_PATTERN_MISSING count={source.count(needle)} snippet={needle!r}")
    out = source.replace(FIXED_LOSS_LINE, _TEL_LOSS, 1)
    out = out.replace(FIXED_REPLAY_LINE, _TEL_REPLAY, 1)
    out = out.replace(step_line, FIXED_STEP + _TEL_FINISH + _STEP_TAIL, 1)
    out = _insert_before(out, _BACKWARD, "tel.begin_update(m, opt)")
    out = _insert_after(out, _CLIP, "tel.note_grads(m)")
    if require_arm:
        for needle in (_ARM_OPT, _ARM_FOR, _ARM_NOTE):
            if source.count(needle) != 1:
                raise ValueError(f"REPLAY_TELEMETRY_PATTERN_MISSING count={source.count(needle)} snippet={needle!r}")
        out = out.replace(_ARM_OPT, _ARM_OPT + "; tel=ReplayTelemetry(m, opt, STEPS)", 1)
        indent, line_start, _index = _indent_at(out, _ARM_FOR)
        loader = (
            f"{indent}import importlib.util as _ilu\n"
            f"{indent}_tspec=_ilu.spec_from_file_location('replay_telemetry','/kaggle/working/replay_telemetry.py')\n"
            f"{indent}_tmod=_ilu.module_from_spec(_tspec); _tspec.loader.exec_module(_tmod); ReplayTelemetry=_tmod.ReplayTelemetry\n"
        )
        out = out[:line_start] + loader + out[line_start:]
        out = _insert_before(out, _ARM_NOTE, "A['expert_telemetry']=tel.summary()")
    if out.count("tel.begin_update(m, opt)") != 1 or out.count("tel.note_grads(m)") != 1:
        raise ValueError("REPLAY_TELEMETRY_RESULT_INVALID")
    if out.count("tel.finish_update(m, opt, tel_first, tel_second, selected)") != 1:
        raise ValueError("REPLAY_TELEMETRY_RESULT_INVALID")
    if "opt.step(rv.expert_selection(" in out or out.count(FIXED_STEP) != 1:
        raise ValueError("REPLAY_TELEMETRY_RESULT_INVALID")
    return out


def apply_unrepaired_observation(source):
    """Record both forwards while the step still receives only the last selection.

    Used by the CPU regression to show the counters catching the original bug.
    This is not the embedded caller.
    """
    if "tel.begin_update(m, opt)" in source:
        raise ValueError("REPLAY_TELEMETRY_ALREADY_APPLIED")
    loss = "loss=lx; ly=None; tel_first=rv.expert_selection(m); tel_second=None\n"
    replay = "loss=lx+lam*ly; tel_second=rv.expert_selection(m)\n"
    step = "opt.step(rv.expert_selection(m))" + _STEP_TAIL
    observed = (
        "_tel_passed=rv.expert_selection(m); opt.step(_tel_passed); "
        "tel.finish_update(m, opt, tel_first, tel_second, _tel_passed)"
        + _STEP_TAIL
    )
    for needle in (BUGGY_LOSS_LINE, BUGGY_REPLAY_LINE, step, _BACKWARD, _CLIP):
        if source.count(needle) != 1:
            raise ValueError(f"REPLAY_TELEMETRY_PATTERN_MISSING count={source.count(needle)} snippet={needle!r}")
    out = source.replace(BUGGY_LOSS_LINE, loss, 1)
    out = out.replace(BUGGY_REPLAY_LINE, replay, 1)
    out = out.replace(step, observed, 1)
    out = _insert_before(out, _BACKWARD, "tel.begin_update(m, opt)")
    out = _insert_after(out, _CLIP, "tel.note_grads(m)")
    if "opt.step(selected)" in out or out.count("opt.step(_tel_passed)") != 1:
        raise ValueError("REPLAY_TELEMETRY_RESULT_INVALID")
    return out
