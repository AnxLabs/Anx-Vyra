"""Extend the expanded40 treatment from 175 to 200 updates with observation only.

Changes: STEPS 175->200, EVAL_AT adds 200, and a post-eval x64_checkpoint.dump
at updates 175 and 200. Seed, LR (constant 3e-4, re-set every update; no
schedule), replay weight, replay indexing, clip, union and eval are untouched.
"""

_HEAD = "HARD=600; SOFT=598; STEPS=175; RB=2; EVAL_AT={75,125,175}"
_HEAD_NEW = "HARD=600; SOFT=598; STEPS=200; RB=2; EVAL_AT={75,125,175,200}"
_LOADER = "ReplayTelemetry=_tmod.ReplayTelemetry\n"
_EVAL = "A['eval_s'][str(step)]=time.time()-te\n"


def apply_update200(source):
    if "x64_checkpoint" in source:
        raise ValueError("UPDATE200_ALREADY_APPLIED")
    for needle in (_HEAD, _LOADER, _EVAL):
        if source.count(needle) != 1:
            raise ValueError(f"UPDATE200_PATTERN_MISSING {needle!r}")
    out = source.replace(_HEAD, _HEAD_NEW, 1)
    i = out.index(_LOADER) + len(_LOADER)
    ls = out.rfind("\n", 0, out.index(_LOADER)) + 1
    ind = out[ls:out.index(_LOADER)]
    ind = ind[:len(ind) - len(ind.lstrip())]
    out = out[:i] + "".join(ind + l + "\n" for l in (
        "_cspec=_ilu.spec_from_file_location('x64_checkpoint','/kaggle/working/x64_checkpoint.py')",
        "_cmod=_ilu.module_from_spec(_cspec); _cspec.loader.exec_module(_cmod); x64_checkpoint=_cmod",
    )) + out[i:]
    j = out.index(_EVAL)
    ls = out.rfind("\n", 0, j) + 1
    ind = out[ls:j]; ind = ind[:len(ind) - len(ind.lstrip())]
    k = j + len(_EVAL)
    out = out[:k] + ind + (
        "if x64_checkpoint.wanted(step): A.setdefault('x64_checkpoints',{})[str(step)]="
        "x64_checkpoint.dump(m, opt, step, XD, SX['plan'], STEPS, OUT, preclip_diag, A['trajectory'][str(step)]); save()\n"
    ) + out[k:]
    if "STEPS=200" not in out or out.count("opt.step(selected)") != 1:
        raise ValueError("UPDATE200_RESULT_INVALID")
    return out
