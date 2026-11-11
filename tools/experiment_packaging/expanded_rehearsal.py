"""Splice the pinned 40-row replay and the four pre-clip reads into treatment.

The union and the aggregate expert telemetry stay in place. Replay still uses
two rows per update. The index modulus changes from 8 to 40. X batches, seed,
loss weight, clip, and the one combined backward are not rewritten.
"""
from expert_union import _indent_at
import kaggle_config as K

_SHAPE = "if tuple(Y72['rows'].shape)!=(8,512) or tuple(Y72['unseen'].shape)!=(40,512): stop('Y72 shape')\n"
_REPLAY = (
    "def replay_at(s):\n"
    "      idx=[(s*RB+j)%8 for j in range(RB)]\n"
    "      return Y72['rows'][idx].long(),Y72['mask'][idx].long(),idx\n"
)
_REPLAY_NEW = (
    "def replay_at(s):\n"
    "      idx=[(s*RB+j)%40 for j in range(RB)]\n"
    "      return RY['rows'][idx].long(),RY['mask'][idx].long(),idx\n"
)
_BLOCKED = "if any(v['row_overlap'] or v['payload_overlap'] for v in dis.values()): stop('BLOCKED overlap '+json.dumps(dis))\n"
_LOSS = "lx=ce(m(tok),tok,msk)"
_LOSS_NEW = "lgx=m(tok); lx=ce(lgx,tok,msk)"
_BEGIN = "tel.begin_update(m, opt)\n"
_CLIP = "torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.grad is not None],1.0)"
_CLIP_NEW = "_clip_params=[p for p in m.parameters() if p.grad is not None]; _preclip_norm=torch.nn.utils.clip_grad_norm_(_clip_params,1.0)"
_NOTE = "tel.note_grads(m)\n"
_LOADER = "ReplayTelemetry=_tmod.ReplayTelemetry\n"
_ARM = "A={'lambda':lam,'steps':0,'loss':[],'trajectory':{},'eval_s':{}}"
_ARM_NEW = "A={'lambda':lam,'steps':0,'loss':[],'trajectory':{},'eval_s':{},'preclip_diagnostics':[],'rehearsal_readout':{}}"
_TRAJ = "A['trajectory'][str(step)]=eval_block(m)"
_BASE = "R['baseline']=eval_block(m); R['baseline_eval_s']=time.time()-t; save()"


def _once(source, needle):
    if source.count(needle) != 1:
        raise ValueError(f"REHEARSAL_PATTERN_MISSING count={source.count(needle)} snippet={needle!r}")


def _insert_after(source, needle, lines):
    indent, _start, index = _indent_at(source, needle)
    end = index + len(needle)
    block = "".join(indent + line + "\n" for line in lines)
    return source[:end] + block + source[end:]


def apply_expanded_rehearsal(source):
    """Return treatment source with the rehearsal pin and sparse diagnostics."""
    if "rehearsal_Y72_40.pt" in source or "preclip_diag.begin_separate" in source:
        raise ValueError("REHEARSAL_ALREADY_APPLIED")
    for needle in (_SHAPE, _REPLAY, _BLOCKED, _LOSS, _BEGIN, _CLIP, _NOTE, _LOADER, _ARM, _TRAJ, _BASE):
        _once(source, needle)
    if source.count(_LOSS) != 1:
        raise ValueError("REHEARSAL_PATTERN_MISSING")
    out = source.replace(_REPLAY, _REPLAY_NEW, 1)
    out = out.replace(_LOSS, _LOSS_NEW, 1)
    out = out.replace(_CLIP, _CLIP_NEW, 1)
    out = out.replace(_ARM, _ARM_NEW, 1)
    out = _insert_after(out, _SHAPE, [
        "import importlib.util as _rilu",
        K.expand("_rspec=_rilu.spec_from_file_location('rehearsal_artifact','@@KROOT@@/working/rehearsal_artifact.py')"),
        "_rmod=_rilu.module_from_spec(_rspec); _rspec.loader.exec_module(_rmod)",
        K.expand("RY=_rmod.load_for_training('@@KROOT@@/working/rehearsal_Y72_40.pt', set(Y72['unseen_payloads']))"),
        "if tuple(RY['rows'].shape)!=(40,512) or tuple(RY['mask'].shape)!=(40,510): stop('rehearsal shape')",
    ])
    out = _insert_after(out, _BLOCKED, [
        "_added=set(RY['payloads'][40:])",
        "if _added & set(RY['payloads'][:40]): stop('added payloads overlap original8')",
        "if _added & set(Y72['unseen_payloads']): stop('added payloads overlap Y72 unseen')",
    ])
    # Load preclip_diag before the baseline readout, which is its first use.
    # It used to load after the telemetry loader, below the baseline line,
    # and the expanded40 Kaggle run died there with a NameError.
    base_at = out.index(_BASE)
    line_start = out.rfind("\n", 0, base_at) + 1
    base_line = out[line_start:base_at]
    indent = base_line[:len(base_line) - len(base_line.lstrip())]
    diag_loader = "".join(
        indent + line + "\n" for line in (
            "import importlib.util as _dilu",
            K.expand("_dspec=_dilu.spec_from_file_location('preclip_diag','@@KROOT@@/working/preclip_diag.py')"),
            "_dmod=_dilu.module_from_spec(_dspec); _dspec.loader.exec_module(_dmod); preclip_diag=_dmod",
        )
    )
    out = out[:line_start] + diag_loader + out[line_start:]
    diag_call = "if preclip_diag.wanted(s): preclip_diag.begin_separate(m, opt, lgx, lx, ly, tok, msk, tel_first, tel_second, selected, lam)"
    indent, line_start, _index = _indent_at(out, _BEGIN)
    out = out[:line_start] + indent + diag_call + "\n" + out[line_start:]
    out = _insert_after(out, _NOTE, [
        "if preclip_diag.wanted(s): A['preclip_diagnostics'].append(preclip_diag.finish_combined(_preclip_norm, s+1))",
    ])
    out = out.replace(
        _TRAJ,
        _TRAJ + "; A['rehearsal_readout'][str(step)]=preclip_diag.rehearsal_readout(m, RY['rows'], RY['mask'])",
        1,
    )
    out = out.replace(
        _BASE,
        "R['baseline']=eval_block(m); R['baseline_rehearsal']=preclip_diag.rehearsal_readout(m, RY['rows'], RY['mask']); R['baseline_eval_s']=time.time()-t; save()",
        1,
    )
    if out.count("m(tok)") != 1 or out.count("%8 for j in range(RB)") != 0:
        raise ValueError("REHEARSAL_RESULT_INVALID")
    if "opt.step(selected)" not in out or "tel.finish_update" not in out:
        raise ValueError("REHEARSAL_RESULT_INVALID")
    return out
