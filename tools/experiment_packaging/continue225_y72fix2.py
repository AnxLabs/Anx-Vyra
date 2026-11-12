"""Turn the 200 -> 300 continuation treatment into the Y72 retention fix 2 experiment (prereg/y72fix2.json): resume at 225, replay weight
UNCHANGED 0.25, per-task gradient clip tau 0.5 instead of the global clip 1.0, stop at 300, 600-segment Y72 set in the evaluation.

Layered on continue300.apply_continue300 like continue225 (exact, count-asserted text edits). Relative to the notebook that produced
checkpoint_225 the edits are exactly:
* the resume point (225, schema v2, resume225_y72fix2.load_checkpoint225; replay weight must equal the saved 0.25);
* ONE experimental variable: `loss.backward(); clip_grad_norm_(1.0)` -> per-task gradients (clip_scheme.task_gradients), per-task clip 0.5 on
  gx and on lam*gy (clip_scheme.combine 'pertask'), assigned as .grad, then the unchanged `opt.step(selected)`; the unclipped combined norm and the
  per-task record are logged (observation only);
* the registered 600-segment Y72 set is added to the held-out dict HO and to EV (counted at every checkpoint like the other sets; its payloads
  go through the same train/held-out disjointness check);
* the pre-registered verdict (y72fix2_gates) replaces the stop rule; the old sparse diagnostics (which assume the global clip) are disabled.
Everything else (plan, seed handling, LR 3e-4, replay rows per update, union, optimizer, data) is untouched.
"""
import continue225 as c5
from continue300 import apply_continue300
import kaggle_config as K

LOAD_NEW = ("cont=_cn_load('continuation'); r225=_cn_load('resume225_y72fix2'); gates=_cn_load('y72fix2_gates'); "
            "y2rt=_cn_load('y72fix2_runtime'); clip_scheme=_cn_load('clip_scheme')\n")
GATE_NEW = c5.GATE_225.replace(
    "'experiment':'replay weight 0.25 -> 0.5, resume at 225, stop at 300 (prereg/y72fix_lambda05.json)','only_variable':'replay_lambda'",
    "'experiment':'per-task gradient clip 0.5 (global clip 1.0 removed), replay weight 0.25 unchanged, resume at 225, stop at 300 (prereg/y72fix2.json)','only_variable':'gradient clip scheme'")
PROV_NEW = ("'resumed_from_sha256':r225.CKPT225_SHA256,'resumed_from_update':START,'replay_lambda':r225.NEW_LAMBDA,"
            "'only_variable':'gradient clip scheme','clip_scheme':'pertask tau 0.5 (the grad_clip_max_norm 1.0 field of the checkpoint config is the legacy global value, unused here)',")
LAM_NEW = "for arm,lam in (('treatment',r225.NEW_LAMBDA),):\n      assert lam==0.25 and r225.NEW_LAMBDA==0.25 and r225.SAVED_LAMBDA==0.25 and gates.TASK_CLIP_TENTHS==5\n"
STEPS_NEW = (c5.STEPS_300 + "          A.setdefault('preclip_norm',[]).append(float(_preclip_norm))\n"
             "          A.setdefault('task_clip',[]).append(y2rt.compact(_rec))\n")
EVAL_NEW = "A.setdefault('y72fix2_gates',{})[str(step)]=gates.evaluate(y2rt.history(HIST)); save()\n"
FINAL_NEW = "  R['y72fix2_verdict']=gates.evaluate(y2rt.history(HIST)); save()\n"

BACKWARD_OLD = (
    "          loss.backward()\n"
    "          if not math.isfinite(float(loss)) or not rv.finite_grads(m): stop(f'nonfinite loss/grad {arm} {s}')\n"
    "          _clip_params=[p for p in m.parameters() if p.grad is not None]; _preclip_norm=torch.nn.utils.clip_grad_norm_(_clip_params,1.0)\n")
BACKWARD_NEW = (
    "          _pp=list(m.parameters())\n"
    "          _gx,_gy=clip_scheme.task_gradients(_pp,lx,ly); _gc,_rec=clip_scheme.combine(_pp,_gx,_gy,lam,'pertask',0.5); clip_scheme.assign(_pp,_gc)\n"
    "          if not math.isfinite(float(loss)) or not rv.finite_grads(m): stop(f'nonfinite loss/grad {arm} {s}')\n"
    "          _preclip_norm=y2rt.preclip_norm(_rec,lam)\n")
HO_OLD = "  HO['nova_diag_5000_5005']=diag\n"
HO_NEW = HO_OLD + "  y2rt=_cn_load('y72fix2_runtime'); HO[y2rt.KEY]=y2rt.load_y600()\n"
EV_OLD = "  EV=['Y72_unseen','X32_unseen','X64_unseen','X109_unseen']\n"
EV_NEW = "  EV=['Y72_unseen','X32_unseen','X64_unseen','X109_unseen',y2rt.KEY]\n"
CNLOAD_OLD = K.expand("  import importlib.util as _cnilu\n  def _cn_load(_n):\n      _sp=_cnilu.spec_from_file_location(_n,'@@KROOT@@/working/'+_n+'.py'); _mo=_cnilu.module_from_spec(_sp); _sp.loader.exec_module(_mo); return _mo\n")
DIAG_OLD = "DIAGNOSTIC_UPDATES = (1, 75, 125, 175, 200)"
DIAG_NEW = "DIAGNOSTIC_UPDATES = ()"

SWAPS = (("head", c5.HEAD_300, c5.HEAD_225), ("load", c5.LOAD_300, LOAD_NEW), ("gate", c5.GATE_300, GATE_NEW),
         ("restore", c5.RESTORE_300, c5.RESTORE_225), ("prov", c5.PROV_300, PROV_NEW), ("lam", c5.LAM_300, LAM_NEW),
         ("steps", c5.STEPS_300, STEPS_NEW), ("eval", c5.EVAL_300, EVAL_NEW), ("final", c5.FINAL_300, FINAL_NEW),
         ("backward", BACKWARD_OLD, BACKWARD_NEW))


def apply_y72fix2(source):
    out = apply_continue300(source)
    # the baseline assignments of the 600-set must precede _cn_load use: HO is built before _cn_load is defined in the real treatment,
    # so the HO/EV edits import the runtime module directly
    ho_new = (K.expand("  import importlib.util as _y2ilu\n  _y2sp=_y2ilu.spec_from_file_location('y72fix2_runtime','@@KROOT@@/working/y72fix2_runtime.py'); y2rt=_y2ilu.module_from_spec(_y2sp); _y2sp.loader.exec_module(y2rt)\n"
              "  HO[y2rt.KEY]=y2rt.load_y600()\n"))
    for label, old, new in SWAPS + (("ho", HO_OLD, HO_OLD + ho_new), ("ev", EV_OLD, EV_NEW)):
        if out.count(old) != 1:
            raise ValueError(f"CONTINUE225_Y72FIX2_PATTERN_MISSING {label} count={out.count(old)}")
        out = out.replace(old, new, 1)
    for banned in ("load_checkpoint200", "stop_rule", "CKPT200", "checkpoint_200.pt", "gates.NEW_LAMBDA", "clip_grad_norm_"):
        if banned in out:
            raise ValueError(f"CONTINUE225_Y72FIX2_RESIDUE {banned}")
    if "for s in range(START, STEPS)" not in out or "START=225" not in out or out.count("opt.step(selected)") != 1:
        raise ValueError("CONTINUE225_Y72FIX2_RESULT_INVALID")
    if out.index("r225.load_checkpoint225(") > out.index("for s in range(START, STEPS)"):
        raise ValueError("CONTINUE225_Y72FIX2_ORDER")
    if out.index("HO[y2rt.KEY]=") > out.index("EV=["):
        raise ValueError("CONTINUE225_Y72FIX2_ORDER")
    return out


def patch_preclip_diag(text):
    if text.count(DIAG_OLD) != 1:
        raise ValueError("CONTINUE225_Y72FIX2_DIAG_PATTERN_MISSING")
    return text.replace(DIAG_OLD, DIAG_NEW, 1)
