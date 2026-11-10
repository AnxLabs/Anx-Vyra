"""Turn the 200 -> 300 continuation treatment into the replay-weight experiment: resume at 225, lambda 0.5, stop at 300.

Layered on continue300.apply_continue300 (itself exact, count-asserted text edits on the 200-update treatment). Changes
relative to the notebook that produced checkpoint_225 are exactly:

* the resume point: START=225, EVAL_AT={250,275,300}, restore from checkpoint_225 (schema v2) through
  resume225.load_checkpoint225 (SHA-256 before torch.load, full-state restore, pinned integer counts at 225 must reproduce);
* ONE experimental variable: the replay weight `lam` 0.25 -> y72fix_gates.NEW_LAMBDA == 0.5;
* observation only: the per-update pre-clip total norm is logged (`A['preclip_norm']`); it is read after clip_grad_norm_ has
  already run and changes nothing;
* the pre-registered verdict (y72fix_gates) replaces the stop rule; diagnostics at 250/275/300.
Everything else (plan, seed handling, LR 3e-4, clip 1.0, replay rows per update, union, optimizer, data) is untouched.
"""
from continue300 import apply_continue300

HEAD_300 = (
    "HARD=600; SOFT=598; STEPS=300; START=200; RB=2; EVAL_AT={225,250,275,300}\n"
    "CKPT_PATH='/kaggle/input/datasets/anjorosimoaj/vyra-expanded40-200u-checkpoint/checkpoint_200.pt'\n"
    "CPU_SMOKE=os.environ.get('VYRA_CONT_CPU_SMOKE')=='1'"
)
HEAD_225 = (
    "HARD=600; SOFT=598; STEPS=300; START=225; RB=2; EVAL_AT={250,275,300}\n"
    "CKPT_PATH='/kaggle/input/datasets/anjorosimoaj/vyra-resume300-ckpt225/checkpoint_225.pt'\n"
    "CPU_SMOKE=os.environ.get('VYRA_CONT_CPU_SMOKE')=='1'"
)
LOAD_300 = "cont=_cn_load('continuation'); stop_rule=_cn_load('stop_rule')\n"
LOAD_225 = "cont=_cn_load('continuation'); r225=_cn_load('resume225'); gates=_cn_load('y72fix_gates')\n"
GATE_300 = (
    "      CKINFO=cont.load_checkpoint200(CKPT_PATH, model=m, opt=opt, plan=SX['plan'], plan_module=plan_extension, device=DEVICE, cpu_smoke=CPU_SMOKE, experts_cls=rv.Experts)\n"
    "  except cont.ContinuationGateError as _ge: stop('CONTINUATION_GATE '+str(_ge))\n"
    "  R['continuation']={'checkpoint':{k:v for k,v in CKINFO.items() if k!='disclosure'},'disclosure':cont.DISCLOSURE,'scope':'update 200 -> 300 continuation; the run itself requires a separate approval'}; save()\n"
)
GATE_225 = (
    "      CKINFO=r225.load_checkpoint225(CKPT_PATH, model=m, opt=opt, plan=SX['plan'], plan_module=plan_extension, lib=rv, device=DEVICE, replay_batch=RB, cpu_smoke=CPU_SMOKE)\n"
    "  except cont.ContinuationGateError as _ge: stop('CONTINUATION_GATE '+str(_ge))\n"
    "  R['continuation']={'checkpoint':CKINFO,'experiment':'replay weight 0.25 -> 0.5, resume at 225, stop at 300 (prereg/y72fix_lambda05.json)','only_variable':'replay_lambda','restored':'complete v2 state: params, optimizer, torch+CUDA RNG, Python RNG, NumPy RNG, router, mode, sampler'}; save()\n"
)
RESTORE_300 = "cont.check_baseline_restoration(m, _bx, _bc, R['baseline'], strict=not CPU_SMOKE, x64_ck=x64_checkpoint, baseline_x64=stop_rule.BASELINE_X64)"
RESTORE_225 = "r225.check_resume_restoration(m, _bx, _bc, strict=not CPU_SMOKE, x64_ck=x64_checkpoint)"
PROV_300 = "'resumed_from_sha256':cont.CKPT200_SHA256,'resumed_from_update':START,"
PROV_225 = "'resumed_from_sha256':r225.CKPT225_SHA256,'resumed_from_update':START,'replay_lambda':gates.NEW_LAMBDA,'only_variable':'replay_lambda',"
LAM_300 = "for arm,lam in (('treatment',0.25),):\n"
LAM_225 = "for arm,lam in (('treatment',gates.NEW_LAMBDA),):\n      assert lam==0.5 and r225.NEW_LAMBDA==0.5 and r225.SAVED_LAMBDA==0.25\n"
STEPS_300 = "A['steps']=s+1; A['loss'].append([round(float(lx),4),None if ly is None else round(float(ly),4)])\n"
STEPS_225 = STEPS_300 + "          A.setdefault('preclip_norm',[]).append(float(_preclip_norm))\n"
EVAL_300 = "A.setdefault('stop_rule',{})[str(step)]=stop_rule.evaluate(HIST); save()\n"
EVAL_225 = "A.setdefault('y72fix_gates',{})[str(step)]=gates.evaluate(HIST); save()\n"
FINAL_300 = "  R['stop_rule_final']=stop_rule.evaluate(HIST); save()\n"
FINAL_225 = "  R['y72fix_verdict']=gates.evaluate(HIST); save()\n"
DIAG_OLD = "DIAGNOSTIC_UPDATES = (1, 75, 125, 175, 200)"
DIAG_NEW = "DIAGNOSTIC_UPDATES = (250, 275, 300)"

SWAPS = (("head", HEAD_300, HEAD_225), ("load", LOAD_300, LOAD_225), ("gate", GATE_300, GATE_225),
         ("restore", RESTORE_300, RESTORE_225), ("prov", PROV_300, PROV_225), ("lam", LAM_300, LAM_225),
         ("steps", STEPS_300, STEPS_225), ("eval", EVAL_300, EVAL_225), ("final", FINAL_300, FINAL_225))


def apply_continue225(source):
    out = apply_continue300(source)
    for label, old, new in SWAPS:
        if out.count(old) != 1:
            raise ValueError(f"CONTINUE225_PATTERN_MISSING {label} count={out.count(old)}")
        out = out.replace(old, new, 1)
    # nothing of the old 200-resume path or the old stop rule may remain
    for banned in ("load_checkpoint200", "stop_rule", "CKPT200", "checkpoint_200.pt", "0.25),)"):
        if banned in out:
            raise ValueError(f"CONTINUE225_RESIDUE {banned}")
    if "for s in range(START, STEPS)" not in out or "START=225" not in out or out.count("opt.step(selected)") != 1:
        raise ValueError("CONTINUE225_RESULT_INVALID")
    if out.index("r225.load_checkpoint225(") > out.index("for s in range(START, STEPS)"):
        raise ValueError("CONTINUE225_ORDER")
    return out


def patch_preclip_diag(text):
    if text.count(DIAG_OLD) != 1:
        raise ValueError("CONTINUE225_DIAG_PATTERN_MISSING")
    return text.replace(DIAG_OLD, DIAG_NEW, 1)
