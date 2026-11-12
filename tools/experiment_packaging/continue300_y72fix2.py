"""Turn the y72fix2 treatment (resume 225 -> 300, per-task clip 0.5, replay weight 0.25) into its 300 -> 400 continuation
(prereg/y72fix2_cont400.json, rule vyra-y72fix2-cont400-v1).

Layered on continue225_y72fix2.apply_y72fix2 (exact, count-asserted text edits). Relative to the notebook that produced checkpoint_300 the
edits are exactly:
* the resume point: START=300, STEPS=400, EVAL_AT={325,350,375,400}, checkpoint_300 (schema v2) through
  resume300_y72fix2.load_checkpoint300 (SHA-256 before torch.load, full-state restore, model-state hash + exact integer counts at 300);
* the plan: plan_extension400 (the v1 extension streams continued for entries 300..399; plan[0:300] unchanged, pinned);
* full checkpoints at resume300_y72fix2.CHECKPOINT_UPDATES (325/350/375/400), their sampler record written with the 400-entry plan;
* the pre-registered verdict y72fix2_cont400_gates replaces y72fix2_gates (the 600-segment set stays in HO/EV unchanged).
Nothing else changes: per-task clip 0.5, replay weight 0.25, 2 replay rows per update, LR 3e-4, union, optimizer, data, seeds.
"""
import continue225 as c5
import continue225_y72fix2 as c2
import kaggle_config as K

HEAD_400 = K.expand(
    "HARD=600; SOFT=598; STEPS=400; START=300; RB=2; EVAL_AT={325,350,375,400}\n"
    "CKPT_PATH='@@KROOT@@/input/datasets/@@KUSER@@/@@SLUG_CKPT300@@/checkpoint_300.pt'\n"
    "CPU_SMOKE=os.environ.get('VYRA_CONT_CPU_SMOKE')=='1'")
PXLOAD_OLD = "  _pxmod=_pxilu.module_from_spec(_pxspec); _pxspec.loader.exec_module(_pxmod); plan_extension=_pxmod\n"
PXLOAD_NEW = PXLOAD_OLD + K.expand(
    "  _p4spec=_pxilu.spec_from_file_location('plan_extension400','@@KROOT@@/working/plan_extension400.py')\n"
    "  plan_extension400=_pxilu.module_from_spec(_p4spec); _p4spec.loader.exec_module(plan_extension400)\n")
PLAN_OLD = "      SX['plan']=plan_extension.build_extended_plan(SX, STEPS)\n"
PLAN_NEW = "      SX['plan']=plan_extension400.build_extended_plan(SX, STEPS)\n"
PLANREC_OLD = ("R['plan_extension']={'policy':plan_extension.POLICY_VERSION,'prefix_sha256':plan_extension.plan_sha256(SX['plan'][:200]),"
               "'full_sha256':plan_extension.plan_sha256(SX['plan']),")
PLANREC_NEW = ("R['plan_extension']={'policy':plan_extension400.POLICY_VERSION,'prefix_sha256':plan_extension400.plan_sha256(SX['plan'][:200]),"
               "'prefix300_sha256':plan_extension400.plan_sha256(SX['plan'][:300]),'full_sha256':plan_extension400.plan_sha256(SX['plan']),")
LOAD_400 = ("cont=_cn_load('continuation'); r300=_cn_load('resume300_y72fix2'); gates=_cn_load('y72fix2_cont400_gates'); "
            "y2rt=_cn_load('y72fix2_runtime'); c4rt=_cn_load('y72fix2_cont400_runtime'); clip_scheme=_cn_load('clip_scheme')\n")
GATE_400 = (
    "      CKINFO=r300.load_checkpoint300(CKPT_PATH, model=m, opt=opt, plan=SX['plan'], lib=rv, device=DEVICE, replay_batch=RB, cpu_smoke=CPU_SMOKE)\n"
    "  except cont.ContinuationGateError as _ge: stop('CONTINUATION_GATE '+str(_ge))\n"
    "  R['continuation']={'checkpoint':CKINFO,'experiment':'y72fix2 continuation: resume at 300, stop at 400, per-task clip 0.5 and replay weight 0.25 unchanged (prereg/y72fix2_cont400.json)',"
    "'only_variable':'none (same configuration, 100 more updates)','restored':'complete v2 state: params, optimizer, torch+CUDA RNG, Python RNG, NumPy RNG, router, mode, sampler'}; save()\n"
)
RESTORE_400 = "r300.check_resume_restoration(m, _bx, _bc, strict=not CPU_SMOKE, x64_ck=x64_checkpoint)"
PROV_400 = ("'resumed_from_sha256':r300.CKPT300_SHA256,'resumed_from_update':START,'replay_lambda':r300.NEW_LAMBDA,"
            "'only_variable':'none (continuation of y72fix2)','clip_scheme':'pertask tau 0.5 (the grad_clip_max_norm 1.0 field of the checkpoint config is the legacy global value, unused here)',")
LAM_400 = "for arm,lam in (('treatment',r300.NEW_LAMBDA),):\n      assert lam==0.25 and r300.NEW_LAMBDA==0.25 and r300.SAVED_LAMBDA==0.25 and gates.TASK_CLIP_TENTHS==5\n"
CKUPD_OLD = "if step in cont.CHECKPOINT_UPDATES:"
CKUPD_NEW = "if step in r300.CHECKPOINT_UPDATES:"
DUMP_OLD = "plan=SX['plan'], plan_module=plan_extension, out_dir=OUT"
DUMP_NEW = "plan=SX['plan'], plan_module=plan_extension400, out_dir=OUT"
EVAL_400 = "A.setdefault('y72fix2_cont400_gates',{})[str(step)]=gates.evaluate(c4rt.history(HIST)); save()\n"
FINAL_400 = "  R['y72fix2_cont400_verdict']=gates.evaluate(c4rt.history(HIST)); save()\n"

SWAPS = (("head", c5.HEAD_225, HEAD_400), ("pxload", PXLOAD_OLD, PXLOAD_NEW), ("plan", PLAN_OLD, PLAN_NEW), ("planrec", PLANREC_OLD, PLANREC_NEW),
         ("load", c2.LOAD_NEW, LOAD_400), ("gate", c2.GATE_NEW, GATE_400), ("restore", c5.RESTORE_225, RESTORE_400),
         ("prov", c2.PROV_NEW, PROV_400), ("lam", c2.LAM_NEW, LAM_400), ("ckupd", CKUPD_OLD, CKUPD_NEW), ("dump", DUMP_OLD, DUMP_NEW),
         ("eval", c2.EVAL_NEW, EVAL_400), ("final", c2.FINAL_NEW, FINAL_400))
BANNED = ("r225", "resume225", "checkpoint_225", "CKPT225", "y72fix2_gates", "y72fix2_verdict", "y2rt.history", "START=225",
          "plan_module=plan_extension,", "cont.CHECKPOINT_UPDATES", "plan_extension.build_extended_plan", "load_checkpoint200",
          "stop_rule", "clip_grad_norm_")


def apply_cont400(source):
    out = c2.apply_y72fix2(source)
    for label, old, new in SWAPS:
        if out.count(old) != 1:
            raise ValueError(f"CONTINUE300_Y72FIX2_PATTERN_MISSING {label} count={out.count(old)}")
        out = out.replace(old, new, 1)
    for banned in BANNED:
        if banned in out:
            raise ValueError(f"CONTINUE300_Y72FIX2_RESIDUE {banned}")
    if ("for s in range(START, STEPS)" not in out or "START=300" not in out or "STEPS=400" not in out
            or out.count("opt.step(selected)") != 1 or out.count("clip_scheme.combine(_pp,_gx,_gy,lam,'pertask',0.5)") != 1):
        raise ValueError("CONTINUE300_Y72FIX2_RESULT_INVALID")
    if out.index("r300.load_checkpoint300(") > out.index("for s in range(START, STEPS)"):
        raise ValueError("CONTINUE300_Y72FIX2_ORDER")
    if out.index("plan_extension400=") > out.index("plan_extension400.build_extended_plan("):
        raise ValueError("CONTINUE300_Y72FIX2_ORDER")
    return out


patch_preclip_diag = c2.patch_preclip_diag
