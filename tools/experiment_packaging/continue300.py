"""Turn the built 200-update treatment.py into the 200 -> 300 continuation.

Same mechanism as update200.py: exact, count-asserted text edits on the
treatment the previous build already produced. Seed, LR (constant 3e-4, re-set
every update), replay weight 0.25, replay indexing, clip 1.0, union, telemetry
and the original eval are untouched. What changes:

* STEPS 200 -> 300, the loop starts at update index START=200, EVAL_AT is
  {225, 250, 275, 300}.
* The plan is extended from 200 to 300 entries by plan_extension (prefix hash
  checked, pinned extension hash checked).
* A fresh model/optimizer is built and restored from checkpoint_200 through
  continuation.load_checkpoint200, which verifies SHA-256 BEFORE torch.load and
  fails closed on every other mismatch. No reseeding and no random.seed(0) at
  the loop start; RNG comes from the checkpoint plus the disclosed init.
* Baseline eval is the restored update-200 model, inside an observer guard,
  and must reproduce the recorded update-200 measurements.
* At 225/250/275/300 eval + readouts run inside the guard and
  continuation.dump_full writes predictions, retention counts and a complete
  v2 checkpoint. The stop rule is evaluated on exact integer counts.
"""
import kaggle_config as K
HEAD = "HARD=600; SOFT=598; STEPS=200; RB=2; EVAL_AT={75,125,175,200}"
HEAD_NEW = (
    K.expand("HARD=600; SOFT=598; STEPS=300; START=200; RB=2; EVAL_AT={225,250,275,300}\n"
    "CKPT_PATH='@@KROOT@@/input/datasets/@@KUSER@@/@@SLUG_CKPT200@@/checkpoint_200.pt'\n"
    "CPU_SMOKE=os.environ.get('VYRA_CONT_CPU_SMOKE')=='1'")
)
PLAN_CHECK = "  if len(SX['plan'])<STEPS: stop(f'plan too short {len(SX[\"plan\"])}<{STEPS}')\n"
BASE_A = "  m=make(); t=time.time(); R['baseline']=eval_block(m); R['baseline_rehearsal']=preclip_diag.rehearsal_readout(m, RY['rows'], RY['mask']); R['baseline_eval_s']=time.time()-t; save()\n"
BASE_B = "  print('BASE',R['baseline'],flush=True); del m\n"
X64_LOADER = "  _cmod=_ilu.module_from_spec(_cspec); _cspec.loader.exec_module(_cmod); x64_checkpoint=_cmod\n"
ARM_INIT = "m=make(); torch.manual_seed(0); torch.cuda.manual_seed_all(0); random.seed(0); rv.LR=3e-4; opt=rv.make_optimizer(m); torch.cuda.reset_peak_memory_stats(); tel=ReplayTelemetry(m, opt, STEPS)"
ARM_INIT_NEW = "rv.LR=3e-4; torch.cuda.reset_peak_memory_stats(); tel=ReplayTelemetry(m, opt, STEPS-START)"
ARM_REC = "A={'lambda':lam,'steps':0,"
ARM_REC_NEW = "A={'lambda':lam,'start_update':START,'resumed_from_update':START,'loss_first_update':START+1,'steps':0,"
LOOP = "      for s in range(STEPS):\n          if time.time()-T0>SOFT: stop(f'time budget at {arm} step {s}')\n"
LOOP_NEW = "      for s in range(START, STEPS):\n          if time.time()-T0>SOFT: stop(f'time budget at {arm} step {s}')\n"
EVAL_LINES = (
    "              te=time.time(); A['trajectory'][str(step)]=eval_block(m); A['rehearsal_readout'][str(step)]=preclip_diag.rehearsal_readout(m, RY['rows'], RY['mask']); A['eval_s'][str(step)]=time.time()-te\n"
    "              if x64_checkpoint.wanted(step): A.setdefault('x64_checkpoints',{})[str(step)]=x64_checkpoint.dump(m, opt, step, XD, SX['plan'], STEPS, OUT, preclip_diag, A['trajectory'][str(step)]); save()\n"
)
VERDICT = "  T=R['arms']['treatment']['final']\n"

_PLAN_NEW = (
    K.expand("  import importlib.util as _pxilu\n"
    "  _pxspec=_pxilu.spec_from_file_location('plan_extension','@@KROOT@@/working/plan_extension.py')\n"
    "  _pxmod=_pxilu.module_from_spec(_pxspec); _pxspec.loader.exec_module(_pxmod); plan_extension=_pxmod\n"
    "  try:\n"
    "      SX['plan']=plan_extension.build_extended_plan(SX, STEPS)\n"
    "  except Exception as _pe: stop('PLAN_EXTENSION '+repr(_pe))\n"
    "  R['plan_extension']={'policy':plan_extension.POLICY_VERSION,'prefix_sha256':plan_extension.plan_sha256(SX['plan'][:200]),'full_sha256':plan_extension.plan_sha256(SX['plan']),'length':len(SX['plan'])}; save()\n")
)

_BASE_NEW = (
    K.expand("  import importlib.util as _cnilu\n"
    "  def _cn_load(_n):\n"
    "      _sp=_cnilu.spec_from_file_location(_n,'@@KROOT@@/working/'+_n+'.py'); _mo=_cnilu.module_from_spec(_sp); _sp.loader.exec_module(_mo); return _mo\n"
    "  cont=_cn_load('continuation'); stop_rule=_cn_load('stop_rule')\n"
    "  _cspec=_cnilu.spec_from_file_location('x64_checkpoint','@@KROOT@@/working/x64_checkpoint.py')\n"
    "  _cmod=_cnilu.module_from_spec(_cspec); _cspec.loader.exec_module(_cmod); x64_checkpoint=_cmod\n"
    "  m=make(); opt=rv.make_optimizer(m); rv.LR=3e-4\n"
    "  try:\n"
    "      CKINFO=cont.load_checkpoint200(CKPT_PATH, model=m, opt=opt, plan=SX['plan'], plan_module=plan_extension, device=DEVICE, cpu_smoke=CPU_SMOKE, experts_cls=rv.Experts)\n"
    "  except cont.ContinuationGateError as _ge: stop('CONTINUATION_GATE '+str(_ge))\n"
    "  R['continuation']={'checkpoint':{k:v for k,v in CKINFO.items() if k!='disclosure'},'disclosure':cont.DISCLOSURE,'scope':'update 200 -> 300 continuation; the run itself requires a separate approval'}; save()\n"
    "  t=time.time()\n"
    "  with cont.observe(m, opt):\n"
    "      R['baseline']=eval_block(m); R['baseline_rehearsal']=preclip_diag.rehearsal_readout(m, RY['rows'], RY['mask'])\n"
    "      _bt,_bh,_bc,_bx=cont.observe_counts(m, XD=XD, HO=HO, EV=EV, plan=SX['plan'], update=START, device=DEVICE, x64_ck=x64_checkpoint)\n"
    "  R['baseline_eval_s']=time.time()-t\n"
    "  try:\n"
    "      R['restoration_check']=cont.check_baseline_restoration(m, _bx, _bc, R['baseline'], strict=not CPU_SMOKE, x64_ck=x64_checkpoint, baseline_x64=stop_rule.BASELINE_X64)\n"
    "  except cont.ContinuationGateError as _ge: stop('CONTINUATION_GATE '+str(_ge))\n"
    "  HIST={START:{'x64':_bx,'counts':_bc}}\n"
    "  PROV={'resumed_from_sha256':cont.CKPT200_SHA256,'resumed_from_update':START,'plan':R['plan_extension'],'treatment_sha256':R['hashes']['treatment.py'],'continuation_sha256':hashlib.sha256(Path('@@KROOT@@/working/continuation.py').read_bytes()).hexdigest(),'disclosure_declared_in':'continuation.DISCLOSURE'}\n"
    "  save()\n")
)

_EVAL_NEW = (
    "              with cont.observe(m, opt):\n"
    "                  te=time.time(); A['trajectory'][str(step)]=eval_block(m); A['rehearsal_readout'][str(step)]=preclip_diag.rehearsal_readout(m, RY['rows'], RY['mask']); A['eval_s'][str(step)]=time.time()-te\n"
    "              if step in cont.CHECKPOINT_UPDATES:\n"
    "                  _rec=cont.dump_full(m, opt, step, XD=XD, HO=HO, EV=EV, plan=SX['plan'], plan_module=plan_extension, out_dir=OUT, diag=preclip_diag, x64_ck=x64_checkpoint, lib=rv, replay_lambda=lam, replay_batch=RB, seed=0, clip=1.0, provenance=PROV, device=DEVICE)\n"
    "                  A.setdefault('continuation_checkpoints',{})[str(step)]=_rec; HIST[step]={'x64':_rec['x64'],'counts':_rec['counts']}\n"
    "                  A.setdefault('stop_rule',{})[str(step)]=stop_rule.evaluate(HIST); save()\n"
)

_VERDICT_NEW = "  R['stop_rule_final']=stop_rule.evaluate(HIST); save()\n" + VERDICT


def _once(source, needle, label):
    if source.count(needle) != 1:
        raise ValueError(f"CONTINUE300_PATTERN_MISSING {label} count={source.count(needle)}")


def apply_continue300(source):
    if "plan_extension" in source or "START=200" in source:
        raise ValueError("CONTINUE300_ALREADY_APPLIED")
    for label, needle in (("head", HEAD), ("plan", PLAN_CHECK), ("base_a", BASE_A), ("base_b", BASE_B),
                          ("x64_loader_unused", X64_LOADER), ("arm_init", ARM_INIT), ("arm_rec", ARM_REC),
                          ("loop", LOOP), ("eval", EVAL_LINES), ("verdict", VERDICT)):
        _once(source, needle, label)
    out = source.replace(HEAD, HEAD_NEW, 1)
    out = out.replace(PLAN_CHECK, _PLAN_NEW + PLAN_CHECK, 1)
    out = out.replace(BASE_A, _BASE_NEW, 1)
    out = out.replace(BASE_B, "  print('BASE',R['baseline'],flush=True)\n", 1)
    # x64_checkpoint is now loaded before the baseline; drop the later duplicate loader pair.
    spec_line = K.expand("  _cspec=_ilu.spec_from_file_location('x64_checkpoint','@@KROOT@@/working/x64_checkpoint.py')\n")
    _once(out, spec_line, "x64_spec_line")
    out = out.replace(spec_line + X64_LOADER, "", 1)
    out = out.replace(ARM_INIT, ARM_INIT_NEW, 1)
    out = out.replace(ARM_REC, ARM_REC_NEW, 1)
    out = out.replace(LOOP, LOOP_NEW, 1)
    out = out.replace(EVAL_LINES, _EVAL_NEW, 1)
    out = out.replace(VERDICT, _VERDICT_NEW, 1)
    # Result checks.
    if out.count("random.seed(0)") != 0 or out.count("torch.manual_seed(0)") != 0:
        raise ValueError("CONTINUE300_RESEED_REMAINS")
    if "for s in range(START, STEPS)" not in out or "STEPS=300" not in out or "x64_checkpoint.dump(" in out:
        raise ValueError("CONTINUE300_RESULT_INVALID")
    if out.count("opt.step(selected)") != 1 or out.count("cont.load_checkpoint200(") != 1:
        raise ValueError("CONTINUE300_RESULT_INVALID")
    if out.index("cont.load_checkpoint200(") > out.index("for s in range(START, STEPS)"):
        raise ValueError("CONTINUE300_ORDER")
    return out


DIAG_OLD = "DIAGNOSTIC_UPDATES = (1, 75, 125, 175, 200)"
DIAG_NEW = "DIAGNOSTIC_UPDATES = (225, 250, 275, 300)"


def patch_preclip_diag(text):
    if text.count(DIAG_OLD) != 1:
        raise ValueError("CONTINUE300_DIAG_PATTERN_MISSING")
    return text.replace(DIAG_OLD, DIAG_NEW, 1)
