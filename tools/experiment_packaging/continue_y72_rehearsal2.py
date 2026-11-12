"""Turn the y72fix2 continuation treatment (300 -> 400) into Y72 rehearsal2 (prereg/y72_rehearsal2.json, rule vyra-y72-rehearsal2-fresh-v1).

Layered on continue300_y72fix2.apply_cont400 (exact, count-asserted text edits). Relative to the cont400 notebook the edits are exactly:
* the replay rows of the updates 301..400 come from the fresh pool F200 (y72_rehearsal2_pool: verified text form, canonical sha256
  d4315c33...): update s+1 uses pool rows 2*(s-300)+j, j = 0,1, so every pool row is presented exactly once. replay_at(s) for s < 300
  still returns the 40-row rows the history used (only the disjointness audit of all presentations calls it there);
* the pool is loaded and verified fail-closed next to the 40-row artifact and is part of the training-presentation disjointness audit
  against every held-out set (incl. the 600-set);
* the rows used at every update are recorded (arms.treatment.replay_rows), and after the restore from checkpoint_300 the sampler record of
  later checkpoints is relabelled with the F200 rule (y72_rehearsal2_pool.wrap_sampler_record);
* the pre-registered verdict y72_rehearsal2_gates (the cont400 gates, unchanged) and the non-gating manipulation checks M1-M3.
Nothing else changes: resume from checkpoint_300 with the same exact restore, plan400, per-task clip 0.5, replay weight 0.25, 2 replay
rows per update, LR 3e-4, union, optimizer, evaluation, the 600-set, seeds.
"""
import continue300_y72fix2 as c4
import kaggle_config as K

POOLLOAD_OLD = "  if tuple(RY['rows'].shape)!=(40,512) or tuple(RY['mask'].shape)!=(40,510): stop('rehearsal shape')\n"
POOLLOAD_NEW = POOLLOAD_OLD + K.expand(
    "  _r2spec=_rilu.spec_from_file_location('y72_rehearsal2_pool','@@KROOT@@/working/y72_rehearsal2_pool.py')\n"
    "  r2pool=_rilu.module_from_spec(_r2spec); _r2spec.loader.exec_module(r2pool)\n"
    "  try: PY2=r2pool.load_pool('@@KROOT@@/working/y72_fresh_pool_F200_set.json')\n"
    "  except Exception as _p2e: stop('REHEARSAL2_POOL '+repr(_p2e))\n"
    "  if RB!=r2pool.ROWS_PER_UPDATE or START!=r2pool.RESUME_UPDATE or STEPS!=r2pool.FINAL_UPDATE: stop('REHEARSAL2_SCHEDULE')\n"
    "  if set(PY2['payloads'])&(set(RY['payloads'])|set(Y72['unseen_payloads'])|set(Y72['payloads'])): stop('REHEARSAL2_POOL_PAYLOAD_OVERLAP')\n"
    "  R['rehearsal2_pool']=PY2['info']; save()\n")
REPLAY_OLD = ("  def replay_at(s):\n"
              "      idx=[(s*RB+j)%40 for j in range(RB)]\n"
              "      return RY['rows'][idx].long(),RY['mask'][idx].long(),idx\n")
REPLAY_NEW = ("  def replay_at(s):\n"
              "      if s<START:\n"
              "          idx=[(s*RB+j)%40 for j in range(RB)]\n"
              "          return RY['rows'][idx].long(),RY['mask'][idx].long(),idx\n"
              "      idx=r2pool.indices(s)\n"
              "      return PY2['rows'][idx].long(),PY2['mask'][idx].long(),idx\n")
CALL_OLD = "if lam: yt,ym,_=replay_at(s); yt=yt.to('cuda')"
CALL_NEW = "if lam: yt,ym,_ridx=replay_at(s); A.setdefault('replay_rows',[]).append([int(_i) for _i in _ridx]); yt=yt.to('cuda')"
LOAD_R2 = ("cont=_cn_load('continuation'); r300=_cn_load('resume300_y72fix2'); gates=_cn_load('y72_rehearsal2_gates'); "
           "y2rt=_cn_load('y72fix2_runtime'); c4rt=_cn_load('y72_rehearsal2_runtime'); clip_scheme=_cn_load('clip_scheme')\n")
GATE_R2 = (
    "      CKINFO=r300.load_checkpoint300(CKPT_PATH, model=m, opt=opt, plan=SX['plan'], lib=rv, device=DEVICE, replay_batch=RB, cpu_smoke=CPU_SMOKE)\n"
    "  except cont.ContinuationGateError as _ge: stop('CONTINUATION_GATE '+str(_ge))\n"
    "  cont.sampler_record=r2pool.wrap_sampler_record(cont.sampler_record)\n"
    "  R['continuation']={'checkpoint':CKINFO,'experiment':'Y72 rehearsal2: resume at 300, stop at 400, replay rows from the fresh once-only pool F200, per-task clip 0.5 and replay weight 0.25 unchanged (prereg/y72_rehearsal2.json)',"
    "'only_variable':'replay rows: F200 rows 2*(s-300)+j, each pool row once (control: the fixed 40-row set)','restored':'complete v2 state: params, optimizer, torch+CUDA RNG, Python RNG, NumPy RNG, router, mode, sampler'}; save()\n"
)
PROV_R2 = ("'resumed_from_sha256':r300.CKPT300_SHA256,'resumed_from_update':START,'replay_lambda':r300.NEW_LAMBDA,"
           "'only_variable':'replay rows from the fresh pool F200 (prereg/y72_rehearsal2.json)','replay_source':PY2['info'],"
           "'clip_scheme':'pertask tau 0.5 (the grad_clip_max_norm 1.0 field of the checkpoint config is the legacy global value, unused here)',")
EVAL_R2 = "A.setdefault('y72_rehearsal2_gates',{})[str(step)]=gates.evaluate(c4rt.history(HIST)); save()\n"
FINAL_R2 = ("  R['y72_rehearsal2_verdict']=gates.evaluate(c4rt.history(HIST)); save()\n"
            "  _TA=R['arms']['treatment']; R['y72_rehearsal2_checks']=r2pool.manipulation_checks(_TA['task_clip'], _TA['loss'], _TA.get('replay_rows',[]), first_update=START+1); save()\n")

SWAPS = (("poolload", POOLLOAD_OLD, POOLLOAD_NEW), ("replay", REPLAY_OLD, REPLAY_NEW), ("call", CALL_OLD, CALL_NEW),
         ("load", c4.LOAD_400, LOAD_R2), ("gate", c4.GATE_400, GATE_R2), ("prov", c4.PROV_400, PROV_R2),
         ("eval", c4.EVAL_400, EVAL_R2), ("final", c4.FINAL_400, FINAL_R2))
BANNED = c4.BANNED + ("y72fix2_cont400_gates", "y72fix2_cont400_runtime", "y72fix2_cont400_verdict", "prereg/y72fix2_cont400.json",
                      "'only_variable':'none", "yt,ym,_=replay_at(s)")


def apply_rehearsal2(source):
    out = c4.apply_cont400(source)
    for label, old, new in SWAPS:
        if out.count(old) != 1:
            raise ValueError(f"REHEARSAL2_PATTERN_MISSING {label} count={out.count(old)}")
        out = out.replace(old, new, 1)
    for banned in BANNED:
        if banned in out:
            raise ValueError(f"REHEARSAL2_RESIDUE {banned}")
    if (out.count("r2pool.indices(s)") != 1 or out.count("replay_at(s)") < 2 or "clip_scheme.combine(_pp,_gx,_gy,lam,'pertask',0.5)" not in out
            or out.count("opt.step(selected)") != 1 or "STEPS=400; START=300" not in out):
        raise ValueError("REHEARSAL2_RESULT_INVALID")
    order = [out.index(x) for x in ("r2pool=_rilu.module_from_spec", "def replay_at(s):", "r300.load_checkpoint300(",
                                    "cont.sampler_record=r2pool.wrap_sampler_record", "for s in range(START, STEPS)")]
    if order != sorted(order):
        raise ValueError("REHEARSAL2_ORDER")
    return out


patch_preclip_diag = c4.patch_preclip_diag
