# Per-task clip candidate: specification (not built)

Status: specification only. Nothing here changes a committed builder, notebook or checkpoint; no notebook for this candidate exists, nothing was uploaded or run.
Evidence: `/workspace/forge/vyra-noise-clip-study-20261110/REPORT.md` (CPU only, from saved checkpoints).

## What the code in this commit is
* `tools/experiment_packaging/clip_scheme.py` - pure functions: per-task gradients read from the graph the step already built (no `.grad` side effects),
  coefficient rules for `global` (identical to `loss.backward(); clip_grad_norm_(1.0)` up to float error, tested), `pertask` (each task clipped to tau) and `pcgrad`.
* `tools/experiment_packaging/pertask_clip_gates.py` + `prereg/pertask_clip_v1.{json,md}` - integer gates and verdict table, pinned together by tests.
* `tests/test_clip_scheme.py`, `tests/test_pertask_clip_gates.py`.

## What a GPU package would still need (why it was not built)
1. A new exact, count-asserted text edit of the training step in `treatment.py`, replacing
   `loss.backward()` ... `_preclip_norm=torch.nn.utils.clip_grad_norm_(_clip_params,1.0)` by
   `_gx,_gy=clip_scheme.task_gradients(params,lx,ly); _g,_rec=clip_scheme.combine(params,_gx,_gy,lam,'pertask',0.5); clip_scheme.assign(params,_g)`;
   the pre-clip norm logged in `A['preclip_norm']`, `tel.note_grads`, `preclip_diag.finish_combined` (which assumes the global clip multiplier) and the finite-gradient check
   must be re-pointed and re-tested (observer invariance included). Two backward passes instead of one (the second only through the 2 replay rows).
2. A resume module that accepts replay_lambda == saved 0.25 (resume225 pins 0.5), a gates import swap, a builder variant, a CPU smoke/validate pair and the two-identical-builds proof.
3. New kernel id; the private dataset `anjorosimoaj/vyra-resume300-ckpt225` can be reused, so no new upload.
Estimated work: about 5 new modules plus tests and a build/validate cycle. Run cost estimate (from the two observed 75-update runs, trainer ~130 s, notebook ~148 s, ~0.09-0.15 h of
displayed quota, $0): about 1.2-1.4x for the second backward on 2 rows, i.e. roughly 3 minutes, still $0 on the free T4 x2.
Approval it would need: building it (code + tests + commit), then a separate approval to run one kernel on free T4 x2.
