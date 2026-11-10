# Replay-weight experiment: resume at 225, lambda 0.25 -> 0.5 (prep only)

Status: **prepared, not approved to run.** Pre-registration: `prereg/y72fix_lambda05.json` / `.md`.
Diagnosis behind the choice: `/workspace/forge/vyra-y72fix-prep-20261110/DIAGNOSIS.md` (not part of the repo).

* One variable: `replay_lambda` 0.25 -> 0.5. Everything else is restored bit-for-bit from `checkpoint_225.pt`
  (complete v2 state written by the approved 200 -> 300 run) and pinned.
* `tools/experiment_packaging/resume225.py` - SHA-256/size gate before `torch.load`, schema v2, update 225, saved lambda exactly
  0.25 and every other optimizer-config field equal, pinned plan hashes, two-T4 device gate, full-state restore, pinned model-state
  hash, zero-tolerance integer restoration check against the counts the 300 run recorded at 225.
* `tools/experiment_packaging/y72fix_gates.py` - the pre-registered gates R/A/N and the 8-way verdict table (integer arithmetic).
* `tools/experiment_packaging/continue225.py` - exact, count-asserted text edits layered on `continue300` (resume point, lambda,
  verdict wiring, per-update pre-clip norm logging that is read after clipping and changes nothing).
* `tools/experiment_packaging/build_y72fix.py` - deterministic builder (kernel id `anjorosimoaj/vyra-y72fix-lambda05-packaged`,
  private dataset `anjorosimoaj/vyra-resume300-ckpt225` containing only `checkpoint_225.pt`).
* Test-only harnesses: `cpu_smoke_y72fix.py` (bubblewrap CPU, real checkpoint, 2 updates), `validate_y72fix.py` (preflight + negatives).
* Tests: `tests/test_y72fix_gates.py`, `tests/test_resume225.py` (gate negatives, lambda resume-equivalence incl. negative control),
  `tests/test_y72fix_build.py` (generator; built notebook with `VYRA_Y72FIX_NOTEBOOK=<replay.ipynb>`).
* Nothing here uploads, submits or runs anything on Kaggle. The old 200 -> 300 package (notebook 22cb420c...) is unchanged.
