# Update 200 -> 300 continuation (prepared, NOT run)

Preparation only. Nothing here trains, uploads, or submits. Running it needs a
separate approval from the repository owner, including the one-time private
Kaggle dataset that holds the unchanged `checkpoint_200.pt`.

## What is built

`tools/experiment_packaging/build_continuation.py` rebuilds the 200-update
treatment exactly as `build.py` does and then rewrites it with
`continue300.py`. The checkpoint is bound by SHA-256 into the dependency lock:

| item | value |
| --- | --- |
| checkpoint | `checkpoint_200.pt`, 197225718 bytes, sha256 `8542c3405309a5fcf9e419a6e3dd733beab346765fa7bd9cbdd61226dd867e81` |
| planned mount | `<kaggle-root>/input/datasets/<user>/<ckpt200-dataset-slug>/checkpoint_200.pt` (private dataset; real values in the local `kaggle_config.json`) |
| updates | 201..300, one treatment arm, seed/LR/clip/lambda/replay unchanged |
| checkpoints | 225, 250, 275, 300 (schema v2, complete state) |
| limits | trainer 600 s, notebook 900 s, unchanged |

## Plan extension (`plan_extension.py`)

The pinned `train_X.pt` plan has 200 entries. Entries 200..299 are built only
from rows that already exist in that file, by the documented policy in the
module docstring, from three SHA-256 counter-mode streams. They use neither
`random`, NumPy nor torch, so `Random(901)` / `Random(9101)` offsets of the
original builder are not touched. Entries 0..199 are copied unchanged; the
prefix hash, the extension hash and the full-plan hash are pinned and checked
before training.

## Disclosed non-restored state

`checkpoint_200.pt` holds only the model, the name-keyed SliceAdam state, and
the torch CPU/CUDA RNG states. See `continuation.DISCLOSURE` for the exact
values the rest is set to (Python RNG `seed(0)`, NumPy `seed(0)`, router
`_selected` absent, `_load` zeros, sampler position reconstructed). The run
records that disclosure in `result.json`. New checkpoints capture all of it.

## Fail-closed checkpoint load

`continuation.load_checkpoint200` verifies SHA-256 and size of the one exact
path before `torch.load`, then loads the verified bytes with
`weights_only=True`, and refuses on: missing file, wrong path, symlink or
directory, hash/size, schema, update != 200, optimizer class or name mismatch,
plan-prefix mismatch, device mismatch. No discovery, no retry, no restart.

### Accelerator
The 200-update run requested `machine_shape: "NvidiaTeslaT4"` (Kaggle CLI 2.2.5
help: "GPU T4 x2"; the CLI passes the field through and the server decides) and
saw 2 x Tesla T4, but `treatment.py` only ever uses `cuda` == `cuda:0`. The
continuation requests the same shape and keeps the two-device gate: exactly 2
visible CUDA devices (equal to the 2 saved CUDA RNG states), all `Tesla T4`,
every parameter on `cuda:0`. The saved `cuda:1` RNG state is byte-identical to
`cuda:0`'s (seed 0, offset 0, never consumed), restored and inert.

### Restoration check (pre-registered, `prereg/restoration_check_200.json`)
After restore, before any training step: parameter sha256 must equal the
recorded update-200 hash (always fatal) and the recomputed integer counts must
equal the recorded ones exactly (zero tolerance): X64 held-out 257/1920 and
train 669/1920 with segment-exact 0/0, and the Y72/X32/X109 integer counts. The
CPU smoke records the same comparisons without stopping.

## Observers

Eval, rehearsal readout, the X64 dump and the checkpoint capture run inside
`continuation.observe`, which restores RNG (Python, NumPy, torch CPU/CUDA),
route metadata and module modes and verifies params, grads and optimizer state
are bit-identical on exit; a write raises and stops the run.

## Stop rule (pre-registered, `prereg/stop_rule_300.{json,md}`)

One rule, integer arithmetic only, evaluated by `stop_rule.py`:
STUCK at 300 iff X64 train and held-out segment-exact counts are both 0 AND
(held-out token acc(300) - acc(200) < 10.0 pp OR train token acc(300) - acc(250)
< 3.0 pp), strict `<`. SLOW-LEARNER evidence iff at any of 225/250/275/300 the
X64 held-out segment-exact > 0 and same-checkpoint Y72 segment-exact >= 80.0%.
RETENTION FAILURE iff Y72 < 80.0% at any checkpoint (blocks any success claim).
Otherwise INCONCLUSIVE. Training always stops at 300.

## Tests

`tests/test_resume_equivalence.py`, `tests/test_resume_equivalence_real.py`
(real model classes, needs the private X bundle), `tests/test_plan_extension.py`,
`tests/test_continuation_gates.py`, `tests/test_observer_invariance.py`,
`tests/test_stop_rule.py`, `tests/test_continue300.py`.
`cpu_smoke_continuation.py` and `validate_continuation.py` are test-only
harnesses that run the built notebook in a bubblewrap CPU sandbox.
