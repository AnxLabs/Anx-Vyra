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
| planned mount | `/kaggle/input/datasets/anjorosimoaj/vyra-expanded40-200u-checkpoint/checkpoint_200.pt` (dataset `anjorosimoaj/vyra-expanded40-200u-checkpoint`, private) |
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
plan-prefix mismatch, device mismatch (CUDA only, device count equal to the
saved CUDA RNG count). No discovery, no retry, no restart. After restoring,
the parameter hash must equal the recorded update-200 model-state hash and,
on the T4, the update-200 evals and X64 counts must reproduce.

## Observers

Eval, rehearsal readout, the X64 dump and the checkpoint capture run inside
`continuation.observe`, which restores RNG (Python, NumPy, torch CPU/CUDA),
route metadata and module modes and verifies params, grads and optimizer state
are bit-identical on exit; a write raises and stops the run.

## Stop rule

`stop_rule.py` encodes the pre-registered rule with integer arithmetic only.
The run always ends at 300. It never starts a follow-up.

## Tests

`tests/test_resume_equivalence.py`, `tests/test_resume_equivalence_real.py`
(real model classes, needs the private X bundle), `tests/test_plan_extension.py`,
`tests/test_continuation_gates.py`, `tests/test_observer_invariance.py`,
`tests/test_stop_rule.py`, `tests/test_continue300.py`.
`cpu_smoke_continuation.py` and `validate_continuation.py` are test-only
harnesses that run the built notebook in a bubblewrap CPU sandbox.
