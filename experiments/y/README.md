# Experiment Y — distance jitter

Reproduces the run recorded in [`results/experiment_y_jitter_fix.json`](../../results/experiment_y_jitter_fix.json).
The outcome was negative: jittering the needle-to-query distance moved the model's
sharp points rather than smoothing them. See [docs/EXPERIMENTS.md](../../docs/EXPERIMENTS.md).

## What changed from Experiment X

Exactly one variable. X trained three fixed distances (32, 64, 109). Y draws each
needle slot's geometry per step from `target ± 8` in steps of 4:

| target | band trained |
|---|---|
| 32 | 24, 28, 32, 36, 40 |
| 64 | 56, 60, 64, 68, 72 |
| 109 | 101, 105, 109, 113, 117 |

Everything else is held fixed: arm R, LR 3e-4, 200 steps, batch 5 (2 general rows +
3 needle slots), 5 passes per (geometry, row), fresh optimizer, starting from the
Experiment X step-200 checkpoint.

## Files

- `make_corpus_Y.py` — builds the training corpus and the unseen/gap probes, and
  asserts the row sets are pairwise disjoint so no probe leaks into training.
- `y_train.py` — the trainer: jitter scheduling, the library-LR trap assertion, the
  GPU/CPU equivalence gate, and the per-geometry evaluation.
- `vyra_shim.py` — portability shim for running the original Windows-authored library
  on Linux/CUDA.

## Running it

`y_train.py` expects the lab root in `VYRA_LAB` and the starting weights at
`bundle/vyra_X_step200_weights.pt` (override with `T2_CKPT`). Model weights are not
distributed here; obtain or retrain them from the architecture in `anx_vyra/`.

```sh
export VYRA_LAB=./lab
python make_corpus_Y.py          # writes train_Y.pt and the probe sets
python y_train.py                # writes the checkpoint series and results JSON
```

The published corpus has sha256 `438dfdc6bca259783784dbbd5af7544c38a7762fef3287267e1db8e3e5667d45`.
If your rebuilt corpus differs, the run is not reproducing this result.
