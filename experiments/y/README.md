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

## Running it — read this before you try

This directory is the code that produced the published result, published so the result can be
inspected and checked. It is **not** a self-contained reproduction package, and the corpus hash
below **cannot** be reproduced from it. That is a limitation of this publication, stated plainly
rather than hidden behind a check that looks rigorous and isn't.

Why the hash cannot match:

1. The saved corpus embeds the lab path it was built at (`meta["shim"]` records `LAB->/tmp/ylab`),
   so its sha256 is a function of where it runs, not only of its contents.
2. The payload embeds an internal sign-off field that the published script fills with a constant
   placeholder; the original run recorded a real value, so the bytes differ regardless of path.
3. Building the corpus also needs inputs that are not published here (the model loader
   `run_vyra1_copy.py`, `verify_shapes.py`, `corpus.pt`, `probe_Q.pt`, and the four training-set tensors from
   experiments T2, V, W and X. `anx_vyra/` is a separate, cleaner implementation and cannot load them.

So: do not compare your hash to the one below and conclude the experiment was wrong. A mismatch is
expected and proves nothing either way.

What this directory is good for: reading exactly how the jitter was scheduled and audited, how the
library-LR trap was asserted against, how the GPU/CPU equivalence gate was applied, and how each
per-geometry number was measured. Those are the mechanisms the write-up depends on.

For reference, the original corpus recorded:

```
sha256 438dfdc6bca259783784dbbd5af7544c38a7762fef3287267e1db8e3e5667d45
built at /tmp/ylab, 2026-10-29, by the author's local run
```

If you want a genuinely reproducible VYRA experiment, the right thing to ask for is a
self-contained bundle: model code, loader, corpora and probes in one dataset with a manifest, and
no absolute paths baked into the artifacts. That is not what this is.
