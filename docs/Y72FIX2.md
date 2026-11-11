# Y72 retention fix 2: the run package (per-task clip)

Specification: [`prereg/y72fix2.md`](../prereg/y72fix2.md) (`prereg/y72fix2.json`, sha256 `8eaace70…5382`, registered at b99677a).
This document describes how that specification was turned into a closed notebook. Nothing in it changes the registered change or gates.

## What was built
* `tools/experiment_packaging/build_y72fix2.py` builds the notebook from the same source notebook, archives and rehearsal artifact as the
  λ0.5 package. It refuses a prereg file whose sha256 differs, verifies the registered 600-segment set (canonical and payload-list sha256)
  and embeds its text form (`prereg/y72_heldout_v2/y72_heldout_v2_set.json`) and the prereg JSON as hashed assets. No other data is uploaded.
* `continue225_y72fix2.py` applies count-asserted text edits to the 200-update treatment. Relative to the run that produced checkpoint 225
  they are: resume at 225; **`loss.backward(); clip_grad_norm_(1.0)` replaced by per-task gradients, a per-task clip of 0.5 on the X gradient and
  on 0.25 × the replay gradient (`clip_scheme`, `'pertask'`), assigned as `.grad`, then the unchanged `opt.step(selected)`**; the 600-segment set added to the held-out
  dict and to the evaluated sets (it passes the same train/held-out disjointness check); the pre-registered verdict (`y72fix2_gates`) replaces the stop
  rule. The old sparse diagnostics, which assume the global clip, are switched off; the per-update per-task record (norms, cosine, clip factors) and the
  unclipped combined norm are logged instead.
* `resume225_y72fix2.py`: the checkpoint-225 resume of the λ0.5 package with the replay weight required to stay at the saved 0.25.
* `y72fix2_runtime.py`: loading the 600-set, turning the run history into the gate inputs (with denominator checks), the logged norm.
* Kernel (slug in the local `kaggle_config.json`), private, GPU `NvidiaTeslaT4`, internet off, datasets: X bundle, Y bundle and the private checkpoint-225 dataset reused from the λ0.5 run.

## Checks before launch
Two builds in different working directories / hash seeds / time zones are byte-identical; bubblewrap preflight (no network, no GPU) passes with
55 locked files incl. the checkpoint and rejects a flipped checkpoint byte, a missing dataset and a wrong file name; a CPU smoke on the real checkpoint
restores with zero tolerance (model-state sha and every integer count equal the recorded ones) and runs real updates. `tests/test_y72fix2_build.py`
pins the generator, the λ-unchanged resume gate, the runtime glue and (env-gated) the built notebook. Notebook / metadata / lock / report hashes: `packages/y72fix2/SHA256SUMS`
(rebuild with the same inputs to reproduce them).
