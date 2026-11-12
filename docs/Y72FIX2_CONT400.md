# y72fix2 continuation 300 → 400: the run package

Specification: [`prereg/y72fix2_cont400.md`](../prereg/y72fix2_cont400.md) (`prereg/y72fix2_cont400.json`, sha256 `f71193df…8c41`, registered at
c2dfa1f before any of this code existed). This document describes how that specification was turned into a closed notebook. Nothing in it
changes the registered design or gates. **Status: built and CPU-preflighted only.** Nothing has been uploaded, no checkpoint-300 dataset exists,
and nothing has run on a GPU. Each of those needs its own approval.

## What was built
* `build_y72fix2_cont400.py` builds the notebook from the same source notebook, archives and rehearsal artifact as the y72fix2 package. It
  refuses a prereg file whose sha256 differs, and it refuses a checkpoint whose size or sha256 differs from checkpoint_300
  (`5df9a3f8…45a3`, 197237651 bytes). It verifies the 600-segment set and embeds it and the prereg JSON as hashed assets.
* `continue300_y72fix2.py` applies count-asserted text edits on top of `continue225_y72fix2` (the y72fix2 treatment). The edits are:
  * resume at 300 and stop at 400, with evals and full checkpoints at 325/350/375/400;
  * the 400-entry plan;
  * `resume300_y72fix2` in place of `resume225_y72fix2`;
  * checkpoint dumps that record the 400-entry plan;
  * `y72fix2_cont400_gates` in place of `y72fix2_gates`.

  The per-task clip 0.5, replay weight 0.25, replay rows, LR, union, optimizer and data are unchanged. Old asset modules are not edited, so
  every earlier notebook rebuilds to its recorded sha256.
* `plan_extension400.py` builds the v1 300-entry plan (unchanged, pinned), then entries 300..399 from the same v1 SHA-256 streams
  (`build_extension(count=200)[100:]`). The pins are: continuation `1cfdd455…2499`, full `d9f13234…d943`, and prefix 300 `2c59794e…86aa`.
* `resume300_y72fix2.py` reads the SHA-256 before `torch.load`. It then checks:
  * schema v2 and update 300;
  * the saved config, with λ exactly 0.25;
  * the saved sampler record, pinned in full;
  * the live 400-entry plan.

  It restores the complete v2 state, giving `restore_full` the 300-entry prefix and the v1 module because those are what the saved
  sampler record refers to. Finally it runs the exact restoration check: the model-state sha is always fatal, and the integer counts at 300
  must equal the y72fix2 run's recorded counts, including the 600-set count of 496.
* `y72fix2_cont400_gates.py` and `y72fix2_cont400_runtime.py` hold the registered R hold / A progress / N gates in integer arithmetic, and
  the history over 350/375/400 with denominator checks.
* Two new optional keys in the private `kaggle_config.json` (template: `kaggle_config.example.json`): `ckpt300` and `kernel_y72fix2_cont400`.
  Older configs still load. A token whose key is absent fails closed.
* Kernel (slug in the local config): private, GPU `NvidiaTeslaT4`, internet off. Datasets: X bundle, Y bundle, and a new private dataset that
  would hold only `checkpoint_300.pt` and **does not exist yet**.

## Checks before launch (CPU only)
* Two builds are byte-identical, and a build from a clean clone with a different hash seed and time zone matches them. Notebook sha256
  `e13b599a…01e3`; all hashes are in `packages/y72fix2_cont400/SHA256SUMS`. The four earlier notebooks still rebuild to their recorded hashes.
* Bubblewrap preflight (no network, no GPU) passes with 56 locked files, including the checkpoint and the 400-entry plan pins. It rejects a
  flipped checkpoint byte, a missing dataset, a wrong file name, and the real checkpoint_225 presented as checkpoint_300.
* CPU smoke on the real checkpoint_300 restores with zero tolerance: the model-state sha and every integer count equal the recorded ones.
  It then runs real updates 301 and 302 and writes v2 checkpoints whose sampler record carries the 400-entry plan.
* `tests/test_y72fix2_cont400.py` pins the gates, boundaries, resume gate, restoration check, plan pins, generator and (env-gated) the built
  notebook to the registration.
