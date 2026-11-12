# Y72 rehearsal2 (fresh once-only pool), 300 → 400: the run package

Specification: [`prereg/y72_rehearsal2.md`](../prereg/y72_rehearsal2.md) (`prereg/y72_rehearsal2.json`, sha256 `b02c4510…c577cab`,
registered at fbc9638 before any of this code existed). This document describes how that specification became a closed notebook. Nothing
in it changes the registered design, pool or gates. **Status: built and CPU-preflighted only.** Nothing has been uploaded, no kernel has
been created, and nothing has run on a GPU. Each of those needs its own approval.

## What was built
* `build_y72_rehearsal2.py` is derived from `build_y72fix2_cont400.py`, so it uses the same source notebook, archives, rehearsal artifact
  and checkpoint_300 (`5df9a3f8…45a3`, 197237651 bytes). It refuses:
  * a prereg JSON or prose file whose sha256 differs;
  * a pool file whose text, canonical or payload-list sha256 differs from the pins registered in the prereg.

  It embeds the F200 pool (`y72_fresh_pool_F200_set.json`, text `604bba8a…3c45`, canonical `d4315c33…76d8`) and the prereg JSON as hashed
  assets.
* `continue_y72_rehearsal2.py` applies count-asserted text edits on top of the cont400 treatment (`continue300_y72fix2`). It changes
  only the source of replay rows:
  * Update s+1 (s = 300..399) uses pool rows `2*(s-300)+j`, j = 0, 1, so each of the 200 rows is presented exactly once.
  * The pool is loaded and verified in the notebook. Its payloads are checked to be disjoint from the 40-row rehearsal set, every held-out
    set and the training payloads.
  * The saved sampler record is checked at restore against the old 40-row rule. It is relabelled to the F200 rule only for dumps after
    update 300.
  * The result records which rows were presented, the cont400 gates (unchanged, under the rehearsal2 rule id), and the non-gating
    manipulation checks M1–M3.

  Clip 0.5, replay weight 0.25, LR, union, optimizer, plan and data are unchanged. No earlier module is edited, so every earlier notebook
  rebuilds to its recorded sha256.
* `y72_rehearsal2_pool.py` handles the pool: verification, the once-only schedule, the sampler relabel and the M1–M3 checks.
  `y72_rehearsal2_gates.py` and `y72_rehearsal2_runtime.py` are the cont400 gates and runtime under the new rule id, with identical logic.
* There is one new optional key in the private `kaggle_config.json` (template: `kaggle_config.example.json`): `kernel_y72_rehearsal2`. A token
  whose key is absent fails closed. The checkpoint_300 dataset key (`ckpt300`) is the one cont400 already uses.

## Checks before launch (CPU only)
* Two builds with different hash seeds and time zones, plus a build from a clean clone, are byte-identical: notebook sha256 `c3ab200b…2be2d8e`.
  All hashes are in `packages/y72_rehearsal2/SHA256SUMS`. The five earlier notebooks still rebuild to their recorded hashes.
* Bubblewrap preflight (no network, no GPU) passes with 58 locked files, including the checkpoint, the pool and the prereg. It rejects:
  * a flipped checkpoint byte;
  * a missing dataset;
  * a wrong checkpoint file name;
  * checkpoint_225 presented as checkpoint_300;
  * a tampered pool asset;
  * a substituted pool with consistent hashes and lock.
* CPU smoke on the real checkpoint_300 (updates 301–302) restores exactly: the model-state sha and every integer count match. It presents
  pool rows [0,1] and [2,3] once each, and writes a checkpoint whose sampler record carries the F200 rule with next rows [4,5]. Smoke numbers
  are not gate measurements.
* `tests/test_y72_rehearsal2.py` covers the following against the registration:
  * prereg, prose and pool pins;
  * gates and boundaries, including identity to the cont400 gates;
  * the once-only schedule, the M1–M3 checks and the sampler relabel;
  * the generator edits and their refusals;
  * the built notebook (env-gated).
