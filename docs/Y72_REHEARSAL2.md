# Y72 rehearsal2 (fresh once-only pool), 300 → 400: the run package

Specification: [`prereg/y72_rehearsal2.md`](../prereg/y72_rehearsal2.md) (`prereg/y72_rehearsal2.json`, sha256 `b02c4510…c577cab`,
registered at fbc9638 before any of this code existed). This document describes how that specification became a closed notebook. Nothing
in it changes the registered design, pool or gates. **Status: built and CPU-preflighted only.** No kernel has been created and nothing has
run on a GPU. Each of those needs its own approval.

## Repack under Kaggle's 1 MB source limit (supersedes notebook `c3ab200b`)
The first build of this package (`c3ab200b…2be2d8e`, commit 00af1ad, 1,708,020 bytes) was refused by Kaggle's SaveKernel with a
bare `400 Bad Request`: Kaggle rejects a kernel source of 1 MB (1,048,576 bytes) or more. The embedded F200 pool (841,576 bytes of
hex JSON) and the 600-segment set made up most of the size. No kernel was created and nothing ran.

* `notebook_pack.py` stores **all** embedded assets as one deterministic blob, built as JSON (sorted keys, compact, ASCII), then
  zlib level 9, then base64. The rehearsal artifact bytes are stored as zlib level 9 plus base64. In the notebook, the standard
  library decompresses both. Before decompressing, the notebook checks the blob's SHA-256 (`BOOTSTRAP_PACK_CORRUPT`). After
  decompressing, the unchanged per-asset SHA-256 loop (`BOOTSTRAP_CORRUPT name`), the dependency lock, the rehearsal pin, and the
  pool module's registered hashes (canonical `d4315c33…76d8`) all run as before.
* The decompressed asset texts and rehearsal bytes are byte-identical to those of `c3ab200b`: the same 29 assets, the same
  `asset_hashes` line, the same dependency lock and the same kernel metadata. Only how they are stored changed.
* Size gate (fail-closed): the builder refuses to write a notebook of 1,048,576 bytes or more (`NOTEBOOK_TOO_LARGE`).
  `validate_y72_rehearsal2.py` re-checks the file on disk before any sandbox run, and it requires an oversized copy to be
  refused. `python notebook_pack.py check NOTEBOOK` runs the same gate on its own.
* The repacked notebook was `3d5756c8…a15e4b`, 383,178 bytes (36.5 % of the limit). It is superseded by `14b5a99a` below.
  Commit 00af1ad still rebuilds `c3ab200b`, kept for the record. It must not be pushed.

## Mount diagnosis and hash-located datasets (supersedes notebook `3d5756c8`)
Notebook `3d5756c8` (commit ec46d3c) was pushed once and stopped about 50 s later with `MISSING_DATASET_ROOT C`. That was after the
asset bootstrap and before any GPU work, so nothing ran and nothing was scored. What the saved outputs determine:
* The package side is identical to the cont400 package that ran successfully. These match byte-for-byte: the dependency lock's
  dataset entries (C/X/Y ids, pinned mounts and file hashes), `dependency_gate.py`, and the kernel metadata's dataset sources and
  settings. Those three mount paths all verified in the cont400 kernel.
* Kaggle's own record of the failed kernel lists all three datasets. In that session, the pinned checkpoint directory did not exist.
* Undetermined: where, or whether, Kaggle mounted the checkpoint dataset in that session. The old notebook never listed its inputs.
  The lock checks C first, so the X/Y mounts are unknown too. Kaggle is publicly reported to mount datasets under either
  `<input>/datasets/<owner>/<slug>` or `<input>/<slug>`, depending on how they are attached. That fits the symptom but is
  unverified for this kernel.

Fix (packaging only; same prereg, pool, gates, checkpoint):
* `mount_resolver.py` prints and saves a redacted listing of the input root before any dataset check (`INPUT_LISTING`; the user
  and slugs are replaced by `<user>` and `<slug:ROLE>`).
* It uses each dataset's pinned directory when that exists. A missing or wrong file there stays fatal in the unchanged gate.
* Only when the pinned directory is absent does it look for the one directory that holds all locked files at the locked SHA-256.
  None found raises `MISSING_DATASET_ROOT`; more than one raises `AMBIGUOUS_DATASET_ROOT`.
* The unchanged `dependency_gate.verify_lock` then hashes every locked file at the resolved mount. X/Y are passed to input
  validation from the resolved roots. The trainer receives the checkpoint path as `VYRA_CKPT300_PATH`, and `load_checkpoint300`
  still verifies its SHA-256 and size before `torch.load`.
* New notebook `14b5a99a…4b902c`, 386,403 bytes, under the 1 MB gate. Preflight adds three cases: a checkpoint at a non-pinned
  path found by hash (passes), a non-pinned wrong-hash copy (refused), and two matching copies (refused as ambiguous).

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
* Two builds with different hash seeds and time zones, plus a build from a clean clone, are byte-identical: notebook sha256
  `14b5a99a…4b902c` (386,403 bytes, under the 1 MB gate). Hashes are in `packages/y72_rehearsal2/SHA256SUMS`.
  All hashes are in `packages/y72_rehearsal2/SHA256SUMS`. The five earlier notebooks still rebuild to their recorded hashes.
* Bubblewrap preflight (no network, no GPU) passes with 59 locked files (now including `mount_resolver.py`), including the checkpoint, the pool and the prereg. It rejects:
  * a flipped checkpoint byte;
  * a missing dataset;
  * a wrong checkpoint file name;
  * checkpoint_225 presented as checkpoint_300;
  * a tampered pool asset;
  * a substituted pool with consistent hashes and lock;
  * a changed compressed blob (`BOOTSTRAP_PACK_CORRUPT`);
  * an oversized notebook (`NOTEBOOK_TOO_LARGE`).
* CPU smoke on the real checkpoint_300 (updates 301–302) restores exactly: the model-state sha and every integer count match. It presents
  pool rows [0,1] and [2,3] once each, and writes a checkpoint whose sampler record carries the F200 rule with next rows [4,5]. Smoke numbers
  are not gate measurements.
* `tests/test_y72_rehearsal2.py` covers the following against the registration:
  * prereg, prose and pool pins;
  * gates and boundaries, including identity to the cont400 gates;
  * the once-only schedule, the M1–M3 checks and the sampler relabel;
  * the generator edits and their refusals;
  * the built notebook (env-gated).
