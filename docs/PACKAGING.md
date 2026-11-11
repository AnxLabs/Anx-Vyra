# Exact replay package

The replay notebook is built from locked inputs. The builder does not regenerate
corpora, does not train, and does not submit a job. A receipt from this tool is
an isolated CPU initialization check. It is not a GPU result and it does not
grant a launch.

Private dataset archives and the source notebook are not in this repository.
Without them the target stops at `REAL_INPUTS_UNAVAILABLE`. That status is not
a package pass.

## Caller fix applied at build

`expert_selection` reports the last forward only. The replay caller runs a
primary forward and a replay forward, then one backward, one global clip, and
one SliceAdam step. The builder rewrites that caller so the first selection is
captured immediately and unioned with the second before the step. A caller that
does not contain the confirmed lines exactly once is refused. Single-forward
updates are unchanged: with the replay term off, the captured selection is the
only forward.

Scientific gates, loss coefficient, learning rate, and the optimizer algebra
are not rewritten.

## Telemetry

The same rewrite stores the two selections and, after the existing clip and
the existing step, records aggregated counters. It does not add a forward, a
backward, or an optimizer step. Pinned `run_vyra1_copy` is still the library
the caller imports.

For every expert slice on every completed update the counters keep four facts
apart: which forward selected it, whether the optimizer dict contained it,
whether its post-clip gradient had a nonzero element, whether SliceAdam's
per-expert clock increased, and whether the int64 sum of that slice's float32
bit patterns changed. A clock increase is not proof the values changed. A
changed bit-sum means at least one stored value changed. An unchanged bit-sum
can hide two writes that cancel in the sum, so it is not proof of no write.
Selection, clocks, and nonzero gradients cover every expert slice, not a
sample. The bit-sum reads each expert slice before and after the step and does
not keep a copy of the slice. Those reductions stay on device until `summary()`
at the end of the arm. An earlier stop does not flush the gradient or bit-sum
cross-tab.

The CPU preflight exits before this loop, so a preflight receipt does not
exercise the counters. `tests/test_expert_union_replay.py` does.

## Expanded rehearsal preparation

Training replay uses a 40-row artifact, hash
`abd8d94d7e105deab2a19c461266c89ab308f7edf67b4394b638afdb7cb474d7`, 88749
bytes. That file is not in the repository. The builder embeds those bytes in
the notebook and the notebook checks the hash before writing them to
`<kaggle-root>/working/rehearsal_Y72_40.pt`. There is no `/submission` path and no
new dataset mount. The original 8 rows stay byte
identical inside it and 32 rows are added. Replay still takes 2 rows each
update, now `(2*s+j) % 40`, 350 presentations, 30 rows repeated 9 times and
10 rows repeated 8 times. X batches, seed 0, 175 updates, loss weight 0.25,
learning rate, clip, optimizer, and heldout eval are unchanged. Heldout rows
are still the original unseen split, not the 40 training rows.

On updates 1, 75, 125, and 175 the caller reads separate pre-clip gradients
with `autograd.grad` on the losses already produced, including one X64 row
from that same X forward. The combined norm is the return value of the
existing clip. These reads are not the expert-telemetry counters. GPU time
for the four reads and the extra rehearsal readouts is unmeasured. A preflight
receipt still stops before any forward.

## Runtime file contract

| File | How it reaches the notebook | Runtime path | Status |
| --- | --- | --- | --- |
| Notebook source, including treatment, gates, and the rehearsal bytes | The notebook Kaggle executes | decoded to `<kaggle-root>/working/rehearsal_Y72_40.pt` only after the pinned SHA matches | Chosen transport. The failed job wrote other `<kaggle-root>/working` assets before it died. |
| `rehearsal_Y72_40.pt` as a file beside the notebook | Not transported | `/submission/rehearsal_Y72_40.pt` | Observed `FileNotFoundError` on Kaggle. The CPU sandbox had invented that mount. It is no longer used. |
| X bundle | Existing dataset attachment named in the kernel metadata | `<kaggle-root>/input/datasets/<user>/<x-dataset-slug>` | Documented package path. This failed job did not reach it. |
| Y bundle | Existing dataset attachment named in the kernel metadata | `<kaggle-root>/input/datasets/<user>/<y-dataset-slug>` | Documented package path. This failed job did not reach it. |
| `/tools`, `/runtime` | CPU sandbox harness only | not read by the notebook | The harness mounts the notebook at `/tools/replay.ipynb` and does not create `/submission`. |

No new dataset is declared and nothing here uploads one. A rehearsal dataset mount is not claimed.

## Command

Run this outside the repository. The target refuses an output path inside the
worktree so private bytes are not written next to the source.

```bash
python3 tools/experiment_packaging/validate_experiment_package.py \
  --source-notebook prior_remote/<replay175-approved-kernel>.ipynb \
  --x-archive archives/X/<x-dataset-slug>.zip \
  --y-archive archives/Y/<y-dataset-slug>.zip \
  --builder-source sources/make_corpus_X.py \
  --archive-audit REMOTE_ARCHIVE_AUDIT.json \
  --replay-artifact /path/to/rehearsal_Y72_40.pt \
  --inputs inputs \
  --runtime /path/to/cpu-venv \
  --output "$HOME/replay-package-out"
```

`inputs` must already contain the extracted archive directories
`<x-dataset-slug>` and `<y-dataset-slug>`. `--runtime` must contain `bin/python` for
the CPU environment mounted read-only in the sandbox. `bwrap` must be on
`PATH`. The builder provenance file has to match the pinned SHA. Nothing in
this command uploads a notebook or starts a Kaggle job.

On a real pass the printed status is `ISOLATED_CPU_PREFLIGHT_PASS` with
`submission_ready` false. `python3 tools/experiment_packaging/launch_gate.py`
checks that receipt and still does not push. `launch_gate.py --push` refuses
before any network call.

## What the pass covers

- Both builds write byte-identical notebook, kernel metadata, and dependency lock.
- The sandbox runs the generated notebook with Python `-I`, network unshared,
  an empty environment, no GPU device, and no host workspace mount.
- The trainer's CPU preflight loads the real library and the real checkpoint,
  builds a fresh SliceAdam, checks the planned rows, and exits before any
  forward, backward, or optimizer step.
- Three mutations are rejected: a missing embedded file, a corrupt embedded
  hash, and an explicit wrong dataset path.
- A notebook byte changed after the receipt is rejected.

The CPU preflight does not exercise the expert-union update. That update is
covered by `tests/test_expert_union_replay.py` on a tiny CPU model. GPU kernels
and a full training run are outside this gate.

## Private Kaggle identifiers (not in the repository)

The Kaggle username, the dataset and kernel slugs and the Kaggle filesystem root are not stored in any tracked
file. Where this document writes `<user>`, `<x-dataset-slug>` and so on, the real values are in
`kaggle_config.json` at the repository root. That file is gitignored and local only; the committed template is
`kaggle_config.example.json`, and `VYRA_KAGGLE_CONFIG` can point at another location.

* Tracked source uses tokens (`@@KUSER@@`, `@@KROOT@@`, `@@SLUG_X_BUNDLE@@`, ...). `tools/experiment_packaging/kaggle_config.py`
  expands them. The builders expand every notebook asset, the notebook program and the dependency lock before hashing,
  so a build with the real config is byte-identical to a build made when the literals were in the source.
* Fail closed: a builder (or any module that expands a token) raises `KaggleConfigError` if the config is missing,
  incomplete, uses an unknown token or leaves a token unexpanded. The placeholder template is refused by the builders
  unless `VYRA_KAGGLE_ALLOW_EXAMPLE=1`, which `tests/conftest.py` sets only when no real config exists.
* `tests/test_no_kaggle_identifiers.py` fails if any tracked file contains the username, a slug, a Kaggle filesystem path or a
  Kaggle URL. Only `kaggle_config.example.json` is allowlisted.
* Commits made before this change still contain the old values; history was not rewritten.
