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

## Command

Run this outside the repository. The target refuses an output path inside the
worktree so private bytes are not written next to the source.

```bash
python3 tools/experiment_packaging/validate_experiment_package.py \
  --source-notebook prior_remote/vyra-replay175-approved.ipynb \
  --x-archive archives/X/vyra-x-bundle.zip \
  --y-archive archives/Y/vyra-y-bundle.zip \
  --builder-source sources/make_corpus_X.py \
  --archive-audit REMOTE_ARCHIVE_AUDIT.json \
  --inputs inputs \
  --runtime /path/to/cpu-venv \
  --output "$HOME/replay-package-out"
```

`inputs` must already contain the extracted archive directories
`vyra-x-bundle` and `vyra-y-bundle`. `--runtime` must contain `bin/python` for
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
