# Publishing an experiment

ANX-VYRA publishes every completed experiment, **including the ones that failed**, under a single
fail-closed workflow. This page documents that workflow and, just as importantly, what it does not do.

## The rule

An experiment is published when, and only when, all of the following hold:

1. its own `SHA256SUMS` manifest verifies;
2. every file about to be committed passes a secret and privacy scan;
3. every file is valid UTF-8 text and not a binary or weight file;
4. the test suite passes;
5. the curated record round-trips as valid JSON and contains no forbidden strings.

If any check fails, **nothing is committed and nothing is pushed**, and the results stay private. The
script exits non-zero and names the failing check.

## One command

```bash
python tools/publish_experiment.py <results-dir> --experiment <TAG> --slug <slug>
```

For example:

```bash
python tools/publish_experiment.py ../expX/results --experiment X --slug multi_distance
```

Useful flags:

| Flag | Effect |
|---|---|
| `--dry-run` | Run every check and write the files, but do not commit or push. |
| `--no-push` | Commit locally, do not push. |
| `--prose-mode insert\|skip` | `insert` adds the generated stub; `skip` leaves prose files alone. |
| `--allow-missing-profile` | Publish without a distance-profile chart. |

The script is safe to re-run. If the curated record, chart and prose are already in place it reports
that there is nothing to do and exits 0.

## What the experiment directory must contain

```
<results-dir>/
  SHA256SUMS            # required; "<sha256>  ./relative/path", one per line
  x_results.json        # required; or the single *_results.json present
  probe_profile.json    # optional; enables the chart
```

`SHA256SUMS` is the anchor. It is verified before anything else happens, and a checksum mismatch
aborts the run. Note that manifests are written relative to their own root and different experiments
use different roots - verify from the directory the manifest was generated in.

## Steps, in order

1. **Verify the manifest.** Every listed file must exist and hash exactly.
2. **Curate the record.** The raw run record is *not* published as-is. It contains Kaggle dataset and
   kernel identifiers, absolute paths, a GPU device string, and internal approval fields. A curation
   function emits an allowlisted record: the question, design, setup, geometry, learning-rate
   verification, metric gate, the per-checkpoint table, final numbers, rule firings and cost.
3. **Render the chart**, if a profile JSON is present. The output is a plain-text SVG, because this
   repository contains no binary files. Degraded points between the two series are circled
   automatically.
4. **Update `docs/EXPERIMENTS.md` and `README.md`.** See the warning below.
5. **Scan.** Every staged file is checked against the pattern groups listed in the script:
   credentials (Kaggle, GitHub, AWS, OpenAI/Anthropic-style, Hugging Face, private-key blocks, bearer
   tokens, passwords, `.env`-style assignments) and private data (email addresses, personal handles,
   absolute host paths, Windows hostnames, Kaggle slugs, internal approval strings, checkpoint
   filenames). Any hit is fatal.
6. **Refuse binaries.** By extension and by content: NUL bytes, invalid UTF-8, or a weight/archive
   extension all abort.
7. **Run the tests.** The full suite, in the repository root.
8. **Commit and push** to the current branch, then print the file list, scan result, test result,
   commit SHA and commit URL.

## The prose step is deliberately not clever

**This is the honest limitation of the tool.** The script appends a *generated* section to
`docs/EXPERIMENTS.md` and inserts one into `README.md`, wrapped in
`<!-- PUBLISH-GENERATED:BEGIN ... -->` / `END` markers, followed by a `TODO(publish)` comment.

The generated block is a faithful dump of the curated record - design text, setup key/values, final
numbers, cost. It is **not** a narrative, and it is not a substitute for one. Writing the experiment
up means:

- replacing the generated block with prose in the same voice as the existing entries: what was asked,
  what actually happened, what failed, and what it means;
- stating failures and regressions plainly rather than burying them;
- scoping every claim to a single-seed run;
- deleting the markers and the TODO.

The tool writes the parts a machine can get right and flags the parts it cannot judge. It will not
silently overwrite hand-written prose: if a section for the experiment already exists, or the README
already links the curated record, that file is left untouched.

Because of this, treat the commit the script produces as a **draft publication**, not a finished one.
Review the prose diff before considering the experiment documented.

## What never gets published

- Model weights or checkpoints, in any format.
- Private datasets, corpora, or their identifiers and slugs.
- Credentials, tokens, keys, or anything that looks like one.
- Personal paths, hostnames, usernames, email addresses.
- Internal approval or authorization strings.

If it is unclear whether something is safe, leave it out and say so. The scan is a backstop, not a
guarantee: it looks for known shapes of secret, and a number or phrase that is private without
matching a pattern will pass. Curation is a per-experiment allowlist for that reason - an experiment
without one falls back to a generic key-based scrubber, and the record it produces says so in a
`_curation_warning` field.

## Known false positives

Two files in this repository trip the scanner if you scan the *whole* repo rather than only the files
being published:

- `tools/publish_experiment.py` — its pattern table necessarily contains the literal strings it
  searches for, so it matches itself.
- `LICENSE` — the Apache-2.0 text contains an approval-flavoured word in its legal wording.

Neither is ever a publish candidate: the publisher stages only the curated JSON, the chart, and the two
prose files. Do not "fix" these by weakening the patterns — the detection is what protects the
repository. If you add a whole-repo audit, allowlist those two paths explicitly and say so.

## Relationship to training

Publishing and training are separate tracks. The publisher never touches a live training directory:
it reads the results directory, and if that directory does not exist or its manifest does not verify,
it stops. It is safe to run while a training job is in progress elsewhere, and safe to re-run after
one finishes.

## Adding a new experiment type

1. Write a curation function and register it in `CURATORS`, keyed by experiment tag. Prefer an
   explicit allowlist over dropping keys.
2. Check the curated output against the original record, field by field, for anything identifying.
3. Run with `--dry-run` and read every file it produced before letting it commit.
