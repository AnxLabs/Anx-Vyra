# Pre-registration: larger Y72 held-out set and rescaled gate (`vyra-y72-heldout-v2-gate-v1`)

Registered by FORGE (Grok Bot), Nov 11 2026 (America/Edmonton, UTC-6), **before any checkpoint is scored on this set**. CPU only; no GPU, upload or Kaggle action.
Machine-readable twin: `prereg/y72_heldout_v2.json` (pinned by `tests/test_y72_heldout_v2.py`); gate code `tools/experiment_packaging/y72_heldout_v2.py`.

## The set
* 120 rows x 5 segments = **600 segments** (4,800 tokens), distance 72, three blocks of 40 rows.
* Generator: the existing Experiment-Y `make_corpus_Y.py` (git c38958c, file sha256 `3a78a791...c2f`) - its `gen_rows` etc. extracted verbatim, not re-implemented. Check: the same
  extraction reproduces the shipped 40 unseen D72 rows exactly (rows, masks, payloads) from seed 20261672.
* Seed policy: `seed = base + distance`. Existing bases 20261500 (train) / 20261600 (unseen) / 20261700 (gap), rehearsal 20262672. **New seeds 20263672, 20263772, 20263872** (bases 20263600/700/800).
* Leakage: original ban list extended with all train_Y payloads and the 40 rehearsal rows' payloads.
* Identity: canonical sha256 (rows+mask int16 LE) `16ff53a839c5c68e2230fe4f96e794f11d7ff7f885cac3e184e4f4770f3ee71b`; file `prereg/y72_heldout_v2/y72_heldout_v2.pt` sha256 `12e034022357e230685a956e47658315797f164b5b11c30bd35ba4f6ca04a826`.
* Overlap check (independent script, `overlap_check.json`): seed level - 0 of 3 new seeds in the 87 used/implied seeds; content level - 0 exact-row, 0 payload, 0 filler-string, 0 sixteen-byte-window matches against 9,627 existing rows
  (corpus, T2/V/W/X/Y, probes, rehearsal, the shipped 200-segment held-out set) and 0 payload occurrences in the corpus byte blobs; no duplicates inside the set. A positive control (one shipped row injected) makes the checker fail.

## Noise math (from the 200-segment study, scaled by k = 600/200 = 3)
Assumption: noise is item-level (observed ICC by row -0.015), so SD in segments scales with sqrt(3), SD in pp with 1/sqrt(3).
| quantity | 200-set | 600-set (predicted) |
|---|---|---|
| single-checkpoint SD, binomial p=0.8 | 5.66 seg (2.83 pp) | 9.80 seg (1.63 pp) |
| single-checkpoint SD, churn / within-item models | 5.13-5.41 seg (2.6-2.7 pp) | 8.9-9.4 seg (1.5-1.6 pp) |
| mean-of-3 SD (independent .. cluster bootstrap) | 3.05-3.95 seg | 5.3-6.8 seg (0.9-1.1 pp) |
| MDE vs floor, single checkpoint (one-sided alpha 0.05, power 0.8) | 7.0 pp | 3.7-4.1 pp (22-24 seg) |
| MDE vs floor, mean of 3 | 3.8-4.9 pp | 2.2-2.8 pp (13-17 seg) |
Full tables and operating characteristics: `prereg/y72_heldout_v2/noise_rescale.json`. The prediction is tested, not assumed, by the (non-gating) scoring below.

## Gate (integer, `>=`)
**R**: sum over checkpoints 250/275/300 of Y72 segment-exact on this set **>= 1440** (mean 480/600 = 80.0%) **and** every checkpoint **>= 432/600 (72.0%)**.
Boundaries 1440/1439 and 432/431. Percentages are unchanged from `pertask_clip_v1` (480 -> 1440, 144 -> 432). Replaces only the Y72 R gate for future experiments; the 200-set is comparison-only and is never pooled.
Normal-model pass probabilities (SD of the 3-checkpoint mean 5.3-6.8 seg): true 84% passes ~99.98-100%, 82% ~96%, 80% ~51%, 78% ~4% (cluster-bootstrap SD).

## Scoring of existing checkpoints (non-gating, may not change this registration)
lambda 0.25: 200/225/250/275/300; lambda 0.5: 250/275/300; checkpoint sha256 verified before `torch.load`; statistics listed in the JSON (`scoring_plan_existing_checkpoints_nongating`).
