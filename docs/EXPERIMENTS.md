# Experiment log

A chronological record of what was actually run, including the experiments that failed and the
bugs that invalidated some of them. Negative results are kept because they are the reason the
findings section says what it does.

All experiments share one architecture (arm R, unchanged throughout) and one code base. Every run
continued from a fixed pretrained checkpoint with a **fresh optimizer**, so no run inherits
momentum from the one before it.

## Terminology

- **record-exact** — all 8 payload bytes of a record are predicted correctly. The primary metric.
- **unseen** — 40 rows never trained on, with payloads disjoint from every training set.
- **distance** — token gap from the last byte of a record to the first byte of the query marker.
- **passes** — how many times each training row is presented. Fixed per experiment so runs are
  compared at matched exposure rather than matched wall time.

## E–R: exploratory phase

Established the architecture, the corpus format, the parameter-count checks (16,427,072 total /
8,169,536 active) and the SliceAdam implementation. Early work was on a Windows CPU host and
focused on getting a bit-reproducible baseline; these runs are not reported numerically here.

## S — curriculum continuation (FAIL)

Gradual ramp of filler length from 32 to 64 tokens, from the pretrained checkpoint, 200 steps.
64-token record-exact stayed **0.000 at all nine checkpoints** on both trained and disjoint rows.
Cross-entropy fell from 11.68 to 5.41 and flattened at ln(256) = 5.545 — the model was emitting an
undecided distribution rather than learning retrieval. Token-level accuracy also never rose above
chance on the answer bytes.

**Conclusion:** a curriculum alone does not teach this task at the rate used.

## T — fixed 64-token continuation (STOPPED EARLY)

40 varied 64-filler rows, 4.75 planned presentations each, plus 40 disjoint unseen rows, learning
rate 1e-4. Stopped at step 1525 of 1525–1700 after only 25 steps: the train F32 probe collapsed
0.5064 → 0.1667 (floor 0.3564) and needle F32 record-exact fell 0.9896 → 0.4146 after 15
consecutive 64-filler steps.

**Design error identified in T:** the 32-token rehearsal load was cut from two rows per step to one
at the same moment 64-token rows arrived. Two variables changed at once, so the 32-token collapse
could not be attributed.

## T2 — the T confound removed (FAIL on 64-token)

Two arms, both holding 32-token rehearsal constant at one row per step for the entire run:

- Arm A: slot 3 carries 64-filler rows (the treatment).
- Arm B: slot 3 carries 32-filler rows (the control, no 64-token content).

200 steps, learning rate 1e-4, answer-only cross-entropy, from preserved step-1500 weights.

Result: Arm A reached **0.0000 record-exact at every checkpoint** on both trained and unseen rows.
The GPU/CPU equivalence gate passed (argmax agreement 1.000).

**Conclusion:** T's 32-token collapse was rehearsal dilution, not 32-token overwriting — with
rehearsal held constant, 32-token performance stayed ≥0.98. But 64-token retrieval still did not
learn at 1e-4.

## U — is the 128-token window the blocker? (NO)

Same 1e-4 rate, two arms: windowed (128 on layers 0,2,4,6) versus full context. Both reached
essentially nothing (train 0.015 and 0.000, unseen 0.0000). No retention floor was breached.

Then the same arms were accidentally run at 3e-4 — the library default, because the trainer never
set the module-level `LR` — and **both learned** (windowed 0.97 train / 0.84 unseen; full context
0.86 / 0.67).

**Conclusion:** the attention window is not the blocker; the learning rate was. This also exposed
the `LR` global bug described in the README.

## V — learning-rate replication (PASS)

The 3e-4 result reproduced on the fixed 128-token-window architecture across two independent
corpora:

| corpus | LR | trained | unseen |
|---|---|---|---|
| d1 | 3e-4 | 0.970 | 0.840 |
| d1 | 1e-4 | 0.015 | 0.000 |
| d2 | 3e-4 | 0.975 | 0.890 |
| d2 | 1e-4 | 0.020 | 0.000 |

Retention held (validation fell only 0.0055 / 0.0025; 32-token rose slightly). Learning rate
application was verified by the optimizer's first update magnitude: `3.000498e-04` for the 3e-4 arms
and `1.000166e-04` for the 1e-4 arms.

**Caveat:** the torch seed has no effect in this trainer (no dropout, fixed batch plan), so
replication was at the data level, not the seed level.

**Bug found here:** checkpoints recorded the module-level `LR` constant rather than the arm's actual
rate, so a 3e-4 arm was stamped "1e-4". Training was correct; only the metadata was wrong. Fixed by
reading the per-arm rate.

## W — distance sweep (the main result)

Six arms, each training one needle-to-query distance, from V's step-200 weights at 3e-4, 200 steps,
5 passes per row:

| distance | records/row | trained | unseen | learned |
|---|---|---|---|---|
| 104 | 4 | 0.994 | 0.950 | yes |
| 108 | 4 | 0.981 | 0.931 | yes |
| 109 | 3 | 0.992 | 0.992 | yes |
| 114 | 3 | 0.983 | 0.767 | yes |
| 120 | 3 | 0.283 | 0.000 | no |
| 136 | 3 | 0.225 | 0.000 | no |

The limit sits between 114 and 120 — below the 128-token window. Record count is ruled out
(distance 109 with 3 records learned; distance 120 with 3 records did not).

**Retention:** 32-token and general validation held on every arm. The 64-token ability was
destroyed on every arm except the closest one (distance 104, where it degraded to 0.67).

**Bugs found here:** a probe was initially bound to the wrong row set, which made a skill the model
never learned look like catastrophic forgetting; and a missing reference key produced a fake gate
failure instead of a config error. Both are now hard failures with explicit messages.

**Reproducibility:** distance 136 was run three separate times and reproduced identically
(0.225 / 0.000) each time.

## X — multi-distance mix (what W's interference actually was)

W trained exactly one distance per arm. X asked whether the interference W measured was a property of
the architecture or of that training configuration, so it trained three distances at once.

One arm, from V's step-200 weights at 3e-4, 200 steps, batch of 5 rows: 2 general replay rows plus 1
row per distance. Each distance therefore received **5 passes** — the same exposure W gave each of its
arms — so the comparison isolates *mixing* rather than extra training. Each distance is scored on its
own 40 disjoint unseen rows, every checkpoint.

| distance | baseline | trained | unseen | W, same distance, trained alone |
|---|---|---|---|---|
| 32 | 0.000 | 0.894 | 0.813 | not trained in W |
| 64 | 0.000 | 0.975 | 0.929 | 0.000 on every distance-learning arm |
| 109 | 0.000 | 0.975 | 0.967 | 0.992 / 0.992 (W arm d101) |

All three started at 0.000 record-exact on both trained and unseen rows.

**Result: W's interference did not reproduce once the distances were trained together.** W drove
64-token retrieval to 0.000 on five of six arms and to 0.67 on the sixth; X holds it at 0.929 while
learning 109 to 0.967. The 32-token retention probe held (0.9917 → 0.9479) and general validation was
flat (0.6995 → 0.6992 token accuracy).

Two things X did **not** show, and should not be read as showing. It is one run per configuration, so
the difference is between two training setups, not a seed-level or architectural result. And it did
not raise W's distance ceiling: at distance 109 the mixed run is slightly *worse* than W's
single-distance arm (0.967 against 0.992 unseen), and X never attempted 120 or 136, the distances W
failed. Mixing buys retention and costs a little accuracy on the distance being trained.

Learning-rate application was verified by first-update magnitude, `3.000498e-04`, and the GPU/CPU
metric gate passed with an absolute delta of 0.0 on all three gate metrics. Cost: 192.2 s of training,
961 ms per step, 1.968 GiB peak GPU allocation.

### The regression, and what the distance profile showed

Stopping rule 2 fired at steps 175 and 200: the `train_F32` rehearsal probe fell 0.6795 → 0.4167
record-exact, under its 0.5295 floor. Rather than guess at the cause, both checkpoints were profiled
across the whole inherited probe set — 13 distances, CPU, deterministic:

| probe | distance | step 0 | step 200 | delta |
|---|---|---|---|---|
| train_F32 | 40 | 0.9183 | 0.6843 | **-0.2340** |
| train_F48 | 56 | 0.0033 | 0.0033 | 0.0000 |
| train_F64 | 72 | 0.2674 | 0.0058 | **-0.2616** |
| train_F80 | 88 | 0.0036 | 0.0036 | 0.0000 |
| train_F96 | 104 | 0.0018 | 0.0018 | 0.0000 |
| train_F128 | 136 | 0.0052 | 0.0078 | +0.0026 |
| train_F160–F256, unseen_F288–F416 | 168–424 | ≤0.0063 | ≤0.0063 | ~0 |

(Token accuracy, not record-exact; probe payloads are disjoint from all training payloads. A probe
named F*n* has filler *n*, and distance = filler + 8.)

Exactly two points moved, and **both sit precisely +8 tokens past a distance this run trained**:
40 = 32 + 8 and 72 = 64 + 8. Every distance further out was already at floor, so it had no headroom
left to lose. No probe sits on 32, 64 or 109 themselves, so this profile does not measure the trained
offsets directly — it measures their neighbours.

This is not rehearsal dilution: the general-replay rows and the 32-token retention rows are scored
separately and both held. Dilution would have moved those too. The damage is confined to the untrained
neighbours.

**Working explanation, not a proven mechanism:** the model forms distance-sharp representations at the
exact offsets it is trained on rather than a smooth interpolation rule, so it does not automatically
generalise to a distance it was never shown even when that distance is 8 tokens away. This is an
inference from two points and one run.

**Not yet run:** the proposed targeted fix is to jitter each training row's filler length across a
small band around its target, so the model cannot key on one exact offset — a change to the data
rather than more steps. It is a new experiment and is listed as the next step, not as a result.

## Open questions

1. ~~Does rehearsing prior distances inside the training mix prevent the interference, or only slow
   it?~~ **Partly answered by X:** a balanced mix learns several distances and keeps the earlier one,
   but it costs accuracy on the distance being trained and leaves untrained neighbours degraded.
   Whether a *jittered* mix generalises across a continuous band is the next experiment.
2. Is the 114–120 boundary a property of the architecture or of this task's encoding?
3. Does the learning-rate sensitivity persist at larger widths, or is it a small-model artifact?
4. Would per-distance heads or a distance-conditioned curriculum avoid overwriting entirely?
