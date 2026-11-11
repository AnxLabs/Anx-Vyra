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

## Y — jittering the distance (FAIL)

### Experiment Y — jittering the distance, and what it did not do

**What was asked.** Experiment X ended with two probes degraded, both sitting exactly +8 tokens past a
distance it had trained. The natural reading was that the model snaps to whatever offset it sees
often, and the natural fix was to stop it snapping: train on jittered filler lengths so each target
becomes a band rather than a point. The test was made falsifiable in advance. If jitter smooths the
representation, distances the model never trains should lift as well. Recovery only at the band edges
would mean the sharp points moved rather than spread.

**What was done.** One variable changed against X. Same arm R, same 3e-4, same 200 steps, same batch
of five rows, same five passes per geometry, fresh optimizer, starting from X's step-200 weights so
the damage was present at step zero. The change: each needle slot draws its geometry per step from a
band of `target ± 8` in steps of 4. Three anchors become fifteen trained geometries — 24 through 40
around 32, 56 through 72 around 64, 101 through 117 around 109 — and eleven further distances
(44, 48, 52, 76, 80, 84, 88, 92, 96, 121, 128) are never trained and serve as the interpolation test.
The jitter is measured, not assumed: all fifteen geometries received exactly 40 presentations.

**What actually happened.** The interpolation test failed. No never-trained distance produced a single
exact record, and their mean token accuracy fell from 0.067 to 0.023 — the loss concentrated almost
entirely in one distance (84, 0.702 → 0.166) while the rest sat at floor throughout. Nothing
interpolated.

The anchors got worse too. On the same 40 disjoint unseen rows per distance, the three distances this
experiment was supposed to repair ended at 0.425 / 0.229 / 0.133 record-exact, down from
0.828 / 0.904 / 0.908 at step 0 and well below X's 0.813 / 0.929 / 0.967 on the same rows. Token
accuracy fell with them, 0.972 / 0.986 / 0.989 → 0.883 / 0.790 / 0.748, so the loss is not an artefact
of the stricter metric.

Eight of the fifteen trained geometries (24, 28, 36, 56, 60, 68, 105, 113) finished at exactly 0.000
*training* record-exact despite 40 presentations each. That is not zero learning — their training token
accuracy runs 0.09 to 0.56 — but none of them ever completed a single full training record.

**The two band edges, read carefully.** The machine verdict recorded edge 40 as recovered. It was
never damaged: 0.922 unseen at step 0, 0.981 at step 200, and the apparent repair comes from
comparing a record-exact column against X's token-accuracy probe. Edge 72 is the one genuine gain in
the run: unseen token accuracy 0.006 → 0.785, record-exact 0.000 → 0.135. That is X's other +8 offset,
the one X had collapsed to zero, and jitter did teach retrieval there.

So the fix bought the two band edges while costing the three anchors. That is the signature of
capacity spent on specific offsets, not of a representation smoothed across a band.

**What it means.** Jitter did not smooth the representation; it redistributed the sharpness. Within a
single 200-step run, exposure to a band of neighbouring distances bought no generalisation across that
band. This is a stronger result than one fix failing: it says X's distance-sharp behaviour is not a
training-mix artefact that another variation of the data mix can regularise away. The next mechanism
worth testing is a different one — distance-relative position features, rather than more variation in
what the data looks like.

**Limitations, including one that is ours.** One seed, one arm, one band width, one jitter
granularity. And a real confound: X gave each distance 40 distinct rows, while Y spread 15 geometries
over 8 rows each, so each geometry's 40 presentations come from 8 rows. Part of the unseen degradation
is plausibly reduced per-geometry row diversity rather than jitter as such, which weakens any claim
that jitter specifically caused those losses. Gap probes use 20 rows apiece, so differences below
about 0.05 record-exact are not resolvable.

The full curated record, including per-geometry and per-gap numbers, is in
[`results/experiment_y_jitter_fix.json`](../results/experiment_y_jitter_fix.json), and the profile
chart is [`distance_profile_y.svg`](distance_profile_y.svg).


## R300 — resuming from update 200 (INCONCLUSIVE; Y72 retention failed)

**What was asked.** At update 200 the 64-token needle task (X64) sat at zero record-exact on its
held-out rows. Was it stuck, or only slow? A stop rule was written down first
(`prereg/stop_rule_300.json`): STUCK if both train and held-out X64 record-exact are still zero at
update 300 and the gains are small; otherwise INCONCLUSIVE; and a retention failure (Y72 below 80%
record-exact at any checkpoint) is reported separately and blocks any success claim. Training always
stops at 300.

**What was done.** One run, one seed. The update-200 checkpoint was restored and checked against the
pinned update-200 numbers with zero tolerance (model hash and every integer count equal). The recipe
was unchanged: replay weight 0.25 on the Y72 rehearsal rows, one global gradient clip at 1.0, learning
rate 3e-4, six X rows and two replay rows per update, the data plan extended deterministically to 300.
Checkpoints and full evaluations at 225, 250, 275 and 300.

**What actually happened.** X64 was slow, not stuck.

| update | X64 held-out exact /240 | X64 held-out tokens /1920 | X64 train exact /240 | Y72 exact /200 | X32 /320 | X109 /120 |
|---|---|---|---|---|---|---|
| 200 (restored) | 0 | 257 | 0 | 165 | 173 | 80 |
| 225 | 11 | 1418 | 28 | 168 | 160 | 102 |
| 250 | 29 | 1542 | 80 | **137** | 206 | 99 |
| 275 | 46 | 1603 | 126 | **137** | 235 | 106 |
| 300 | 92 | 1687 | 182 | **153** | 264 | 109 |

Held-out X64 rows are disjoint from every training row, so the rise is not memorisation of those rows.
Y72 is the retention probe: 200 held-out segments of the 72-token needle task. It sat at 165 to 168
until update 225 and then fell below the 160 floor at 250, 275 and 300.

**What it means.** The rule's verdict is **INCONCLUSIVE**: not stuck (held-out token gain 200 → 300 of 74.5
points and train token gain 250 → 300 of 8.0 points, against STUCK thresholds of 10.0 and 3.0), but
retention failed from update 250 on, so no success is claimed. The more interesting finding is the
trade: the same recipe that teaches X64 costs Y72, and Y72 had recovered only partly by 300.

**Limitations.** One seed, one run, no retries. The older joint gates of the notebook's 200-update
protocol also failed and are separate from the stop rule. Python and NumPy RNG state are not restored
across the resume (disclosed; both were observed unused). Record:
[`results/experiment_r300_continuation.json`](../results/experiment_r300_continuation.json).

## Y72-fix-λ0.5 — doubling the replay weight (FAIL_RETENTION)

**What was asked.** If Y72 falls because replay is too weak, raising the replay weight should hold it.
One variable changed: 0.25 → 0.5, everything else identical, resumed from the update-225 checkpoint
of R300 (restored with the same zero-tolerance check) and stopped at 300. Gates were registered first
(`prereg/y72fix_lambda05.json`): R, Y72 ≥ 160/200 at 250, 275 and 300; A, X64 held-out ≥ 11/240 at every
checkpoint and a token gain 225 → 300 of at least 10.0 points; N, X32 ≥ 128/320 and X109 ≥ 90/120.
The control is the 0.25 run from the same checkpoint and plan.

**What actually happened.**

| update | Y72 /200 (0.5 / 0.25) | X64 held-out exact /240 (0.5 / 0.25) | X64 held-out tokens /1920 | X32 /320 | X109 /120 |
|---|---|---|---|---|---|
| 250 | **152** / 137 | 17 / 29 | 1532 / 1542 | 211 / 206 | 100 / 99 |
| 275 | **147** / 137 | 29 / 46 | 1571 / 1603 | 239 / 235 | 99 / 106 |
| 300 | **152** / 153 | 42 / 92 | 1618 / 1687 | 273 / 264 | 114 / 109 |

R failed at all three checkpoints (short of 160 by 8, 13 and 8). A passed (the token gain was 200
tokens, 10.4 points, against a 10.0 requirement). N passed. The verdict is **FAIL_RETENTION**.
Weight 0.5 narrowed the drop at 250 and 275, was level with the control at 300 (152 vs 153), and cost
held-out X64 acquisition: 42 vs 92 exact at 300.

**What it means.** A doubled replay weight is not enough, and it is not free. It also says something
narrower: at update 300 more weight on the rehearsal rows did not translate into more held-out Y72
(152 vs 153), so weight alone is not the missing ingredient.

**Limitations.** One seed. A metric that flips 49 to 61 of 200 items per 25 updates (see the noise
study below) can move by more than the 8 to 13 item shortfall by chance, so the result shows that 0.5
missed the bar, not that the weight has no effect. Record:
[`results/experiment_y72fix_lambda05.json`](../results/experiment_y72fix_lambda05.json).

## Y72-noise — how noisy is the Y72 gate, and is the shared clip to blame? (measurement)

**What was asked.** Before another run: how large is the checkpoint-to-checkpoint noise in the Y72
metric, can the old rule ("every checkpoint ≥ 160 of 200") tell a fix from chance, and does the single
global clip, which scales the X and replay gradients by one number, plausibly squeeze replay out?

**What was done.** CPU only, no training, from the eight existing checkpoints (λ0.25 at 200, 225,
250, 275, 300; λ0.5 at 250, 275, 300). The CPU evaluation reproduced every GPU-emitted Y72 count.
Noise was modelled four ways (binomial, row-clustered via ICC, adjacent-checkpoint churn, within-item
Bernoulli) and bootstrapped over the 40 rows. Gradients were split per task from separate per-row
backward passes on 25 pinned-plan batches per state, and one real optimizer step was taken per clip
scheme to compare effects.

**What actually happened.**

- *Churn.* Between adjacent checkpoints 49, 61, 60 and 52 of the 200 segments change correctness for
  λ0.25 (net +3, −31, 0, +16) and 51 and 53 for λ0.5; the mean is 54.3 (27%). One real optimizer step
  flips 13.3 ± 4.0 segments.
- *Noise floor.* Single-checkpoint SD is 5.2 to 5.7 segments (2.6 to 2.8 points); the row ICC is
  −0.015, so there is no detectable row clustering. The minimum effect one checkpoint can resolve
  (one-sided α 0.05, power 0.8) is about 14 segments (7.0 points); the mean of three checkpoints has a
  cluster-bootstrap SE of 3.7 (λ0.25) and 3.9 (λ0.5) segments.
- *The movement is more than noise.* The RMS of the six adjacent net changes is 14.6 against 7.4
  expected from churn alone. Against the 225 level, the λ0.25 mean of three is down 24.2 segments and
  the λ0.5 mean of three down 16.2; λ0.5 beats λ0.25 by 8.0 (z 2.06).
- *Gradients.* At the λ0.25 checkpoints cos(X gradient, replay gradient) averages −0.106 at 225,
  −0.061 at 250, −0.088 at 275 and +0.031 at 300; the gradients conflict on 76%, 72%, 72% and 36% of
  batches; the global clip binds on 96 to 100%. Replay takes 11 to 29% of the update norm under the
  global clip against 28 to 46% under a per-task clip of 0.5.
- *One-step probes (12 real updates).* Per-task clipping gains Y72 +1.42 segments (SE 0.54) and loses
  roughly 40% of the X loss decrease; PCGrad-style projection gains +0.5 (SE 0.29). Both are far
  below what one run can resolve.

**What it means.** The old gate could not separate a fix from chance at the effect sizes at stake, and
the 200-segment set cannot be made better from saved data (no further Y72 held-out items exist). The
clip evidence points at a plausible mechanism but not at a cure.

**Limitations.** The first-order gradient model is a direction diagnostic, not a forecast of a 75-update
trajectory; nothing here shows that any clip scheme changes one. Record:
[`results/experiment_y72-noise_noise_floor_clip_split.json`](../results/experiment_y72-noise_noise_floor_clip_split.json).

## Y72-heldout-v2 — a larger, pre-registered held-out set (both runs fail it)

**What was asked.** Lower the noise floor with more items, without looking at the new items first.

**What was done.** 600 new Y72 segments (120 rows × 5, three blocks of 40 rows) from the existing
experiment-Y generator with its seed policy (seed = base + distance, new bases 20263600, 20263700,
20263800) and a leakage ban list extended with every training payload. The set's hashes, an overlap
check (no new seed among 87 used seeds; 0 exact-row, payload, filler or 16-byte-window matches against
9,627 old rows; 600 distinct payloads; a positive control that fails on purpose), a rescaled gate
(sum over 250, 275, 300 ≥ 1440 of 1800, each ≥ 432 of 600) and noise predictions were committed before
any checkpoint was scored (`prereg/y72_heldout_v2.json`). The existing checkpoints were then scored on
CPU with their file hashes verified before loading, and the 200-set was rescored in the same pass
(identical to the earlier counts for all eight).

**What actually happened.**

| checkpoint | 600-set exact | % | old 200-set |
|---|---|---|---|
| λ0.25 200 / 225 | 498 / 490 | 83.0 / 81.7 | 165 / 168 |
| λ0.25 250 / 275 / 300 | 363 / 390 / 454 | 60.5 / 65.0 / 75.7 | 137 / 137 / 153 |
| λ0.5 250 / 275 / 300 | 450 / 421 / 420 | 75.0 / 70.2 / 70.0 | 152 / 147 / 152 |

Applying the registered gate to the two finished runs: **λ0.25 sums to 1207 and λ0.5 to 1291 against
1440, both fail**, and the 432 guard fails at two of three checkpoints in each run. This is an
illustration, not a gated experiment. λ0.5 minus λ0.25 on the mean of the three checkpoints is +28
segments (+4.67 points), cluster SE 9.3, z 3.01. The new set is about 4 points harder than the old one
at the later checkpoints (mean of three: 67.1% / 71.7% against 71.2% / 75.2%); at 200 and 225 the two
sets agree within 2.3 points.

*Noise.* Single-checkpoint SD came out at 9.34 (churn model) and 9.73 (within-item) segments, inside
the registered 8.9 to 9.8, so noise does scale with √n for one checkpoint (1.6 points); row ICC is
+0.013. The mean of three did not behave: cluster-bootstrap SE 8.96 (λ0.25) and 8.26 (λ0.5)
segments, against a registered 5.3 to 6.8, because the three checkpoints share a component. The
registered prediction was wrong here. Observed minimum detectable effect: 24.2 segments (4.0 points)
for one checkpoint, 20.5 to 22.3 (3.4 to 3.7 points) for the mean of three, against 7.0 and 3.8 to 4.9
points on the old set.

**What it means.** The larger set gives a firmer reading of an effect that was borderline, and a gate
that both earlier runs clearly miss. It does not give the precision the √n prediction promised for a
three-checkpoint gate.

**Limitations.** The set is still one distance, one generator, and the scoring was of existing
checkpoints: no new experiment was run. The set itself is committed in text form
(`prereg/y72_heldout_v2/y72_heldout_v2_set.json`), verified against the registered hashes. Record:
[`results/experiment_y72-heldout-v2_heldout_600.json`](../results/experiment_y72-heldout-v2_heldout_600.json).

## Open questions

1. ~~Does rehearsing prior distances inside the training mix prevent the interference, or only slow
   it?~~ **Partly answered by X:** a balanced mix learns several distances and keeps the earlier one,
   but it costs accuracy on the distance being trained and leaves untrained neighbours degraded.
   **Answered by Y: no.** Jittering filler by +-8 around each target did not generalise across the band; it moved capability to the band edges at the cost of the anchors.
2. Is the 114–120 boundary a property of the architecture or of this task's encoding?
3. Does the learning-rate sensitivity persist at larger widths, or is it a small-model artifact?
4. Would per-distance heads or a distance-conditioned curriculum avoid overwriting entirely?
5. Can Y72 be retained while X64 is still learned? X and Y gradients conflict on the same parameters (mean
   cosine −0.49; the Fisher diagonals of the two tasks are nearly the same coordinates), so there is no separate
   set of weights to protect. CPU probes (not a run) favour a per-task gradient clip over a heavier replay
   weight, an L2 anchor, frozen layers or gradient projection, at the price of much slower X64 learning. The
   change and its gates are registered in [`prereg/y72fix2.md`](../prereg/y72fix2.md) and have **not** been run.
