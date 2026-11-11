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

## Y — (generated draft)

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


## Y72-noise — (generated draft)

<!-- PUBLISH-GENERATED:BEGIN Y72-noise -->
**Question.** How large is the checkpoint-to-checkpoint noise in the Y72 held-out retention metric (200 segments), is the old per-checkpoint gate itself noisy, and does a shared global clip between the X and replay gradients plausibly explain the Y72 drop?

**Design.** CPU-only re-evaluation and gradient measurements from eight existing checkpoints (replay weight 0.25: updates 200/225/250/275/300; replay weight 0.5: updates 250/275/300). No training run, no GPU. Segment-exact = all 8 answer tokens correct. Noise models: binomial, row-cluster (ICC), adjacent-checkpoint churn, within-item Bernoulli, cluster bootstrap over 40 rows. Gradient split from separate per-row backward passes on 25 pinned-plan batches per state; one real optimizer step probed per scheme.

**Setup.**

- `heldout_segments`: 200
- `rows`: 40
- `segments_per_row`: 5
- `tokens_per_segment`: 8
- `cpu_eval_reproduces_gpu_counts`: True
- `bootstrap`: cluster over rows
- `alpha_one_sided`: 0.05
- `power`: 0.8
- `z_sum`: 2.4865
- `clip_split_batches_per_state`: 25
- `real_step_probe_steps`: 12

**Final numbers.**

- `segment_exact_by_checkpoint`: `{"l025_200": 165, "l025_225": 168, "l025_250": 137, "l025_275": 137, "l025_300": 153, "l05_250": 152, "l05_275": 147, "l05_300": 152}`
- `adjacent_churn`: `{"l025_200->l025_225": {"changed_to_wrong": 23, "changed_to_right": 26, "net": 3, "flips": 49, "churn_rate": 0.245}, "l025_225->l025_250": {"changed_to_wrong": 46, "changed_to_right": 15, "net": -31, "flips": 61, "churn_rate": 0.305}, "l025_250->l025_275": {"changed_to_wrong": 30, "changed_to_right": 30, "net": 0, "flips": 60, "churn_rate": 0.3}, "l025_275->l025_300": {"changed_to_wrong": 18, "changed_to_right": 34, "net": 16, "flips": 52, "churn_rate": 0.26}, "l05_250->l05_275": {"changed_to_wrong": 28, "changed_to_right": 23, "net": -5, "flips": 51, "churn_rate": 0.255}, "l05_275->l05_300": {"changed_to_wrong": 24, "changed_to_right": 29, "net": 5, "flips": 53, "churn_rate": 0.265}}`
- `mean_adjacent_flips`: `54.333333333333336`
- `single_checkpoint_sd_segments`: `{"binomial_p0.8": 5.656854249492381, "churn_model": 5.2122, "within_item_all_8_observations": 5.2864, "row_icc_mean": -0.0148}`
- `rms_adjacent_net_change_vs_churn_expectation`: `{"observed": 14.5831, "expected_from_churn_alone": 7.3711}`
- `mean_of_checkpoints_250_275_300`: `{"replay_0.25": 142.3333, "replay_0.5": 150.3333, "sum_replay_0.25": 427, "sum_replay_0.5": 451, "cluster_bootstrap_se_replay_0.25": 3.6856, "cluster_bootstrap_se_replay_0.5": 3.9471}`
- `paired_replay_0.5_minus_0.25_mean_of_3`: `{"diff_segments": 8.0, "cluster_boot_se": 3.8788127886548573, "z": 2.0624867545552097, "ci95": [1.7708057242771297e-15, 15.333333333333336], "frac_boot_le_0": 0.0216}`
- `minimum_detectable_effect_single_checkpoint`: `{"binomial_iid": {"sd_S": 5.656854249492381, "mde_segments": 14.065768091362804, "mde_pp": 7.032884045681402, "mde_segments_alpha025": 15.848242865377854, "mde_pp_alpha025": 7.924121432688927}, "clustered": {"sd_S": 5.656854249492381, "mde_segments": 14.065768091362804, "mde_pp": 7.032884045681402, "mde_segments_alpha025": 15.848242865377854, "mde_pp_alpha025": 7.924121432688927}, "churn_i_single": {"sd_S": 5.212165257037297, "mde_segments": 12.960048911623238, "mde_pp": 6.480024455811619, "mde_segments_alpha025": 14.602402184115691, "mde_pp_alpha025": 7.301201092057846}}`
- `one_real_update_flips_of_200`: `{"n_steps": 12, "flips_mean": 13.333333333333334, "flips_sd": 3.961940143032161, "flip_rate_per_step": 0.06666666666666667}`
- `recommended_gate_for_200_set`: `"sum of Y72 segment-exact over checkpoints 250/275/300 >= 480 and every checkpoint >= 144 (superseded for new experiments by the 600-segment set, see Y72-heldout-v2)"`
- `shared_clip_split_per_state`: `{"l025_225": {"update": 225, "replay_weight": 0.25, "batches": 25, "mean_norm_x_grad": 4.5192, "mean_norm_replay_grad": 3.0838, "median_norm_replay_grad": 1.6363, "mean_cos_x_vs_replay": -0.1061, "frac_batches_conflict": 0.76, "mean_global_clip_multiplier": 0.2442, "frac_batches_clip_binding": 1.0, "frac_batches_replay_exceeds_x_norm": 0.04, "replay_share_of_update_norm_global_clip": 0.1506, "replay_share_of_update_norm_per_task_clip_0p5": 0.3548}, "l025_250": {"update": 250, "replay_weight": 0.25, "batches": 25, "mean_norm_x_grad": 2.6103, "mean_norm_replay_grad": 3.2022, "median_norm_replay_grad": 2.1736, "mean_cos_x_vs_replay": -0.0614, "frac_batches_conflict": 0.72, "mean_global_clip_multiplier": 0.393, "frac_batches_clip_binding": 1.0, "frac_batches_replay_exceeds_x_norm": 0.0, "replay_share_of_update_norm_global_clip": 0.2217, "replay_share_of_update_norm_per_task_clip_0p5": 0.4526}, "l025_275": {"update": 275, "replay_weight": 0.25, "batches": 25, "mean_norm_x_grad": 3.0336, "mean_norm_replay_grad": 5.2119, "median_norm_replay_grad": 4.3211, "mean_cos_x_vs_replay": -0.0884, "frac_batches_conflict": 0.72, "mean_global_clip_multiplier": 0.4014, "frac_batches_clip_binding": 1.0, "frac_batches_replay_exceeds_x_norm": 0.08, "replay_share_of_update_norm_global_clip": 0.2909, "replay_share_of_update_norm_per_task_clip_0p5": 0.4555}, "l025_300": {"update": 300, "replay_weight": 0.25, "batches": 25, "mean_norm_x_grad": 3.0806, "mean_norm_replay_grad": 1.2824, "median_norm_replay_grad": 0.6041, "mean_cos_x_vs_replay": 0.0306, "frac_batches_conflict": 0.36, "mean_global_clip_multiplier": 0.412, "frac_batches_clip_binding": 0.96, "frac_batches_replay_exceeds_x_norm": 0.04, "replay_share_of_update_norm_global_clip": 0.1136, "replay_share_of_update_norm_per_task_clip_0p5": 0.2782}, "l05_300": {"update": 300, "replay_weight": 0.5, "batches": 25, "mean_norm_x_grad": 2.4696, "mean_norm_replay_grad": 3.6765, "median_norm_replay_grad": 2.3708, "mean_cos_x_vs_replay": 0.0133, "frac_batches_conflict": 0.4, "mean_global_clip_multiplier": 0.3578, "frac_batches_clip_binding": 1.0, "frac_batches_replay_exceeds_x_norm": 0.32, "replay_share_of_update_norm_global_clip": 0.3512, "replay_share_of_update_norm_per_task_clip_0p5": 0.4639}}`
- `one_step_scheme_effects`: `{"per_task_clip_minus_global": {"n": 12, "dS_mean": 1.4166666666666667, "dS_se": 0.5429762723390704, "dNLL_mean": -0.00427224300801754, "dNLL_se": 0.0019741419994247886, "dLx_actual_ratio_to_A_mean": 0.5963725512547295, "dLx_actual_ratio_median": 0.6029059136034365}, "pcgrad_then_global_minus_global": {"n": 12, "dS_mean": 0.5, "dS_se": 0.2886751345948129, "dNLL_mean": -0.0013105794787406921, "dNLL_se": 0.0004403833497011578, "dLx_actual_ratio_to_A_mean": 0.9935884788968092, "dLx_actual_ratio_median": 1.0}}`
- `no_more_heldout_items_in_saved_data`: `true`
- `not_established`: `"any effect of a clip scheme on a 75-update trajectory; relation of the Y72 sawtooth phase to the replay-row cycle"`

**Cost.** gpu_hours=0, usd=0, cpu_seconds_y72_eval=460.9, cpu_seconds_clip_split=8132, cpu_seconds_step_probes=1907

<!-- PUBLISH-GENERATED:END Y72-noise -->

<!-- TODO(publish): human judgement required. The block above is a faithful dump of the curated record, not prose. Replace it with a narrative section in the same voice as the E-W entries: what was asked, what actually happened, what failed, and what it means. Keep the generated numbers; do not restate any number that is not in results/. Then delete these markers. -->

## Open questions

1. ~~Does rehearsing prior distances inside the training mix prevent the interference, or only slow
   it?~~ **Partly answered by X:** a balanced mix learns several distances and keeps the earlier one,
   but it costs accuracy on the distance being trained and leaves untrained neighbours degraded.
   **Answered by Y: no.** Jittering filler by +-8 around each target did not generalise across the band; it moved capability to the band edges at the cost of the anchors.
2. Is the 114–120 boundary a property of the architecture or of this task's encoding?
3. Does the learning-rate sensitivity persist at larger widths, or is it a small-model artifact?
4. Would per-distance heads or a distance-conditioned curriculum avoid overwriting entirely?
