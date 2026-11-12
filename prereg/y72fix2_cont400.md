# Pre-registration: y72fix2 continuation 300 → 400 (`vyra-y72fix2-cont400-v1`)

Machine-readable twin: [`y72fix2_cont400.json`](y72fix2_cont400.json). Registered by FORGE (Grok Bot) on Nov 12 2026 after AJ's go on
VERA card t608s2520 (12:03 AM UTC-7), before any update 301..400 exists, before any package build and before any scoring. The approved
scope is to pre-register, build the run package and pass CPU preflight. A GPU run, an upload or a new Kaggle dataset each need a separate approval.

## Question

The parent run (`prereg/y72fix2.json`, SUCCESS) held Y72 with a per-task clip of 0.5 and replay weight 0.25 from 225 to 300: 528, 524 and 496
on the 600-segment set (sum 1548), while X64 held-out exact went 13 → 28 → 44 and tokens reached 1665/1920. Y72 was falling at the end
(−32 over the last 50 updates). This run continues the same configuration for 100 more updates and asks two things:
does Y72 **hold**, and does X64 **keep progressing**?

## Design

* Resume from the parent's `checkpoint_300.pt` (sha256 `5df9a3f8bcf04dbeac6b15d96a312e3fa3b425aef60639a603ece3c19fea45a3`, 197237651 bytes,
  model-state sha256 `8fecaa8b159d2979b93322b896b9709824ea354a2002c5eafae104cbcaeceb9d`). It is a complete v2 state (parameters, SliceAdam
  state, both CUDA RNGs, Python/NumPy RNG, router, sampler at 300). The sha is checked before `torch.load`.
* Exact restore: the model-state hash must match (always fatal), and the restored model's integer counts at 300 must equal the parent's
  recorded counts (zero tolerance, fatal on GPU). This is the same discipline as the parent's restore from 225.
* No change to the training rule. The per-task clip is 0.5 on gx and on 0.25·gy. Replay is 2 rows per update, LR 3e-4, SliceAdam, and the
  expert union is unchanged.
* Plan: entries 300..399 continue the same v1 extension streams (`build_extension(..., count=200)[100:]`). plan[0:300] is byte-identical to
  the pinned 300-entry plan. Full 400-entry plan sha256 `d9f13234…d943`.
* Evaluations and complete checkpoints at 325, 350, 375 and 400. The gates use 350/375/400; 325 is reported only. The run always stops at
  400 with one report and no retry.

## Gates (integer arithmetic, `>=`)

| Gate | Rule |
| --- | --- |
| R hold | Y72 600-set segment-exact: sum over 350, 375, 400 ≥ 1440 of 1800 **and** each ≥ 432 of 600 |
| A progress | X64 held-out exact ≥ 44 of 240 (its value at 300) at each of 350, 375, 400 **and** token gain 300 → 400 ≥ 3.0 pp (tokens(400) ≥ 1723 of 1920) |
| N non-regression | X32 ≥ 128/320 and X109 ≥ 90/120 at each of 350, 375, 400 |

The verdict is SUCCESS only if R, A and N all pass. Otherwise it is FAIL_<failed gates> (FAIL_HOLD, FAIL_PROGRESS, FAIL_NONREGRESSION, or a
combination), and NOT_EVALUATED_BEFORE_400 if any gate checkpoint is missing.

R reuses the 600-set rule (`vyra-y72-heldout-v2-gate-v1`: mean ≥ 80.0 %, each ≥ 72.0 %) on the last three checkpoints. It is again a sum over
three checkpoints of 600 segments, so no rescaling is needed. A has the parent's structure: never below the resume-point value, plus a token
gain from the resume point. The gain bar falls from 7.0 pp to 3.0 pp because only 13.3 pp of headroom remain. N is unchanged.

## Noise math

**Y72, 600-set.** The single-checkpoint SD is about 9.5 segments (measured 9.3–9.7). Checkpoints share a noise component, so the
mean-of-3 SE is 8.3–9.0 (measured), not 9.5/√3 = 5.5, and the SE of the sum is 25–27. The floor 1440 corresponds to a true per-checkpoint
mean of 480. P(R_sum passes | true mean m) is ≈ 0.12 at 470, 0.29 at 475, 0.51 at 480, 0.72 at 485, 0.88 at 490, 0.97 at 496 (the value at
300) and 0.99 at 500. The guard of 432 sits 5 SDs below 480 and only catches a one-checkpoint collapse.

Measurement noise is not what limits this run. The parent's Y72 slope over its last 50 updates was −16 per 25 updates. If that slope
continued, the sum over 350/375/400 would be about 1344 and R would fail. The gate therefore tests whether Y72 levels off at or above 80 %.
Non-gating paired report: mean(350..400) − 496, SE ≈ √(9.5² + 8.7²) ≈ 12.9.

**X64 held-out** (40 rows × 6 segments, 1920 tokens). The single-measurement SD is about 15 tokens (0.8 pp). This is the binomial value,
and the row-cluster SDs from the parent's predictions at 250/275/300 are 13.8, 17.4 and 12.3. Segment-exact SD is about 5–6. A
300 → 400 gain has an SD of about 21 tokens (1.1 pp), so the 3.0 pp bar (58 tokens) is 2.7 difference-SDs. P(pass) is ≈ 0.003 with no
true progress, 0.20 for a true +2.1 pp, 0.51 for +3.0 pp, 0.86 for +4.2 pp and 0.999 for +6.25 pp. For comparison, the parent gained 12.9 pp
in 75 updates. A plateau exactly at 44 passes each checkpoint's segment floor only about half the time, so that floor also demands progress.

**X32/X109.** The floors are 101 and 19 segments below the values at 300 and only catch a regression.

## Expected outcome (judgment, not computed)

P(SUCCESS) is about 0.4 (range 0.25–0.6): R ≈ 0.5, A ≈ 0.8, N ≈ 0.95, positively correlated. No CPU probe of 301..400 was run before
registration. This run asks a real question. A FAIL_HOLD would mean that the per-task clip slows Y72 forgetting but does not stop it
over longer training.
