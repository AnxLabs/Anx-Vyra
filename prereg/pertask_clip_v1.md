# Pre-registration (specification only): per-task gradient clip from checkpoint_225 (`vyra-pertask-clip-v1`)

Registered by FORGE (Grok Bot) after the CPU-only noise-floor / shared-clip study (`/workspace/forge/vyra-noise-clip-study-20261110/REPORT.md`).
**No notebook is built. Not approved to build or run.** Machine-readable twin: `prereg/pertask_clip_v1.json`; implemented in
`tools/experiment_packaging/pertask_clip_gates.py` (a test pins the two together); the gradient scheme is in `clip_scheme.py`.

## Question
Does clipping the X gradient and the replay gradient separately (each to norm 0.5, replay weight 0.25 unchanged) instead of one global
clip at 1.0 keep Y72 held-out segment-exact at a mean of >= 80.0% over checkpoints 250/275/300 while X64 held-out acquisition continues?

## The one variable
Gradient combination only: `u = clip(gx, 0.5) + 0.25 * clip(gy, 0.5/0.25)` (equivalently each task's contribution is limited to norm 0.5; `||u|| <= 1.0`)
instead of `u = clip(gx + 0.25 * gy, 1.0)`. Same resume file (`checkpoint_225.pt`, sha256 `a84d35be...b283`), plan, SliceAdam, LR, replay rows, union,
seed, data, evaluation and hardware. 75 updates (226..300), checkpoints 250/275/300, one report, no retry.

## Gates (integer arithmetic, `>=` only)
| Gate | Statement | Boundary |
|---|---|---|
| **R** | sum over 250/275/300 of Y72 segment-exact >= 480 (mean 160/200) **and** every checkpoint >= 144/200 (72.0%) | 480 / 479; 144 / 143 |
| **A** | X64 held-out segment-exact >= 11 at every checkpoint **and** token gain 225->300 >= 7.0 pp (>= 1553/1920 from 1418) | seg 11/10; tokens 1553/1552 |
| **N** | X32 >= 128/320 and X109 >= 90/120 at every checkpoint | 128/127; 90/89 |

Why R changed from the lambda-0.5 rule: one checkpoint's Y72 count has SD ~5.3 segments of iterate noise (binomial 5.66, churn model 5.2,
within-item 5.3; one real update flips 13.3 +- 4.0 of 200 segments), so "every checkpoint >= 160" fails a true-168 model 7-19% of the time (fully correlated .. independent checkpoints).
The three-checkpoint mean has SD ~3-4 (cluster bootstrap 3.7-3.95). Why A's gain is 7.0 pp: control +14.0 pp, lambda 0.5 +10.4 pp, per-task clip makes ~0.60x the
X progress per step in the real one-step probe; 7.0 pp is 50% of the control's gain.

## Verdict table
Same eight-way table as `y72fix_lambda05.md` (SUCCESS only if R and A and N; otherwise FAIL_ + failed gates; NOT_EVALUATED_BEFORE_300 if any checkpoint is missing).
Observed global-clip runs under this R rule: lambda 0.25 sum 427, lambda 0.5 sum 451, both fail.

## Caveats (pre-registered)
One run, one seed. The minimum detectable difference of the three-checkpoint Y72 mean vs 160 is 7.6-9.8 segments (3.8-4.9 pp) at one-sided alpha 0.05 / power 0.8, so a
pass or a fail says only whether this run reached 160/200 on average; it does not measure the size of any improvement. The one-step probe is first-order
evidence about a single update, not about 75.
