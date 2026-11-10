# Pre-registration: replay weight 0.5 from checkpoint_225 (`vyra-y72fix-lambda05-v1`)

Registered by FORGE (Grok Bot) before any update 226..300 with replay weight 0.5 exists. **Not approved to run.**
Machine-readable twin: `prereg/y72fix_lambda05.json`; implemented in `tools/experiment_packaging/y72fix_gates.py`
(a test pins the two together).

## Question
Does a larger replay weight keep Y72 held-out segment-exact at or above 80.0% while X64 held-out acquisition continues?

## The one variable
`replay_lambda` 0.25 -> **0.5** (`loss = lx + lambda * ly`). Nothing else changes: same checkpoint state (complete v2 state at
update 225), same plan, SliceAdam, LR 3e-4, global clip 1.0, 2 replay rows per update with index `(s*2+j)%40`, expert union,
seed 0, data, evaluation, free Kaggle T4 x2.

* Resume file: `checkpoint_225.pt`, sha256 `a84d35bee3323cf60d0f0a09f0eece5ffb4c1e7bddbd196d4c478df15cb1b283`, 197,237,331 bytes
  (written by the approved 200 -> 300 run; model-state sha256 `0990d7c25ec3108bd05f214e1cbfb5495cf12cfceec68e9fddfd7f404bc61482`).
* Updates 226..300 (75). Checkpoints/evals at 250, 275, 300. Always stops at 300; one report; no retry.
* Paired control: the already-observed lambda 0.25 trajectory 225 -> 300 (same plan, same state) - comparison only, never a gate.
* Restoration check at 225 (zero tolerance, before any update): model-state sha and every integer count must match the 225
  values recorded by the 300 run (Y72 168/200 seg, 1565/1600 tok; X32 160/320, 2336/2560; X109 102/120, 934/960; X64 held-out
  11/240, 1418/1920; X64 train 28/240, 1562/1920).

## Arithmetic
Integer cross-multiplication only. No float, no rounding. Every floor and the gain threshold use `>=`.

## Gates
| Gate | Statement | Boundary |
|---|---|---|
| **R** retention | Y72 segment-exact >= 80.0% (>= 160/200) at **every** checkpoint 250, 275, 300 | 160 pass, 159 fail |
| **A** acquisition | X64 held-out segment-exact >= 11 (resume value) at every checkpoint **and** held-out token gain 225 -> 300 >= 10.0 pp (tokens_correct(300) >= 1610 of 1920, from 1418) | seg 11 pass / 10 fail; tokens 1610 pass / 1609 fail |
| **N** non-regression | X32 segment-exact >= 40.0% (>= 128/320) and X109 >= 75.0% (>= 90/120) at every checkpoint | 128/127 and 90/89 |

Justification: A's floor means held-out X64 may not fall below where it resumed, and its 10.0 pp token gain is the size the earlier
stop rule used for "not stuck" (the control reached 92/240 and 1687 tokens, +14.0 pp, so A asks for continued learning, not for
matching the control). N's bounds are 10 pp below the resume values (X32 50.0%, X109 85.0%); the control stayed far above them.

## Verdict table (strict, computed only by `y72fix_gates.evaluate`)
| R | A | N | Verdict |
|---|---|---|---|
| pass | pass | pass | **SUCCESS** |
| fail | pass | pass | FAIL_RETENTION |
| pass | fail | pass | FAIL_ACQUISITION |
| pass | pass | fail | FAIL_NONREGRESSION |
| fail | fail | pass | FAIL_RETENTION_ACQUISITION |
| fail | pass | fail | FAIL_RETENTION_NONREGRESSION |
| pass | fail | fail | FAIL_ACQUISITION_NONREGRESSION |
| fail | fail | fail | FAIL_RETENTION_ACQUISITION_NONREGRESSION |
| - | - | - | NOT_EVALUATED_BEFORE_300 (any of 250/275/300 missing; per-checkpoint booleans still reported) |

## Caveats (pre-registered)
One run, one seed: SUCCESS is not a replication; a Y72 miss of a few segments does not prove lambda ineffective (Y72 changes
correctness for 49, 61, 60, 52 of 200 items in the four observed 25-update intervals; net +3, -31, 0, +16). A FAIL_RETENTION would
say only that lambda 0.5 did not hold 160/200 at every checkpoint in this run. After the verdict there is no automatic follow-up.
