# Pre-registered stop rule, update 200 -> 300 (`vyra-stop-rule-300-v1`)

Registered before any update 201..300 data from the real GPU run existed. FORGE chose this reading
(the exact-0 condition gates **both** gain clauses). Machine-readable copy: `stop_rule_300.json`.
Implemented in `tools/experiment_packaging/stop_rule.py`; a test pins the two together.

Data: X64 teacher-forced masked-byte readout, train and held-out splits, 1920 tokens and 240 segments each.
Y72 retention probe: 200 segments. Checkpoints 225, 250, 275, 300. Baselines at update 200:
held-out 257/1920, train 669/1920. `token_acc = tokens_correct / tokens * 100`, evaluated by exact integer
cross-multiplication (no float, no rounding).

- **STUCK at update 300** iff
  (X64 train segment-exact count == 0 **AND** X64 held-out segment-exact count == 0)
  **AND**
  (held-out token acc(300) − held-out token acc(200) **< 10.0 pp** OR train token acc(300) − train token acc(250) **< 3.0 pp**).
  Strict `<`.
- **SLOW-LEARNER evidence** iff, at any of 225/250/275/300, X64 held-out segment-exact count > 0 AND Y72 segment-exact at that same checkpoint >= 80.0% (>= 160/200).
- **RETENTION FAILURE** iff Y72 segment-exact < 80.0% (< 160/200) at any checkpoint. This blocks any success claim.
- Otherwise the verdict is **INCONCLUSIVE**.
- Training **always stops at 300**, whatever the verdict. No automatic follow-up, no threshold relaxation.
- The original scientific joint gates are separate and unchanged.
