# Y72 retention fix 2: per-task gradient clip (specification only)

Rule id `vyra-y72fix2-pertask-v2`. Machine-readable form: [`y72fix2.json`](y72fix2.json); gates in
`tools/experiment_packaging/y72fix2_gates.py`, pinned by `tests/test_y72fix2_gates.py`. Nothing here builds, uploads or runs
anything: no notebook or run package exists for this change.

## The change

Resume from the update-225 checkpoint (sha256 `a84d35be…b283`), replay weight stays 0.25, and replace the single global gradient clip
(1.0 on `gx + 0.25·gy`) with a **per-task clip**: `gx` is clipped to norm 0.5 and `0.25·gy` is clipped to norm 0.5 separately, then
the two are summed. Everything else (plan, SliceAdam, learning rate 3e-4, expert union, restored optimizer state) is unchanged.
Updates 226..300, evals at 250/275/300, stop at 300.

The value 0.5 was fixed on Nov 10 in `pertask_clip_v1.json`, before the evidence below existed, so it was not tuned on it. A dedicated
extra replay budget was **not** added: the probes did not need it and a second variable would make a pass uninterpretable.

## Why this and not the others (CPU probes, not a run)

All probes start from the sha256-verified update-225 checkpoint; evaluation is on the registered 600-segment Y72 set and the X64
held-out set. They ran without the deterministic-MKL setting, so they are not bit-reproducible; each arm is one trajectory.

* **The two tasks fight over the same parameters.** Mean X-vs-Y gradient cosine is −0.49 at 225 (−0.57 layer 0, −0.42 layer 1,
  −0.48 embedding, about 0 in layers 2–7). 99% of the Y Fisher mass is in the embedding and layers 0–2, and the X and Y Fisher
  diagonals are nearly the same coordinates (Spearman 0.82; 61% of the top-1% coordinates coincide against 1% by chance).
  There is no separate set of weights to protect, which counts against protected-parameter and subspace-projection fixes.
* **No subset of the drift restores Y for free.** Undoing only layer 0 of the 225→250 drift restores Y (486 vs 490 at 225) and erases
  the X64 gain (12 vs 29). Undoing the Fisher-top-2% coordinates gives Y 476 and X64 17; random 2% gives Y 371. Straight
  interpolation back toward 225 does about as well (α 0.5: Y 481, X64 21). Y falls non-linearly along the drift path.
* **Arms (75 updates unless noted; Y over 250/275/300, sum, X64 held-out exact):**

| arm | Y600 | sum | X64 |
|---|---|---|---|
| global clip, λ0.25 (CPU; GPU run 363/390/454 = 1207) | 363/395/324 | 1082 | 29/40/105 |
| global clip, λ0.5 (CPU; GPU 1291) | 450/366/429 | 1245 | 17/51/55 |
| L2 anchor to 225, μ1 | 564/353/388 | 1305 | 2/29/25 |
| freeze layer 0–1 q/k | 387/211/306 | 904 | 25/78/74 |
| **per-task clip 0.5, λ0.25** | **528/519/528** | **1575** | 13/28/27 |
| same, perturbation replicate | 528/517/494 | 1539 | 13/28/41 |

  25-update screens at 250 only: L2 μ0.3 487, EWC μ1 487 (X64 at the floor, 11), Fisher-2% freeze 472, PCGrad 360 (no help, as the
  earlier one-step probe predicted). The CPU run reproduces the GPU at update 250 exactly for both controls (363/29 and 450/17) and
  then drifts away from it, so these are samples of a chaotic trajectory, not predictions.

## Gates (integer arithmetic, `>=`)

* **R** retention: Y72 on the 600-set, sum over 250/275/300 ≥ 1440 and each ≥ 432 (rule `vyra-y72-heldout-v2-gate-v1`).
* **A** acquisition: X64 held-out exact ≥ 11/240 at every checkpoint and token gain 225→300 ≥ 7.0 pp (tokens at 300 ≥ 1553 of 1920).
  Same bar as `pertask_clip_v1`, chosen there because this change is expected to give about 0.6× the X progress. The 10.0 pp bar of
  the λ0.5 rule is reported, not gating.
* **N** non-regression: X32 ≥ 128/320 and X109 ≥ 90/120 at every checkpoint.

SUCCESS only if all three hold.

## Expected effect, noise, probability

The CPU per-task arms beat the CPU λ0.25 control by 457–493 segments of 1800 (GPU controls: +248 to +368 against λ0.5 / λ0.25
GPU runs). Measurement noise of the three-checkpoint sum is about 25–27 segments, so the effect is many times it; the observed sums
clear 1440 by 99–135. The limiting noise is trajectory noise: the CPU control was 130 segments from the GPU run at update 300, and the
two per-task trajectories differ by 34.

**Probability of passing all gates on the GPU run: about 0.6 (plausible 0.4–0.8), a judgment, not a computed number.** Retention is
the likelier pass (about 0.85–0.9). Acquisition is the tight gate: X64 held-out was 13 at update 250 against a floor of 11. Caveats:
the arm was picked from about nine one-trajectory arms; the perturbation replicate only decorrelates late (identical at 250), so there
are about 1.5 independent samples; CPU-to-GPU transfer is unverified.

**What a pass would mean:** Y72 held at the cost of much slower X64 acquisition (27–41 held-out exact at 300 in the CPU arms vs 92 for the
λ0.25 run). That meets the registered floors but is a trade-off, not a free fix. Nothing in the evidence suggests any of the
candidates delivers the old X64 pace and Y72 retention together.
