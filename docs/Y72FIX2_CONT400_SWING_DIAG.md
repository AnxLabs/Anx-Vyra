# Y72-fix2 cont400: diagnosis of the 375 opposite swing (CPU, measurement only)

Scope: why held-out Y72 fell and X64 spiked at update 375, and whether 400 is the same fight reversed. Inputs are only the saved
checkpoints (300, 325, 350, 375, 400) and the run's logged per-update clip record. Nothing was trained, no weights were interpolated,
no treatment was scored. CPU re-scoring of every checkpoint reproduces the run's counts exactly.

## What happened (600-set and X64)

| update | Y72 600-set | X64 exact | X64 tokens | Y600 mean row CE | X64 mean row CE | REP40 CE |
|---|---|---|---|---|---|---|
| 300 | 496 | 44 | 1665 | .140 | .837 | .0089 |
| 325 | 490 | 55 | 1671 | .151 | .760 | .0073 |
| 350 | 504 | 46 | 1618 | .116 | .890 | .0057 |
| 375 | 396 | 124 | 1764 | .316 | .458 | .0237 |
| 400 | 573 | 82 | 1658 | .023 | .756 | .0025 |

- 350->375: Y72 lost 128 segments, gained 20 (net -108, about 11 single-checkpoint SDs of 9.5). Losses were diffuse: 82 of 120 rows, at most 4 per row.
- 375->400: Y72 gained 185, lost 8; 123 of the 128 segments lost at 375 came back. X64 lost 56 exact, gained 14; 48 of the 83 exact gained at 375 were lost again.

## Mechanism

1. **The two held-out objectives pull along one shared axis.** Raw cosine between the held-out Y72 (600-set) gradient and the X64
   held-out gradient: -0.88 (325), -0.79 (350), -0.81 (375), -0.41 (400). It is the same in shared and expert parameters
   (shared -0.88/-0.79/-0.81/-0.42, expert -0.88/-0.74/-0.81/-0.40); the router carries little of it. An Adam-preconditioned step on
   X64 raises Y72 loss (normalized cross-effect -0.34/-0.31/-0.34/-0.19). The X64 needle training row conflicts too (raw -0.72/-0.62/-0.32/-0.39).
2. **The clip caps the replay pull but does not hold it up.** X's clipped contribution is exactly 0.5 on every update (X clipped
   98/100). The replay contribution is min(0.25*|g_rep|, 0.5), so it tracks the replay rows' own loss. Window means of the replay
   contribution / replay share of the update: 301-325 0.23/0.26, 326-350 0.27/0.30, **351-375 0.15/0.19**, 376-400 0.24/0.27
   (parent run 226-300: 0.34, 0.24, 0.25 / 0.38, 0.27, 0.27). 351-375 is the lowest window in either run: 16 of 25 updates below 0.1,
   replay clip bound on only 3, max replay gradient norm 3.8 (vs 41.3 in 326-350). Correlation of replay contribution with replay loss is 0.73.
3. **When the 40 replay rows are memorized, their gradient also stops carrying held-out Y72.** At 350 the replay gradient norm is 0.31
   and its preconditioned effect on held-out Y72 is 0.035 (0.16 at 325, 0.31 at 375, 0.056 at 400). So 351-375 was X-dominated and the
   replay signal that remained barely transferred to Y72; X64 moved along the conflict axis and Y72 fell.
4. **Restoring force with a lag.** By 375 the replay rows have degraded (CE .0057 -> .0237), the replay gradient is 3.0 and aligned with
   held-out Y72 (raw +0.83, preconditioned +0.31), and it pushes X64 back (preconditioned -0.25 on X64). 376-400 replay contribution
   rises to 0.24 (clip bound on 9 updates) and Y72 recovers while X64 gives back its gains.
5. **Not a routing event.** Per-layer routing total variation at scored positions for 350->375 (Y600 .05-.13, X64 .03-.13) is the same
   size as 300->325 (.04-.10, .03-.14); Y600 and X64 share experts at every checkpoint (TV .01-.17). Expert-union overlap telemetry unchanged.

## Is 400 the reverse of the same fight?

Yes in function space, no in parameter space. The same rows and the same axis flip back (123/128 Y72 segments, 48/83 X64 exact), driven
by the replay pull that the 375 erosion re-armed. Parameters do not retrace: segment displacement cosines are +0.29 (325-350 vs 350-375)
and +0.16 (350-375 vs 375-400); 350->400 has norm 5.63 and cosine 0.75 with 350->375. At 400 the system is back in the 350-like state
(replay gradient norm 0.10, held-out Y72 gradient 0.63, replay->Y72 transfer 0.056, momentum raising X64 loss by first order), so the
oscillation is expected to continue on further training; this is a prediction, not a measurement.

## Is a pre-registered fix warranted? No.

- The swing is not evaluation noise (11 SD), but it is one trajectory of one seed with one collapse. Low-replay windows without a
  collapse exist (301-325 replay-share median 0.16; parent 251-300 0.21), so replay starvation is not sufficient on its own and there is
  no second instance to size an effect from.
- The obvious lever (a floor or fixed-norm normalization on the replay gradient) would amplify a direction whose transfer to held-out
  Y72 is weakest exactly when it is small (0.035 at 350, 0.056 at 400). The mechanism predicts weak efficacy, not a detectable gain.
- The underlying conflict (held-out cosine about -0.8) is structural. Rebalancing the clip moves the operating point along the trade-off;
  a Y72 gain would be paid for in X64 (the 375 X64 gain is exactly the Y72 loss). Under the cont400 gates, which need both Y72 hold and
  X64 progress, there is no change whose expected effect can be stated against the 600-set noise (single SD 9.5, mean-of-3 SE 8.3-9.0)
  without inventing it.
- This is a fight the current per-task clip cannot stabilize. Addressing it would be a redesign of the replay signal (for example a
  rehearsal set that is not memorized to zero loss), which is a new experiment, not a pre-registered tweak of this one.
