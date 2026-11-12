# Y72 held-out v2: scoring of existing checkpoints (non-gating)

Scored AFTER the pre-registration (prereg/y72_heldout_v2.json, commit d7e6147). CPU only. Every checkpoint file sha256 was verified against the pinned value before `torch.load`. The legacy 200-segment set was re-scored in the same pass and matched the earlier study exactly for all 8 checkpoints.

Set: canonical sha256 `16ff53a839c5c68e2230fe4f96e794f11d7ff7f885cac3e184e4f4770f3ee71b`, file sha256 `12e034022357e230685a956e47658315797f164b5b11c30bd35ba4f6ca04a826`, 600 segments.

| checkpoint | 600-set seg-exact | % | blocks of 200 | legacy 200-set | tokens correct /4800 | mean NLL |
|---|---|---|---|---|---|---|
| l025_200 | 498 | 83.0 | [173, 160, 165] | 165 (82.5%) | 4685 | 0.1167 |
| l025_225 | 490 | 81.7 | [172, 158, 160] | 168 (84.0%) | 4680 | 0.1343 |
| l025_250 | 363 | 60.5 | [124, 126, 113] | 137 (68.5%) | 4526 | 0.3036 |
| l025_275 | 390 | 65.0 | [139, 124, 127] | 137 (68.5%) | 4576 | 0.2703 |
| l025_300 | 454 | 75.7 | [160, 148, 146] | 153 (76.5%) | 4652 | 0.1719 |
| l05_250 | 450 | 75.0 | [152, 154, 144] | 152 (76.0%) | 4645 | 0.1889 |
| l05_275 | 421 | 70.2 | [129, 149, 143] | 147 (73.5%) | 4609 | 0.2137 |
| l05_300 | 420 | 70.0 | [139, 147, 134] | 152 (76.0%) | 4607 | 0.2011 |

## Noise floor vs the registered predictions
- single-checkpoint SD (segments): churn model 9.34, within-item (all 8) 9.73; registered range 8.89-9.8 -> within range: True (both inside).
- mean-of-3 cluster-bootstrap SE: lambda0.25 8.96, lambda0.5 8.26 segments; registered range 5.29-6.84 -> ABOVE range (ratio to sqrt(3) x legacy: 1.40, 1.21). Averaging 3 checkpoints shrinks the SE less than independence predicts (ratio mean3/single 0.75, 0.75; independence would give 0.58).
- level vs the legacy 200-set (disjoint sample, same generator): mean of 250/275/300 is 67.1% (lambda0.25) and 71.7% (lambda0.5) on the 600-set versus 71.2% and 75.2% on the 200-set, i.e. about 4 pp lower for both arms; at 225 and 200 the two sets agree within 2.3 pp.
- row ICC mean 0.013 (legacy -0.015): still about 0.
- registered gate on the observed runs: lambda0.25 sum3=1207 (floor 1440) R_FAIL; lambda0.5 sum3=1291 R_FAIL. Illustration only; the gate is for future experiments.
- lambda0.5 minus lambda0.25 (mean of 250/275/300, same items): 28.0 segments = 4.67 pp, cluster-bootstrap SE 9.30, z 3.01.

Full numbers: `y72_v2_noise_floor.json`; raw per-segment token matches: `y72_v2_scores.json`. Reproduce: `score_heldout_v2.py` then `analyze_v2.py` (paths inside are the box paths used).
