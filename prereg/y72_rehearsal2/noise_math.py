"""Pass probabilities of the registered gates under the measured noise (Monte Carlo, 400k draws, seed 0).
Y72 600-set: single-checkpoint SD 9.5; checkpoints share a noise component: mean-of-3 SE 8.3-9.0 -> intra-run correlation rho from
SE^2 = 9.5^2 (1+2 rho)/3 : rho 0.64 (SE 8.3) .. 0.74 (8.65) .. 0.85 (9.0). X64 tokens: SD 21 for the 300->400 gain (sqrt2 x 15), exact SD 6."""
import json, numpy as np
rng = np.random.default_rng(0); N = 400_000; SD = 9.5
def rho(se): return (3 * se * se / SD ** 2 - 1) / 2
def p_hold(mu, r):
    mu = np.asarray(mu, float); C = SD ** 2 * (r * np.ones((3, 3)) + (1 - r) * np.eye(3))
    z = rng.multivariate_normal(mu, C, N); return float(((z.sum(1) >= 1440) & (z >= 432).all(1)).mean())
from math import erf, sqrt
Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
out = {"rho": {"SE8.3": round(rho(8.3), 3), "SE8.65": round(rho(8.65), 3), "SE9.0": round(rho(9.0), 3)}, "R_hold_flat_true_level": {}, "R_hold_profiles": {}}
for m in (470, 475, 480, 485, 490, 496, 505, 520):
    out["R_hold_flat_true_level"][m] = {k: round(p_hold([m] * 3, rho(se)), 3) for k, se in (("SE8.3", 8.3), ("SE9.0", 9.0))}
prof = {"control_observed_504_396_573": [504, 396, 573], "dip_to_440_at_375": [504, 440, 520], "dip_to_450": [500, 450, 510], "dip_to_460": [500, 460, 500],
        "flat_496": [496, 496, 496], "slow_slide_490_480_470": [490, 480, 470], "slide_like_parent_480_464_448": [480, 464, 448]}
for k, mu in prof.items(): out["R_hold_profiles"][k] = {"SE8.3": round(p_hold(mu, rho(8.3)), 3), "SE9.0": round(p_hold(mu, rho(9.0)), 3)}
out["A_tokens_gain_ge58_by_true_gain"] = {g: round(1 - Phi((57.5 - g) / 21), 3) for g in (-50, -7, 0, 20, 40, 58, 80, 120)}
out["A_exact_each_ge44_by_true_flat_level_indep"] = {m: round((1 - Phi((43.5 - m) / 6)) ** 3, 3) for m in (44, 50, 55, 60, 70)}
out["N_floors"] = "X32 >= 128 and X109 >= 90 sit far below observed values (control 248-266 and 105-111); P(fail | no regression) < 0.001"
json.dump(out, open("noise_math.json", "w"), indent=1); print(json.dumps(out, indent=1))
