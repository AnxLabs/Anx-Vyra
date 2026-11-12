"""Rescale the 200-item noise-floor study to n_new segments. Every input number is read from the study JSONs; arithmetic only (no scoring)."""
import json, math, sys
from math import erf, sqrt
N_OLD = 200; N_NEW = int(sys.argv[1]); K = N_NEW / N_OLD
nf = json.load(open("/workspace/forge/vyra-noise-clip-study-20261110/diag/noise_floor.json"))
gd = json.load(open("/workspace/forge/vyra-noise-clip-study-20261110/diag/gate_design.json"))
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
ZSUM = 1.6449 + 0.8416     # one-sided alpha 0.05 + power 0.8 (same constants as the 200-item study)
single = {"binomial_p0.8": sqrt(N_OLD * 0.8 * 0.2), "churn_model_i": nf["churn_model_i"]["implied_single_checkpoint_sd"],
          "within_item_ii_all8": nf["churn_model_ii_within_item_variance"]["all_8_obs_k8_sd"],
          "within_item_ii_l05_k3": nf["churn_model_ii_within_item_variance"]["l05_250_275_300_k3_sd"],
          "within_item_ii_l025_k4": nf["churn_model_ii_within_item_variance"]["l025_225_to_300_5ckpt_k4_sd"]}
mean3 = {"independent_ckpts_5.29/sqrt3": nf["churn_model_ii_within_item_variance"]["all_8_obs_k8_sd"] / sqrt(3),
         "cluster_bootstrap_l025": nf["mean_of_3_checkpoints"]["l025_250_275_300"]["cluster_bootstrap_se"],
         "cluster_bootstrap_l05": nf["mean_of_3_checkpoints"]["l05_250_275_300"]["cluster_bootstrap_se"]}
out = {"n_old": N_OLD, "n_new": N_NEW, "scale_factor_k": K,
       "assumption": "iterate/sampling noise is item-level (independent across segments): OBSERVED ICC by row -0.015 -> design effect 1 in the 200-item study, so SD in segments scales with sqrt(n) and SD in pp with 1/sqrt(n). "
                     "A checkpoint-level component shared by all items would NOT shrink; that is the thing the scoring of existing checkpoints on the new set tests (prediction made BEFORE scoring).",
       "single_checkpoint_sd_segments": {k: {"old_200": v, "new": v * sqrt(K), "old_pp": 100 * v / N_OLD, "new_pp": 100 * v * sqrt(K) / N_NEW} for k, v in single.items()},
       "mean_of_3_sd_segments": {k: {"old_200": v, "new": v * sqrt(K), "old_pp": 100 * v / N_OLD, "new_pp": 100 * v * sqrt(K) / N_NEW} for k, v in mean3.items()}}
# MDE vs the floor (segments and pp)
out["mde_vs_floor"] = {"single_checkpoint": {k: {"segments": ZSUM * v * sqrt(K), "pp": 100 * ZSUM * v * sqrt(K) / N_NEW} for k, v in single.items()},
                       "mean_of_3": {k: {"segments": ZSUM * v * sqrt(K), "pp": 100 * ZSUM * v * sqrt(K) / N_NEW} for k, v in mean3.items()}, "z_alpha_plus_z_beta": ZSUM}
# gate rescaled (integer thresholds)
FLOOR_T, GUARD_T = 800, 720
floor = FLOOR_T * N_NEW // 1000; guard = GUARD_T * N_NEW // 1000
assert floor * 1000 == FLOOR_T * N_NEW and guard * 1000 == GUARD_T * N_NEW
out["gate"] = {"per_checkpoint_floor_segments": floor, "sum3_floor_segments": 3 * floor, "per_checkpoint_guard_segments": guard,
               "old_gate_200": {"sum3": 480, "guard": 144}, "n_new": N_NEW}
# operating characteristics of the new sum gate (normal model, SD of mean-of-3 in segments; continuity correction on the integer sum)
oc = {}
for lab, sd1 in (("sd_mean3_independent", mean3["independent_ckpts_5.29/sqrt3"]), ("sd_mean3_cluster_boot_l05", mean3["cluster_bootstrap_l05"]), ("sd_mean3_fully_correlated_single_sd", single["within_item_ii_all8"])):
    sd = sd1 * sqrt(K); row = {}
    for pct in (86, 84, 82, 81, 80, 79, 78, 76, 74, 72):
        L = N_NEW * pct / 100
        row[str(pct)] = {"P_pass_sum3": 1 - Phi(((3 * floor - 0.5) / 3 - L) / sd)}
    oc[lab] = {"sd_mean3_segments": sd, "by_true_level_pct": row}
out["operating_characteristics_sum3"] = oc
# the old 'every checkpoint >= floor' form at the new n for comparison (independent / fully-correlated single-ckpt SD, within-item all-8)
old = {}
sd1 = single["within_item_ii_all8"] * sqrt(K)
for pct in (84, 82, 80, 78, 76):
    L = N_NEW * pct / 100; p1 = 1 - Phi((floor - 0.5 - L) / sd1)
    old[str(pct)] = {"P_all3_independent": p1 ** 3, "P_all3_fully_correlated": p1}
out["operating_characteristics_every_ckpt_ge_floor"] = {"sd_single_segments": sd1, "by_true_level_pct": old}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps({k: out[k] for k in ("single_checkpoint_sd_segments", "mean_of_3_sd_segments", "mde_vs_floor", "gate")}, indent=1)[:3500])
print(json.dumps(out["operating_characteristics_sum3"]["sd_mean3_cluster_boot_l05"]))
