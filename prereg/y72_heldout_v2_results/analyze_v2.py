"""Noise-floor analysis of existing checkpoints on the pre-registered 600-segment set (numpy only). Plan fixed in prereg/y72_heldout_v2.json (scoring_plan_existing_checkpoints_nongating)."""
import json, sys
from math import sqrt, erf
import numpy as np
sys.path.insert(0, "/workspace/forge/vyra-resume300-prep-20261110/repo/tools/experiment_packaging")
import y72_heldout_v2 as g
D = json.load(open("/workspace/forge/vyra-y72heldout-20261111/score/y72_v2_scores.json"))
PRE = json.load(open("/workspace/forge/vyra-resume300-prep-20261110/repo/prereg/y72_heldout_v2.json"))
NF = json.load(open("/workspace/forge/vyra-noise-clip-study-20261110/diag/noise_floor.json"))
names = list(D["ckpt"]); seg = {k: (np.array(D["ckpt"][k]["new"]["tok"]) == 8).astype(int) for k in names}   # [120,5]
assert all(int(seg[k].sum()) == D["ckpt"][k]["new"]["segment_exact"] for k in names)
prev = json.load(open("/workspace/forge/vyra-noise-clip-study-20261110/diag/y72_tok.json"))
assert all(int((np.array(prev[k]["tok"])==8).sum())==D["ckpt"][k]["legacy200"]["segment_exact"] for k in names)
seg_old = {k: (np.array(prev[k]["tok"]) == 8).astype(int) for k in names}
out = {"n_segments": 600}
out["counts"] = {k: {"new600": int(seg[k].sum()), "pct": 100 * seg[k].sum() / 600, "blocks_of_200": [int(seg[k][i*40:(i+1)*40].sum()) for i in range(3)], "legacy200": int(seg_old[k].sum()), "legacy_pct": 100 * seg_old[k].sum() / 200,
                 "tokens_correct": D["ckpt"][k]["new"]["tokens_correct"], "mean_nll": float(np.mean(D["ckpt"][k]["new"]["nll"]))} for k in names}
pairs = [("l025_200", "l025_225"), ("l025_225", "l025_250"), ("l025_250", "l025_275"), ("l025_275", "l025_300"), ("l05_250", "l05_275"), ("l05_275", "l05_300")]
cross = [("l025_250", "l05_250"), ("l025_275", "l05_275"), ("l025_300", "l05_300")]
def churn(a, b):
    a, b = seg[a], seg[b]; return {"c2w": int(((a == 1) & (b == 0)).sum()), "w2c": int(((a == 0) & (b == 1)).sum()), "flips": int((a != b).sum()), "net": int(b.sum() - a.sum()), "churn_rate": float((a != b).mean())}
out["churn_adjacent"] = {f"{a}->{b}": churn(a, b) for a, b in pairs}; out["churn_cross_arm"] = {f"{a}|{b}": churn(a, b) for a, b in cross}
fl = [out["churn_adjacent"][k]["flips"] for k in out["churn_adjacent"]]; nets = [out["churn_adjacent"][k]["net"] for k in out["churn_adjacent"]]
F_ = float(np.mean(fl))
out["churn_model_i"] = {"mean_flips": F_, "sd_adjacent_diff": sqrt(F_), "implied_single_sd": sqrt(F_ / 2), "rms_observed_net": float(np.sqrt(np.mean(np.square(nets)))), "nets": nets,
                        "legacy200": {"implied_single_sd": NF["churn_model_i"]["implied_single_checkpoint_sd"], "rms_net": NF["churn_model_i"]["rms_observed_adjacent_net"], "sd_adjacent_diff": NF["churn_model_i"]["sd_of_adjacent_difference"]}}
def within(keys):
    A = np.stack([seg[k].reshape(-1) for k in keys]); k = len(keys); p = A.mean(0)
    var_item = p * (1 - p) * k / (k - 1); return float(var_item.sum())
out["within_item_var"] = {}
for lab, keys in (("l025_225_300_k4", ["l025_225", "l025_250", "l025_275", "l025_300"]), ("l025_200_300_k5", ["l025_200", "l025_225", "l025_250", "l025_275", "l025_300"]), ("l05_k3", ["l05_250", "l05_275", "l05_300"]), ("all8_k8", names)):
    v = within(keys); out["within_item_var"][lab] = {"var": v, "sd": sqrt(v)}
out["binomial_sd_at_observed_p"] = {k: sqrt(600 * (seg[k].mean()) * (1 - seg[k].mean())) for k in names}
# scaling check vs prediction (sqrt(3) x legacy)
legacy_sd = {"churn_i": NF["churn_model_i"]["implied_single_checkpoint_sd"], "within_all8": NF["churn_model_ii_within_item_variance"]["all_8_obs_k8_sd"]}
pred_lo, pred_hi = PRE["noise_math"]["predictions_registered_before_scoring"]["single_checkpoint_sd_segments_range"]
out["scaling_check"] = {"predicted_single_sd_range_600": [pred_lo, pred_hi], "observed_churn_i": out["churn_model_i"]["implied_single_sd"], "observed_within_all8": out["within_item_var"]["all8_k8"]["sd"],
                        "ratio_observed_to_legacy_times_sqrt3": {"churn_i": out["churn_model_i"]["implied_single_sd"] / (legacy_sd["churn_i"] * sqrt(3)), "within_all8": out["within_item_var"]["all8_k8"]["sd"] / (legacy_sd["within_all8"] * sqrt(3))},
                        "within_prediction_range": bool(pred_lo - 0.5 <= out["within_item_var"]["all8_k8"]["sd"] <= pred_hi + 0.5)}
# row ICC (one-way ANOVA estimator, m=5) and cluster bootstrap
def icc(a):
    m = a.shape[1]; rm = a.mean(1); gm = a.mean(); msb = m * ((rm - gm) ** 2).sum() / (a.shape[0] - 1); msw = ((a - rm[:, None]) ** 2).sum() / (a.shape[0] * (m - 1)); return float((msb - msw) / (msb + (m - 1) * msw))
out["icc_by_row"] = {k: icc(seg[k]) for k in names}; out["icc_mean"] = float(np.mean(list(out["icc_by_row"].values())))
rng = np.random.default_rng(20261111); B = 20000; idx = rng.integers(0, 120, size=(B, 120))
def bootsum(a): return a.sum(1)[idx].sum(1)
out["cluster_boot_single_se"] = {k: float(bootsum(seg[k]).std(ddof=1)) for k in names}
def mean3(arm): return sum(seg[f"{arm}_{c}"] for c in (250, 275, 300)) / 3.0
m25, m05 = mean3("l025"), mean3("l05")
out["mean3"] = {}
for nm, a in (("l025", m25), ("l05", m05)):
    b = bootsum(a); sing = np.mean([out["cluster_boot_single_se"][f"{nm}_{c}"] for c in (250, 275, 300)])
    out["mean3"][nm] = {"mean_S": float(a.sum()), "mean_pct": float(100 * a.sum() / 600), "cluster_boot_se": float(b.std(ddof=1)), "single_ckpt_cluster_se_mean": float(sing), "ratio_mean3_to_single": float(b.std(ddof=1) / sing),
                        "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "legacy200_cluster_boot_se": NF["mean_of_3_checkpoints"][f"{nm}_250_275_300"]["cluster_bootstrap_se"],
                        "ratio_new_to_legacy_times_sqrt3": float(b.std(ddof=1) / (NF["mean_of_3_checkpoints"][f"{nm}_250_275_300"]["cluster_bootstrap_se"] * sqrt(3)))}
d = bootsum(m05 - m25); out["paired_l05_minus_l025_mean3"] = {"diff_segments": float((m05 - m25).sum()), "diff_pp": float(100 * (m05 - m25).sum() / 600), "cluster_boot_se": float(d.std(ddof=1)), "z": float((m05 - m25).sum() / d.std(ddof=1)), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "frac_boot_le_0": float((d <= 0).mean())}
base = (seg["l025_200"] + seg["l025_225"]) / 2.0
for nm, a in (("l025", m25), ("l05", m05)):
    dd = bootsum(a - base); out[f"{nm}_mean3_minus_mean_200_225"] = {"diff_segments": float((a - base).sum()), "diff_pp": float(100 * (a - base).sum() / 600), "cluster_boot_se": float(dd.std(ddof=1)), "z": float((a - base).sum() / dd.std(ddof=1)), "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))], "baseline_mean_200_225": float(base.sum())}
# block-level coherence: do the three 200-blocks move together between adjacent checkpoints? (shared model-level component)
blocks = lambda k: np.array([seg[k][i*40:(i+1)*40].sum() for i in range(3)])
dn = np.array([blocks(b) - blocks(a) for a, b in pairs])      # [6,3]
cors = [float(np.corrcoef(dn[:, i], dn[:, j])[0, 1]) for i, j in ((0, 1), (0, 2), (1, 2))]
out["block_coherence"] = {"adjacent_block_net_changes": dn.tolist(), "pairwise_corr_of_net_changes_across_6_intervals": cors, "mean_corr": float(np.mean(cors)),
                          "note": "positive mean corr = a model-level component moves all blocks together; ~0 = item-level noise (n=6 intervals, very noisy)"}
# MDE from observed SDs
Z = 1.6449 + 0.8416
sd_single = out["within_item_var"]["all8_k8"]["sd"]; sd_m3 = max(out["mean3"]["l025"]["cluster_boot_se"], out["mean3"]["l05"]["cluster_boot_se"])
out["mde_vs_floor_observed"] = {"single_checkpoint_within_item_sd": {"segments": Z * sd_single, "pp": 100 * Z * sd_single / 600}, "single_checkpoint_churn_i": {"segments": Z * out["churn_model_i"]["implied_single_sd"], "pp": 100 * Z * out["churn_model_i"]["implied_single_sd"] / 600},
                                "mean_of_3_cluster_boot_max": {"segments": Z * sd_m3, "pp": 100 * Z * sd_m3 / 600}, "mean_of_3_cluster_boot_min": {"segments": Z * min(out["mean3"]["l025"]["cluster_boot_se"], out["mean3"]["l05"]["cluster_boot_se"]), "pp": 100 * Z * min(out["mean3"]["l025"]["cluster_boot_se"], out["mean3"]["l05"]["cluster_boot_se"]) / 600}}
# pre-registered gate applied to the two observed runs (non-gating illustration)
out["gate_on_observed"] = {"l025": g.evaluate({c: out["counts"][f"l025_{c}"]["new600"] for c in (250, 275, 300)}), "l05": g.evaluate({c: out["counts"][f"l05_{c}"]["new600"] for c in (250, 275, 300)})}
# one-checkpoint pass rates of the old-style rule on the observed sequence, for context
out["every_ckpt_ge_480"] = {"l025": [out["counts"][f"l025_{c}"]["new600"] >= 480 for c in (250, 275, 300)], "l05": [out["counts"][f"l05_{c}"]["new600"] >= 480 for c in (250, 275, 300)]}
# legacy vs new difficulty
out["legacy_vs_new_pct"] = {k: [out["counts"][k]["legacy_pct"], out["counts"][k]["pct"]] for k in names}
json.dump(out, open("/workspace/forge/vyra-y72heldout-20261111/score/y72_v2_noise_floor.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in out if k not in ("churn_adjacent", "icc_by_row")}, indent=1)[:9000])
