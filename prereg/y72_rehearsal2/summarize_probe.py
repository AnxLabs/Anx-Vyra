"""Summary of signal_probe.json: medians, ratios F200/REP40, X-cost per Y-benefit. Design proxy Y200 only (the 600-set is not used)."""
import json, statistics as st
D = json.load(open("signal_probe.json")); out = {}
for u, r in D.items():
    b = r["batches"]; row = {"ce_mean": {k: round(v, 4) for k, v in r["ce_mean"].items()},
        "F200_over_Y200_ce": round(r["ce_mean"]["F200"] / r["ce_mean"]["Y200"], 3), "REP40_over_Y200_ce": round(r["ce_mean"]["REP40"] / r["ce_mean"]["Y200"], 3)}
    for n in ("REP40", "F200"):
        pb = b[n]["per_batch"]
        row[n] = {"contrib_mean": round(b[n]["mean_contrib"], 3), "contrib_median": round(st.median(x["contrib"] for x in pb), 3),
                  "contrib_lt_0.1": sum(x["contrib"] < 0.1 for x in pb), "clip_bind_frac": b[n]["clip_bind_frac"], "loss_median": round(st.median(x["loss"] for x in pb), 4),
                  "cosP_Y200": round(b[n]["mean_cosP_Y200"], 3), "cosP_X64ho": round(b[n]["mean_cosP_X64ho"], 3),
                  "x_cost_per_y_benefit": round(b[n]["mean_dL_X64ho"] / -b[n]["mean_dL_Y200"], 2)}
    row["REPtop_cosP_Y200"] = round(b["REPtop"]["mean_cosP_Y200"], 3)
    row["ratio_F200_over_REP40"] = {"Y200_pull": round(b["F200"]["mean_dL_Y200"] / b["REP40"]["mean_dL_Y200"], 2), "X64_cost": round(b["F200"]["mean_dL_X64ho"] / b["REP40"]["mean_dL_X64ho"], 2),
                                    "contrib": round(b["F200"]["mean_contrib"] / b["REP40"]["mean_contrib"], 2), "cosP_Y200": round(b["F200"]["mean_cosP_Y200"] / b["REP40"]["mean_cosP_Y200"], 2)}
    out[u] = row; print(u, json.dumps(row))
json.dump(out, open("signal_probe_summary.json", "w"), indent=1)
