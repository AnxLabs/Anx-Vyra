"""The fresh, once-only Y72 rehearsal pool F200 of prereg/y72_rehearsal2.json (rule vyra-y72-rehearsal2-fresh-v1).

Notebook asset and builder-side check. Fail closed:
* `verify_text_set` recomputes the registered canonical sha256 (int16 little-endian rows, then mask) and the payload-list sha256 of the
  committed text form prereg/y72_rehearsal2/y72_fresh_pool_F200_set.json; any other file is refused.
* `indices(s)` is the registered schedule: the update s+1 (s = 300..399) uses pool rows 2*(s-300) and 2*(s-300)+1, so every one of the
  200 rows is presented exactly once in updates 301..400. Any other step is refused.
* `wrap_sampler_record` relabels the replay rule in the sampler records of the checkpoints written after 300 (installed only after the
  restore from checkpoint_300, whose saved record keeps the 40-row rule it was written with).
* `manipulation_checks` computes the registered NON-GATING checks M1-M3 from the run's own per-update record. They never change the verdict.
"""
import hashlib
import json
import statistics
from pathlib import Path

RULE_ID = "vyra-y72-rehearsal2-fresh-v1"
POOL_ASSET = "y72_fresh_pool_F200_set.json"
SCHEMA = "vyra-y72-rehearsal2-fresh-pool-F200-text-1"
TEXT_SHA256 = "604bba8a47c81bd3edc0f347487d908bc54769f48981fb0dac1f9252c96c3b45"   # file sha256 registered in prereg design_evidence.pool_text_form_sha256
CANONICAL_SHA256 = "d4315c338f44855e1162099c4a42792f0368c60a854f6d047930fa865ef976d8"
PAYLOAD_LIST_SHA256 = "c860f59bf25862b706ba36a6277ff45310ab65c4d1130c75945e8e92c43dc4b8"
SEEDS = (20264672, 20264772, 20264872, 20264972, 20265072)
POOL_ROWS = 200
ROW_LEN, MASK_LEN = 512, 510
COPIES = 5
RESUME_UPDATE = 300
FINAL_UPDATE = 400
ROWS_PER_UPDATE = 2
REPLAY_RULE = "F200 idx=2*(s-300)+j (update s+1, s=300..399, j=0,1): every pool row exactly once"
WINDOWS = ((301, 325), (326, 350), (351, 375), (376, 400))
M1_MIN_PULL = 0.10        # median alpha_y*|g_y| per window (non-gating)
M2_MIN_LOSS = 0.005       # median replay loss per window (non-gating)


class PoolError(ValueError):
    pass


def _canon(rows_hex, mask_hex):
    return hashlib.sha256(bytes.fromhex(rows_hex) + bytes.fromhex(mask_hex)).hexdigest()


def verify_text_set(path):
    """Check the text form; return its parsed dict. Raises PoolError on any mismatch."""
    path = Path(path)
    if not path.is_file():
        raise PoolError("POOL_MISSING")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TEXT_SHA256:
        raise PoolError("POOL_TEXT_SHA256")
    d = json.loads(raw)
    if d.get("schema") != SCHEMA or d.get("distance") != 72 or d.get("copies") != COPIES or list(d.get("seeds", [])) != list(SEEDS):
        raise PoolError("POOL_SCHEMA")
    if d.get("rows_shape") != [POOL_ROWS, ROW_LEN] or d.get("mask_shape") != [POOL_ROWS, MASK_LEN]:
        raise PoolError("POOL_SHAPE")
    if len(d["rows_int16le_hex"]) != POOL_ROWS * ROW_LEN * 4 or len(d["mask_int16le_hex"]) != POOL_ROWS * MASK_LEN * 4:
        raise PoolError("POOL_SHAPE")
    if _canon(d["rows_int16le_hex"], d["mask_int16le_hex"]) != CANONICAL_SHA256 or d.get("canonical_rows_mask_sha256") != CANONICAL_SHA256:
        raise PoolError("POOL_CANONICAL")
    p = d.get("payloads", [])
    if len(p) != POOL_ROWS * COPIES or len(set(p)) != len(p) or hashlib.sha256("".join(p).encode()).hexdigest() != PAYLOAD_LIST_SHA256:
        raise PoolError("POOL_PAYLOADS")
    return d


def load_pool(path):
    """Verified pool as int16 tensors (rows [200,512], mask [200,510]) + info for result.json."""
    import torch
    d = verify_text_set(path)
    rows = torch.frombuffer(bytearray(bytes.fromhex(d["rows_int16le_hex"])), dtype=torch.int16).view(POOL_ROWS, ROW_LEN).clone()
    mask = torch.frombuffer(bytearray(bytes.fromhex(d["mask_int16le_hex"])), dtype=torch.int16).view(POOL_ROWS, MASK_LEN).clone()
    if int(mask.sum()) != POOL_ROWS * 8 * COPIES:
        raise PoolError("POOL_MASK_COUNT")
    info = {"rule": RULE_ID, "text_sha256": TEXT_SHA256, "canonical_sha256": CANONICAL_SHA256, "payload_list_sha256": PAYLOAD_LIST_SHA256,
            "rows": POOL_ROWS, "seeds": list(SEEDS), "replay_rule": REPLAY_RULE}
    return {"rows": rows, "mask": mask, "payloads": list(d["payloads"]), "info": info}


def indices(s):
    """Pool rows for the update s+1 (s is the 0-based step index of the training loop)."""
    if isinstance(s, bool) or not isinstance(s, int) or not RESUME_UPDATE <= s < FINAL_UPDATE:
        raise PoolError("POOL_STEP_RANGE %r" % (s,))
    base = ROWS_PER_UPDATE * (s - RESUME_UPDATE)
    return [base + j for j in range(ROWS_PER_UPDATE)]


def schedule():
    return [(s + 1, indices(s)) for s in range(RESUME_UPDATE, FINAL_UPDATE)]


def check_once(rows_used):
    flat = [int(i) for idx in rows_used for i in idx]
    return {"presented": len(flat), "distinct": len(set(flat)), "no_repeat": len(flat) == len(set(flat)),
            "complete_once": sorted(flat) == list(range(POOL_ROWS)), "matches_schedule": [list(map(int, x)) for x in rows_used] == [ix for _, ix in schedule()][:len(rows_used)]}


def wrap_sampler_record(orig):
    def sampler_record(update, plan, plan_module, replay_batch, replay_rows=40):
        rec = orig(update, plan, plan_module, replay_batch, replay_rows)
        if update <= RESUME_UPDATE:
            return rec
        rec = dict(rec)
        rec["replay_rule"] = REPLAY_RULE
        rec["replay_next_idx"] = indices(update) if update < FINAL_UPDATE else []
        rec["replay_source"] = {"pool": "F200", "canonical_sha256": CANONICAL_SHA256, "rule": RULE_ID}
        return rec
    sampler_record.rehearsal2 = True
    return sampler_record


def manipulation_checks(task_clip, losses, rows_used, first_update=RESUME_UPDATE + 1):
    """Registered non-gating checks from the run's own record. task_clip[i]/losses[i] belong to update first_update+i."""
    if not (len(task_clip) == len(losses) == len(rows_used)):
        raise PoolError("CHECK_LENGTHS")
    win = []
    for a, b in WINDOWS:
        ii = [i for i in range(len(losses)) if a <= first_update + i <= b]
        pulls = [float(task_clip[i]["alpha_y"]) * float(task_clip[i]["norm_gy"]) for i in ii]
        ly = [float(losses[i][1]) for i in ii]
        rec = {"window": [a, b], "updates": len(ii), "complete": len(ii) == b - a + 1}
        if ii:
            rec.update({"median_replay_pull": statistics.median(pulls), "median_replay_loss": statistics.median(ly)})
            rec["M1_pull_ge_0.10"] = rec["median_replay_pull"] >= M1_MIN_PULL
            rec["M2_loss_ge_0.005"] = rec["median_replay_loss"] >= M2_MIN_LOSS
        win.append(rec)
    once = check_once(rows_used)
    full = all(w["complete"] for w in win)
    return {"rule": RULE_ID, "nongating": True, "windows": win,
            "M1_no_starved_window": all(w.get("M1_pull_ge_0.10", False) for w in win) if full else None,
            "M2_not_memorized": all(w.get("M2_loss_ge_0.005", False) for w in win) if full else None,
            "M3_presentation_once": once["complete_once"] and once["matches_schedule"] if full else (once["no_repeat"] and once["matches_schedule"]),
            "once": once, "complete_run": full}
