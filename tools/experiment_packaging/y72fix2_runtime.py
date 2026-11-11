"""Run-time glue for the Y72 retention fix 2 notebook (prereg/y72fix2.json): the 600-segment set, the gate history and the logged norm.

No training logic lives here. `load_y600` verifies the registered text set (canonical sha256 + payload-list sha256, fail closed) and returns
(row, mask) pairs scored with the notebook's own teacher-forced segment rule. `history` turns the notebook's HIST into the integer inputs of
y72fix2_gates.evaluate and refuses unexpected denominators.
"""
import math

import torch

import y72_heldout_v2 as heldout

KEY = "Y72v2_600"
SET_PATH = "/kaggle/working/y72_heldout_v2_set.json"


def load_y600(path=SET_PATH):
    s = heldout.verify_text_set(path)
    rows = torch.frombuffer(bytearray(bytes.fromhex(s["rows_int16le_hex"])), dtype=torch.int16).view(heldout.N_ROWS, 512).clone()
    mask = torch.frombuffer(bytearray(bytes.fromhex(s["mask_int16le_hex"])), dtype=torch.int16).view(heldout.N_ROWS, 510).clone()
    return list(zip(rows, mask))


def history(HIST):
    out = {}
    for u, rec in HIST.items():
        if u not in heldout.CHECKPOINTS:
            continue
        c, x = rec["counts"], rec["x64"]["heldout"]
        want = ((c[KEY]["segments"], 600), (x["segments"], 240), (x["tokens"], 1920), (c["X32_unseen"]["segments"], 320), (c["X109_unseen"]["segments"], 120))
        if any(a != b for a, b in want):
            raise ValueError("DENOMINATOR_MISMATCH")
        out[u] = {"y600": c[KEY]["segments_exact"], "x64_seg": x["segments_exact"], "x64_tokens": x["tokens_correct"],
                  "x32": c["X32_unseen"]["segments_exact"], "x109": c["X109_unseen"]["segments_exact"]}
    return out


def preclip_norm(rec, lam):
    """Norm of the UNclipped weighted sum gx + lam*gy, from the per-task record (observation only; nothing is clipped with it)."""
    p, q = rec["norm_gx"] ** 2, rec["norm_gy"] ** 2
    r = 0.0 if rec["cos_gx_gy"] is None else rec["cos_gx_gy"] * math.sqrt(p * q)
    return math.sqrt(max(p + 2 * lam * r + lam * lam * q, 0.0))


def compact(rec):
    return {k: (None if v is None else round(float(v), 6)) if not isinstance(v, str) else v for k, v in rec.items()}
