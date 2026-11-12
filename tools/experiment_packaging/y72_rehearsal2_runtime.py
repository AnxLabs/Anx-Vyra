"""Run-time glue for Y72 rehearsal2 300 -> 400 (prereg/y72_rehearsal2.json): the gate history only (same as cont400).

`history` turns the notebook's HIST into the integer inputs of y72_rehearsal2_gates.evaluate (gate checkpoints only; 325 is
observation only) and refuses unexpected denominators. The 600-segment set itself is loaded by y72fix2_runtime, unchanged.
"""
import y72_rehearsal2_gates as gates
import y72fix2_runtime as y2rt

KEY = y2rt.KEY


def history(HIST):
    out = {}
    for u, rec in HIST.items():
        if u not in gates.CHECKPOINTS:
            continue
        c, x = rec["counts"], rec["x64"]["heldout"]
        want = ((c[KEY]["segments"], 600), (x["segments"], 240), (x["tokens"], 1920), (c["X32_unseen"]["segments"], 320), (c["X109_unseen"]["segments"], 120))
        if any(a != b for a, b in want):
            raise ValueError("DENOMINATOR_MISMATCH")
        out[u] = {"y600": c[KEY]["segments_exact"], "x64_seg": x["segments_exact"], "x64_tokens": x["tokens_correct"],
                  "x32": c["X32_unseen"]["segments_exact"], "x109": c["X109_unseen"]["segments_exact"]}
    return out
