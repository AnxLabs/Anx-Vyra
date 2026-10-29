#!/usr/bin/env python3
"""Reproduce the ANX-VYRA retrieval benchmark: does the model learn a needle at distance D?

Usage
-----
    python benchmarks/reproduce_retrieval.py --distance 32  --steps 200
    python benchmarks/reproduce_retrieval.py --distance 109 --steps 200
    python benchmarks/reproduce_retrieval.py --distance 120 --steps 200   # expected to fail

What it does
------------
Builds the arm-R model from scratch, trains it on one needle-to-query distance for `--steps`
optimizer steps at the published learning rate, and reports *record-exact* accuracy on the 40 rows
it trained on and on 40 rows it has never seen. Record-exact means all 8 payload bytes of a record
are predicted correctly, not per-token accuracy.

Two honest cautions
-------------------
1. From-scratch training is NOT the published setup. The published results are continuations from a
   pretrained checkpoint that is not part of this repository. What this script reproduces is the
   *relative* behaviour that the finding rests on: which distances are learnable and which are not.
   Absolute numbers here will differ from the table in the README, and the README says so.
2. Steps are capped so a laptop run finishes. Use --steps 200 to match the published exposure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anx_vyra import optim                                            # noqa: E402
from anx_vyra.data import build_corpus, build_distance_set            # noqa: E402
from anx_vyra.model import (SEQ, VOCAB, build_arm, expert_selection,  # noqa: E402
                            stable_experts_forward)


def masked_ce(logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on the answer bytes only; everything else is ignored."""
    nll = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), tokens[:, 1:].reshape(-1),
                          reduction="none").view(tokens.shape[0], tokens.shape[1] - 1)[:, :SEQ - 2]
    weight = mask.to(nll.dtype)
    return ((nll * weight).sum(1) / weight.sum(1).clamp_min(1)).mean()


@torch.no_grad()
def evaluate(model, rows, mask, batch=4) -> dict:
    model.eval()
    ok = total = records = whole = 0
    for start in range(0, rows.shape[0], batch):
        x = rows[start:start + batch].long()
        m = mask[start:start + batch].bool()
        pred = model(x)[:, :SEQ - 2].argmax(-1)
        target = x[:, 1:SEQ - 1]
        for row in range(x.shape[0]):
            hit = (pred[row] == target[row])[m[row]]
            ok += int(hit.sum())
            total += int(hit.numel())
            usable = (int(hit.numel()) // 8) * 8
            if usable:
                whole += int(hit[:usable].view(-1, 8).all(1).sum())
                records += usable // 8
    model.train()
    return {"token_acc": ok / total if total else None,
            "record_exact": whole / records if records else None,
            "records": records}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--distance", type=int, default=32, help="needle-to-query distance in tokens")
    p.add_argument("--steps", type=int, default=200, help="optimizer steps")
    p.add_argument("--lr", type=float, default=3e-4, help="learning rate (published: 3e-4)")
    p.add_argument("--rows", type=int, default=40, help="train rows (and same count unseen)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default="", help="optional path for a JSON result")
    args = p.parse_args()

    torch.set_num_threads(args.threads)   # linalg in this model is not safe with many threads
    torch.manual_seed(args.seed)

    # Build the train/unseen pair, keeping payloads disjoint from the pretraining corpus.
    corpus = build_corpus()
    banned = set(bytes(row).replace(b"\x00", b"") for row in [])
    del banned
    ds = build_distance_set(args.distance, args.rows, seed=args.seed * 1000 + args.distance)

    model = build_arm("R")
    optim.LR = float(args.lr)
    assert optim.LR == float(args.lr), "SliceAdam reads the module-level LR"
    opt = optim.make_optimizer(model, lr=args.lr)

    base_train = evaluate(model, ds["rows"], ds["mask"])
    base_unseen = evaluate(model, ds["unseen"], ds["unseen_mask"])
    print(f"distance {args.distance}  filler {ds['geometry']['filler']}  "
          f"records/row {ds['copies']}  lr {args.lr:g}  steps {args.steps}")
    print(f"  untrained   train {base_train['record_exact']:.3f}   "
          f"unseen {base_unseen['record_exact']:.3f}")

    order = list(range(ds["rows"].shape[0]))
    t0 = time.time()
    first_update = None
    for step in range(args.steps):
        optim.LR = float(args.lr)          # re-assert: the optimizer reads this global every step
        assert optim.LR == float(args.lr)
        idx = [order[(step * 4 + k) % len(order)] for k in range(4)]
        x = ds["rows"][idx].long()
        m = ds["mask"][idx]
        model.zero_grad(set_to_none=True)
        loss = masked_ce(model(x), x, m)
        loss.backward()
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss at step {step}")
        grads = [q for q in model.parameters() if q.grad is not None]
        torch.nn.utils.clip_grad_norm_(grads, 1.0)
        if step == 0:
            before = {n: q.detach().clone() for n, q in model.named_parameters() if q.grad is not None}
        opt.step(expert_selection(model))
        if step == 0:
            first_update = max((q.detach() - before[n]).abs().max().item()
                               for n, q in model.named_parameters() if n in before)

    train = evaluate(model, ds["rows"], ds["mask"])
    unseen = evaluate(model, ds["unseen"], ds["unseen_mask"])
    secs = time.time() - t0
    result = {"distance": args.distance, "filler": ds["geometry"]["filler"],
              "records_per_row": ds["copies"], "lr": args.lr, "steps": args.steps,
              "seed": args.seed, "rows": args.rows,
              "first_update_max_abs": first_update,
              "untrained_train": base_train["record_exact"],
              "untrained_unseen": base_unseen["record_exact"],
              "train_record_exact": train["record_exact"],
              "unseen_record_exact": unseen["record_exact"],
              "seconds": round(secs, 1),
              "learned": bool(train["record_exact"] is not None
                              and train["record_exact"] >= 0.5
                              and unseen["record_exact"] is not None
                              and unseen["record_exact"] >= 0.10)}
    print(f"  trained     train {train['record_exact']:.3f}   "
          f"unseen {unseen['record_exact']:.3f}   ({secs:.0f}s, "
          f"first update {first_update:.2e})")
    print(f"  learned: {result['learned']}")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=1)
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
