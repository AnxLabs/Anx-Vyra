"""Shared evaluation helper for the tests (kept tiny so the tests stay independent of the CLI)."""
import torch

from anx_vyra.model import SEQ


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
            "record_exact": whole / records if records else None}
