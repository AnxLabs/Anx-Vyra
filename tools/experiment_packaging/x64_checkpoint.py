"""Checkpoints and X64 byte readouts at updates 175 and 200. Observational only.

Runs after the update's optimizer step and after the existing eval, with the
model in eval mode under no_grad. Restores train/eval mode, Python/torch/CUDA
RNG, and route metadata. Never touches .grad or the optimizer. torch.save only
reads tensors. Any checkpoint write failure is recorded, not raised, so the
training run is never stopped by instrumentation.
"""
import hashlib
import json
import time
from pathlib import Path

import torch

CHECKPOINT_UPDATES = (175, 200)
X64 = 64


def wanted(update_one_indexed):
    return int(update_one_indexed) in CHECKPOINT_UPDATES


def _state_sha(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes() if tensor.dtype != torch.bfloat16 else tensor.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def _opt_state(model, opt):
    """SliceAdam has no state_dict(); key its per-param state by parameter name."""
    names = {id(p): n for n, p in model.named_parameters()}
    return {names.get(k, str(k)): dict(v) for k, v in opt.state.items()}


def train_x64_rows(plan, steps):
    """Unique X64 needle indices presented in updates 1..steps, with first update."""
    first = {}
    for s in range(int(steps)):
        k = int(plan[s]["needles"][X64])
        first.setdefault(k, s + 1)
    return first


def _readout(model, rows, masks, device):
    out = []
    ok = n = seg = segs = 0
    for i in range(0, len(rows), 4):
        x = torch.stack(rows[i:i + 4]).long().to(device)
        mm = torch.stack(masks[i:i + 4]).bool().to(device)
        lg = model(x)[:, :510].float()
        pr = lg.argmax(-1)
        tg = x[:, 1:511]
        for r in range(x.shape[0]):
            sel = mm[r]
            p = pr[r][sel].tolist()
            t = tg[r][sel].tolist()
            o = (pr[r] == tg[r])[sel]
            f = (o.numel() // 8) * 8
            row_seg = int(o[:f].view(-1, 8).all(1).sum()) if f else 0
            ok += int(o.sum()); n += o.numel(); seg += row_seg; segs += f // 8
            out.append({
                "pred_bytes_hex": bytes(v % 256 for v in p).hex(),
                "pred_ids": p if any(v > 255 for v in p) else None,
                "target_bytes_hex": bytes(v % 256 for v in t).hex(),
                "byte_correct": int(o.sum()),
                "bytes": int(o.numel()),
                "segments_exact": row_seg,
                "segments": f // 8,
            })
    return {
        "rows": out,
        "byte_acc": (ok / n) if n else None,
        "exact_segment_acc": (seg / segs) if segs else None,
        "bytes": n,
        "segments": segs,
    }


def dump(model, opt, update, XD, plan, steps, out_dir, diag, eval_record=None):
    t0 = time.time()
    out_dir = Path(out_dir)
    mode = model.training
    rng = diag._rng()
    routes = diag._capture_routes(model)
    device = next(model.parameters()).device
    record = {"update": int(update)}
    try:
        model.eval()
        with torch.no_grad():
            first = train_x64_rows(plan, steps)
            idx = sorted(first)
            train = _readout(model, [XD[X64]["rows"][k] for k in idx], [XD[X64]["mask"][k] for k in idx], device)
            for k, row in zip(idx, train["rows"]):
                row["train_index"] = k
                row["first_presented_update"] = first[k]
                row["presented_by_this_update"] = first[k] <= int(update)
            held = _readout(model, list(XD[X64]["unseen"]), list(XD[X64]["unseen_mask"]), device)
        pred_path = out_dir / f"x64_predictions_{int(update)}.json"
        pred_path.write_text(json.dumps({"update": int(update), "train": train, "heldout": held,
                                         "scope": "teacher-forced argmax over masked answer positions, same as ev()"}))
        record.update({
            "train_rows": len(idx), "train_byte_acc": train["byte_acc"], "train_exact": train["exact_segment_acc"],
            "heldout_rows": len(held["rows"]), "heldout_byte_acc": held["byte_acc"], "heldout_exact": held["exact_segment_acc"],
            "predictions_file": pred_path.name,
        })
        if eval_record is not None and "X64_unseen" in eval_record:
            record["heldout_matches_eval"] = (
                round(held["byte_acc"], 4) == eval_record["X64_unseen"]["token_acc"]
                and round(held["exact_segment_acc"], 4) == eval_record["X64_unseen"]["record_exact"]
            )
    finally:
        diag._restore_rng(rng)
        diag._restore_routes(routes)
        model.train(mode)
    try:
        ck = out_dir / f"checkpoint_{int(update)}.pt"
        torch.save({"update": int(update), "model": model.state_dict(), "optimizer": _opt_state(model, opt),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, str(ck))
        record.update({"checkpoint_file": ck.name, "checkpoint_bytes": ck.stat().st_size,
                       "checkpoint_sha256": hashlib.sha256(ck.read_bytes()).hexdigest(),
                       "model_state_sha256": _state_sha(model)})
    except Exception as exc:  # instrumentation must never stop training
        record["checkpoint_error"] = repr(exc)
    record["dump_s"] = time.time() - t0
    return record
