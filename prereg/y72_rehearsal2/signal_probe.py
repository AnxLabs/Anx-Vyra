"""Candidate rehearsal signals measured at the control (cont400) checkpoints, CPU, read-only restore, nothing trained.
Design held-out proxy = Y200 (the original 40-row/200-segment Y72 unseen set, NOT gated). The 600-set is not touched here.
Candidates per update s in the window after checkpoint u (s = u..u+24, 2 rows each):
  REP40  : the control's rows idx=(2s+j)%40 (what actually ran)
  F200   : fresh pool rows 2(s-300)+j (each pool row used exactly once in 301..400)
  REPtop : the 2 highest-loss REP40 rows at checkpoint u (error-weighted pick, fixed for the window)
Per 2-row batch: grad norm |g|, clipped replay contribution c=min(0.25|g|,0.5) (scale a=c/|g|), and the first-order change of
Y200 / X64-heldout mean loss from the replay part of one step under the checkpoint's Adam preconditioner P:
  dL = -LR * g_eval . P (a g)   (negative = loss falls). Also cos(P g, g_eval)."""
import json, sys, time, torch
sys.path.insert(0, "."); import lab3
sys.path.insert(0, "/workspace/forge/vyra-cont400-swing-diag-20261112/diag"); import importlib
G = importlib.import_module("grads")  if False else None
from pathlib import Path
POOL = torch.load(Path(__file__).resolve().parent.parent / "pool/y72_fresh_pool_F200.pt", map_location="cpu", weights_only=False)
PR, PM = POOL["rows"].long(), POOL["mask"].long()
B1, B2, EPS = lab3.rv.B1, lab3.rv.B2, lab3.rv.EPS

def grad(m, rows, masks, bs=8):
    ps = list(m.parameters()); acc = [torch.zeros_like(p) for p in ps]; n = rows.shape[0]; tot = 0.0; m.train()
    for i in range(0, n, bs):
        x = rows[i:i + bs]; k = masks[i:i + bs]
        l = lab3.ce_rows(m(x), x, k).sum() / n; tot += float(l.detach())
        for a, g in zip(acc, torch.autograd.grad(l, ps, allow_unused=True)):
            if g is not None: a.add_(g)
    return torch.cat([a.reshape(-1) for a in acc]), tot

def precond(m, opt):
    out = []
    for p in m.parameters():
        st = opt.state[id(p)]; v = st["exp_avg_sq"]
        if st["expert_steps"] is None:
            b2 = 1 - B2 ** max(st["step"], 1); out.append((1.0 / ((v / b2).sqrt() + EPS)).reshape(-1))
        else:
            steps = st["expert_steps"].clamp_min(1).double(); sh = (-1,) + (1,) * (v.dim() - 1)
            b2 = (1 - B2 ** steps).float().view(sh); out.append((1.0 / ((v / b2).sqrt() + EPS)).reshape(-1))
    return torch.cat(out)

@torch.no_grad()
def row_ce(m, rows, masks, bs=8):
    m.eval(); out = []
    for i in range(0, rows.shape[0], bs): out += lab3.ce_rows(m(rows[i:i + bs]), rows[i:i + bs], masks[i:i + bs]).tolist()
    return out

Y2r = torch.stack([a for a, _ in lab3.SETS["Y200"]]).long(); Y2m = torch.stack([b for _, b in lab3.SETS["Y200"]]).long()
X6r = torch.stack([a for a, _ in lab3.SETS["X64"]]).long(); X6m = torch.stack([b for _, b in lab3.SETS["X64"]]).long()
RR, RM = lab3.RY["rows"].long(), lab3.RY["mask"].long()
res = {}
for u in (300, 325, 350, 375, 400):
    t = time.time(); m, opt, _ = lab3.load(u); P = precond(m, opt); LR = lab3.rv.LR
    ce = {"REP40": row_ce(m, RR, RM), "F200": row_ce(m, PR, PM), "Y200": row_ce(m, Y2r, Y2m)}
    gY, lY = grad(m, Y2r, Y2m); gX, lX = grad(m, X6r, X6m)
    top = sorted(range(40), key=lambda i: -ce["REP40"][i])[:2]
    steps = range(u, min(u + 25, 400)) if u < 400 else range(375, 400)   # at 400: re-use the last window's indices as a probe
    rec = {"ce_mean": {k: sum(v) / len(v) for k, v in ce.items()}, "ce_rows": ce, "Y200_loss": lY, "X64ho_loss": lX, "REPtop_rows": top, "batches": {}}
    for name in ("REP40", "F200", "REPtop"):
        L = []
        for s in steps:
            if name == "REP40": idx = [(s * 2 + j) % 40 for j in range(2)]; r, k = RR[idx], RM[idx]
            elif name == "F200": idx = [2 * (s - 300) + j for j in range(2)]; r, k = PR[idx], PM[idx]
            else: idx = top; r, k = RR[idx], RM[idx]
            g, l = grad(m, r, k); n = float(g.norm()); c = min(0.25 * n, 0.5); a = c / max(n, 1e-30); pg = P * g
            L.append({"s": s + 1, "rows": idx, "loss": l, "norm": n, "contrib": c, "clip_binds": 0.25 * n > 0.5,
                      "dL_Y200": float(-LR * a * (gY @ pg)), "dL_X64ho": float(-LR * a * (gX @ pg)),
                      "cosP_Y200": float((pg @ gY) / (pg.norm() * gY.norm())), "cosP_X64ho": float((pg @ gX) / (pg.norm() * gX.norm()))})
            if name == "REPtop": break        # fixed rows: one batch is the whole window's signal at this state
        agg = lambda key: sum(x[key] for x in L) / len(L)
        rec["batches"][name] = {"n": len(L), "mean_loss": agg("loss"), "mean_norm": agg("norm"), "mean_contrib": agg("contrib"), "clip_bind_frac": agg("clip_binds"),
                                "mean_dL_Y200": agg("dL_Y200"), "mean_dL_X64ho": agg("dL_X64ho"), "mean_cosP_Y200": agg("cosP_Y200"), "mean_cosP_X64ho": agg("cosP_X64ho"),
                                "frac_dL_Y200_negative": sum(x["dL_Y200"] < 0 for x in L) / len(L), "per_batch": L}
    res[u] = rec; json.dump(res, open("signal_probe.json", "w"), indent=1)
    print(u, round(time.time() - t), "s CE", {k: round(v, 4) for k, v in rec["ce_mean"].items()}, flush=True)
    for name, b in rec["batches"].items():
        print("  ", name, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in b.items() if k != "per_batch"}, flush=True)
