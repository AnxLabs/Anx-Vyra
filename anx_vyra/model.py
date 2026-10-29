"""ANX-VYRA: Versatile Yet Resource-Aware Architecture - reference model.

This is the exact architecture used in every reported VYRA experiment. It is a 16.4M-parameter
research model (8.17M active per token) with hybrid attention mixers, top-2-of-8 sparse experts,
absolute RoPE and a tied embedding/output projection. It exists to study retrieval behaviour and
sparse-routing / optimizer interactions, not to serve traffic and not as a product model.

ANX-VYRA is an experimental research architecture. It is NOT the future Anx-U:50B model and is
not a preliminary version of it.

Written by AnxLabs. Apache-2.0.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# ---- fixed architecture constants (do not tune casually: reported results depend on them) ----
D = 512
N_LAYERS = 8
N_Q = 8
N_KV = 2
HEAD_DIM = 64
N_EXPERTS = 8
TOP_K = 2
EXPERT_M = 112
FFN_M = 340
GDN_HEADS = 4
GDN_DK = 64
GDN_DV = 64
CONV_K = 4
VOCAB = 259
BOS, EOS, PAD = 256, 257, 258
SEQ = 512
N_TRAIN = 1000
N_VALID = 64
N_NEEDLE = 8
INIT_STD = 0.02

# window of each local-attention layer; None = full causal attention
ARM_WINDOWS = {"R": [128, None, 128, None, 128, None, 128, None]}

def init_normal(param: torch.Tensor, seed: int) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    with torch.no_grad():
        param.normal_(0.0, INIT_STD, generator=generator)


def rope(x: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    half = x.shape[-1] // 2
    freq = theta ** (-torch.arange(half, dtype=x.dtype) / half)
    angles = torch.arange(x.shape[-2], dtype=x.dtype)[:, None] * freq[None, :]
    cos = angles.cos().to(x.dtype)
    sin = angles.sin().to(x.dtype)
    a = x[..., :half]
    b = x[..., half:]
    return torch.cat((a * cos - b * sin, a * sin + b * cos), dim=-1)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def bit_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    gamma = weight.abs().mean().clamp_min(1e-8)
    hat = torch.round(weight / gamma).clamp(-1, 1)
    quantized = weight + (hat * gamma - weight).detach()
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    x_hat = torch.round(x / scale).clamp(-128, 127) * scale
    x_q = x + (x_hat - x).detach()
    return F.linear(x_q, quantized)
class Attention(torch.nn.Module):
    def __init__(self, layer: int, window: int | None, ternary: bool) -> None:
        super().__init__()
        self.window = window
        self.ternary = ternary
        self.q = torch.nn.Parameter(torch.empty(N_Q * HEAD_DIM, D))
        self.k = torch.nn.Parameter(torch.empty(N_KV * HEAD_DIM, D))
        self.v = torch.nn.Parameter(torch.empty(N_KV * HEAD_DIM, D))
        self.o = torch.nn.Parameter(torch.empty(D, N_Q * HEAD_DIM))
        self.sink = torch.nn.Parameter(torch.zeros(N_Q))
        init_normal(self.q, 5000 + layer * 10 + 1)
        init_normal(self.k, 5000 + layer * 10 + 2)
        init_normal(self.v, 5000 + layer * 10 + 3)
        init_normal(self.o, 5000 + layer * 10 + 4)

    def project(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.ternary:
            return bit_linear(x, weight)
        return F.linear(x, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, steps, _ = x.shape
        q = rope(self.project(x, self.q).view(bsz, steps, N_Q, HEAD_DIM).transpose(1, 2))
        k = rope(self.project(x, self.k).view(bsz, steps, N_KV, HEAD_DIM).transpose(1, 2))
        v = self.project(x, self.v).view(bsz, steps, N_KV, HEAD_DIM).transpose(1, 2)
        k = k.repeat_interleave(N_Q // N_KV, dim=1)
        v = v.repeat_interleave(N_Q // N_KV, dim=1)
        scale = HEAD_DIM**-0.5
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        causal = torch.ones(steps, steps, dtype=torch.bool)
        causal = torch.triu(causal, diagonal=1)
        if self.window is not None:
            pos = torch.arange(steps)
            causal = causal | ((pos[None, :] < pos[:, None] - self.window))
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        sink = self.sink.view(1, N_Q, 1, 1)
        denom = torch.logsumexp(torch.cat([scores, sink.expand(bsz, N_Q, steps, 1)], dim=-1), dim=-1)
        weights = torch.exp(scores - denom.unsqueeze(-1))
        out = torch.matmul(weights, v).transpose(1, 2).reshape(bsz, steps, D)
        return self.project(out, self.o)


class GatedDelta(torch.nn.Module):
    def __init__(self, layer: int) -> None:
        super().__init__()
        width = GDN_HEADS * GDN_DK
        self.q = torch.nn.Parameter(torch.empty(width, D))
        self.k = torch.nn.Parameter(torch.empty(width, D))
        self.v = torch.nn.Parameter(torch.empty(GDN_HEADS * GDN_DV, D))
        self.o = torch.nn.Parameter(torch.empty(D, GDN_HEADS * GDN_DV))
        self.g = torch.nn.Parameter(torch.empty(GDN_HEADS * GDN_DV, D))
        self.a = torch.nn.Parameter(torch.empty(GDN_HEADS, D))
        self.b = torch.nn.Parameter(torch.empty(GDN_HEADS, D))
        self.conv_q = torch.nn.Parameter(torch.empty(width, 1, CONV_K))
        self.conv_k = torch.nn.Parameter(torch.empty(width, 1, CONV_K))
        self.conv_v = torch.nn.Parameter(torch.empty(GDN_HEADS * GDN_DV, 1, CONV_K))
        self.a_log = torch.nn.Parameter(torch.zeros(GDN_HEADS))
        self.dt_bias = torch.nn.Parameter(torch.zeros(GDN_HEADS))
        self.norm = torch.nn.Parameter(torch.ones(GDN_HEADS * GDN_DV))
        for slot, tensor in enumerate((self.q, self.k, self.v, self.o, self.g, self.a, self.b, self.conv_q, self.conv_k, self.conv_v)):
            init_normal(tensor, 6000 + layer * 10 + slot)

    def conv(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        channels = x.shape[-1]
        flat = x.transpose(1, 2)
        padded = F.pad(flat, (CONV_K - 1, 0))
        mixed = F.conv1d(padded, weight, groups=channels)
        return mixed.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = F.silu(self.conv(F.linear(x, self.q), self.conv_q))
        k = F.silu(self.conv(F.linear(x, self.k), self.conv_k))
        v = F.silu(self.conv(F.linear(x, self.v), self.conv_v))
        bsz, steps, _ = q.shape
        q = q.view(bsz, steps, GDN_HEADS, GDN_DK)
        k = k.view(bsz, steps, GDN_HEADS, GDN_DK)
        v = v.view(bsz, steps, GDN_HEADS, GDN_DV)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        alpha_in = F.softplus(F.linear(x, self.a) + self.dt_bias)
        alpha = torch.exp(-torch.exp(self.a_log) * alpha_in).permute(0, 2, 1)
        beta = torch.sigmoid(F.linear(x, self.b)).permute(0, 2, 1)
        state = x.new_zeros(bsz, GDN_HEADS, GDN_DV, GDN_DK)
        outputs = []
        for t in range(steps):
            kt = k[:, :, t, :]
            vt = v[:, :, t, :]
            at = alpha[:, :, t]
            bt = beta[:, :, t]
            sk = torch.einsum("bhvd,bhd->bhv", state, kt)
            state = at[:, :, None, None] * (state - bt[:, :, None, None] * sk[:, :, :, None] * kt[:, :, None, :])
            state = state + bt[:, :, None, None] * vt[:, :, :, None] * kt[:, :, None, :]
            outputs.append(torch.einsum("bhvd,bhd->bhv", state, q[:, :, t, :]))
        mixed = torch.stack(outputs, dim=2).permute(0, 2, 1, 3).reshape(bsz, steps, GDN_HEADS * GDN_DV)
        gated = rms_norm(mixed, self.norm) * torch.sigmoid(F.linear(x, self.g))
        return F.linear(gated, self.o)


class Experts(torch.nn.Module):
    def __init__(self, layer: int) -> None:
        super().__init__()
        self.gate = torch.nn.Parameter(torch.empty(N_EXPERTS, EXPERT_M, D))
        self.up = torch.nn.Parameter(torch.empty(N_EXPERTS, EXPERT_M, D))
        self.down = torch.nn.Parameter(torch.empty(N_EXPERTS, D, EXPERT_M))
        self.router = torch.nn.Parameter(torch.empty(N_EXPERTS, D))
        init_normal(self.gate, 3000 + layer * 10 + 1)
        init_normal(self.up, 3000 + layer * 10 + 2)
        init_normal(self.down, 3000 + layer * 10 + 3)
        init_normal(self.router, 4000 + layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = F.linear(x, self.router)
        values, index = torch.topk(logits, TOP_K, dim=-1)
        weights = torch.softmax(values, dim=-1)
        bsz, steps, _ = x.shape
        flat = x.reshape(bsz * steps, D)
        chosen = index.reshape(bsz * steps, TOP_K)
        mix = flat.new_zeros(flat.shape)
        for slot in range(TOP_K):
            for expert in range(N_EXPERTS):
                mask = chosen[:, slot] == expert
                if not bool(mask.any()):
                    continue
                hidden = F.silu(F.linear(flat[mask], self.gate[expert])) * F.linear(flat[mask], self.up[expert])
                mix[mask] = mix[mask] + weights.reshape(bsz * steps, TOP_K)[mask, slot, None] * F.linear(hidden, self.down[expert])
        return mix.view(bsz, steps, D)


class DenseReLU2(torch.nn.Module):
    def __init__(self, layer: int) -> None:
        super().__init__()
        self.up = torch.nn.Parameter(torch.empty(FFN_M, D))
        self.down = torch.nn.Parameter(torch.empty(D, FFN_M))
        init_normal(self.up, 7000 + layer * 10 + 1)
        init_normal(self.down, 7000 + layer * 10 + 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(bit_linear(x, self.up)).square()
        return bit_linear(hidden, self.down)


class Block(torch.nn.Module):
    def __init__(self, layer: int, arm: str) -> None:
        super().__init__()
        self.norm1 = torch.nn.Parameter(torch.ones(D))
        self.norm2 = torch.nn.Parameter(torch.ones(D))
        if arm == "H" and layer % 4 != 3:
            self.mixer = GatedDelta(layer)
        else:
            window = None if (arm == "H" or layer % 2 == 1) else 128
            self.mixer = Attention(layer, window, ternary=(arm == "T"))
        self.ffn = DenseReLU2(layer) if arm == "T" else Experts(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(rms_norm(x, self.norm1))
        x = x + self.ffn(rms_norm(x, self.norm2))
        return x


class ToyLM(torch.nn.Module):
    def __init__(self, arm: str) -> None:
        super().__init__()
        self.arm = arm
        self.embed = torch.nn.Parameter(torch.empty(VOCAB, D))
        init_normal(self.embed, 1000)
        self.blocks = torch.nn.ModuleList(Block(layer, arm) for layer in range(N_LAYERS))
        self.final_norm = torch.nn.Parameter(torch.ones(D))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = F.embedding(tokens, self.embed)
        for block in self.blocks:
            x = block(x)
        x = rms_norm(x, self.final_norm)
        return F.linear(x, self.embed)


def counts(model: ToyLM) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    expert = 0
    for module in model.modules():
        if isinstance(module, Experts):
            expert += module.gate.numel() + module.up.numel() + module.down.numel()
    active = total - expert + expert * TOP_K // N_EXPERTS
    return {"total": total, "active": active, "expert_catalog": expert}


def linear_muls(model: ToyLM) -> int:
    muls = 0
    for name, param in model.named_parameters():
        if param.ndim == 2 and "embed" not in name and "norm" not in name and "sink" not in name and "a_log" not in name and "dt_bias" not in name:
            if "gate" in name or "up" in name or "down" in name:
                muls += param.shape[-1] * param.shape[-2] * TOP_K // N_EXPERTS
            elif name.endswith(".a") or name.endswith(".b"):
                muls += param.numel()
            else:
                muls += param.numel()
    muls += VOCAB * D
    return muls


# ---- stable expert routing (the form used in every reported run) -------------------------------
# torch.topk breaks ties by index on CPU but not guaranteed-identically elsewhere, which made
# routing depend on kernel numerics. This forward sorts explicitly with stable=True, so the same
# logits always select the same experts, and records which experts were used so the slicing
# optimizer can step only those slices.


def stable_experts_forward(self: Experts, x: torch.Tensor) -> torch.Tensor:
    logits = F.linear(x, self.router)
    index = torch.argsort(logits, dim=-1, descending=True, stable=True)[..., :TOP_K]
    values = torch.gather(logits, -1, index)
    weights = torch.softmax(values, dim=-1)
    bsz, steps, _ = x.shape
    flat = x.reshape(bsz * steps, D)
    chosen = index.reshape(bsz * steps, TOP_K)
    mix = flat.new_zeros(flat.shape)
    selected: set[int] = set()
    slot_weights = weights.reshape(bsz * steps, TOP_K)
    for slot in range(TOP_K):
        for expert in range(N_EXPERTS):
            mask = chosen[:, slot] == expert
            if not bool(mask.any()):
                continue
            selected.add(expert)
            hidden = F.silu(F.linear(flat[mask], self.gate[expert])) * F.linear(
                flat[mask], self.up[expert]
            )
            mix[mask] = mix[mask] + slot_weights[mask, slot, None] * F.linear(
                hidden, self.down[expert]
            )
    self._selected = selected
    load = getattr(self, "_load", None)
    binc = torch.bincount(chosen.detach().reshape(-1), minlength=N_EXPERTS).cpu()
    self._load = binc.clone() if load is None else load + binc
    return mix.view(bsz, steps, D)


def reset_expert_load(model: ToyLM) -> None:
    for module in model.modules():
        if isinstance(module, Experts):
            module._load = torch.zeros(N_EXPERTS, dtype=torch.long)


def expert_selection(model: ToyLM) -> dict[int, list[int]]:
    """Which experts each FFN actually used on the last forward pass."""
    selected: dict[int, list[int]] = {}
    for module in model.modules():
        if not isinstance(module, Experts):
            continue
        chosen = getattr(module, "_selected", None)
        if not chosen:
            raise RuntimeError("an expert layer recorded no selection")
        for name in ("gate", "up", "down"):
            selected[id(getattr(module, name))] = sorted(chosen)
    return selected


def build_arm(name: str = "R") -> ToyLM:
    """Construct an arm with stable routing installed."""
    if name not in ARM_WINDOWS:
        raise ValueError(f"unknown arm {name!r}; available: {sorted(ARM_WINDOWS)}")
    model = ToyLM(name)
    Experts.forward = stable_experts_forward
    windows = [getattr(b.mixer, "window", None) for b in model.blocks]
    if windows != ARM_WINDOWS[name]:
        raise RuntimeError(f"window layout drifted: {windows}")
    return model
