"""Local inference-only experiments for Qwen3.5 MLP and one-token convolution."""

from contextlib import contextmanager
import torch
from torch import nn
import triton
import triton.language as tl


class PackedMLP(nn.Module):
    def __init__(self, original):
        super().__init__()
        if original.gate_proj.bias is not None or original.up_proj.bias is not None:
            raise ValueError("Expected the original bias-free Qwen3.5 projections")
        # Keep output order gate, up. The persisted checkpoint stays untouched.
        self.register_buffer(
            "weight",
            torch.cat(
                [original.gate_proj.weight.detach(), original.up_proj.weight.detach()]
            ),
        )
        self.down_proj = original.down_proj
        self.activation = original.act_fn

    def forward(self, x):
        gate, up = nn.functional.linear(x, self.weight).chunk(2, dim=-1)
        return self.down_proj(self.activation(gate) * up)


@triton.jit
def _step(
    X,
    S,
    W,
    B,
    Y,
    C: tl.constexpr,
    K: tl.constexpr,
    STRIDE_XB: tl.constexpr,
    STRIDE_XC: tl.constexpr,
    STRIDE_SB: tl.constexpr,
    STRIDE_SC: tl.constexpr,
    STRIDE_SK: tl.constexpr,
    STRIDE_WC: tl.constexpr,
    STRIDE_WK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(1)
    c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = c < C
    st = S + batch * STRIDE_SB + c * STRIDE_SC
    x = tl.load(X + batch * STRIDE_XB + c * STRIDE_XC, mask=ok, other=0)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(K - 1):
        old = tl.load(st + (k + 1) * STRIDE_SK, mask=ok, other=0)
        w = tl.load(W + c * STRIDE_WC + k * STRIDE_WK, mask=ok, other=0)
        acc += old.to(tl.float32) * w.to(tl.float32)
        tl.store(st + k * STRIDE_SK, old, mask=ok)
    w = tl.load(W + c * STRIDE_WC + (K - 1) * STRIDE_WK, mask=ok, other=0)
    acc += x.to(tl.float32) * w.to(tl.float32)
    if HAS_BIAS:
        acc += tl.load(B + c, mask=ok, other=0).to(tl.float32)
    # Native depthwise conv materializes the BF16 intermediate before SiLU.
    acc = acc.to(x.dtype).to(tl.float32)
    tl.store(st + (K - 1) * STRIDE_SK, x, mask=ok)
    tl.store(Y + batch * C + c, acc * tl.sigmoid(acc), mask=ok)


@contextmanager
def variant(model, name):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    if name not in {"original", "packed", "packed_conv"}:
        raise ValueError(f"Unknown local inference variant {name!r}")
    layers = model.model.language_model.layers
    originals = []
    native = qwen.causal_conv1d_update

    def conv(x, state, weight, bias, activation):
        if (
            x.shape[-1] != 1
            or activation not in ("silu", "swish")
            or state.shape[-1] != weight.shape[-1]
            or x.dtype != torch.bfloat16
        ):
            return native(x, state, weight, bias, activation)
        b, c, _ = x.shape
        out = torch.empty((b, c, 1), device=x.device, dtype=x.dtype)
        _step[(triton.cdiv(c, 128), b)](
            x,
            state,
            weight,
            bias if bias is not None else x,
            out,
            c,
            weight.shape[-1],
            *x.stride()[:2],
            *state.stride(),
            *weight.stride(),
            bias is not None,
            128,
        )
        return out

    try:
        if "packed" in name:
            for layer in layers:
                originals.append(layer.mlp)
                layer.mlp = PackedMLP(layer.mlp)
        if "conv" in name:
            qwen.causal_conv1d_update = conv
        yield
    finally:
        for layer, orig in zip(layers, originals):
            layer.mlp = orig
        qwen.causal_conv1d_update = native
