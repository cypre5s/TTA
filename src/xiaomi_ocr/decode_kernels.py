"""Decode-only kernels for the local BF16 Xiaomi model, leaving prefill untouched."""

from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _norm(
    X,
    W,
    G,
    Y,
    D: tl.constexpr,
    EPS: tl.constexpr,
    GATED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    active = col < D
    x = tl.load(X + row * D + col, mask=active, other=0).to(tl.float32)
    w = tl.load(W + col, mask=active, other=0).to(tl.float32)
    v = tl.sum(x * x, axis=0) / D
    y = x * tl.rsqrt(v + EPS)
    if GATED:
        # The reference casts normalized activations and the scale product to BF16.
        y = (y.to(tl.bfloat16).to(tl.float32) * w).to(tl.bfloat16).to(tl.float32)
        g = tl.load(G + row * D + col, mask=active, other=0).to(tl.float32)
        y *= g * tl.sigmoid(g)
    else:
        y *= 1.0 + w
    tl.store(Y + row * D + col, y, mask=active)


class DecodeNorm(nn.Module):
    def __init__(self, original, gated):
        super().__init__()
        self.original = original
        self.gated = gated
        self.eps = original.variance_epsilon if gated else original.eps

    def forward(self, x, gate=None):
        d = x.shape[-1]
        rows = x.numel() // d
        if rows > 256 or x.dtype != torch.bfloat16 or not x.is_contiguous():
            return self.original(x, gate) if self.gated else self.original(x)
        if self.gated and (gate is None or not gate.is_contiguous()):
            return self.original(x, gate)
        out = torch.empty_like(x)
        _norm[(rows,)](
            x,
            self.original.weight,
            gate if self.gated else x,
            out,
            d,
            self.eps,
            self.gated,
            triton.next_power_of_2(d),
            enable_fp_fusion=False,
        )
        return out


class DecodeDeltaNet(nn.Module):
    def __init__(self, original, recurrence):
        super().__init__()
        self.original = original
        self.recurrence = recurrence
        parts = [
            original.in_proj_qkv,
            original.in_proj_z,
            original.in_proj_b,
            original.in_proj_a,
        ]
        if any(p.bias is not None for p in parts):
            raise ValueError("Only bias-free delta projections are supported")
        self.register_buffer("weights", torch.cat([p.weight.detach() for p in parts]))
        self.widths = [p.out_features for p in parts]
        self.register_buffer("decay", -original.A_log.detach().float().exp())

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q

        m = self.original
        if (
            hidden_states.shape[1] != 1
            or cache_params is None
            or not cache_params.has_previous_state(m.layer_idx)
            or cache_params.layers[m.layer_idx].record_past
            or m.num_k_heads != m.num_v_heads
        ):
            return m(
                hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
                **kwargs,
            )
        hidden_states = q.apply_mask_to_padding_states(hidden_states, attention_mask)
        bsz = hidden_states.shape[0]
        projected = nn.functional.linear(hidden_states, self.weights)
        mixed, z, b, a = projected.split(self.widths, dim=-1)
        state = cache_params.layers[m.layer_idx]
        mixed = q.causal_conv1d_update(
            mixed.transpose(1, 2),
            state.conv_states[0],
            m.conv1d.weight.squeeze(1),
            m.conv1d.bias,
            m.activation,
        ).transpose(1, 2)
        query, key, value = mixed.split([m.key_dim, m.key_dim, m.value_dim], dim=-1)
        query = query.reshape(bsz, 1, m.num_k_heads, m.head_k_dim)
        key = key.reshape_as(query)
        value = value.reshape(bsz, 1, m.num_v_heads, m.head_v_dim)
        decay = self.decay * nn.functional.softplus(a.float() + m.dt_bias)
        kwargs.pop("cu_seq_lens_q", None)
        core, latest = self.recurrence(
            query,
            key,
            value,
            g=decay,
            beta=b.sigmoid(),
            initial_state=state.recurrent_states[0],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            **kwargs,
        )
        cache_params.update_recurrent_state(latest, m.layer_idx)
        y = m.norm(core.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim))
        return m.out_proj(y.reshape(bsz, 1, -1))


def fused_decode_delta(
    query,
    key,
    value,
    *,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel,
    **kwargs,
):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    if use_qk_l2norm_in_kernel:
        query, key = q.l2norm(query), q.l2norm(key)
    return fused_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=False,
    )


@contextmanager
def accelerated_decode(model, mode):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q

    supported = {
        "norm",
        "projections",
        "norm_projections",
        "norm_projections_recurrent",
    }
    if mode not in supported:
        raise ValueError(mode)
    with ExitStack() as stack:
        # Install onto individual layers. Preserve the upstream prefill and visual encoder.
        if "norm" in mode:
            for module in list(model.model.language_model.modules()):
                for key, child in list(module.named_children()):
                    if isinstance(child, (q.Qwen3_5RMSNorm, q.Qwen3_5RMSNormGated)):
                        stack.enter_context(
                            patch.object(
                                module,
                                key,
                                DecodeNorm(
                                    child, isinstance(child, q.Qwen3_5RMSNormGated)
                                ),
                            )
                        )
        if "projections" in mode:
            step = (
                fused_decode_delta
                if "recurrent" in mode
                else q.torch_recurrent_gated_delta_rule
            )
            for layer in model.model.language_model.layers:
                if hasattr(layer, "linear_attn"):
                    stack.enter_context(
                        patch.object(
                            layer,
                            "linear_attn",
                            DecodeDeltaNet(layer.linear_attn, step),
                        )
                    )
        yield
