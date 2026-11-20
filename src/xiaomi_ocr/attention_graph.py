"""Capture fixed-shape arithmetic around native variable-length full attention."""

from contextlib import contextmanager
import torch
from torch import nn

from src.xiaomi_ocr.graph_decode import graph_linear_stacks


class StaticFunction:
    def __init__(self):
        self.signature = self.inputs = self.result = self.graph = None

    def reset(self):
        if self.graph is not None:
            self.graph.reset()
        self.signature = self.inputs = self.result = self.graph = None

    def __call__(self, fn, *args):
        key = tuple((x.shape, x.dtype, x.device) for x in args)
        if key != self.signature:
            self.reset()
            self.signature = key
            return fn(*args)
        if self.graph is None:
            self.inputs = [torch.empty_like(x) for x in args]
            for target, source in zip(self.inputs, args):
                target.copy_(source)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.result = fn(*self.inputs)
        else:
            for target, source in zip(self.inputs, args):
                target.copy_(source)
        self.graph.replay()
        return self.result


class SplitAttention(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.before = StaticFunction()
        self.after = StaticFunction()

    def pre(self, hidden, cos, sin):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        layer = self.original
        m = layer.self_attn
        x = layer.input_layernorm(hidden)
        shape = (*x.shape[:-1], -1, m.head_dim)
        q, gate = m.q_proj(x).view(*x.shape[:-1], -1, m.head_dim * 2).chunk(2, dim=-1)
        q = m.q_norm(q.reshape(shape)).transpose(1, 2)
        k = m.k_norm(m.k_proj(x).view(shape)).transpose(1, 2)
        v = m.v_proj(x).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v, gate.reshape(*x.shape[:-1], -1)

    def post(self, residual, attention, gate):
        layer = self.original
        attention = attention.reshape(*residual.shape[:-1], -1).contiguous()
        y = layer.self_attn.o_proj(attention * torch.sigmoid(gate))
        x = residual + y
        return x + layer.mlp(layer.post_attention_layernorm(x))

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kwargs,
    ):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q

        if (
            hidden_states.shape[1] != 1
            or past_key_values is None
            or self.training
            or not past_key_values.has_previous_state()
        ):
            return self.original(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
        m = self.original.self_attn
        qs, ks, vs, gate = self.before(self.pre, hidden_states, *position_embeddings)
        ks, vs = past_key_values.update(ks, vs, m.layer_idx)
        fn = q.ALL_ATTENTION_FUNCTIONS.get_interface(
            m.config._attn_implementation, q.eager_attention_forward
        )
        attention, _ = fn(
            m,
            qs,
            ks,
            vs,
            attention_mask,
            dropout=0.0,
            scaling=m.scaling,
            position_ids=position_ids,
            **kwargs,
        )
        return self.after(self.post, hidden_states, attention, gate)

    def reset(self):
        self.before.reset()
        self.after.reset()


@contextmanager
def graph_hybrid_decode(model):
    layers = model.model.language_model.layers
    full = [(i, l) for i, l in enumerate(layers) if l.block_type == "full_attention"]
    with graph_linear_stacks(model):
        try:
            for i, layer in full:
                layers[i] = SplitAttention(layer).eval()
            yield
        finally:
            for i, layer in full:
                layers[i].reset()
                layers[i] = layer
