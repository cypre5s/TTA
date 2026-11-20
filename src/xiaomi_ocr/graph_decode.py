"""Replay the 18 fixed-shape linear-attention decoder layers on one CUDA stream."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch
import torch
from torch import nn


class GraphedLayer(nn.Module):
    def __init__(self, original, pool):
        super().__init__()
        self.original = original
        self.pool = pool
        self.cache = self.graph = self.static_input = self.result = None

    def reset(self):
        if self.graph is not None:
            self.graph.reset()
        self.cache = self.graph = self.static_input = self.result = None

    def forward(self, hidden_states, **kwargs):
        cache = kwargs.get("past_key_values")
        # The linear-attention block consumes no positional tensors. Decode-only masks
        # are normally None. Keep the upstream path for any other mask or shape.
        if (
            hidden_states.shape[1] != 1
            or cache is None
            or kwargs.get("attention_mask") is not None
        ):
            self.reset()
            return self.original(hidden_states, **kwargs)
        if self.cache is not cache:
            self.reset()
            self.cache = cache
            return self.original(
                hidden_states, **kwargs
            )  # one real step warms the kernels
        if self.graph is None:
            self.static_input = torch.empty_like(hidden_states)
            self.static_input.copy_(hidden_states)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, pool=self.pool):
                self.result = self.original(self.static_input, **kwargs)
        else:
            self.static_input.copy_(hidden_states)
        self.graph.replay()
        # The shared pool can reuse storage in the next layer.
        return self.result.clone()


@contextmanager
def graph_linear_layers(model):
    layers = model.model.language_model.layers
    pool = torch.cuda.graph_pool_handle()
    installed = []
    with ExitStack() as stack:
        try:
            for i, layer in enumerate(list(layers)):
                if layer.block_type == "linear_attention":
                    wrapper = GraphedLayer(layer, pool)
                    stack.enter_context(patch.dict(layers._modules, {str(i): wrapper}))
                    installed.append(wrapper)
            yield
        finally:
            for wrapper in installed:
                wrapper.reset()


def in_unmasked_decode(x, kwargs):
    return (
        x.shape[1] == 1
        and kwargs.get("past_key_values") is not None
        and kwargs.get("attention_mask") is None
    )


class LinearStack(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x


class StackedLeader(nn.Module):
    def __init__(self, group, pool):
        super().__init__()
        self.engine = GraphedLayer(LinearStack(group), pool)

    def forward(self, hidden_states, **kwargs):
        if in_unmasked_decode(hidden_states, kwargs):
            return self.engine(hidden_states, **kwargs)
        self.engine.reset()
        return self.engine.original.layers[0](hidden_states, **kwargs)


class StackedFollower(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, hidden_states, **kwargs):
        if in_unmasked_decode(hidden_states, kwargs):
            return hidden_states
        return self.original(hidden_states, **kwargs)


@contextmanager
def graph_linear_stacks(model):
    # The six full-attention layers separate six runs of linear-attention blocks.
    # For one-token decode there is no RoPE or padding-mask dependence within a run.
    # Replaying three blocks together also removes two temporary activation clones.
    from itertools import groupby

    layers = model.model.language_model.layers
    original = list(layers)
    pool = torch.cuda.graph_pool_handle()
    leaders = []
    try:
        for linear, group in groupby(
            enumerate(original),
            key=lambda item: item[1].block_type == "linear_attention",
        ):
            if not linear:
                continue
            items = list(group)
            index, _ = items[0]
            leader = StackedLeader([layer for _, layer in items], pool)
            leaders.append(leader)
            layers[index] = leader
            for i, layer in items[1:]:
                layers[i] = StackedFollower(layer)
        yield
    finally:
        for leader in leaders:
            leader.engine.reset()
        for i, original_layer in enumerate(original):
            layers[i] = original_layer
