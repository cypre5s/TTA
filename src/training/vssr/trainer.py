"""One explicit training step, suitable for a smoke test or an external loop."""

from contextlib import nullcontext

import torch

from .losses import compute_losses


def move_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def train_step(model, batch, optimizer, bf16=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    batch = move_batch(batch, device)
    context = torch.autocast(device.type, dtype=torch.bfloat16) if bf16 else nullcontext()
    with context:
        out = model(batch)
    losses = compute_losses(out, batch, model.config)
    if not all(torch.isfinite(x).all() for x in losses.values()):
        raise FloatingPointError("Non-finite VSSR loss")
    losses["total"].backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError("Missing or non-finite VSSR gradients")
    optimizer.step()
    return {k: float(v.detach()) for k, v in losses.items()}
