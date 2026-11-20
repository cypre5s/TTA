"""One explicit training step, suitable for a smoke test or an external loop."""

from contextlib import nullcontext

import torch

from .losses import compute_losses


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=device.type == "cuda") if torch.is_tensor(v) else v
            for k, v in batch.items()}


def batch_losses(model, batch, bf16=False):
    device = batch["images"].device

    def amp():
        return torch.autocast(device.type, dtype=torch.bfloat16) if bf16 else nullcontext()

    def descend(vision, parents):
        with amp():
            return model.hierarchy.decode_level(vision, parents)

    with amp():
        out = model(batch)
    return out, compute_losses(out, batch, model.config, descend if hasattr(model, "hierarchy") else None)


def train_step(model, batch, optimizer, bf16=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    batch = move_batch(batch, device)
    _, losses = batch_losses(model, batch, bf16)
    loss_values = torch.stack(list(losses.values())).detach()
    if not torch.isfinite(loss_values).all().item():
        raise FloatingPointError("Non-finite VSSR loss")
    losses["total"].backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not torch.isfinite(torch.nn.utils.get_total_norm(gradients, norm_type=float("inf"))).item():
        raise FloatingPointError("Missing or non-finite VSSR gradients")
    optimizer.step()
    return dict(zip(losses, loss_values.tolist(), strict=True))
