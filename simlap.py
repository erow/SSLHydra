"""Distributed InfoNCE / SupCon / SimLAP / X-CLR losses."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        if not dist.is_available() or not dist.is_initialized():
            return tensor
        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor.contiguous())
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        if not dist.is_available() or not dist.is_initialized():
            return grad
        world = dist.get_world_size()
        rank = dist.get_rank()
        grad = grad.contiguous()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        return grad.chunk(world, dim=0)[rank]


def gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    return _Gather.apply(tensor)


@torch.no_grad()
def gather_labels(labels: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return labels
    gathered = [torch.empty_like(labels) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, labels.contiguous())
    return torch.cat(gathered, dim=0)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """SimCLR NT-Xent over the global 2N views."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([gather_with_grad(z1), gather_with_grad(z2)], dim=0)
    n = z.size(0)
    logits = (z @ z.T) / temperature
    logits.fill_diagonal_(float("-inf"))
    half = n // 2
    labels = torch.cat(
        [torch.arange(half, n, device=z.device), torch.arange(half, device=z.device)]
    )
    return F.cross_entropy(logits, labels)

def xclr(
    z1: torch.Tensor,
    z2: torch.Tensor,
    y: torch.Tensor,
    class_sim: torch.Tensor,
    temperature: float,
    tau_s: float,
) -> torch.Tensor:
    """X-CLR (Sobal et al., ICLR 2025): InfoNCE with a class-name similarity graph.

    Target G is class-wise (paper A.7): precomputed C×C caption cosines, then
    ``G_ij = class_sim[y_i, y_j]``. Same-class pairs (including the two views)
    get 1; related classes get (0, 1). Predicted p is view-wise ``z z^T``, the
    quantity being trained — same as SimCLR / SupCon. Diagonal is dropped so
    both distributions sit on the other 2N-1 views (A.9). ``tau_s -> 0``
    recovers SupCon (identical class captions), not per-sample SimCLR.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([gather_with_grad(z1), gather_with_grad(z2)], dim=0)
    y = torch.cat([gather_labels(y), gather_labels(y)], dim=0)
    self_mask = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    logits = ((z @ z.T) / temperature).masked_fill(self_mask, float("-inf"))
    # class-wise lookup, not sample embeds: every (dog, cat) pair shares one G
    graph = class_sim.to(device=z.device, dtype=logits.dtype)[y[:, None], y[None, :]]
    graph = (graph / tau_s).masked_fill(self_mask, float("-inf"))
    log_p = logits.log_softmax(dim=1).masked_fill(self_mask, 0)
    return -(graph.softmax(dim=1) * log_p).sum(dim=1).mean()


def _multipos_ce(logits: torch.Tensor, pos_mask: torch.Tensor, exclude_mask: torch.Tensor) -> torch.Tensor:
    logits = logits - logits.mean(dim=1, keepdim=True)
    sim = logits.exp()
    neg = (sim * (~exclude_mask)).sum(dim=1, keepdim=True)
    return (pos_mask * (torch.log(sim + neg) - logits)).sum() / pos_mask.sum().clamp_min(1)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    torch.manual_seed(0)
    b, d, c = 8, 16, 4
    z1 = F.normalize(torch.randn(b, d), dim=1)
    z2 = F.normalize(z1 + 0.05 * torch.randn(b, d), dim=1)
    # TODO: add test cases