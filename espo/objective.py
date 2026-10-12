"""ESPO objective: group advantages, sequence-level ratio, clipped surrogate, KL."""
from __future__ import annotations

from typing import Optional

import torch


def group_advantages(rewards: torch.Tensor, group_size: int, baseline: str = "loo",
                     scale: bool = False, eps: float = 1e-4) -> torch.Tensor:
    """Group-relative advantages for rewards laid out as consecutive groups of size G.

    baseline="mean": A = r - mean(group)              (paper, Sec. 2.2)
    baseline="loo" : A = r - mean(group without r)    (authors' released code)
    scale=True divides by the group std (+eps).
    """
    G = group_size
    if G < 2 or rewards.numel() % G != 0:
        raise ValueError(f"need group_size >= 2 dividing {rewards.numel()}, got {G}")
    r = rewards.view(-1, G)
    if baseline == "mean":
        b = r.mean(dim=1, keepdim=True)
    elif baseline == "loo":
        b = (r.sum(dim=1, keepdim=True) - r) / (G - 1)
    else:
        raise ValueError(f"unknown baseline {baseline!r}")
    adv = r - b
    if scale:
        adv = adv / (r.std(dim=1, keepdim=True) + eps)
    # exactly zero for constant groups (avoids -0.0 / rounding noise)
    const = (r == r[:, :1]).all(dim=1, keepdim=True)
    adv = torch.where(const, torch.zeros_like(adv), adv)
    return adv.reshape(-1)


def sequence_ratio(elbo: torch.Tensor, old_elbo: torch.Tensor, length: int) -> torch.Tensor:
    """Per-token-normalised sequence ratio rho = exp((L_theta - L_old) / L) (Eq. 8)."""
    return torch.exp((elbo - old_elbo) / length)


def clipped_surrogate(ratio: torch.Tensor, advantages: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Per-sequence PPO surrogate min(rho*A, clip(rho, 1-eps, 1+eps)*A) (to maximise)."""
    return torch.minimum(ratio * advantages, ratio.clamp(1 - epsilon, 1 + epsilon) * advantages)


def kl_estimate(elbo: torch.Tensor, ref_elbo: torch.Tensor, length: int, kind: str = "k2") -> torch.Tensor:
    """Per-sequence KL(pi_theta || pi_ref) estimate from ELBOs, divided by L.

    With D = L_theta - L_ref:  k1 = D,  k2 = D^2/2 (Eq. 10),  k3 = exp(-D) - 1 + D (Eq. 9).
    """
    d = elbo - ref_elbo
    if kind == "k1":
        v = d
    elif kind == "k2":
        v = 0.5 * d * d
    elif kind == "k3":
        v = torch.exp(-d) - 1 + d
    else:
        raise ValueError(f"unknown KL estimator {kind!r}")
    return v / length


def espo_loss(elbo: torch.Tensor, old_elbo: torch.Tensor, ref_elbo: Optional[torch.Tensor],
              advantages: torch.Tensor, length: int, epsilon: float, beta: float, denom: int,
              kl_kind: str = "k2"):
    """Loss (to minimise) for a (micro-)batch: -sum_i J_i/denom + beta * sum_i KL_i/denom.

    ``denom`` is the number of sequences in the full batch so that summing micro-batch
    losses gives the full-batch mean.  Returns ``(loss, stats)``.
    """
    if beta > 0 and ref_elbo is None:
        raise ValueError("beta > 0 requires ref_elbo")
    ratio = sequence_ratio(elbo, old_elbo.detach(), length)
    adv = advantages.to(ratio.dtype)
    surrogate = clipped_surrogate(ratio, adv, epsilon)
    loss = -surrogate.sum() / denom
    if beta > 0:
        kl = kl_estimate(elbo, ref_elbo.detach(), length, kl_kind)
        loss = loss + beta * kl.sum() / denom
    else:
        kl = torch.zeros_like(ratio) if ref_elbo is None else kl_estimate(
            elbo.detach(), ref_elbo.detach(), length, kl_kind)
    with torch.no_grad():
        r = ratio.detach()
        clipped = ((r > 1 + epsilon) & (adv > 0)) | ((r < 1 - epsilon) & (adv < 0))
        stats = {
            "ratio_min": r.min().item() if r.numel() else 1.0,
            "ratio_max": r.max().item() if r.numel() else 1.0,
            "clip_count": clipped.sum().item(),
            "kl_sum": kl.detach().sum().item(),
            "n": r.numel(),
        }
    return loss, stats
