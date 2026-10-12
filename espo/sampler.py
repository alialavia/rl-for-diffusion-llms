"""Masked-diffusion sampling (LLaDA-style low-confidence remasking, semi-AR blocks)."""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .elbo import LogitsFn


def transfer_schedule(n_masked: torch.Tensor, steps: int) -> torch.Tensor:
    """Tokens to unmask per step: ``[B, steps]``, as even as possible, larger first."""
    n = n_masked.view(-1, 1).long()
    base = n // steps
    rem = n % steps
    idx = torch.arange(steps, device=n.device).view(1, -1)
    return base + (idx < rem).long()


def _uniform(shape, generator, device) -> torch.Tensor:
    gen_device = generator.device if generator is not None else device
    return torch.rand(shape, generator=generator, dtype=torch.float64, device=gen_device).to(device)


def _sample_tokens(logits: torch.Tensor, temperature: float, generator) -> torch.Tensor:
    """Gumbel-max sampling from softmax(logits / temperature) in float64; argmax at 0."""
    if temperature == 0.0:
        return logits.argmax(-1)
    x = logits.to(torch.float64)
    u = _uniform(x.shape, generator, x.device)
    gumbel = -torch.log(-torch.log(u.clamp_min(1e-300)))
    return (x / temperature + gumbel).argmax(-1)


@torch.no_grad()
def generate(logits_fn: LogitsFn, prompt_ids: torch.Tensor, prompt_mask: Optional[torch.Tensor],
             gen_length: int, block_length: int, steps: int, mask_id: int,
             temperature: float = 0.0, remasking: str = "low_confidence",
             generator: Optional[torch.Generator] = None,
             trace: Optional[Callable[[int, int, torch.Tensor], None]] = None) -> torch.Tensor:
    """Generate ``gen_length`` tokens after ``prompt_ids``; returns ``[B, gen_length]``.

    The completion is split into blocks of ``block_length`` decoded left to right; each
    block gets ``steps / num_blocks`` denoising steps, and at each step the
    highest-confidence predictions among the block's masked positions are committed.
    ``remasking``: "low_confidence" (prob of sampled token), "entropy" (negative
    entropy, Dream-style) or "random".
    ``trace(step, block, completion)`` is called after every step (for testing/logging).
    """
    if block_length <= 0 or gen_length % block_length != 0:
        raise ValueError(f"gen_length={gen_length} must be a positive multiple of block_length={block_length}")
    num_blocks = gen_length // block_length
    if steps <= 0 or steps % num_blocks != 0:
        raise ValueError(f"steps={steps} must be a positive multiple of num_blocks={num_blocks}")
    if remasking not in ("low_confidence", "entropy", "random"):
        raise ValueError(f"unknown remasking {remasking!r}")
    steps_per_block = steps // num_blocks

    B, P = prompt_ids.shape
    device = prompt_ids.device
    x = torch.cat([prompt_ids, torch.full((B, gen_length), mask_id, dtype=prompt_ids.dtype,
                                          device=device)], dim=1)
    attn = None
    if prompt_mask is not None:
        attn = torch.cat([prompt_mask, torch.ones(B, gen_length, dtype=prompt_mask.dtype,
                                                  device=device)], dim=1)
    step = 0
    for blk in range(num_blocks):
        lo, hi = P + blk * block_length, P + (blk + 1) * block_length
        n_masked = (x[:, lo:hi] == mask_id).sum(1)
        schedule = transfer_schedule(n_masked, steps_per_block)
        for i in range(steps_per_block):
            logits = logits_fn(x, attn)[:, lo:hi]  # only the current block can change
            cur = x[:, lo:hi]
            eligible = cur == mask_id
            x0 = _sample_tokens(logits, temperature, generator)
            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                conf = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "entropy":
                logp = F.log_softmax(logits.to(torch.float64), dim=-1)
                conf = (logp.exp() * logp).sum(-1)  # negative entropy
            else:
                conf = _uniform(x0.shape, generator, device)
            conf = torch.where(eligible, conf, torch.full_like(conf, -float("inf")))
            k = schedule[:, i]
            transfer = torch.zeros_like(eligible)
            kmax = int(k.max())
            if kmax > 0:
                top = conf.topk(kmax, dim=-1).indices  # [B, kmax]
                keep = torch.arange(kmax, device=device).view(1, -1) < k.view(-1, 1)
                transfer.scatter_(1, top, keep)
                transfer &= eligible
            x[:, lo:hi] = torch.where(transfer, x0, cur)
            if trace is not None:
                trace(step, blk, x[:, P:])
            step += 1
    return x[:, P:]
