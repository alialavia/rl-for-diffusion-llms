"""Monte-Carlo ELBO estimator for masked diffusion LMs (paper Eq. 3, 32-34).

Conventions
-----------
* A completion has ``L`` tokens. A mask is a bool tensor over the completion only;
  prompt tokens are never masked.
* ``masks`` passed to :func:`sequence_elbo` have shape ``[M, B, L]`` (M Monte-Carlo
  samples). Re-using the same masks for theta, theta_old and the reference model is the
  "antithetic sampling" of the paper.
* ``logits_fn(input_ids, attention_mask) -> logits[B, T, V]``.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

LogitsFn = Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]


def stratified_counts(k: int, batch: int, L: int) -> torch.Tensor:
    """Numbers of masked tokens for ``batch`` rows, stratified over ``{0..L}``.

    Slot ``b`` gets ``(k + floor(b (L+1) / batch)) mod (L+1)``. With ``k`` uniform on
    ``{0..L}`` each slot's count is exactly uniform (integer offsets keep the map
    ``k -> count`` a bijection), and counts are spread evenly across the batch.
    """
    offsets = torch.div(torch.arange(batch) * (L + 1), batch, rounding_mode="floor")
    return (int(k) + offsets) % (L + 1)


def random_subsets(counts: torch.Tensor, L: int, generator: Optional[torch.Generator] = None,
                   device=None) -> torch.Tensor:
    """Bool masks ``[N, L]`` where row ``n`` is a uniformly random ``counts[n]``-subset."""
    n = counts.shape[0]
    noise = torch.rand(n, L, generator=generator)
    ranks = noise.argsort(dim=1).argsort(dim=1)  # uniform random permutation ranks
    masks = ranks < counts.view(-1, 1)
    return masks.to(device) if device is not None else masks


def sample_masks(num_mc: int, batch: int, L: int, generator: Optional[torch.Generator] = None,
                 device=None) -> torch.Tensor:
    """Sample ``[num_mc, batch, L]`` masks: l ~ U{0..L} (stratified over the batch),
    positions uniformly random given l."""
    out = []
    for _ in range(num_mc):
        k = int(torch.randint(0, L + 1, (1,), generator=generator))
        counts = stratified_counts(k, batch, L)
        out.append(random_subsets(counts, L, generator=generator))
    masks = torch.stack(out) if out else torch.zeros(0, batch, L, dtype=torch.bool)
    return masks.to(device) if device is not None else masks


def apply_mask(completion_ids: torch.Tensor, mask: torch.Tensor, mask_id: int) -> torch.Tensor:
    """Replace masked completion positions with ``mask_id``."""
    return torch.where(mask, torch.full_like(completion_ids, mask_id), completion_ids)


def _masked_token_logps(logits_fn: LogitsFn, prompt_ids, prompt_mask, completion_ids, mask,
                        mask_id) -> torch.Tensor:
    """Sum over masked completion positions of log p(y_i | noisy input), per row.

    Rows are processed in one forward pass; only logits at masked positions are
    gathered (memory: the vocabulary is large for real models).
    """
    P = prompt_ids.shape[1]
    noisy = torch.cat([prompt_ids, apply_mask(completion_ids, mask, mask_id)], dim=1)
    attn = None
    if prompt_mask is not None:
        attn = torch.cat([prompt_mask, torch.ones_like(completion_ids, dtype=prompt_mask.dtype)], 1)
    logits = logits_fn(noisy, attn)[:, P:]
    rows, cols = mask.nonzero(as_tuple=True)
    out_dtype = torch.promote_types(logits.dtype, torch.float32)
    per_row = torch.zeros(completion_ids.shape[0], dtype=out_dtype, device=logits.device)
    if rows.numel() == 0:
        return per_row + logits.sum() * 0  # keep graph connectivity
    sel = logits[rows, cols].to(out_dtype)
    logp = -F.cross_entropy(sel, completion_ids[rows, cols], reduction="none")
    return per_row.index_add(0, rows, logp)


def sequence_elbo(logits_fn: LogitsFn, prompt_ids: torch.Tensor, prompt_mask: Optional[torch.Tensor],
                  completion_ids: torch.Tensor, masks: torch.Tensor, mask_id: int,
                  coupled: bool = True) -> torch.Tensor:
    """MC estimate of the sequence ELBO ``L(y|x)`` for each row; returns ``[B]``.

    For each MC mask with ``l`` masked tokens:
      plain   : (L+1)/l * sum_{masked} log p(y_i | y_l)               (0 if l == 0)
      coupled : 0.5 * [ above + (L+1)/(L-l) * sum_{unmasked} log p(y_i | y~_l) ]
    where ``y~_l`` masks the complementary positions. Both are unbiased (Eq. 32-34).
    """
    M, B, L = masks.shape
    if completion_ids.shape != (B, L):
        raise ValueError(f"completion_ids {tuple(completion_ids.shape)} vs masks {tuple(masks.shape)}")
    masks = masks.to(completion_ids.device)
    total = None
    for m in range(M):
        mask = masks[m]
        l = mask.sum(-1)
        w = (L + 1) / l.clamp(min=1).to(torch.float64)
        est = _masked_token_logps(logits_fn, prompt_ids, prompt_mask, completion_ids, mask, mask_id)
        est = est * w.to(est.dtype) * (l > 0)
        if coupled:
            wc = (L + 1) / (L - l).clamp(min=1).to(torch.float64)
            est_c = _masked_token_logps(logits_fn, prompt_ids, prompt_mask, completion_ids, ~mask,
                                        mask_id)
            est = 0.5 * (est + est_c * wc.to(est_c.dtype) * (l < L))
        total = est if total is None else total + est
    return total / M
