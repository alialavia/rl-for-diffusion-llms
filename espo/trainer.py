"""ESPO training loop (framework-free; works with DDP, LoRA or a toy model).

One *rollout phase* samples ``prompts_per_rank`` prompts x ``num_generations``
completions per rank, scores them, computes group advantages, samples ``num_iterations``
x ``num_mc`` masks per sequence and pre-computes ``L_old`` and ``L_ref`` on those masks.
It is followed by ``num_iterations`` (= mu) optimizer steps; step ``j`` re-uses masks
``j`` for the current policy (antithetic sampling).

Policy interface (see espo.toy.ToyPolicy, espo.model.HFPolicy):
  logits(ids, attn)            trainable forward (may go through DDP)
  generation_logits(ids, attn) forward used for sampling (no grad)
  reference_logits()           context manager yielding the reference logits_fn
  no_sync()                    context manager to skip DDP all-reduce (accumulation)
  trainable_parameters(), train(), eval(), mask_id

Task interface (see espo.toy.ToyTask, espo.tasks.TextTask):
  sample_prompts(n) -> rows; encode_prompts(rows, device) -> (ids, attn)
  rewards(rows, completion_ids) -> (list[float], list[str] | None)
"""
from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import torch

from .elbo import sample_masks, sequence_elbo
from .objective import espo_loss, group_advantages
from .sampler import generate


@dataclass
class ESPOConfig:
    num_generations: int = 6          # G
    prompts_per_rank: int = 2         # prompts per rollout phase per rank
    micro_batch_size: int = 3         # sequences per forward/backward
    num_iterations: int = 8           # mu: optimizer steps per rollout phase
    num_mc: int = 2                   # M: MC masks per ELBO
    coupled: bool = True              # complementary-mask variance reduction
    epsilon: float = 0.2
    beta: float = 0.01
    kl_estimator: str = "k2"
    baseline: str = "loo"
    scale_rewards: bool = False
    gen_length: int = 256
    block_length: int = 32
    diffusion_steps: int = 128
    temperature: float = 0.9
    remasking: str = "low_confidence"
    generation_batch_size: int = 6
    max_grad_norm: float = 0.2
    seed: int = 42

    def __post_init__(self):
        if self.num_generations < 2:
            raise ValueError("num_generations must be >= 2 for group advantages")
        if self.block_length <= 0 or self.gen_length % self.block_length:
            raise ValueError("gen_length must be a multiple of block_length")
        if self.diffusion_steps % (self.gen_length // self.block_length):
            raise ValueError("diffusion_steps must be a multiple of the number of blocks")
        if self.kl_estimator not in ("k1", "k2", "k3"):
            raise ValueError(f"unknown kl_estimator {self.kl_estimator!r}")
        if self.baseline not in ("mean", "loo"):
            raise ValueError(f"unknown baseline {self.baseline!r}")
        if min(self.prompts_per_rank, self.micro_batch_size, self.num_iterations, self.num_mc) < 1:
            raise ValueError("batch sizes, num_iterations and num_mc must be >= 1")


@dataclass
class Rollout:
    prompt_ids: torch.Tensor          # [N, P]
    prompt_mask: torch.Tensor         # [N, P]
    completion_ids: torch.Tensor      # [N, L]
    rewards: torch.Tensor             # [N]
    advantages: torch.Tensor          # [N]
    masks: torch.Tensor               # [mu, M, N, L]
    old_elbo: Optional[torch.Tensor]  # [mu, N] (None when mu == 1: on-policy)
    ref_elbo: Optional[torch.Tensor]  # [mu, N] (None when beta == 0)
    texts: Optional[list] = None
    stats: dict = field(default_factory=dict)


class ESPOTrainer:
    def __init__(self, policy, task, optimizer, cfg: ESPOConfig, scheduler=None,
                 device: str | torch.device = "cpu", rank: int = 0, world_size: int = 1,
                 all_reduce: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 log_fn: Optional[Callable[[dict], None]] = None):
        self.policy, self.task, self.opt, self.cfg = policy, task, optimizer, cfg
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.rank, self.world_size = rank, world_size
        self.all_reduce = all_reduce  # sums a tensor across ranks (metrics only)
        self.log_fn = log_fn
        self.global_step = 0
        self.num_rollouts = 0
        gen_dev = self.device if self.device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=gen_dev).manual_seed(cfg.seed + 1000 * rank)
        self.mask_generator = torch.Generator().manual_seed(cfg.seed + 1000 * rank + 1)

    # ------------------------------------------------------------------ helpers
    def _chunks(self, n: int):
        b = self.cfg.micro_batch_size
        return [slice(i, min(i + b, n)) for i in range(0, n, b)]

    @torch.no_grad()
    def _elbo_table(self, logits_fn, r_prompt, r_pmask, r_comp, masks) -> torch.Tensor:
        """ELBO for every (iteration, sequence) with the same micro-batching as training."""
        mu, N = masks.shape[0], r_comp.shape[0]
        out = torch.zeros(mu, N, dtype=torch.float32, device=self.device)
        for j in range(mu):
            for s in self._chunks(N):
                out[j, s] = sequence_elbo(logits_fn, r_prompt[s], r_pmask[s], r_comp[s],
                                          masks[j][:, s], self.policy.mask_id,
                                          coupled=self.cfg.coupled).float()
        return out

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def make_rollout(self) -> Rollout:
        cfg, G = self.cfg, self.cfg.num_generations
        t0 = time.time()
        rows = self.task.sample_prompts(cfg.prompts_per_rank)
        rows_rep = [r for r in rows for _ in range(G)]  # groups are contiguous
        prompt_ids, prompt_mask = self.task.encode_prompts(rows_rep, device=self.device)
        self.policy.eval()
        comps = []
        for i in range(0, len(rows_rep), cfg.generation_batch_size):
            s = slice(i, i + cfg.generation_batch_size)
            comps.append(generate(self.policy.generation_logits, prompt_ids[s], prompt_mask[s],
                                  cfg.gen_length, cfg.block_length, cfg.diffusion_steps,
                                  self.policy.mask_id, temperature=cfg.temperature,
                                  remasking=cfg.remasking, generator=self.generator))
        completion_ids = torch.cat(comps)
        t_gen = time.time() - t0
        # L_old / L_ref must be evaluated in the same module mode as the update forward
        # (eval-mode fast paths or dropout would otherwise make rho != 1 on-policy).
        self.policy.train()
        rewards, texts = self.task.rewards(rows_rep, completion_ids)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        adv = group_advantages(rewards, G, baseline=cfg.baseline, scale=cfg.scale_rewards)

        N, L = completion_ids.shape
        masks = torch.stack([sample_masks(cfg.num_mc, N, L, generator=self.mask_generator)
                             for _ in range(cfg.num_iterations)]).to(self.device)
        old = None
        if cfg.num_iterations > 1:
            old = self._elbo_table(self.policy.logits, prompt_ids, prompt_mask, completion_ids, masks)
        ref = None
        if cfg.beta > 0:
            with self.policy.reference_logits() as ref_fn:
                ref = self._elbo_table(ref_fn, prompt_ids, prompt_mask, completion_ids, masks)
        self.num_rollouts += 1
        std = rewards.view(-1, G).std(1)
        return Rollout(prompt_ids, prompt_mask, completion_ids, rewards, adv, masks, old, ref, texts,
                       stats={"reward": rewards.mean().item(), "reward_std": std.mean().item(),
                              "zero_std_frac": (std < 1e-6).float().mean().item(),
                              "time_rollout": time.time() - t0, "time_generate": t_gen})

    # ------------------------------------------------------------------ update
    def optimizer_step(self, ro: Rollout, iteration: int) -> dict:
        cfg = self.cfg
        N, L = ro.completion_ids.shape
        self.opt.zero_grad(set_to_none=True)
        chunks = self._chunks(N)
        agg = {"ratio_min": math.inf, "ratio_max": -math.inf, "clip_count": 0, "kl_sum": 0.0,
               "loss": 0.0, "elbo_sum": 0.0}
        for ci, s in enumerate(chunks):
            sync = ci == len(chunks) - 1
            with (nullcontext() if sync else self.policy.no_sync()):
                elbo = sequence_elbo(self.policy.logits, ro.prompt_ids[s], ro.prompt_mask[s],
                                     ro.completion_ids[s], ro.masks[iteration][:, s],
                                     self.policy.mask_id, coupled=cfg.coupled).float()
                old = ro.old_elbo[iteration, s] if ro.old_elbo is not None else elbo.detach()
                ref = ro.ref_elbo[iteration, s] if ro.ref_elbo is not None else None
                loss, st = espo_loss(elbo, old, ref, ro.advantages[s], L, cfg.epsilon,
                                     cfg.beta, denom=N, kl_kind=cfg.kl_estimator)
                loss.backward()
            agg["ratio_min"] = min(agg["ratio_min"], st["ratio_min"])
            agg["ratio_max"] = max(agg["ratio_max"], st["ratio_max"])
            agg["clip_count"] += st["clip_count"]
            agg["kl_sum"] += st["kl_sum"]
            agg["loss"] += loss.item()
            agg["elbo_sum"] += elbo.detach().sum().item()
        params = self.policy.trainable_parameters()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm) if cfg.max_grad_norm \
            else torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None))
        self.opt.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1
        mem = {}
        if self.device.type == "cuda":
            mem = {"max_mem_gb": torch.cuda.max_memory_allocated(self.device) / 2**30}
        return {**mem, "loss": agg["loss"], "kl": agg["kl_sum"] / N, "clip_frac": agg["clip_count"] / N,
                "ratio_min": agg["ratio_min"], "ratio_max": agg["ratio_max"],
                "elbo_per_token": agg["elbo_sum"] / (N * L), "grad_norm": float(grad_norm)}

    # ------------------------------------------------------------------ loop
    def _reduce(self, stats: dict) -> dict:
        if self.all_reduce is None or self.world_size == 1:
            return stats
        keys = [k for k in stats if k not in ("ratio_min", "ratio_max")]
        t = torch.tensor([float(stats[k]) for k in keys], device=self.device)
        t = self.all_reduce(t) / self.world_size
        out = dict(zip(keys, t.tolist()))
        for k, sign in (("ratio_min", -1), ("ratio_max", 1)):
            if k in stats:  # min/max via max of signed values
                v = self.all_reduce_max(torch.tensor([sign * stats[k]], device=self.device))
                out[k] = sign * v.item()
        return out

    def all_reduce_max(self, t):
        import torch.distributed as dist

        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t

    def train(self, num_steps: int, on_step: Optional[Callable[["ESPOTrainer", dict], None]] = None):
        logs = []
        ro = None
        while self.global_step < num_steps:
            j = self.global_step % self.cfg.num_iterations
            if j == 0 or ro is None:
                ro = self.make_rollout()
            stats = self.optimizer_step(ro, j)
            stats.update(ro.stats)
            stats["iteration"] = j
            stats = self._reduce(stats)
            stats["step"] = self.global_step
            if self.scheduler is not None:
                stats["lr"] = self.scheduler.get_last_lr()[0]
            logs.append(stats)
            if self.log_fn is not None:
                self.log_fn(stats)
            if on_step is not None:
                on_step(self, stats)
        return logs
