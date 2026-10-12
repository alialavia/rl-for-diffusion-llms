"""Evaluate a base model or a trained ESPO adapter (d1-style generation settings).

    torchrun --standalone --nproc_per_node=8 -m espo.evaluate --task sudoku \
        --adapter outputs/sudoku_llada/checkpoint-10000 --gen_lengths 128 256 512

Planning tasks (sudoku, countdown) unmask 2 tokens per step (steps = L/2); math tasks
use steps = L (paper Appendix E.1). LLaDA: low-confidence remasking, block length 32,
greedy decoding.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from .data import load_rows_from_file, load_task_rows
from .model import HFPolicy, ModelArgs, load_model_and_tokenizer
from .sampler import generate
from .tasks import TextTask, reward_for_row
from .train import setup_distributed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["sudoku", "countdown", "gsm8k", "math"])
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--test_file", default=None, help="sudoku csv / countdown jsonl (d1 format)")
    ap.add_argument("--gen_lengths", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--remasking", default="low_confidence")
    ap.add_argument("--max_prompt_length", type=int, default=400)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--output", default=None, help="jsonl with per-example generations")
    a = ap.parse_args(argv)

    rank, world, _, device = setup_distributed()
    margs = ModelArgs(model_name_or_path=a.model, adapter_path=a.adapter, use_lora=a.adapter is not None)
    model, tok, family, mask_id, _ = load_model_and_tokenizer(margs, device, trainable=False)
    policy = HFPolicy(model, family, mask_id)
    rows = load_rows_from_file(a.task, a.test_file) if a.test_file else load_task_rows(
        a.task, "test", seed=a.data_seed, countdown_source="synthetic")
    rows = rows[: a.limit] if a.limit else rows
    shard = rows[rank::world]
    task = TextTask(a.task, shard, tok, max_prompt_length=a.max_prompt_length)

    results = {}
    out_f = open(f"{a.output}.rank{rank}", "w") if a.output else None
    for L in a.gen_lengths:
        steps = L // 2 if a.task in ("sudoku", "countdown") else L
        block = min(a.block_length, L) if family == "llada" else L
        tot = torch.zeros(3, dtype=torch.float64, device=device)  # [sum reward, sum solved, n]
        for i in range(0, len(shard), a.batch_size):
            batch = shard[i:i + a.batch_size]
            ids, attn = task.encode_prompts(batch, device=device)
            comp = generate(policy.generation_logits, ids, attn, L, block, steps, mask_id,
                            temperature=a.temperature, remasking=a.remasking if family == "llada" else "entropy")
            texts = task.decode(comp)
            for r, t in zip(batch, texts):
                rew = reward_for_row(a.task, r, t)
                tot += torch.tensor([rew, float(rew == 1.0), 1.0], dtype=torch.float64, device=device)
                if out_f:
                    out_f.write(json.dumps({"gen_length": L, "reward": rew, "completion": t,
                                            **{k: v for k, v in r.items() if k != "messages"}}) + "\n")
        if world > 1:
            import torch.distributed as dist

            dist.all_reduce(tot)
        results[L] = {"mean_reward": (tot[0] / tot[2]).item(), "solved": (tot[1] / tot[2]).item(),
                      "n": int(tot[2].item())}
        if rank == 0:
            print(json.dumps({"task": a.task, "gen_length": L, **results[L]}), flush=True)
    if out_f:
        out_f.close()
    if world > 1:
        import torch.distributed as dist

        dist.destroy_process_group()
    return results


if __name__ == "__main__":
    main()
