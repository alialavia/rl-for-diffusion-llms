"""Text tasks: prompt sampling, chat-template encoding and reward dispatch."""
from __future__ import annotations

import random

import torch

from .rewards import countdown_reward, gsm8k_reward, math_reward, sudoku_reward


def reward_for_row(task: str, row: dict, text: str) -> float:
    if task == "sudoku":
        return sudoku_reward(text, row["puzzle"], row["solution"])
    if task == "countdown":
        return countdown_reward(text, row["numbers"], row["target"])
    if task == "gsm8k":
        return gsm8k_reward(text, row["answer"])
    if task == "math":
        return math_reward(text, row["answer"])
    raise ValueError(f"unknown task {task!r}")


class TextTask:
    """Samples prompts without replacement (per-rank shard of a shuffled epoch)."""

    def __init__(self, task: str, rows: list[dict], tokenizer, max_prompt_length: int = 400,
                 seed: int = 0, rank: int = 0, world_size: int = 1):
        self.task, self.rows, self.tok = task, rows, tokenizer
        self.max_prompt_length = max_prompt_length
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.epoch, self.cursor = 0, 0
        self._order = self._shuffle()

    def _shuffle(self):
        order = list(range(len(self.rows)))
        random.Random(self.seed + self.epoch).shuffle(order)  # same on every rank
        return order[self.rank::self.world_size]

    def sample_prompts(self, n: int) -> list[dict]:
        out = []
        while len(out) < n:
            if self.cursor >= len(self._order):
                self.epoch += 1
                self.cursor = 0
                self._order = self._shuffle()
            out.append(self.rows[self._order[self.cursor]])
            self.cursor += 1
        return out

    def render(self, row: dict) -> str:
        return self.tok.apply_chat_template(row["messages"], add_generation_prompt=True, tokenize=False)

    def encode_prompts(self, rows, device="cpu"):
        texts = [self.render(r) for r in rows]
        self.tok.padding_side = "left"
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        ids, attn = enc["input_ids"], enc["attention_mask"]
        if self.max_prompt_length:
            ids, attn = ids[:, -self.max_prompt_length:], attn[:, -self.max_prompt_length:]
        return ids.to(device), attn.to(device)

    def decode(self, completion_ids: torch.Tensor) -> list[str]:
        return self.tok.batch_decode(completion_ids, skip_special_tokens=True)

    def rewards(self, rows, completion_ids):
        texts = self.decode(completion_ids)
        return [reward_for_row(self.task, r, t) for r, t in zip(rows, texts)], texts

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self.cursor}

    def load_state_dict(self, s):
        self.epoch, self.cursor = s["epoch"], s["cursor"]
        self._order = self._shuffle()
