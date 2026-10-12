"""A tiny bidirectional transformer and toy task, for CPU tests and smoke runs."""
from __future__ import annotations

import copy
import random
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import torch
import torch.nn as nn


class TinyDiffusionLM(nn.Module):
    """Bidirectional (non-causal) transformer encoder over tokens + positions."""

    def __init__(self, vocab_size: int, max_len: int, d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 4):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        T = input_ids.shape[1]
        h = self.tok(input_ids) + self.pos(torch.arange(T, device=input_ids.device))
        pad = None if attention_mask is None else attention_mask == 0
        return SimpleNamespace(logits=self.head(self.encoder(h, src_key_padding_mask=pad)))


class ToyPolicy:
    """Full-parameter policy with a frozen deep-copied reference model."""

    def __init__(self, model: nn.Module, mask_id: int, ddp: bool = False):
        self.model = model
        self.reference = copy.deepcopy(model)
        self.ddp = None
        if ddp:
            from torch.nn.parallel import DistributedDataParallel

            self.ddp = DistributedDataParallel(model)
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.mask_id = mask_id

    def logits(self, input_ids, attention_mask=None):
        m = self.ddp if self.ddp is not None else self.model
        return m(input_ids, attention_mask).logits

    def generation_logits(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask).logits

    @contextmanager
    def reference_logits(self):
        yield lambda ids, am=None: self.reference(ids, am).logits

    def no_sync(self):
        return self.ddp.no_sync() if self.ddp is not None else nullcontext()

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def train(self):  # the reference follows the policy's mode (identical numerics)
        self.model.train()
        self.reference.train()

    def eval(self):
        self.model.eval()
        self.reference.eval()


class ToyTask:
    """Prompt: random tokens. Reward: fraction of completion tokens equal to ``target_token``."""

    def __init__(self, vocab_size: int = 5, prompt_length: int = 2, target_token: int = 1, seed: int = 0):
        self.real_vocab = vocab_size
        self.vocab_size = vocab_size + 1  # + mask token
        self.mask_id = vocab_size
        self.prompt_length = prompt_length
        self.target_token = target_token
        self.rng = random.Random(seed)

    def sample_prompts(self, n: int):
        return [{"tokens": [self.rng.randrange(self.real_vocab) for _ in range(self.prompt_length)]}
                for _ in range(n)]

    def encode_prompts(self, rows, device="cpu"):
        ids = torch.tensor([r["tokens"] for r in rows], dtype=torch.long, device=device)
        return ids, torch.ones_like(ids)

    def rewards(self, rows, completion_ids):
        return (completion_ids == self.target_token).float().mean(1).tolist(), None
