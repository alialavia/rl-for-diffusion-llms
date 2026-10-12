"""Integration properties on a tiny, randomly initialised LLaDA built from the official
remote modeling code + tokenizer (downloaded from the Hub, a few MB; skipped offline).

Set ESPO_TINY_LLADA_DIR to reuse a prepared directory.
"""
import json
import os
import shutil

import pytest
import torch
from hypothesis import given, settings, strategies as st

pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from espo.elbo import sample_masks, sequence_elbo  # noqa: E402
from espo.model import HFPolicy, ModelArgs, load_model_and_tokenizer  # noqa: E402

FILES = ["modeling_llada.py", "configuration_llada.py", "config.json", "tokenizer.json",
         "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"]


@pytest.fixture(scope="module")
def tiny_llada(tmp_path_factory):
    d = os.environ.get("ESPO_TINY_LLADA_DIR")
    if d and os.path.exists(os.path.join(d, "config.json")):
        return d
    d = str(tmp_path_factory.mktemp("tiny_llada"))
    try:
        from huggingface_hub import hf_hub_download

        for f in FILES:
            shutil.copy(hf_hub_download("GSAI-ML/LLaDA-8B-Instruct", f), d)
    except Exception as e:  # offline
        pytest.skip(f"cannot fetch LLaDA code/tokenizer: {e}")
    cfg = json.load(open(os.path.join(d, "config.json")))
    cfg.update(d_model=64, n_layers=2, n_heads=4, n_kv_heads=4, mlp_hidden_size=128,
               max_sequence_length=1024)
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    from transformers import AutoConfig, AutoModel

    torch.manual_seed(0)
    AutoModel.from_config(AutoConfig.from_pretrained(d, trust_remote_code=True),
                          trust_remote_code=True).save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def policy(tiny_llada):
    margs = ModelArgs(model_name_or_path=tiny_llada, torch_dtype="float32", lora_r=8, lora_alpha=16)
    model, tok, family, mask_id, _ = load_model_and_tokenizer(margs, "cpu", trainable=True)
    assert family == "llada" and mask_id == 126336
    # make the adapter non-trivial copies possible later; start from B = 0 (PEFT init)
    return HFPolicy(model, family, mask_id), tok


def test_HF_T2_reference_equals_policy_at_init(policy):
    """T2 (HF). With PEFT's B = 0 init, ELBO with adapters == ELBO with adapters disabled,
    exactly, on shared masks.  Oracle: identity.
    """
    pol, _ = policy
    g = torch.Generator().manual_seed(0)
    prompt = torch.randint(0, 1000, (3, 5), generator=g)
    comp = torch.randint(0, 1000, (3, 12), generator=g)
    masks = sample_masks(2, 3, 12, generator=g)
    pol.train()
    with torch.no_grad():
        a = sequence_elbo(pol.logits, prompt, None, comp, masks, pol.mask_id)
        with pol.reference_logits() as ref_fn:
            b = sequence_elbo(ref_fn, prompt, None, comp, masks, pol.mask_id)
    assert torch.equal(a, b)


@settings(max_examples=5)
@given(st.integers(1, 6), st.integers(0, 10**6))
def test_HF_left_padding_invariance(policy, pad, seed):
    """Left-padding a prompt (with attention_mask 0 on the pads) leaves the ELBO unchanged:
    RoPE depends only on relative positions and pads are masked out.
    Oracle: the unpadded computation.  Catches: attention-mask plumbing bugs.
    """
    pol, tok = policy
    g = torch.Generator().manual_seed(seed)
    prompt = torch.randint(0, 1000, (1, 6), generator=g)
    comp = torch.randint(0, 1000, (1, 10), generator=g)
    masks = sample_masks(2, 1, 10, generator=g)
    padded = torch.cat([torch.full((1, pad), tok.pad_token_id), prompt], 1)
    attn = torch.cat([torch.zeros(1, pad, dtype=torch.long), torch.ones(1, 6, dtype=torch.long)], 1)
    with torch.no_grad():
        a = sequence_elbo(pol.logits, prompt, torch.ones_like(prompt), comp, masks, pol.mask_id)
        b = sequence_elbo(pol.logits, padded, attn, comp, masks, pol.mask_id)
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-3)


def test_HF_T5_lora_training_moves_adapter_not_base(tiny_llada):
    """T5 (HF). Training with non-zero advantages changes LoRA weights only; base weights
    (the reference) are bit-identical afterwards.
    """
    from espo.trainer import ESPOConfig, ESPOTrainer

    margs = ModelArgs(model_name_or_path=tiny_llada, torch_dtype="float32", lora_r=4, lora_alpha=8)
    model, tok, family, mask_id, _ = load_model_and_tokenizer(margs, "cpu", trainable=True)
    pol = HFPolicy(model, family, mask_id)
    base = {k: v.clone() for k, v in model.named_parameters() if "lora_" not in k}
    lora = {k: v.clone() for k, v in model.named_parameters() if "lora_" in k}

    class Task:  # reward = fraction of even token ids (gives non-zero advantages)
        def sample_prompts(self, n):
            return [{}] * n

        def encode_prompts(self, rows, device="cpu"):
            ids = torch.randint(0, 1000, (len(rows), 4))
            return ids, torch.ones_like(ids)

        def rewards(self, rows, comp):
            return (comp % 2 == 0).float().mean(1).tolist(), None

    cfg = ESPOConfig(num_generations=3, prompts_per_rank=1, micro_batch_size=3, num_iterations=2,
                     num_mc=1, beta=0.01, gen_length=4, block_length=4, diffusion_steps=2,
                     temperature=1.0, generation_batch_size=3)
    opt = torch.optim.AdamW(pol.trainable_parameters(), lr=1e-2)
    logs = ESPOTrainer(pol, Task(), opt, cfg).train(num_steps=2)
    assert logs[0]["kl"] == 0.0 and logs[0]["ratio_max"] == 1.0
    for k, v in model.named_parameters():
        if k in base:
            assert torch.equal(v, base[k]), k
    assert any(not torch.equal(v, lora[k]) for k, v in model.named_parameters() if k in lora)
