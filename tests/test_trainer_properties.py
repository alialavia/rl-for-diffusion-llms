"""Stateful / integration properties of the ESPO trainer (PROPERTIES.md section 5, T1-T5).

These use a tiny bidirectional transformer (espo.toy) so they run on CPU.
"""
import copy
import math

import pytest
import torch
from hypothesis import example, given, settings, strategies as st

from espo.toy import TinyDiffusionLM, ToyPolicy, ToyTask
from espo.trainer import ESPOConfig, ESPOTrainer


def make_trainer(lr=0.0, mu=2, M=1, G=4, prompts=2, micro=3, beta=0.01, gen_length=6,
                 block=3, steps=6, seed=0, coupled=True, temperature=1.0, vocab=5):
    torch.manual_seed(seed)
    task = ToyTask(vocab_size=vocab, prompt_length=2, target_token=1)
    model = TinyDiffusionLM(vocab_size=task.vocab_size, max_len=2 + gen_length, d_model=32,
                            n_layers=1, n_heads=2)
    policy = ToyPolicy(model, mask_id=task.mask_id)
    cfg = ESPOConfig(num_generations=G, prompts_per_rank=prompts, micro_batch_size=micro,
                     num_iterations=mu, num_mc=M, coupled=coupled, beta=beta,
                     gen_length=gen_length, block_length=block, diffusion_steps=steps,
                     temperature=temperature, generation_batch_size=4, seed=seed,
                     max_grad_norm=1.0)
    opt = torch.optim.AdamW(policy.trainable_parameters(), lr=lr, weight_decay=0.01)
    return ESPOTrainer(policy, task, opt, cfg), policy


# Tolerances: the update forward runs with autograd, L_old / L_ref under no_grad. On some
# CPUs (GitHub's runners) the two modes give last-bit-different results (KL ~ 1e-14, seen
# in CI), so bit-exactness is not portable. The bugs this property targets (old/current
# ELBO on different masks or iterations) give |Delta L| of order 1, i.e. |rho - 1| ~ 1e-1.
RATIO_TOL = 1e-6   # |rho - 1|        <=> |Delta L| <~ L * 1e-6
KL_TOL = 1e-9      # 0.5 Delta^2 / L  <=> |Delta L| <~ 1e-4


@settings(max_examples=8)
@given(st.integers(1, 4), st.integers(1, 2), st.integers(1, 5), st.booleans(),
       st.integers(0, 1000))
@example(mu=1, M=1, micro=2, coupled=False, seed=0)  # CI counterexample to the bit-exact version
def test_T1_T2_zero_lr_keeps_ratio_one_and_kl_zero(mu, M, micro, coupled, seed):
    """T1. With lr = 0, for every inner iteration the sequence ratio is 1 (to float
    tolerance) and the clip fraction is 0: old and current ELBO use the same
    per-iteration masks.
    T2. Since the policy starts at the reference, the KL estimate is 0 (to tolerance).
    Oracle: identity (theta never changes).
    History: first stated bit-exact; CI showed KL = 9.5e-15 because grad / no-grad forwards
    differ in the last bits on some CPUs, so the property now uses RATIO_TOL / KL_TOL.
    """
    trainer, _ = make_trainer(lr=0.0, mu=mu, M=M, micro=micro, coupled=coupled, seed=seed)
    logs = trainer.train(num_steps=2 * mu)
    assert len(logs) == 2 * mu
    for s in logs:
        assert abs(s["ratio_min"] - 1.0) <= RATIO_TOL and abs(s["ratio_max"] - 1.0) <= RATIO_TOL
        assert s["clip_frac"] == 0.0
        assert 0.0 <= s["kl"] <= KL_TOL


@settings(max_examples=6)
@given(st.integers(1, 4), st.integers(1, 9))
def test_T3_rollouts_regenerated_once_every_mu_steps(mu, num_steps):
    """T3. Optimizer steps == num_steps and rollouts == ceil(num_steps / mu).
    Oracle: counter model.
    """
    trainer, _ = make_trainer(lr=1e-3, mu=mu)
    calls = []
    orig = trainer.make_rollout
    trainer.make_rollout = lambda: (calls.append(trainer.global_step), orig())[1]
    trainer.train(num_steps=num_steps)
    assert trainer.global_step == num_steps
    assert len(calls) == math.ceil(num_steps / mu)
    assert calls == [i * mu for i in range(len(calls))]


def test_T4_toy_task_reward_increases():
    """T4. Integration: on the toy task (reward = fraction of target token) the mean
    rollout reward rises substantially after training.  Oracle: reward before vs after.
    Catches: sign errors that every unit property would miss.
    """
    trainer, _ = make_trainer(lr=3e-3, mu=2, M=2, G=8, prompts=4, micro=8, beta=0.0,
                              gen_length=6, block=6, steps=6, seed=0, vocab=5)
    logs = trainer.train(num_steps=40)
    first = logs[0]["reward"]  # rollout from the initial policy
    last = sum(s["reward"] for s in logs[-6:]) / 6
    assert first < 0.4
    assert last > first + 0.4, (first, last)


def test_T5_reference_is_never_updated_and_policy_is():
    """T5. Reference parameters are bit-identical after training; trainable ones moved."""
    trainer, policy = make_trainer(lr=1e-2, mu=2)
    ref_before = copy.deepcopy(policy.reference.state_dict())
    pol_before = copy.deepcopy(policy.model.state_dict())
    trainer.train(num_steps=4)
    for k, v in policy.reference.state_dict().items():
        assert torch.equal(v, ref_before[k])
    assert any(not torch.equal(v, pol_before[k]) for k, v in policy.model.state_dict().items())


def test_T_config_validation():
    """Partiality: inconsistent configurations are rejected up front."""
    with pytest.raises(ValueError):
        ESPOConfig(num_generations=1)
    with pytest.raises(ValueError):
        ESPOConfig(gen_length=10, block_length=4)
    with pytest.raises(ValueError):
        ESPOConfig(kl_estimator="k4")
