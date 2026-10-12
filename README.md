# ESPO — RL for diffusion LLMs, from a sequence-level perspective

A from-scratch PyTorch implementation of **ELBO-based Sequence-level Policy Optimization**
([Ou et al., 2025, arXiv:2512.03759](https://arxiv.org/abs/2512.03759)) for masked
diffusion LLMs (LLaDA-8B-Instruct, Dream-7B-Instruct). It is tested with **property-based
testing** following [`alialavia/pbt-skills`](https://github.com/alialavia/pbt-skills).

* `PROPERTIES.md` holds the specification: every property, its oracle and the bugs it
  catches. It was written before any implementation code.
* `tests/` holds the Hypothesis property tests (each docstring cites its property ID).
* `scripts/mutation_test.py` checks that the suite catches 22 injected plausible bugs.

## The method

For each prompt, sample `G` completions with the diffusion sampler. Then each completion
is treated as **one action**:

| Piece | Formula | Code |
|---|---|---|
| Likelihood proxy | coupled MC ELBO, `l ~ U{0..L}`, weights `(L+1)/l` and `(L+1)/(L−l)` (Eq. 32–34) | `espo/elbo.py` |
| Variance reduction | same masks for θ, θ_old and ref (antithetic); complementary masks (coupled); `l` stratified over the batch | `espo/elbo.py`, `espo/trainer.py` |
| Ratio | `ρ = exp((L_θ − L_old)/L)` (Eq. 8) | `espo/objective.py` |
| Objective | `min(ρÂ, clip(ρ,1±ε)Â) − β·½(L_θ − L_ref)²/L` (Eq. 7, 10) | `espo/objective.py` |
| Advantage | group leave-one-out (released code) or group mean (paper) | `espo/objective.py` |
| Rollouts | LLaDA low-confidence remasking, semi-AR blocks of 32, T = 0.9 | `espo/sampler.py` |
| Loop | µ = 8 optimizer steps per rollout batch, fresh masks per step | `espo/trainer.py` |

## Install

```bash
pip install -e ".[test,math]"    # torch>=2.4, transformers 4.x, peft, datasets
```

LLaDA/Dream ship remote modeling code written for transformers 4.x and do not load
under transformers 5.

## Train on one node with 8×H100

```bash
scripts/train_8xh100.sh sudoku_llada
```

This runs `torchrun --standalone --nproc_per_node=8 -m espo.train --config configs/sudoku_llada.yaml`.
Other configs: `countdown_llada`, `gsm8k_llada`, `math_llada`, `sudoku_dream`.
Override any value with dotted keys:

```bash
scripts/train_8xh100.sh countdown_llada train.max_steps=2000 espo.beta=0.005
```

Default setup (paper Sec. 5.1 / App. E, authors' recipes):

* LoRA with r = 128 and α = 64 on bf16 LLaDA-8B-Instruct; one full replica per GPU under DDP.
  The reference model is the base with adapters disabled, so it costs no extra memory.
* Per optimizer step: 8 ranks × 2 prompts × G = 6 gives 96 completions (MATH: G = 16, 256 completions).
* µ = 8, M = 2, coupled sampling, ε = 0.2, β = 0.01 (countdown 3e-3 with ε = 0, GSM8K/MATH 1e-3).
* AdamW (0.9, 0.99), lr = 3e-6 constant with warmup, weight decay 0.01, gradient clipping 0.2.
* Generation length 256 with blocks of 32. Diffusion steps: 128 for sudoku/countdown, 256 for math.
* Run length: 10k steps (sudoku/countdown), 3k steps (GSM8K/MATH).

Outputs go to `outputs/<config>/`:
* `metrics.jsonl`: reward, KL, clip fraction, ratio range, grad norm and timings.
* `completions.jsonl`: sample generations.
* `checkpoint-N/`: LoRA adapter, optimizer and scheduler state.

Resume with `train.resume_from=outputs/<config>/checkpoint-N`. Log to Weights & Biases with `train.wandb_project=espo`.

Gradient checkpointing is on by default, because each micro-batch keeps M × 2 (coupled) forward passes for one backward. If you still hit OOM, lower `espo.micro_batch_size`.
Gradients are accumulated exactly over micro-batches (property O10), so lowering it does not change the objective.

Data:
* Sudoku: synthetic 4×4 puzzles with unique solutions (properties D1–D3). You can also pass the d1 CSV with `train.train_file=4x4_sudoku_unique_puzzles.csv`.
* Countdown: `Jiayi-Pan/Countdown-Tasks-3to4` (3 numbers), or `train.countdown_source=synthetic`.
* GSM8K: `openai/gsm8k`.
* MATH: `EleutherAI/hendrycks_math` for training, `HuggingFaceH4/MATH-500` for test.

## Evaluate

```bash
scripts/eval_8xh100.sh sudoku outputs/sudoku_llada/checkpoint-10000 --test_file 4x4_test_sudoku.csv
scripts/eval_8xh100.sh countdown outputs/countdown_llada/checkpoint-10000 --test_file countdown_cd3_test.jsonl
```

This uses greedy decoding at lengths 128, 256 and 512. Planning tasks unmask 2 tokens per step (App. E.1).
For sudoku, `mean_reward` is cell accuracy. For the other tasks, read `solved`.
The test files are the ones in the authors' repo (`dataset/`).

## Tests

```bash
pytest                                   # ~1 min on CPU
HYPOTHESIS_PROFILE=thorough pytest       # 500 examples per property
python scripts/mutation_test.py          # every injected bug must be killed
```

| File | What it pins down |
|---|---|
| `test_elbo_properties.py` | E[estimator] equals the exact `log p(y)` for consistent denoisers, and the brute-force any-order-AR likelihood for arbitrary ones. Also: bound, antithetic exactness, exactly uniform stratified `l`, batch independence. |
| `test_objective_properties.py` | Advantage laws, length-invariant ratio, PPO clipping (finite-difference gradients), KL estimators (k1/k3 unbiasedness against an exact KL), loss gradients, exact micro-batch accumulation. |
| `test_sampler_properties.py` | Transfer schedule, decoding-trace state machine (committed tokens fixed, blocks left to right), point-mass oracle, confidence ordering, tempered-softmax sampling (χ²). |
| `test_rewards_properties.py`, `test_data_properties.py` | Reward oracles from constructed answers, safe arithmetic (no `eval`), boxed-answer round trips, unique-solution puzzles (brute force over all 288 grids), reachable countdown targets. |
| `test_trainer_properties.py` | With lr = 0: ρ ≡ 1 and KL ≡ 0 over all µ iterations. Also: rollout cadence, the reference never moves, and reward rises on a toy task. |
| `test_distributed.py` | DDP replicas stay bitwise identical with different data per rank. |
| `test_hf_integration.py` | On a tiny LLaDA built from the official modeling code: adapter-off equals reference, left-padding invariance, LoRA trains only the adapter. |

## Deviations from the released code

All are documented in `PROPERTIES.md`:
* Stratified mask counts use integer offsets, so each sequence's `l` is exactly uniform.
  The released `torch.round(linspace(...))` is biased when `(L+1)/B` ends in .5.
* LoRA dropout is 0, so that ρ is exactly 1 on-policy.
* `L_old` and `L_ref` are computed in the same module mode as the update.
* Prompts are left-padded with an attention mask in both generation and ELBO evaluation.
  The released code masks only during generation.
* Coding tasks (AceCoder with a code-execution sandbox) are not included.
