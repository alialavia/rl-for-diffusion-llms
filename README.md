# ESPO: reinforcement learning for diffusion LLMs

A PyTorch implementation of **ESPO** (ELBO-based Sequence-level Policy Optimization,
[Ou et al., 2025](https://arxiv.org/abs/2512.03759)). ESPO fine-tunes masked diffusion
language models such as LLaDA-8B with RL on tasks that have checkable answers: Sudoku,
Countdown, GSM8K and MATH.

Diffusion LLMs don't produce text left to right, so they have no per-token probabilities
for GRPO-style training. ESPO treats each whole answer as one action instead, and uses the
model's ELBO (a lower bound on log-likelihood) as a stand-in for the answer's log-probability.

The code was written spec-first with property-based testing
([`alialavia/pbt-skills`](https://github.com/alialavia/pbt-skills)). See [Testing](#testing).

> **Status.** All tests pass on CPU. The full training command has also run end to end on a
> tiny LLaDA with 2 processes. It has **not yet been run on GPUs or with the real 8B
> weights**, so do the [preflight](#2-preflight-1020-min) before starting a long run.

## Quickstart (one node, 8×H100)

**Requirements:** 8 GPUs with 80 GB each, Python ≥ 3.10, and about 20 GB of disk for the
LLaDA weights, which download automatically from the Hugging Face Hub.

### 1. Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # any CUDA build of torch >= 2.4
pip install -e ".[test,math]"
```

Exact `transformers` and `peft` versions are pinned on purpose. LLaDA's model code does not
load under transformers 5.

### 2. Preflight (10–20 min)

```bash
scripts/smoke_8xh100.sh sudoku_llada
```

This runs 16 training steps with the real model and the full batch size, and fails fast if
something is wrong. Afterwards, check peak GPU memory and timings:

```bash
tail -1 outputs/smoke_sudoku_llada/metrics.jsonl    # see max_mem_gb, time_generate, time_rollout
```

### 3. Train

```bash
scripts/train_8xh100.sh sudoku_llada
```

| Config | Task | Steps | Rough time |
|---|---|---|---|
| `sudoku_llada` | 4×4 Sudoku | 10k | ~1 day |
| `countdown_llada` | Countdown (3 numbers) | 10k | ~1 day |
| `gsm8k_llada` | GSM8K | 3k | — |
| `math_llada` | MATH | 3k | — |
| `sudoku_dream` | Sudoku on Dream-7B (experimental, never run) | 8k | — |

Times are back-of-envelope estimates, not measurements.

Results go to `outputs/<config>/`:
* `metrics.jsonl` has one line per step: reward, KL, clip fraction, gradient norm, memory and timings.
* `completions.jsonl` has sample generations.
* `checkpoint-N/` has the LoRA adapter plus optimizer state.

### 4. Evaluate

The authors' test sets:

```bash
curl -LO https://raw.githubusercontent.com/ML-GSAI/ESPO/main/dataset/4x4_test_sudoku.csv
curl -LO https://raw.githubusercontent.com/ML-GSAI/ESPO/main/dataset/countdown_cd3_test.jsonl

scripts/eval_8xh100.sh sudoku outputs/sudoku_llada/checkpoint-10000 --test_file 4x4_test_sudoku.csv
scripts/eval_8xh100.sh countdown outputs/countdown_llada/checkpoint-10000 --test_file countdown_cd3_test.jsonl
```

This prints one JSON line per generation length (128, 256, 512), for example
`{"task": "sudoku", "gen_length": 256, "mean_reward": …, "solved": …, "n": 500}`.
Compare with Table 1 of the paper as follows:
* Sudoku: compare `mean_reward`, which is the fraction of empty cells filled correctly.
* All other tasks: compare `solved`, the fraction fully correct.

Leave out the checkpoint argument to evaluate the base model. GSM8K and MATH use their
standard test splits and need no `--test_file`.

## Configuration

Every setting lives in `configs/<name>.yaml`. Override any of them from the command line
with `section.key=value`:

```bash
scripts/train_8xh100.sh countdown_llada train.max_steps=2000 espo.beta=0.005
```

Useful overrides:

| Setting | What it does |
|---|---|
| `espo.micro_batch_size` | Lower it if you run out of GPU memory. Results are unchanged, because gradients are accumulated exactly. |
| `train.resume_from=outputs/<config>/checkpoint-N` | Resume a run. |
| `train.wandb_project=espo` | Log to Weights & Biases. |
| `train.train_file=4x4_sudoku_unique_puzzles.csv` | Use the authors' Sudoku training set instead of generated puzzles. |
| `train.countdown_source=synthetic` | Generate Countdown problems instead of downloading them. |

The defaults follow the paper (Sec. 5.1, App. E) and the authors' run scripts:
* LoRA with rank 128 and α 64 on LLaDA-8B-Instruct.
* 96 answers per step: 16 prompts × 6 answers each. MATH uses 16 prompts × 16 answers.
* 8 updates per batch of generated answers, and 2 random masks per ELBO estimate.
* Learning rate 3e-6.
* Answers of 256 tokens, generated in blocks of 32.

Gradient checkpointing is on by default, so that a micro-batch fits on an 80 GB GPU.

## How it works

Each training round has two phases:
1. **Generate.** Every GPU takes 2 prompts and samples 6 answers for each with the
   diffusion sampler (`espo/sampler.py`).
2. **Score and update.** Each answer is scored by a task reward (`espo/rewards.py`) and
   compared with the other answers to the same prompt. That gives its *advantage*: better
   than its siblings is positive, worse is negative. The model then takes 8 optimizer steps
   on this batch (`espo/trainer.py`).

The objective (`espo/objective.py`) is a PPO-style clipped surrogate, applied to the whole
answer rather than to each token:

| Piece | Definition | Paper |
|---|---|---|
| Answer log-likelihood | ELBO, estimated by masking a random number of tokens and scoring the model's predictions (`espo/elbo.py`) | Eq. 3, 32–34 |
| Importance ratio | `ρ = exp((ELBO_new − ELBO_old) / L)`, where `L` is answer length; dividing by `L` keeps ρ near 1 | Eq. 8 |
| Loss | `−min(ρ·A, clip(ρ, 1−ε, 1+ε)·A) + β·½(ELBO_new − ELBO_ref)² / L`, with ε = 0.2 (0 for Countdown) | Eq. 7, 10 |
| Variance reduction | the old, new and reference models are scored on the same masks; each mask is paired with its complement | App. D |

The reference model is the base model with the LoRA adapter switched off, so it costs no
extra memory.

## Testing

```bash
pytest                                  # ~1 min on CPU, no GPU needed
HYPOTHESIS_PROFILE=thorough pytest      # 500 random cases per property instead of 40
python scripts/mutation_test.py         # checks that the tests catch 22 deliberately injected bugs
```

The tests are *property-based*: each one states a rule that must hold for all inputs, and
Hypothesis generates inputs that try to break it. The rules are written down in
[`PROPERTIES.md`](PROPERTIES.md) and were written before the implementation. Examples:
* On small problems where the exact log-likelihood can be computed by enumeration, the
  ELBO estimator's average equals it.
* With learning rate 0, the importance ratio is exactly 1 for all 8 updates. This catches
  old and new likelihoods being scored on different masks.
* Rewards match answers built to have a known score, and model output is never executed
  as code.
* GPU replicas stay bit-identical under DDP.

The mutation script injects plausible bugs one at a time, such as a wrong weight, a flipped
sign or a missing normalisation, and confirms the tests fail for each. It found two gaps in
the original test suite; both are now covered (see `PROPERTIES.md`).

## Contributing

Read [`AGENTS.md`](AGENTS.md) first. It applies to people, Claude Code (through
`CLAUDE.md`), Codex and other agents. The short version:
1. Change the spec in `PROPERTIES.md`.
2. Reproduce any bug as a failing test.
3. Fix the code.
4. Add a matching injected bug to `scripts/mutation_test.py`.

Never weaken a test just to make it pass. The pbt-skills guides the process relies on are
included in `.claude/skills/`. CI runs the tests and the mutation check on every push.

## Repository layout

```
espo/           elbo.py, objective.py, sampler.py, trainer.py   core method
                rewards.py, data.py, tasks.py                   tasks and rewards
                model.py, train.py, evaluate.py                 LLaDA/Dream loading and CLIs
                toy.py                                          tiny model for CPU tests
configs/        one YAML per task
scripts/        train / eval / smoke launchers, mutation_test.py
tests/          property tests (each cites its ID from PROPERTIES.md)
PROPERTIES.md   the specification
AGENTS.md       how to change this code
```

## Differences from the authors' code

Details and reasons are in `PROPERTIES.md`.
* **Mask counts.** The number of masked tokens is drawn exactly uniformly. The authors' rounding skews it slightly for some batch sizes.
* **LoRA dropout** is 0, so the ratio is exactly 1 before any update.
* **Model mode.** Old and reference likelihoods are computed in the same mode as training (train vs. eval).
* **Attention mask.** The padding mask is applied when scoring as well as when generating.
* **Coding tasks** are not included (they need a code-execution sandbox).
