# ESPO — property specification

This file is the executable-spec-in-English for the implementation of
**ESPO: ELBO-based Sequence-level Policy Optimization** (Ou et al., 2025,
arXiv:2512.03759). It was written *before* any implementation code, following the
property-driven-design mode of
[`alialavia/pbt-skills`](https://github.com/alialavia/pbt-skills): every property
states what *any* correct implementation must satisfy, names an oracle that is
independent of the code under test, and lists the bugs it is meant to catch.

Each property ID (`E1`, `O3`, ...) is referenced from the docstring of the test that
checks it (`tests/test_*.py`).

---

## 0. The method in one page

For a prompt `x`, sample `G` completions `y⁽¹⁾..y⁽ᴳ⁾ ~ π_old` with a masked
diffusion LM. Each completion has length `L` (fixed generation length; EOS padding
is part of the sequence).

* **ELBO** (paper Eq. 3, coupled form Eq. 32–34). With `l ~ U{0..L}`, `y_l` = `y`
  with a uniformly random `l`-subset of completion positions replaced by `[MASK]`,
  and `ỹ_l` the complementary mask (the other `L−l` positions masked):

      ℓ(y_l, l) = (L+1)/l · Σ_{i masked} log p_θ(yⁱ | y_l, x)      (0 if l = 0)
      L̂_θ(y|x)  = ½ [ ℓ(y_l, l) + ℓ(ỹ_l, L−l) ]                     (coupled)

  averaged over `M` Monte-Carlo masks.
* **Antithetic sampling**: `L_θ`, `L_θold`, `L_ref` for the same sequence are evaluated
  on the *same* masks.
* **Sequence-level ratio** (Eq. 8): `ρ = exp((L_θ − L_old) / L)`.
* **Advantage**: group-relative, `Â = R − baseline` (group mean in the paper text;
  leave-one-out mean in the released code; no std normalisation by default).
* **Objective** (Eq. 7 + Eq. 10): maximise
  `1/G Σ_i [ min(ρ Â, clip(ρ, 1−ε, 1+ε) Â) − β · ½ (L_θ − L_ref)² / L ]`.
  (The `/L` on the KL term follows the authors' released code.)
* `µ` policy updates per rollout batch; each update uses its own fresh masks, shared
  across θ / θ_old / ref.
* Rollouts: LLaDA low-confidence remasking, semi-autoregressive blocks of 32,
  temperature 0.9.

---

## 1. ELBO estimator (`espo/elbo.py`)

**Contract.** `sample_masks(num_mc, batch, L, generator) -> bool[M, B, L]` draws masks
whose per-row count `l` is marginally uniform on `{0..L}`, stratified across the batch,
and whose positions given `l` are a uniformly random `l`-subset.
`sequence_elbo(logits_fn, prompt_ids, prompt_mask, completion_ids, masks, mask_id,
coupled) -> float[B]` returns the MC-averaged ELBO estimate. Masks never touch prompt
tokens.

| ID | Category | Property (For all … ) | Oracle | Catches |
|----|----------|-----------------------|--------|---------|
| **E1** | oracle | For all tiny joint distributions `p` over `V^L` (V≤3, L≤4) and all `y`, if the denoiser returns the *exact* conditionals `p(yⁱ | revealed tokens)`, then the **exact expectation** of the estimator over the mask distribution (enumerate every `(l, subset)` with prob `1/(L+1)·1/C(L,l)`) equals `log p(y)` — for coupled and uncoupled estimators. | `log p(y)` read off the joint table — independent of the estimator. | Wrong `(L+1)/l` weight, wrong complement weight, off-by-one in `l` range, masking prompt tokens, wrong target gather. |
| **E2** | oracle | For all *arbitrary* (inconsistent, context-dependent) denoisers, the exact expectation equals the any-order-autoregressive log-likelihood averaged over all `L!` orderings `E_σ Σ_k log p(y^{σ(k)} | y^{σ(<k)})` (Ou et al. 2024). | Brute-force permutation enumeration — a different formula. | Same as E1 but for models whose conditionals are not consistent (the real case). |
| **E3** | boundedness | For all denoisers, `Σ_y exp(E[L̂(y)]) ≤ 1` (the ELBO is a bound on a normalised likelihood), and every single-sample estimate is `≤ 0`. | Probability normalisation; sign of log-probs. | Positive "log-likelihoods", sign errors, weights that over-count. |
| **E4** | invariant (antithetic) | For all models, sequences and masks, evaluating the **same** model twice on the **same** masks gives `L_θ − L_old = 0` exactly (bitwise). | Identity. | Old/current ELBO computed on different masks (breaks antithetic variance reduction and makes ρ ≠ 1 at θ=θ_old). |
| **E5** | invariant | For all `M, B, L`: mask row sums equal the drawn counts; counts ∈ `[0, L]`; the coupled complement is disjoint from and jointly covers all `L` positions. | Set algebra on the bool tensors. | Wrong complement, masks overlapping. |
| **E6** | oracle (exact distribution) | For all `B, L` and every batch slot `b`, the map from the shared stratification offset `k ∈ {0..L}` to slot `b`'s count is a **bijection** on `{0..L}` ⇒ each slot's `l` is exactly uniform. For `B ≤ L+1`, counts within one draw are pairwise distinct (stratified). | Counting / bijection check. | Biased `l` marginal (e.g. banker's rounding in `round(k + b(L+1)/B)` — present in the reference code, see notes). |
| **E7** | oracle (distribution) | Given `l`, every `l`-subset is equally likely (seeded χ² on small `L`). | Uniform over `C(L,l)` subsets. | Positional bias in mask placement. |
| **E8** | invariant | For all inputs, positions outside the mask are left untouched and prompt tokens are never masked. The noisy input differs from the clean input exactly on masked positions, which hold `mask_id`. | Element-wise comparison. | Masking the prompt, corrupting unmasked tokens. |
| **E9** | metamorphic (batch independence) | For all batches, the ELBO of row `b` computed in a batch equals the ELBO of row `b` computed alone with the same mask (allclose). | The single-row computation. | Cross-row leakage, broadcasting bugs, wrong per-row `l`. |
| **E10** | algebraic | For all `M` masks, permuting the order of the MC samples does not change the estimate; the M-sample estimate equals the mean of M single-sample estimates. | Mean of independent calls. | Summing instead of averaging over MC, MC/batch axis mix-ups. |

## 2. Objective (`espo/objective.py`)

**Contract.** `group_advantages(rewards[N], G, baseline∈{mean, loo}, scale) -> [N]`;
`sequence_ratio(elbo, old_elbo, L)`; `clipped_surrogate(ratio, adv, ε)`;
`kl_estimate(elbo, ref_elbo, L, kind∈{k1,k2,k3})`;
`espo_loss(elbo, old_elbo, ref_elbo, adv, L, ε, β, denom) -> (loss, stats)`.

| ID | Category | Property | Oracle | Catches |
|----|----------|----------|--------|---------|
| **O1** | invariant | For all rewards and `G ≥ 2`: advantages sum to zero within each group; all-equal group ⇒ all zeros; the best sample has `Â ≥ 0` and the worst `Â ≤ 0`. | Arithmetic facts of a centred baseline. | Using global mean instead of group mean, sign flip. |
| **O2** | algebraic / metamorphic | Shift invariance: adding `c_g` to every reward of group `g` leaves advantages unchanged. Scale: multiplying all rewards by `a>0` multiplies advantages by `a` (unscaled) or leaves them unchanged (std-scaled). Permuting samples within a group permutes advantages identically. Changing group `g` never affects group `h≠g`. | Metamorphic relations. | Groups mixed across boundaries, reshape order bugs (`view(G,-1)` vs `view(-1,G)`). |
| **O3** | metamorphic | `loo = G/(G−1) · mean-baseline` advantages, element-wise. | Algebraic identity `r − (S−r)/(G−1) = G/(G−1)(r − S/G)`. | Off-by-one in leave-one-out denominator. |
| **O4** | invariant / metamorphic | `ρ > 0`; `ρ = 1` iff `ΔL = 0`; ρ strictly increasing in `ΔL`; **length invariance**: if `ΔL = L·δ` then `ρ = exp(δ)` for every `L` (Eq. 8's per-token normalisation). | `math.exp` on the per-token log ratio. | Missing `/L` (the "astronomically large ratio" failure of Eq. 6), normalising by the wrong length. |
| **O5** | boundedness / monotonicity | Clipped surrogate `J(ρ, Â)`: equals `ρÂ` inside `[1−ε, 1+ε]`; `J ≤ ρÂ` and `J ≤ clip(ρ)Â` always (pessimism); `J` is monotone in ρ in the direction of `sign(Â)`; for `Â>0`, `J ≤ (1+ε)Â`. | PPO definition decomposed into independent inequalities. | `max` instead of `min`, clipping the wrong side, clipping advantages. |
| **O6** | oracle (gradient) | `∂J/∂L_θ` equals the central finite difference; it is **exactly 0** when the ratio is clipped on the beneficial side (`Â>0, ρ>1+ε` or `Â<0, ρ<1−ε`) and equals `Â·ρ/L` otherwise. | Finite differences (independent of autograd). | Clipping implemented with a non-detached `clamp` on the wrong branch, gradient leakage. |
| **O7** | invariant / algebraic | `k2 = ½Δ²/L ≥ 0`, zero iff `Δ = 0`, symmetric under `Δ → −Δ`; `k3 ≥ 0`; `k1` takes negative values for `Δ<0`; for small `Δ`, `|k3 − k2|·L ≤ C|Δ|³` (Taylor agreement). | Inequalities + Taylor bound. | Wrong sign convention in k3 (`exp(ref−θ)`), negative k2. |
| **O7c** | oracle | For exact categorical `π_θ, π_ref`: `E_{y∼π_θ}[L·k1] = E[L·k3] = KL(π_θ‖π_ref)` (unbiasedness), `E[k2] ≥ 0`. *(added after mutation testing — see below)* | KL from its definition. | k3 written for the reverse direction `exp(D)−1−D` (passes O7). |
| **O8** | oracle (gradient) | `∂(β·k2)/∂L_θ = β (L_θ − L_ref)/L` (finite differences). At `θ = ref` the KL contributes zero gradient. | Finite differences. | KL gradient pointing away from the reference. |
| **O8b** | oracle (gradient) | With zero advantages, `∂loss/∂L_θ⁽ⁱ⁾ = +β(L_θ⁽ⁱ⁾ − L_ref⁽ⁱ⁾)/(L·denom)`: descent pulls the policy toward the reference. *(added after mutation testing)* | Hand derivative. | KL entering the loss with the wrong sign. |
| **O9** | invariant (on-policy) | When `L_θ = L_old` (on-policy, first update) and `β=0`: `∂loss/∂L_θ⁽ⁱ⁾ = −Âᵢ/(L·denom)`. Zero advantages + `β=0` ⇒ zero gradient. | Hand-derived gradient of Eq. 7 at ρ=1. | Wrong loss sign (gradient ascent vs descent), wrong normalisation. |
| **O10** | algebraic (aggregation) | The loss is invariant to permuting the samples of the batch, and the sum of micro-batch losses (each with the full-batch denominator) equals the full-batch loss — gradient accumulation is exact. | Full-batch computation. | Averaging per micro-batch instead of over the full batch. |

## 3. Diffusion sampler (`espo/sampler.py`)

**Contract.** `transfer_schedule(n_masked, steps) -> [steps]`;
`generate(logits_fn, prompt_ids, prompt_mask, gen_length, block_length, steps,
mask_id, temperature, remasking, generator, trace) -> completion[B, gen_length]`.
Raises `ValueError` if `gen_length % block_length ≠ 0` or `steps` is not a positive
multiple of the number of blocks.

| ID | Category | Property | Oracle | Catches |
|----|----------|----------|--------|---------|
| **S1** | invariant | For all `n ≥ 0, steps ≥ 1`: schedule sums to `n`, entries ≥ 0, max−min ≤ 1, non-increasing. | Arithmetic. | Tokens left masked at the end, uneven schedule. |
| **S2** | invariant | Output has no `[MASK]`, the prompt is unchanged, shape is `[B, gen_length]`. | Direct inspection. | Leaving masks, overwriting the prompt. |
| **S3** | stateful invariant | Over the trace of steps: each step reveals exactly `schedule[step]` tokens per row; a revealed token never changes afterwards; while block `b` is being decoded every position in blocks `> b` is still masked. | Trace replay against a simple model of the decoding state. | Remasking committed tokens, decoding outside the current block (breaks semi-AR). |
| **S4** | oracle | If the denoiser is a point mass on a target sequence (one-hot logits), the output equals the target — for any temperature, block length, steps. | The target. | Wrong argmax axis, off-by-one between logits and positions. |
| **S5** | oracle | Greedy low-confidence decoding with a context-free model of known per-position confidences reveals positions in decreasing-confidence order within each block. | `argsort` of the known confidences. | Revealing lowest-confidence first, ignoring confidences. |
| **S6** | determinism | `temperature = 0` ⇒ identical outputs across calls; `temperature > 0` with the same generator seed ⇒ identical outputs. | Re-run. | Hidden global RNG use. |
| **S7** | oracle (distribution) | With `temperature = 1` and a context-free model, the sampled token frequency at a position matches `softmax(logits)` (seeded χ²). With `temperature τ`, it matches `softmax(logits/τ)`. | Softmax. | Gumbel-max trick with wrong temperature exponent or precision. |
| **S8** | error | Invalid `gen_length/block_length/steps` raise `ValueError`. | Precondition. | Silent truncation. |

## 4. Rewards and tasks (`espo/rewards.py`, `espo/data.py`)

| ID | Category | Property | Oracle | Catches |
|----|----------|----------|--------|---------|
| **R1** | boundedness | All rewards ∈ `[0, 1]` for arbitrary (fuzzed) completion text; no exceptions. | Range. | Crashes on garbage model output, unbounded rewards. |
| **R2** | oracle | Sudoku: for a valid puzzle/solution and `k` wrong digits placed in empty cells, reward = `1 − k/#empty`; the full solution scores 1. | Count of corruptions, by construction. | Counting prefilled cells, wrong denominator. |
| **R3** | invariant | Sudoku reward ignores whitespace/separators inside `<answer>` and uses the last `<answer>` block; no answer ⇒ 0. | Construction. | Fragile parsing. |
| **R4** | oracle | Countdown: for random expression trees using each given number exactly once, with target = exact `Fraction` value of the tree, reward = 1; inserting spaces does not change it. | `fractions.Fraction` evaluation of the generated tree. | Operator precedence bugs, float vs exact compare. |
| **R5** | error / safety | Countdown: wrong multiset of numbers ⇒ reward 0.1 (format only); divide-by-zero or arbitrary Python (`__import__('os')…`) never executes and never scores 1. | Construction. | `eval` on model output. |
| **R6** | round-trip | GSM8K/MATH: for numbers `n`, formatting `n` with commas / `$` / trailing `.0` inside `\boxed{}` and extracting ⇒ equals `n`; a different number ⇒ 0; last `\boxed{}` wins; nested braces parsed. | The generated number. | Regex that breaks on commas/nesting. |
| **D1** | invariant | Generated 4×4 sudoku solutions are valid (rows, columns, boxes are permutations of 1–4). | Independent validity checker. | Generator bugs. |
| **D2** | oracle | Generated puzzles agree with their solution on given cells and have **exactly one** solution. | Independent backtracking solution counter. | Ambiguous puzzles (reward penalising valid answers). |
| **D3** | invariant | Train and test sudoku splits are disjoint. | Set intersection. | Test leakage. |
| **D4** | oracle | Synthetic countdown targets are reachable using all numbers exactly once. | Brute-force search over expression trees. | Unsolvable training prompts. |

## 5. Trainer (`espo/trainer.py`) — stateful properties

| ID | Category | Property | Oracle | Catches |
|----|----------|----------|--------|---------|
| **T1** | stateful invariant | With learning rate 0, for every one of the `µ` inner iterations the ratio is exactly 1 and the clipped fraction is 0 (old ELBO and current ELBO use the same per-iteration masks). | Identity (θ never changes). | Mask/iteration bookkeeping bugs between rollout time and update time. |
| **T2** | invariant | At step 0 (policy initialised from the reference) the KL estimate is exactly 0. | Identity. | Reference ≠ initial policy, ref computed on different masks. |
| **T3** | stateful | Rollouts are regenerated exactly once every `µ` optimizer steps; optimizer steps = `num_steps`. | Counter model. | Re-using stale rollouts too long / regenerating too often. |
| **T4** | integration (seeded) | On a toy task (reward = fraction of a target token) with a tiny bidirectional transformer, mean reward increases substantially after training. | Reward before vs after. | Sign errors that pass every unit property but ascend the wrong way. |
| **T5** | invariant | Gradient flows only into trainable parameters; reference parameters are never updated. | Parameter snapshots. | Updating the reference / frozen base weights. |
| **DDP** | invariant (distributed) | With different prompts on each of 2 ranks, trainable parameters stay bitwise identical across ranks after several steps (several forwards per backward, `no_sync` accumulation). | Rank-0 parameters. | Forward bypassing DDP, missing all-reduce on accumulation boundary. |

### Real-architecture properties (`tests/test_hf_integration.py`)

Run on a tiny randomly initialised LLaDA built from the **official** remote modeling code
and tokenizer (fetched from the Hub, skipped offline):

| ID | Property | Catches |
|----|----------|---------|
| **HF-T2** | With PEFT's `B = 0` init, ELBO with adapters == ELBO with adapters disabled (the reference), bitwise, on shared masks. | Reference ≠ initial policy for LoRA. |
| **HF-pad** | Left-padding the prompt (attention mask 0 on pads) leaves the ELBO unchanged (RoPE is relative; pads are masked). | Attention-mask plumbing into LLaDA. |
| **HF-T5** | LoRA training with non-zero advantages moves adapter weights and leaves every base weight bitwise unchanged. | Training the frozen base. |

---

## Completeness check (pbt-skills Step 6.5)

| Category | Covered by |
|----------|-----------|
| 1 Round-trip | R6 (format → extract). Sampler/ELBO have no encode/decode pair — n/a. |
| 2 Algebraic laws | E10 (MC permutation invariance), O2 (shift/scale/permutation), O3, O10 (order independence + accumulation). |
| 3 Oracle comparison | E1 (exact likelihood), E2 (permutation AO-AR), S4/S5/S7, R2/R4, D2/D4, O6/O8 (finite differences). |
| 4 Metamorphic | O4 (length invariance), E9 (batch vs single), O2. |
| 5 Invariants preserved | E5, E8, S2, O1, O7, D1. |
| 6 Stateful | S3 (decoding trace), T1–T3, T5. |
| 7 Error / partiality | S8, R1, R5. |
| 8 Monotonicity / boundedness | E3, O4 (monotone ρ), O5, R1. |

## Mutation testing (pbt-skills Step 6: "would a plausible bug be caught?")

`python scripts/mutation_test.py` injects 22 plausible bugs one at a time and checks that
the suite fails for each. The first run killed 18/21; the three survivors were analysed:

* *Banker's-rounding stratification* — the first mutant only rounded the integer offset,
  which is an **equivalent mutant** (still a bijection). Re-modelled as the released
  code's `round(k + b(L+1)/B)`, it is killed by E6's `(B=2, L=256)` example.
* *k3 for the reverse direction* — still `≥ 0` and second-order equal to k2, so O7 could
  not see it. **New property O7c** (unbiasedness against an exact KL) kills it.
* *KL term with the wrong sign in the loss* — O8 only tested `kl_estimate`, and the
  trainer tests run with `β = 0` or `KL ≡ 0`. **New property O8b** kills it.

Current status: all mutants killed, including a DDP mutant (forward bypassing the DDP
wrapper), which the DDP replica-consistency property kills.

## Notes on deviations / clarifications

* **Stratified count rounding.** The released code computes stratified counts with
  `torch.round(linspace(k, ...))`, which uses round-half-to-even and makes the
  per-slot marginal of `l` non-uniform when `(L+1)/B` has a `.5` fractional part. We use
  integer offsets `⌊b(L+1)/B⌋`, which keeps the marginal exactly uniform (E6).
* **LoRA dropout.** Defaults to 0 so that `L_θ` and `L_θold` are evaluated on the same
  deterministic function (needed for exact ρ = 1 on-policy, T1).
* **KL normalisation.** `½(L_θ − L_ref)²/L` per sequence, as in the released code.
* **Baseline.** `loo` (released code) by default; `mean` (paper text) available. They
  differ by the constant factor `G/(G−1)` (O3).
* **Module mode for `L_old`/`L_ref`.** T1/T2 initially failed with `KL = 3e-14` at
  step 0: the first trainer draft computed `L_old`/`L_ref` with the module in `eval()`
  mode, where PyTorch's transformer uses a different fused kernel than in `train()`
  mode. With dropout this would bias ρ. Old/ref ELBOs are now computed in the same mode
  as the update forward.
* **LoRA targets.** LLaDA's linear layers are `q_proj, k_proj, v_proj, attn_out,
  ff_proj, up_proj, ff_out`; the released config lists Llama names, so only
  `q/k/v/up_proj` were actually adapted there. We default to that effective set.
