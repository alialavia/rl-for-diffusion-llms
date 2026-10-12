"""Property tests for the ELBO estimator (PROPERTIES.md section 1, E1-E10)."""
import itertools
import math

import pytest
import torch
from hypothesis import example, given, strategies as st

from espo.elbo import (
    apply_mask,
    random_subsets,
    sample_masks,
    sequence_elbo,
    stratified_counts,
)
from tests.denoisers import (
    all_masks_with_probs,
    any_order_ar_loglik,
    context_denoiser,
    joint_conditional_denoiser,
)


@st.composite
def joint_tables(draw):
    V = draw(st.integers(2, 3))
    L = draw(st.integers(1, 4 if V == 2 else 3))
    w = draw(st.lists(st.floats(0.05, 1.0), min_size=V**L, max_size=V**L))
    y = draw(st.lists(st.integers(0, V - 1), min_size=L, max_size=L))
    return V, L, torch.tensor(w), y


@st.composite
def context_models(draw, max_L=4, prompt_len=0):
    V = draw(st.integers(2, 3))
    L = draw(st.integers(1, max_L if V == 2 else min(max_L, 3)))
    n = L * (L + 1) * (V + 1) * V
    w = draw(st.lists(st.floats(-2.0, 2.0), min_size=n, max_size=n))
    y = draw(st.lists(st.integers(0, V - 1), min_size=L, max_size=L))
    weights = torch.tensor(w).view(L, L + 1, V + 1, V)
    return V, L, weights, y


def exact_expectation(logits_fn, y, V, coupled):
    """Exact E[estimator] by enumerating every mask with its probability."""
    L = len(y)
    masks, probs = all_masks_with_probs(L)
    K = masks.shape[0]
    comp = torch.tensor(y).repeat(K, 1)
    est = sequence_elbo(
        logits_fn,
        prompt_ids=torch.zeros(K, 0, dtype=torch.long),
        prompt_mask=None,
        completion_ids=comp,
        masks=masks[None],
        mask_id=V,
        coupled=coupled,
    )
    return float((est.double() * probs).sum())


@given(joint_tables(), st.booleans())
@example((2, 1, torch.tensor([0.3, 0.7]), [1]), True)
@example((2, 1, torch.tensor([0.3, 0.7]), [1]), False)
def test_E1_elbo_is_exact_for_consistent_denoiser(case, coupled):
    """E1. For a denoiser that returns the exact conditionals of a joint p over V^L,
    the exact expectation of the (coupled or plain) estimator equals log p(y).
    Oracle: log p(y) read directly from the joint table.
    Catches: wrong (L+1)/l or (L+1)/(L-l) weights, wrong l range, wrong gather.
    """
    V, L, table, y = case
    logits_fn, log_prob = joint_conditional_denoiser(table, V, L)
    assert exact_expectation(logits_fn, y, V, coupled) == pytest.approx(log_prob(y), abs=1e-9)


@given(context_models(max_L=4), st.booleans())
def test_E2_elbo_equals_any_order_ar_likelihood(case, coupled):
    """E2. For an arbitrary (inconsistent) context-dependent denoiser, E[estimator]
    equals the any-order-AR log-likelihood averaged over all L! orderings.
    Oracle: brute-force permutation enumeration (a different formula, Ou et al. 2024).
    """
    V, L, weights, y = case
    logits_fn = context_denoiser(weights, V, L)
    expected = any_order_ar_loglik(logits_fn, y, V)
    assert exact_expectation(logits_fn, y, V, coupled) == pytest.approx(expected, abs=1e-9)


@given(context_models(max_L=3), st.booleans())
def test_E3_elbo_is_a_bound_on_a_normalised_likelihood(case, coupled):
    """E3. sum_y exp(E[ELBO(y)]) <= 1 for any denoiser, and every single-sample
    estimate is <= 0.  Oracle: probability normalisation; sign of log-probs.
    """
    V, L, weights, _ = case
    logits_fn = context_denoiser(weights, V, L)
    total = 0.0
    for y in itertools.product(range(V), repeat=L):
        total += math.exp(exact_expectation(logits_fn, list(y), V, coupled))
    assert total <= 1.0 + 1e-9

    masks, _ = all_masks_with_probs(L)
    comp = torch.tensor([0] * L).repeat(masks.shape[0], 1)
    est = sequence_elbo(logits_fn, torch.zeros(len(comp), 0, dtype=torch.long), None,
                        comp, masks[None], V, coupled=coupled)
    assert (est <= 1e-12).all()


@given(context_models(max_L=4, prompt_len=2), st.integers(1, 3), st.integers(1, 4),
       st.integers(0, 2**31 - 1), st.booleans())
def test_E4_antithetic_same_model_same_masks_gives_zero_log_ratio(case, M, B, seed, coupled):
    """E4. Same model + same masks => L_theta - L_old == 0 exactly (bitwise).
    Oracle: identity.  Catches: estimators that resample masks internally.
    """
    V, L, weights, y = case
    logits_fn = context_denoiser(weights, V, L, prompt_len=2)
    g = torch.Generator().manual_seed(seed)
    masks = sample_masks(M, B, L, generator=g)
    prompt = torch.randint(0, V, (B, 2), generator=g)
    comp = torch.tensor(y).repeat(B, 1)
    a = sequence_elbo(logits_fn, prompt, None, comp, masks, V, coupled=coupled)
    b = sequence_elbo(logits_fn, prompt, None, comp, masks.clone(), V, coupled=coupled)
    assert torch.equal(a - b, torch.zeros_like(a))


@given(st.integers(1, 4), st.integers(1, 9), st.integers(0, 40), st.integers(0, 2**31 - 1))
@example(1, 1, 0, 0)
def test_E5_masks_have_drawn_counts_and_complement_partitions(M, B, L, seed):
    """E5. Mask row sums are in [0, L]; complement of apply_mask partitions positions:
    every completion position is masked in exactly one of (noisy, noisy_complement).
    Oracle: set algebra.
    """
    g = torch.Generator().manual_seed(seed)
    masks = sample_masks(M, B, L, generator=g)
    assert masks.shape == (M, B, L) and masks.dtype == torch.bool
    counts = masks.sum(-1)
    assert ((counts >= 0) & (counts <= L)).all()
    y = torch.randint(0, 5, (B, L), generator=g)
    for m in range(M):
        noisy = apply_mask(y, masks[m], mask_id=99)
        noisy_c = apply_mask(y, ~masks[m], mask_id=99)
        masked_once = (noisy == 99) ^ (noisy_c == 99)
        assert masked_once.all()


@given(st.integers(1, 64), st.integers(0, 300))
@example(2, 256)  # (L+1)/B = 128.5: the round-half-to-even trap
@example(1, 0)
def test_E6_stratified_counts_have_exactly_uniform_marginals(B, L):
    """E6. For every batch slot b, k -> stratified_counts(k, B, L)[b] is a bijection on
    {0..L}, so each slot's l is exactly uniform when k is; for B <= L+1 counts within a
    draw are pairwise distinct.  Oracle: counting.
    Catches: round-half-to-even bias of round(k + b(L+1)/B).
    """
    table = torch.stack([stratified_counts(k, B, L) for k in range(L + 1)])  # [L+1, B]
    for b in range(B):
        assert sorted(table[:, b].tolist()) == list(range(L + 1))
    if B <= L + 1:
        for k in range(L + 1):
            assert len(set(table[k].tolist())) == B


def _chi2(observed, expected):
    return sum((o - e) ** 2 / e for o, e in zip(observed, expected))


@pytest.mark.parametrize("L,l", [(4, 2), (5, 1), (5, 3)])
def test_E7_subsets_given_count_are_uniform(L, l):
    """E7. Given l, every l-subset is equally likely (seeded chi-square, p ~ 1e-4 cutoff).
    Oracle: uniform over C(L, l).  Catches: positional bias.
    """
    g = torch.Generator().manual_seed(1234)
    n = 1500 * math.comb(L, l)
    masks = random_subsets(torch.full((n,), l), L, generator=g)
    assert (masks.sum(-1) == l).all()
    keys = [tuple(r) for r in masks.int().tolist()]
    subsets = list(itertools.combinations(range(L), l))
    obs = [keys.count(tuple(int(i in S) for i in range(L))) for S in subsets]
    dof = len(subsets) - 1
    if dof:
        assert _chi2(obs, [n / len(subsets)] * len(subsets)) < 3 * dof + 20


def test_E7b_count_marginal_is_uniform_for_single_row():
    """E7b. With B=1, l is uniform on {0..L} (seeded chi-square).  Oracle: uniform."""
    g = torch.Generator().manual_seed(7)
    L, n = 6, 14000
    counts = sample_masks(n, 1, L, generator=g).sum(-1).flatten().tolist()
    obs = [counts.count(v) for v in range(L + 1)]
    assert _chi2(obs, [n / (L + 1)] * (L + 1)) < 3 * L + 20


@given(context_models(max_L=4, prompt_len=3), st.integers(1, 3), st.integers(0, 2**31 - 1),
       st.booleans())
def test_E8_model_sees_prompt_intact_and_mask_only_on_masked_positions(case, B, seed, coupled):
    """E8. Every input the model sees has the prompt untouched and the completion equal
    to the clean completion except at masked positions, which hold mask_id; the set of
    masked positions is the given mask or (coupled) its complement.
    Oracle: a spy logits_fn recording its inputs.
    """
    V, L, weights, y = case
    inner = context_denoiser(weights, V, L, prompt_len=3)
    seen = []

    def spy(ids, am=None):
        seen.append(ids.clone())
        return inner(ids, am)

    g = torch.Generator().manual_seed(seed)
    masks = sample_masks(2, B, L, generator=g)
    prompt = torch.randint(0, V, (B, 3), generator=g)
    comp = torch.tensor(y).repeat(B, 1)
    sequence_elbo(spy, prompt, None, comp, masks, V, coupled=coupled)

    allowed = [masks[m] for m in range(2)] + ([~masks[m] for m in range(2)] if coupled else [])
    assert seen
    for ids in seen:
        for row in ids:
            p, c = row[:3], row[3:]
            is_mask = c == V
            # some batch slot b explains this row: same prompt, same unmasked tokens,
            # and a mask pattern drawn for slot b (or its complement when coupled)
            assert any(
                torch.equal(p, prompt[b])
                and torch.equal(torch.where(is_mask, comp[b], c), comp[b])
                and any(torch.equal(is_mask, a[b]) for a in allowed)
                for b in range(B)
            )


@given(context_models(max_L=4, prompt_len=2), st.integers(2, 4), st.integers(0, 2**31 - 1),
       st.booleans())
def test_E9_batch_rows_are_independent(case, B, seed, coupled):
    """E9. ELBO of row b in a batch == ELBO of row b computed alone with its mask.
    Oracle: single-row computation.  Catches: cross-row leakage, per-row l mix-ups.
    """
    V, L, weights, _ = case
    logits_fn = context_denoiser(weights, V, L, prompt_len=2)
    g = torch.Generator().manual_seed(seed)
    masks = sample_masks(2, B, L, generator=g)
    prompt = torch.randint(0, V, (B, 2), generator=g)
    comp = torch.randint(0, V, (B, L), generator=g)
    batch = sequence_elbo(logits_fn, prompt, None, comp, masks, V, coupled=coupled)
    for b in range(B):
        single = sequence_elbo(logits_fn, prompt[b:b + 1], None, comp[b:b + 1],
                               masks[:, b:b + 1], V, coupled=coupled)
        assert torch.allclose(batch[b:b + 1], single, atol=1e-9)


@given(context_models(max_L=4), st.integers(2, 4), st.integers(0, 2**31 - 1), st.booleans(),
       st.randoms(use_true_random=False))
def test_E10_mc_average_is_order_free_mean_of_single_estimates(case, M, seed, coupled, rnd):
    """E10. Permuting MC samples does not change the estimate; the M-sample estimate is
    the mean of the M single-sample estimates.  Oracle: mean of independent calls.
    """
    V, L, weights, y = case
    logits_fn = context_denoiser(weights, V, L)
    g = torch.Generator().manual_seed(seed)
    B = 2
    masks = sample_masks(M, B, L, generator=g)
    comp = torch.tensor(y).repeat(B, 1)
    empty = torch.zeros(B, 0, dtype=torch.long)
    full = sequence_elbo(logits_fn, empty, None, comp, masks, V, coupled=coupled)
    perm = list(range(M))
    rnd.shuffle(perm)
    permuted = sequence_elbo(logits_fn, empty, None, comp, masks[perm], V, coupled=coupled)
    singles = torch.stack([
        sequence_elbo(logits_fn, empty, None, comp, masks[m:m + 1], V, coupled=coupled)
        for m in range(M)
    ]).mean(0)
    assert torch.allclose(full, permuted, atol=1e-9)
    assert torch.allclose(full, singles, atol=1e-9)
