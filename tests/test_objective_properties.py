"""Property tests for the ESPO objective (PROPERTIES.md section 2, O1-O10)."""
import math

import pytest
import torch
from hypothesis import assume, example, given, strategies as st

from espo.objective import (
    clipped_surrogate,
    espo_loss,
    group_advantages,
    kl_estimate,
    sequence_ratio,
)

rewards_val = st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False)


@st.composite
def grouped_rewards(draw, min_groups=1, max_groups=4):
    G = draw(st.integers(2, 6))
    n = draw(st.integers(min_groups, max_groups))
    r = draw(st.lists(rewards_val, min_size=G * n, max_size=G * n))
    return G, torch.tensor(r, dtype=torch.float64)


# ---------------------------------------------------------------- advantages
@given(grouped_rewards(), st.sampled_from(["mean", "loo"]), st.booleans())
@example((3, torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)), "loo", True)
def test_O1_advantages_are_centred_per_group(case, baseline, scale):
    """O1. Within each group: advantages sum to 0; all-equal => zeros; the best sample
    has A >= 0 and the worst A <= 0.  Oracle: arithmetic of a centred baseline.
    """
    G, r = case
    a = group_advantages(r, G, baseline=baseline, scale=scale).view(-1, G)
    rg = r.view(-1, G)
    assert torch.allclose(a.sum(1), torch.zeros(a.shape[0], dtype=a.dtype), atol=1e-6)
    for i in range(rg.shape[0]):
        if torch.all(rg[i] == rg[i, 0]):
            assert torch.all(a[i] == 0)
        assert a[i, rg[i].argmax()] >= -1e-9
        assert a[i, rg[i].argmin()] <= 1e-9


@given(grouped_rewards(), st.sampled_from(["mean", "loo"]),
       st.lists(st.floats(-3, 3), min_size=4, max_size=4), st.floats(0.1, 10.0),
       st.randoms(use_true_random=False))
def test_O2_advantage_metamorphic_relations(case, baseline, shifts, a_scale, rnd):
    """O2. Shift invariance per group; scaling by a>0 scales unscaled advantages by a
    and leaves std-scaled ones unchanged (up to eps); within-group permutation is
    equivariant; changing one group never affects another.  Oracle: metamorphic.
    """
    G, r = case
    n = r.numel() // G
    base = group_advantages(r, G, baseline=baseline)
    shift = torch.tensor(shifts[:n], dtype=r.dtype).repeat_interleave(G)
    assert torch.allclose(group_advantages(r + shift, G, baseline=baseline), base, atol=1e-6)
    assert torch.allclose(group_advantages(r * a_scale, G, baseline=baseline), base * a_scale,
                          atol=1e-6)
    # permutation within each group
    perm = []
    for g in range(n):
        idx = list(range(g * G, (g + 1) * G))
        rnd.shuffle(idx)
        perm += idx
    assert torch.allclose(group_advantages(r[perm], G, baseline=baseline), base[perm], atol=1e-9)
    # group isolation
    if n >= 2:
        r2 = r.clone()
        r2[:G] = r2[:G] * 3 + 1
        assert torch.equal(group_advantages(r2, G, baseline=baseline)[G:], base[G:])


@given(grouped_rewards(), st.floats(0.5, 10.0))
def test_O2b_std_scaled_advantages_are_scale_free(case, a_scale):
    """O2b. With std scaling, multiplying rewards by a>0 leaves advantages unchanged
    for groups whose std dominates eps.  Oracle: metamorphic.
    """
    G, r = case
    rg = r.view(-1, G)
    assume(bool((rg.std(1) > 0.5).all()))
    a1 = group_advantages(r, G, baseline="mean", scale=True, eps=1e-12)
    a2 = group_advantages(r * a_scale, G, baseline="mean", scale=True, eps=1e-12)
    assert torch.allclose(a1, a2, atol=1e-6)


@given(grouped_rewards())
def test_O3_leave_one_out_is_rescaled_mean_baseline(case):
    """O3. loo advantages == G/(G-1) * mean-baseline advantages.
    Oracle: r - (S - r)/(G-1) = G/(G-1) (r - S/G).
    """
    G, r = case
    loo = group_advantages(r, G, baseline="loo")
    mean = group_advantages(r, G, baseline="mean")
    assert torch.allclose(loo, mean * G / (G - 1), atol=1e-9)


def test_O_error_group_size_must_divide_batch():
    """Partiality: a batch that is not a whole number of groups is rejected."""
    with pytest.raises(ValueError):
        group_advantages(torch.zeros(5), 2)
    with pytest.raises(ValueError):
        group_advantages(torch.zeros(4), 1)


# ---------------------------------------------------------------- ratio
@given(st.floats(-50, 50), st.integers(1, 2048), st.floats(-30, 30))
@example(0.0, 1, 0.0)
def test_O4_sequence_ratio_is_per_token_normalised(delta, L, old):
    """O4. rho > 0; rho == 1 iff Delta == 0; if Delta = L*delta then rho = exp(delta)
    for every L (length invariance of Eq. 8).  Oracle: math.exp of the per-token ratio.
    Catches: missing /L (Eq. 6's exploding ratio).
    """
    old_t = torch.tensor([old], dtype=torch.float64)
    rho = sequence_ratio(old_t + L * delta, old_t, L)
    assert rho.item() > 0
    assert rho.item() == pytest.approx(math.exp(delta), rel=1e-9)
    if delta == 0:
        assert rho.item() == 1.0


@given(st.floats(-100, 100), st.floats(-100, 100), st.integers(1, 512))
def test_O4b_ratio_is_monotone_in_elbo(e1, e2, L):
    """O4b. rho is strictly increasing in L_theta.  Oracle: ordering."""
    assume(abs(e1 - e2) > 1e-6)
    old = torch.tensor([0.0], dtype=torch.float64)
    r1 = sequence_ratio(torch.tensor([e1], dtype=torch.float64), old, L)
    r2 = sequence_ratio(torch.tensor([e2], dtype=torch.float64), old, L)
    assert (r1 < r2).item() == (e1 < e2)


# ---------------------------------------------------------------- surrogate
ratios = st.floats(1e-3, 10.0)
advs = st.floats(-5, 5)
eps_s = st.floats(0.01, 0.5)


@given(ratios, advs, eps_s)
@example(1.0, 1.0, 0.2)
@example(1.5, 1.0, 0.2)
@example(0.5, -1.0, 0.2)
def test_O5_clipped_surrogate_bounds(rho, a, eps):
    """O5. J == rho*A inside [1-eps, 1+eps]; J <= rho*A and J <= clip(rho)*A always;
    for A > 0, J <= (1+eps)*A.  Oracle: PPO definition split into inequalities.
    """
    r = torch.tensor([rho], dtype=torch.float64)
    A = torch.tensor([a], dtype=torch.float64)
    J = clipped_surrogate(r, A, eps).item()
    clipped = min(max(rho, 1 - eps), 1 + eps)
    assert J <= rho * a + 1e-12
    assert J <= clipped * a + 1e-12
    if 1 - eps <= rho <= 1 + eps:
        assert J == pytest.approx(rho * a, abs=1e-12)
    if a > 0:
        assert J <= (1 + eps) * a + 1e-12


@given(ratios, ratios, advs, eps_s)
def test_O5b_surrogate_monotone_in_direction_of_advantage(r1, r2, a, eps):
    """O5b. J is non-decreasing in rho when A > 0 and non-increasing when A < 0."""
    lo, hi = min(r1, r2), max(r1, r2)
    A = torch.tensor([a], dtype=torch.float64)
    J_lo = clipped_surrogate(torch.tensor([lo], dtype=torch.float64), A, eps).item()
    J_hi = clipped_surrogate(torch.tensor([hi], dtype=torch.float64), A, eps).item()
    if a > 0:
        assert J_hi >= J_lo - 1e-12
    elif a < 0:
        assert J_hi <= J_lo + 1e-12


@given(st.floats(-3, 3), advs, eps_s, st.integers(1, 64))
def test_O6_surrogate_gradient_matches_finite_differences(log_ratio_per_tok, a, eps, L):
    """O6. dJ/dL_theta equals the central finite difference; it is exactly 0 when the
    ratio is clipped on the beneficial side and A*rho/L otherwise.
    Oracle: finite differences (independent of autograd).
    """
    rho = math.exp(log_ratio_per_tok)
    assume(abs(rho - (1 + eps)) > 1e-3 and abs(rho - (1 - eps)) > 1e-3)
    old = torch.tensor([0.0], dtype=torch.float64)
    A = torch.tensor([a], dtype=torch.float64)

    def f(e):
        return clipped_surrogate(sequence_ratio(e, old, L), A, eps)

    e = torch.tensor([L * log_ratio_per_tok], dtype=torch.float64, requires_grad=True)
    f(e).sum().backward()
    h = 1e-6 * L
    fd = (f(e.detach() + h) - f(e.detach() - h)).item() / (2 * h)
    assert e.grad.item() == pytest.approx(fd, abs=1e-6)
    beneficial_clip = (a > 0 and rho > 1 + eps) or (a < 0 and rho < 1 - eps)
    if beneficial_clip:
        assert e.grad.item() == 0.0
    else:
        assert e.grad.item() == pytest.approx(a * rho / L, rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------- KL
@given(st.floats(-40, 40), st.floats(-40, 40), st.integers(1, 512))
@example(0.0, 0.0, 1)
def test_O7_kl_estimator_invariants(e, ref, L):
    """O7. k2 = 0.5*D^2/L >= 0, zero iff D == 0, symmetric in D -> -D; k3 >= 0;
    k1 < 0 when D < 0.  Oracle: inequalities.
    """
    E = torch.tensor([e], dtype=torch.float64)
    R = torch.tensor([ref], dtype=torch.float64)
    k2 = kl_estimate(E, R, L, "k2").item()
    assert k2 >= 0
    if e == ref:
        assert k2 == 0
    elif abs(e - ref) > 1e-100:  # below this, D^2 underflows in float64
        assert k2 > 0
    assert k2 == pytest.approx(kl_estimate(R, E, L, "k2").item(), rel=1e-12, abs=0)
    assert kl_estimate(E, R, L, "k3").item() >= -1e-12
    if e < ref:
        assert kl_estimate(E, R, L, "k1").item() < 0


@given(st.floats(-0.5, 0.5), st.integers(1, 256))
def test_O7b_k3_agrees_with_k2_to_second_order(d, L):
    """O7b. |k3 - k2| * L <= |D|^3 for |D| <= 0.5 (Taylor: k3 = D^2/2 - D^3/6 + ...)."""
    E = torch.tensor([d], dtype=torch.float64)
    Z = torch.zeros(1, dtype=torch.float64)
    k2 = kl_estimate(E, Z, L, "k2").item()
    k3 = kl_estimate(E, Z, L, "k3").item()
    assert abs(k3 - k2) * L <= abs(d) ** 3 + 1e-15


@given(st.floats(-20, 20), st.floats(-20, 20), st.integers(1, 256), st.floats(0.0, 1.0))
def test_O8_k2_gradient_points_to_reference(e, ref, L, beta):
    """O8. d(beta*k2)/dL_theta == beta*(L_theta - L_ref)/L (finite differences)."""
    R = torch.tensor([ref], dtype=torch.float64)
    E = torch.tensor([e], dtype=torch.float64, requires_grad=True)
    (beta * kl_estimate(E, R, L, "k2")).sum().backward()
    h = 1e-5
    fd = beta * (kl_estimate(E.detach() + h, R, L, "k2") - kl_estimate(E.detach() - h, R, L, "k2"))
    assert E.grad.item() == pytest.approx(fd.item() / (2 * h), abs=1e-6)
    assert E.grad.item() == pytest.approx(beta * (e - ref) / L, abs=1e-9)


# ---------------------------------------------------------------- full loss
@st.composite
def loss_batches(draw):
    n = draw(st.integers(1, 8))
    L = draw(st.integers(1, 256))
    f = st.floats(-300, 0)
    elbo = torch.tensor(draw(st.lists(f, min_size=n, max_size=n)), dtype=torch.float64)
    d_old = torch.tensor(draw(st.lists(st.floats(-0.3, 0.3), min_size=n, max_size=n)),
                         dtype=torch.float64)
    ref = torch.tensor(draw(st.lists(f, min_size=n, max_size=n)), dtype=torch.float64)
    adv = torch.tensor(draw(st.lists(advs, min_size=n, max_size=n)), dtype=torch.float64)
    return elbo, elbo - L * d_old, ref, adv, L


@given(loss_batches(), st.integers(1, 16))
def test_O9_on_policy_gradient(batch, denom):
    """O9. At L_theta == L_old with beta = 0: dloss/dL_theta_i == -A_i/(L*denom); zero
    advantages => zero gradient.  Oracle: hand-derived gradient of Eq. 7 at rho = 1.
    """
    elbo, _, ref, adv, L = batch
    e = elbo.clone().requires_grad_(True)
    loss, _ = espo_loss(e, elbo.clone(), ref, adv, L, epsilon=0.2, beta=0.0, denom=denom)
    loss.backward()
    assert torch.allclose(e.grad, -adv / (L * denom), atol=1e-12)
    e2 = elbo.clone().requires_grad_(True)
    loss0, _ = espo_loss(e2, elbo.clone(), ref, torch.zeros_like(adv), L, 0.2, 0.0, denom)
    loss0.backward()
    assert torch.all(e2.grad == 0)


@given(loss_batches(), st.floats(0.0, 0.5), st.randoms(use_true_random=False), st.data())
def test_O10_loss_is_order_free_and_accumulates_exactly(batch, beta, rnd, data):
    """O10. Permuting samples leaves the loss unchanged; summing micro-batch losses that
    use the full-batch denominator equals the full-batch loss (exact accumulation).
    Oracle: full-batch computation.
    """
    elbo, old, ref, adv, L = batch
    n = elbo.numel()
    full, _ = espo_loss(elbo, old, ref, adv, L, 0.2, beta, denom=n)
    perm = list(range(n))
    rnd.shuffle(perm)
    permuted, _ = espo_loss(elbo[perm], old[perm], ref[perm], adv[perm], L, 0.2, beta, denom=n)
    assert permuted.item() == pytest.approx(full.item(), rel=1e-9, abs=1e-9)
    cut = data.draw(st.integers(0, n))
    parts = [slice(0, cut), slice(cut, n)]
    acc = sum(espo_loss(elbo[s], old[s], ref[s], adv[s], L, 0.2, beta, denom=n)[0]
              for s in parts if s.stop > s.start)
    assert float(acc) == pytest.approx(full.item(), rel=1e-9, abs=1e-9)


def test_O_loss_rejects_missing_ref_when_beta_positive():
    """Partiality: beta > 0 requires reference ELBOs."""
    z = torch.zeros(2)
    with pytest.raises(ValueError):
        espo_loss(z, z, None, z, 4, 0.2, beta=0.1, denom=2)


@st.composite
def categorical_pairs(draw):
    n = draw(st.integers(2, 6))
    a = draw(st.lists(st.floats(0.05, 1.0), min_size=n, max_size=n))
    b = draw(st.lists(st.floats(0.05, 1.0), min_size=n, max_size=n))
    p = torch.tensor(a, dtype=torch.float64)
    q = torch.tensor(b, dtype=torch.float64)
    return p / p.sum(), q / q.sum()


@given(categorical_pairs(), st.integers(1, 64))
def test_O7c_k1_k3_are_unbiased_for_forward_kl(pq, L):
    """O7c. With exact log-likelihoods, E_{y~pi_theta}[L * k] == KL(pi_theta || pi_ref) for
    k1 and k3 (Schulman 2017), and E[L * k2] >= 0.
    Oracle: KL computed from its definition over the outcomes.
    Catches: k3 written for the reverse direction (exp(D) - 1 - D).
    """
    p, q = pq
    kl = float((p * (p.log() - q.log())).sum())
    for kind in ("k1", "k3"):
        est = float((p * kl_estimate(p.log(), q.log(), L, kind) * L).sum())
        assert est == pytest.approx(kl, rel=1e-9, abs=1e-12)
    assert float((p * kl_estimate(p.log(), q.log(), L, "k2")).sum()) >= 0


@given(loss_batches(), st.floats(0.01, 1.0), st.integers(1, 16))
def test_O8b_loss_kl_gradient_pulls_toward_reference(batch, beta, denom):
    """O8b. With zero advantages, dloss/dL_theta_i == +beta*(L_theta_i - L_ref_i)/(L*denom),
    so a gradient-descent step moves L_theta toward L_ref.  Oracle: derivative of
    beta*0.5*D^2/L, by hand.  Catches: KL term entering the loss with the wrong sign.
    """
    elbo, old, ref, adv, L = batch
    e = elbo.clone().requires_grad_(True)
    loss, _ = espo_loss(e, old, ref, torch.zeros_like(adv), L, 0.2, beta, denom=denom)
    loss.backward()
    assert torch.allclose(e.grad, beta * (elbo - ref) / (L * denom), atol=1e-12)
