"""Property tests for the masked-diffusion sampler (PROPERTIES.md section 3, S1-S8)."""
import pytest
import torch
from hypothesis import example, given, strategies as st

from espo.sampler import generate, transfer_schedule
from tests.denoisers import context_denoiser

NEG = -1e9


@given(st.lists(st.integers(0, 600), min_size=1, max_size=6), st.integers(1, 300))
@example([0], 1)
@example([32], 16)
@example([5], 7)
def test_S1_transfer_schedule(ns, steps):
    """S1. Schedule sums to n, entries >= 0, max - min <= 1, non-increasing.
    Oracle: arithmetic.
    """
    sched = transfer_schedule(torch.tensor(ns), steps)
    assert sched.shape == (len(ns), steps)
    assert torch.equal(sched.sum(1), torch.tensor(ns))
    assert (sched >= 0).all()
    assert ((sched.max(1).values - sched.min(1).values) <= 1).all()
    assert (sched[:, :-1] >= sched[:, 1:]).all()


@st.composite
def gen_configs(draw, max_len=12):
    block = draw(st.integers(1, 4))
    nblocks = draw(st.integers(1, 3))
    gen_length = block * nblocks
    steps_per_block = draw(st.integers(1, block + 1))
    return gen_length, block, steps_per_block * nblocks


def _random_model(seed, V, L, P):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(L, L + 1, V + 1, V, generator=g)
    return context_denoiser(w, V, L, prompt_len=P)


@given(gen_configs(), st.integers(1, 3), st.integers(0, 2**31 - 1),
       st.sampled_from([0.0, 0.7, 1.0]), st.sampled_from(["low_confidence", "random", "entropy"]))
def test_S2_S3_output_and_decoding_trace(cfg, B, seed, temperature, remasking):
    """S2. No [MASK] in the output, prompt unchanged, shape [B, gen_length].
    S3. Over the trace: each step reveals exactly schedule[step] tokens per row,
    revealed tokens never change, and positions in later blocks stay masked.
    Oracle: replay of the trace against a simple model of the decoding state.
    """
    gen_length, block, steps = cfg
    V, P = 3, 2
    mask_id = V
    logits_fn = _random_model(seed, V, gen_length, P)
    g = torch.Generator().manual_seed(seed)
    prompt = torch.randint(0, V, (B, P), generator=g)
    prompt_copy = prompt.clone()
    states = []
    out = generate(logits_fn, prompt, None, gen_length=gen_length, block_length=block,
                   steps=steps, mask_id=mask_id, temperature=temperature, remasking=remasking,
                   generator=g, trace=lambda step, blk, x: states.append((blk, x.clone())))
    assert out.shape == (B, gen_length)
    assert not (out == mask_id).any()
    assert torch.equal(prompt, prompt_copy)

    nblocks = gen_length // block
    spb = steps // nblocks
    sched = transfer_schedule(torch.tensor([block]), spb)[0]
    assert len(states) == steps
    prev = torch.full((B, gen_length), mask_id)
    for s, (blk, x) in enumerate(states):
        assert blk == s // spb
        revealed_before = prev != mask_id
        assert torch.equal(x[revealed_before], prev[revealed_before])  # committed tokens fixed
        new = (x != mask_id) & ~revealed_before
        assert (new.sum(1) == sched[s % spb]).all()
        assert not (x[:, (blk + 1) * block:] != mask_id).any()  # later blocks untouched
        assert not new[:, : blk * block].any()  # earlier blocks were already complete
        prev = x
    assert torch.equal(prev, out)


@given(gen_configs(), st.integers(1, 3), st.integers(0, 2**31 - 1),
       st.sampled_from([0.0, 0.5, 1.0]), st.sampled_from(["low_confidence", "random", "entropy"]))
def test_S4_point_mass_denoiser_reproduces_target(cfg, B, seed, temperature, remasking):
    """S4. A denoiser that is a point mass on a target yields exactly the target.
    Oracle: the target sequence.
    """
    gen_length, block, steps = cfg
    V, P = 5, 3
    g = torch.Generator().manual_seed(seed)
    target = torch.randint(0, V, (B, gen_length), generator=g)
    prompt = torch.randint(0, V, (B, P), generator=g)

    def logits_fn(ids, am=None):
        out = torch.full((ids.shape[0], ids.shape[1], V + 1), 0.0)
        out[..., V] = NEG
        out[:, P:].scatter_(-1, target[..., None], 1e3)
        return out

    out = generate(logits_fn, prompt, None, gen_length, block, steps, mask_id=V,
                   temperature=temperature, remasking=remasking, generator=g)
    assert torch.equal(out, target)


@given(st.integers(1, 4), st.integers(1, 3), st.data())
def test_S5_low_confidence_reveals_most_confident_first(block, nblocks, data):
    """S5. Greedy low-confidence decoding of a context-free model with known per-position
    peak logits reveals, at each step, positions at least as confident as every
    still-masked position of the current block.  Oracle: the known confidences.
    """
    gen_length = block * nblocks
    spb = data.draw(st.integers(1, block))
    V = 4
    # distinct, well-separated peaks > 0 (so token 0 is the unique argmax and the
    # confidences are distinguishable after softmax in float32)
    peaks = data.draw(st.permutations([1.0 + 0.5 * k for k in range(gen_length)]))
    peaks_t = torch.tensor(peaks)

    def logits_fn(ids, am=None):
        out = torch.zeros(ids.shape[0], ids.shape[1], V + 1)
        out[..., V] = NEG
        out[0, :, 0] = peaks_t  # token 0 is the argmax everywhere, with peak height c_j
        return out

    states = []
    generate(logits_fn, torch.zeros(1, 0, dtype=torch.long), None, gen_length, block,
             spb * nblocks, mask_id=V, temperature=0.0, remasking="low_confidence",
             trace=lambda s, b, x: states.append((b, x.clone())))
    prev = torch.full((gen_length,), V)
    for blk, x in states:
        x = x[0]
        new = (x != V) & (prev == V)
        still = x == V
        lo, hi = blk * block, (blk + 1) * block
        cand = still.clone()
        cand[:lo] = False
        cand[hi:] = False
        if new.any() and cand.any():
            assert peaks_t[new].min() >= peaks_t[cand].max()
        prev = x


@given(gen_configs(), st.integers(0, 2**31 - 1), st.sampled_from([0.0, 0.9]))
def test_S6_determinism(cfg, seed, temperature):
    """S6. temperature 0 => identical outputs; temperature > 0 with the same generator
    seed => identical outputs.  Oracle: re-run.
    """
    gen_length, block, steps = cfg
    V, P = 3, 2
    logits_fn = _random_model(seed, V, gen_length, P)
    prompt = torch.randint(0, V, (2, P), generator=torch.Generator().manual_seed(seed))
    outs = [
        generate(logits_fn, prompt, None, gen_length, block, steps, mask_id=V,
                 temperature=temperature, generator=torch.Generator().manual_seed(seed + 1))
        for _ in range(2)
    ]
    assert torch.equal(outs[0], outs[1])


@pytest.mark.parametrize("tau", [1.0, 0.5, 2.0])
def test_S7_sampling_matches_tempered_softmax(tau):
    """S7. With a context-free model, sampled token frequencies match softmax(logits/tau)
    (seeded chi-square).  Oracle: softmax.  Catches: wrong Gumbel temperature exponent.
    """
    V, n = 4, 20000
    logits = torch.tensor([1.0, 0.0, -0.5, 0.7])

    def logits_fn(ids, am=None):
        out = torch.zeros(ids.shape[0], ids.shape[1], V + 1)
        out[..., :V] = logits
        out[..., V] = NEG
        return out

    out = generate(logits_fn, torch.zeros(n, 0, dtype=torch.long), None, 1, 1, 1, mask_id=V,
                   temperature=tau, generator=torch.Generator().manual_seed(0))
    obs = torch.bincount(out.flatten(), minlength=V).double()
    exp = torch.softmax(logits.double() / tau, 0) * n
    chi2 = ((obs - exp) ** 2 / exp).sum().item()
    assert chi2 < 25  # 3 dof; p << 1e-4


@pytest.mark.parametrize("gen_length,block,steps", [(10, 4, 4), (8, 4, 3), (8, 4, 0), (8, 0, 2)])
def test_S8_invalid_configs_raise(gen_length, block, steps):
    """S8. gen_length % block != 0, or steps not a positive multiple of #blocks, raise."""
    with pytest.raises(ValueError):
        generate(lambda ids, am=None: torch.zeros(*ids.shape, 3), torch.zeros(1, 0, dtype=torch.long),
                 None, gen_length, block, steps, mask_id=2)
