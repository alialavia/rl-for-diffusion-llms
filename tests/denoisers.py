"""Hand-built denoisers used as oracles in the property tests.

These are deliberately written independently of `espo` (pure loops over small
tables) so they can serve as oracles for the ELBO estimator and the sampler.
All denoisers follow the `logits_fn(input_ids, attention_mask) -> logits` contract,
with vocabulary {0..V-1} plus a mask token `V` (whose logit is -1e9).
"""
import itertools
import math

import torch

NEG = -1e9


def joint_conditional_denoiser(table: torch.Tensor, V: int, L: int, prompt_len: int = 0):
    """Exact conditionals p(y_i = v | revealed tokens) of a joint table over V^L.

    `table` is a flat tensor of V**L positive weights (normalised here).
    """
    p = table.double() / table.double().sum()
    configs = list(itertools.product(range(V), repeat=L))

    def logits_fn(input_ids, attention_mask=None):
        B, T = input_ids.shape
        out = torch.zeros(B, T, V + 1, dtype=torch.float64)
        out[..., V] = NEG
        for b in range(B):
            comp = input_ids[b, prompt_len:].tolist()
            consistent = [
                (c, p[k].item())
                for k, c in enumerate(configs)
                if all(t == V or t == c[j] for j, t in enumerate(comp))
            ]
            for i in range(L):
                if comp[i] != V:
                    continue
                for v in range(V):
                    num = sum(w for c, w in consistent if c[i] == v)
                    out[b, prompt_len + i, v] = math.log(num) if num > 0 else NEG
        return out

    def log_prob(y):
        return math.log(p[configs.index(tuple(y))].item())

    return logits_fn, log_prob


def context_denoiser(weights: torch.Tensor, V: int, L: int, prompt_len: int = 0):
    """Arbitrary (generally inconsistent) context-dependent denoiser.

    weights: [L, L+1, V+1, V].  logit[i, v] = sum_j weights[i, j, token_j, v]
    over completion positions j (mask token included as a value), plus a
    prompt-dependent term via slot L (sum of prompt tokens mod (V+1)).
    """
    w = weights.double()

    def logits_fn(input_ids, attention_mask=None):
        B, T = input_ids.shape
        out = torch.zeros(B, T, V + 1, dtype=torch.float64)
        out[..., V] = NEG
        for b in range(B):
            comp = input_ids[b, prompt_len:].tolist()
            ptok = int(input_ids[b, :prompt_len].sum().item()) % (V + 1) if prompt_len else 0
            for i in range(L):
                acc = w[i, L, ptok].clone()
                for j, t in enumerate(comp):
                    acc = acc + w[i, j, t]
                out[b, prompt_len + i, :V] = acc
        return out

    return logits_fn


def any_order_ar_loglik(logits_fn, y, V):
    """E_sigma sum_k log p(y_sigma(k) | y_sigma(<k)) by brute-force over all L! orders."""
    L = len(y)
    total = 0.0
    for perm in itertools.permutations(range(L)):
        x = torch.full((1, L), V, dtype=torch.long)
        s = 0.0
        for pos in perm:
            lp = torch.log_softmax(logits_fn(x, None)[0, pos].double(), -1)[y[pos]].item()
            s += lp
            x[0, pos] = y[pos]
        total += s
    return total / math.factorial(L)


def all_masks_with_probs(L):
    """Every (l, subset) of the l~U{0..L}, uniform-subset mask distribution."""
    masks, probs = [], []
    for l in range(L + 1):
        for S in itertools.combinations(range(L), l):
            m = torch.zeros(L, dtype=torch.bool)
            m[list(S)] = True
            masks.append(m)
            probs.append(1.0 / ((L + 1) * math.comb(L, l)))
    return torch.stack(masks), torch.tensor(probs, dtype=torch.float64)
