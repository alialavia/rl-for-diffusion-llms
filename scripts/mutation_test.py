"""Mutation testing for the property suite.

Each mutant is a plausible implementation bug. The property tests must fail ("kill")
for every mutant; a surviving mutant means a missing or too-weak property.

    python scripts/mutation_test.py            # all mutants
    python scripts/mutation_test.py k3 elbo    # mutants whose name contains a substring
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
MUTANTS = [
  ("elbo: L/l instead of (L+1)/l", "espo/elbo.py", "w = (L + 1) / l.clamp", "w = L / l.clamp", "tests/test_elbo_properties.py"),
  ("elbo: complement weight L/(L-l)", "espo/elbo.py", "wc = (L + 1) / (L - l)", "wc = L / (L - l)", "tests/test_elbo_properties.py"),
  ("elbo: sum MC instead of mean", "espo/elbo.py", "return total / M", "return total", "tests/test_elbo_properties.py"),
  ("elbo: banker's rounding stratification", "espo/elbo.py",
   'return (int(k) + offsets) % (L + 1)',
   'return torch.round(int(k) + torch.arange(batch) * (L + 1) / batch).long() % (L + 1)', "tests/test_elbo_properties.py"),
  ("elbo: mask also first prompt token", "espo/elbo.py",
   "noisy = torch.cat([prompt_ids, apply_mask(completion_ids, mask, mask_id)], dim=1)",
   "noisy = torch.cat([prompt_ids, apply_mask(completion_ids, mask, mask_id)], dim=1); noisy[:, :min(P,1)] = mask_id",
   "tests/test_elbo_properties.py"),
  ("objective: ratio without /L", "espo/objective.py", "torch.exp((elbo - old_elbo) / length)", "torch.exp(elbo - old_elbo)", "tests/test_objective_properties.py"),
  ("objective: max instead of min", "espo/objective.py", "return torch.minimum(ratio * advantages", "return torch.maximum(ratio * advantages", "tests/test_objective_properties.py"),
  ("objective: loo divides by G", "espo/objective.py", "b = (r.sum(dim=1, keepdim=True) - r) / (G - 1)", "b = (r.sum(dim=1, keepdim=True) - r) / G", "tests/test_objective_properties.py"),
  ("objective: k3 sign flipped", "espo/objective.py", "v = torch.exp(-d) - 1 + d", "v = torch.exp(d) - 1 - d", "tests/test_objective_properties.py"),
  ("objective: kl not detached ref -> wrong sign", "espo/objective.py", "loss = loss + beta * kl.sum() / denom", "loss = loss - beta * kl.sum() / denom", "tests/test_objective_properties.py tests/test_trainer_properties.py"),
  ("objective: loss sign (gradient ascent)", "espo/objective.py", "loss = -surrogate.sum() / denom", "loss = surrogate.sum() / denom", "tests/test_objective_properties.py tests/test_trainer_properties.py"),
  ("objective: per-microbatch mean", "espo/objective.py", "loss = -surrogate.sum() / denom", "loss = -surrogate.mean()", "tests/test_objective_properties.py"),
  ("sampler: lowest confidence first", "espo/sampler.py", "top = conf.topk(kmax, dim=-1).indices", "top = (-conf.nan_to_num(neginf=float('inf'))).topk(kmax, dim=-1).indices", "tests/test_sampler_properties.py"),
  ("sampler: temperature multiplies", "espo/sampler.py", "(x / temperature + gumbel)", "(x * temperature + gumbel)", "tests/test_sampler_properties.py"),
  ("sampler: decode beyond block", "espo/sampler.py", "logits = logits_fn(x, attn)[:, lo:hi]", "hi = P + gen_length; logits = logits_fn(x, attn)[:, lo:hi]", "tests/test_sampler_properties.py"),
  ("trainer: fresh masks for update (no antithetic)", "espo/trainer.py", "ro.completion_ids[s], ro.masks[iteration][:, s],",
   "ro.completion_ids[s], sample_masks(cfg.num_mc, s.stop - s.start, L, generator=self.mask_generator),", "tests/test_trainer_properties.py"),
  ("trainer: old elbo from wrong iteration", "espo/trainer.py", "old = ro.old_elbo[iteration, s]", "old = ro.old_elbo[0, s]", "tests/test_trainer_properties.py"),
  ("rewards: sudoku counts all cells", "espo/rewards.py", "empty = [i for i, c in enumerate(puzzle) if c == \"0\"]", "empty = list(range(16))", "tests/test_rewards_properties.py"),
  ("rewards: countdown ignores number multiset", "espo/rewards.py", "if Counter(used) != Counter(int(n) for n in numbers):", "if False:", "tests/test_rewards_properties.py"),
  ("rewards: first boxed instead of last", "espo/rewards.py", 'start = text.rfind("\\\\boxed{")', 'start = text.find("\\\\boxed{")', "tests/test_rewards_properties.py"),
  ("ddp: forward bypasses DDP (no grad all-reduce)", "espo/toy.py",
   "m = self.ddp if self.ddp is not None else self.model", "m = self.model", "tests/test_distributed.py"),
  ("data: no uniqueness check", "espo/data.py", "if _count_solutions(cells[:]) == 1:", "if True:", "tests/test_data_properties.py"),
]
env = dict(os.environ, HYPOTHESIS_PROFILE="ci", PYTHONDONTWRITEBYTECODE="1")
survivors = []
only = sys.argv[1:]
for name, path, old, new, tests in MUTANTS:
    if only and not any(o in name for o in only): continue
    p = os.path.join(ROOT, path); src = open(p).read()
    assert src.count(old) == 1, (name, src.count(old))
    open(p, "w").write(src.replace(old, new))
    try:
        r = subprocess.run([PY, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *tests.split()],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
        killed = r.returncode != 0
        failing = [l for l in r.stdout.splitlines() if l.startswith("FAILED")][:1]
    finally:
        open(p, "w").write(src)
    print(f"{'KILLED ' if killed else 'SURVIVED'} {name}  {failing[0][7:] if failing else ''}", flush=True)
    if not killed: survivors.append(name)
print("survivors:", survivors)
sys.exit(1 if survivors else 0)
