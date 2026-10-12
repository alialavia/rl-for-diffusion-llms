# Agent and contributor instructions

These instructions apply to every coding agent (Claude Code, Codex, Cursor, …) and to people.

This is an implementation of ESPO (arXiv:2512.03759). It is developed **property-first**
with the [`alialavia/pbt-skills`](https://github.com/alialavia/pbt-skills) methodology.
A copy of those skills is vendored in this repo (MIT, see `.claude/skills/LICENSE-pbt-skills`,
upstream commit 7ff41bb). `.agents/skills` is a symlink to the same files.

**Before changing code or tests, read these files.** If your agent does not auto-load
skills, open them directly:
- `.claude/skills/property-based-testing/SKILL.md`: the core workflow. Then read:
  - `references/property-driven-design.md` for new behaviour;
  - `references/interpreting-failures.md` when a property fails;
  - `references/anti-patterns.md` before writing any test.
- `.claude/skills/pbt-hypothesis/SKILL.md`: Hypothesis specifics.

The rules below are the repo-specific minimum on top of those.

## Sources of truth
- `PROPERTIES.md` is the spec. Every property has an ID, a "for all … holds" statement, an
  **independent oracle**, and the bugs it catches. Tests cite the ID in their docstring.
- The paper is the spec behind the spec. Where we follow the authors' released code
  instead, or deviate from it, the "Notes on deviations" section of `PROPERTIES.md` says so.

## Workflow for any change

1. **Spec before code.** For a new feature or behaviour change, first add or edit the
   property in `PROPERTIES.md`, using the skill's property-driven-design mode. Then write
   the test, then the code.
2. **Bug fix = failing property first.** For a bug seen on GPUs or in a real run:
   - Reproduce it on CPU as a property that fails today. Use the toy model (`espo/toy.py`),
     the tiny LLaDA (`tests/test_hf_integration.py`) or 2-rank gloo DDP (`tests/test_distributed.py`).
   - Ask which category the bug escaped through, and which property *should* have caught it.
     A bug that passed the suite means the spec had a gap. Record the new property in `PROPERTIES.md`.
   - Then fix the code.
   - Pure infrastructure failures (OOM, NCCL timeouts, downloads) need no property. Note them
     in the README "what is and isn't verified" section.
3. **A failing property is a diagnosis, not a nuisance.** Follow the skill's
   `interpreting-failures.md`: the implementation, the property, the generator, the oracle or
   the precision may be wrong. Never weaken or delete a property just to go green. If the
   property itself was wrong, say why in the test docstring and in `PROPERTIES.md`.
   Examples already in the history: the float32 oracle normalisation and the space inserted
   inside a number. Pin each found counterexample with `@example(...)`.
4. **Oracles must be independent.** Do not reimplement the function under test in the
   assertion. The existing good oracles are exact likelihoods from joint tables,
   permutation enumeration, finite differences, brute-force solvers, `fractions.Fraction`,
   χ² against softmax, and metamorphic relations.
5. **Generators match the domain.** Use `@st.composite` instead of `assume`/filter chains.
   All randomness goes through Hypothesis (draw seeds or values), never `random` or global torch RNG.
6. **Every new property gets a mutant.** Add a plausible bug it should catch to
   `scripts/mutation_test.py` and check that it is KILLED. A surviving mutant means a missing
   or weak property (see the "Mutation testing" section of `PROPERTIES.md`).
7. **Completeness.** When adding a module, walk the 8 categories (round-trip, algebraic,
   oracle, metamorphic, invariant, stateful, error, monotonic/bounded). Write down which test
   covers each, or why it doesn't apply, in the completeness table of `PROPERTIES.md`.

## Commands
```bash
pytest                                  # all properties, ~1-2 min on CPU
HYPOTHESIS_PROFILE=thorough pytest      # 500 examples per property
python scripts/mutation_test.py         # must print "survivors: []"
scripts/smoke_8xh100.sh sudoku_llada    # 16-step GPU preflight with the real model
```

## Invariants that are easy to break (each one has caught a real bug)
- θ, θ_old and ref ELBOs for a sequence use the **same masks** for each inner iteration (E4, T1).
- `L_old` and `L_ref` are computed in the **same module mode** as the update forward (T1/T2).
- The micro-batch loss divides by the full-batch size, so accumulation is exact (O10).
- Mask counts are stratified with integer offsets. Rounding biases them (E6).
