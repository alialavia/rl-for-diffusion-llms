"""Property tests for synthetic task generators (PROPERTIES.md section 4, D1-D4)."""
import random
from fractions import Fraction

from hypothesis import given, strategies as st

from espo.data import make_sudoku_puzzle, random_countdown, random_sudoku_solution, sudoku_splits
from tests.test_rewards_properties import ALL_GRIDS, _valid

seeds = st.integers(0, 2**32 - 1)


@given(seeds)
def test_D1_generated_solutions_are_valid(seed):
    """D1. Generated solutions are valid 4x4 sudoku grids.
    Oracle: independent validity checker (rows/cols/boxes are permutations of 1-4).
    """
    sol = random_sudoku_solution(random.Random(seed))
    assert len(sol) == 16 and _valid(sol)


@given(seeds)
def test_D2_puzzles_agree_with_solution_and_are_unique(seed):
    """D2. Given cells match the solution, at least one cell is empty, and exactly one
    valid grid is consistent with the puzzle.
    Oracle: brute-force count over all 288 valid grids (independent of the generator).
    """
    rng = random.Random(seed)
    sol = random_sudoku_solution(rng)
    puzzle = make_sudoku_puzzle(sol, rng)
    assert "0" in puzzle
    assert all(p in ("0", s) for p, s in zip(puzzle, sol))
    consistent = [g for g in ALL_GRIDS if all(p in ("0", c) for p, c in zip(puzzle, g))]
    assert consistent == [sol]


@given(st.integers(1, 200), st.integers(1, 100), seeds)
def test_D3_sudoku_splits_are_disjoint(n_train, n_test, seed):
    """D3. Train/test puzzle sets are disjoint and have the requested sizes."""
    train, test = sudoku_splits(n_train, n_test, seed=seed)
    assert len(train) == n_train and len(test) == n_test
    assert not ({r["puzzle"] for r in train} & {r["puzzle"] for r in test})


def _reachable(values, target):
    """Brute-force: can `values` (all used once) be combined with + - * / to hit target?"""
    if len(values) == 1:
        return values[0] == target
    for i in range(len(values)):
        for j in range(len(values)):
            if i == j:
                continue
            a, b = values[i], values[j]
            rest = [v for k, v in enumerate(values) if k not in (i, j)]
            outs = [a + b, a - b, a * b] + ([a / b] if b != 0 else [])
            if any(_reachable(rest + [o], target) for o in outs):
                return True
    return False


@given(seeds, st.integers(3, 4))
def test_D4_countdown_targets_are_reachable(seed, n):
    """D4. Synthetic countdown targets are positive integers reachable using every number
    exactly once.  Oracle: brute-force search over all combination orders.
    """
    nums, target = random_countdown(random.Random(seed), n_numbers=n)
    assert len(nums) == n and isinstance(target, int) and target > 0
    assert _reachable([Fraction(x) for x in nums], Fraction(target))
