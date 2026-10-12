"""Property tests for task rewards (PROPERTIES.md section 4, R1-R6)."""
import itertools
from fractions import Fraction

import pytest
from hypothesis import assume, example, given, strategies as st

from espo.rewards import (
    countdown_reward,
    extract_boxed,
    gsm8k_reward,
    math_reward,
    sudoku_reward,
)

# --------------------------------------------------------------- sudoku helpers
# Independent oracle: all 288 valid 4x4 grids, enumerated by brute force.
_ROWS = list(itertools.permutations("1234"))


def _valid(grid):
    rows = [grid[i * 4:(i + 1) * 4] for i in range(4)]
    cols = ["".join(r[c] for r in rows) for c in range(4)]
    boxes = ["".join(rows[r + dr][c + dc] for dr in (0, 1) for dc in (0, 1))
             for r in (0, 2) for c in (0, 2)]
    return all(sorted(u) == list("1234") for u in rows + cols + boxes)


ALL_GRIDS = [g for g in ("".join(sum(map(list, rs), [])) for rs in itertools.product(_ROWS, repeat=4))
             if _valid(g)]
assert len(ALL_GRIDS) == 288


@st.composite
def sudoku_cases(draw):
    sol = draw(st.sampled_from(ALL_GRIDS))
    empty = draw(st.lists(st.integers(0, 15), min_size=1, max_size=16, unique=True))
    puzzle = "".join("0" if i in empty else c for i, c in enumerate(sol))
    return puzzle, sol, sorted(empty)


def wrap(ans):
    return f"<reasoning>\nthinking...\n</reasoning>\n<answer>\n{ans}\n</answer>"


@given(sudoku_cases(), st.data())
def test_R2_sudoku_reward_counts_correct_empty_cells(case, data):
    """R2. With k wrong digits placed in empty cells, reward == 1 - k/#empty; the full
    solution scores 1.  Oracle: number of corruptions, by construction.
    """
    puzzle, sol, empty = case
    assert sudoku_reward(wrap(sol), puzzle, sol) == 1.0
    wrong = data.draw(st.lists(st.sampled_from(empty), unique=True))
    ans = list(sol)
    for i in wrong:
        ans[i] = data.draw(st.sampled_from([d for d in "1234" if d != sol[i]]))
    assert sudoku_reward(wrap("".join(ans)), puzzle, sol) == pytest.approx(1 - len(wrong) / len(empty))


@given(sudoku_cases(), st.data())
def test_R2b_sudoku_prefilled_cells_do_not_count(case, data):
    """R2b. Changing a prefilled (given) cell of the answer does not change the reward."""
    puzzle, sol, empty = case
    given_cells = [i for i in range(16) if i not in empty]
    assume(given_cells)
    i = data.draw(st.sampled_from(given_cells))
    ans = list(sol)
    ans[i] = "1" if sol[i] != "1" else "2"
    assert sudoku_reward(wrap("".join(ans)), puzzle, sol) == 1.0


@given(sudoku_cases(), st.lists(st.sampled_from([" ", "\n", "|", "-", "+"]), max_size=20),
       st.randoms(use_true_random=False))
def test_R3_sudoku_parsing_ignores_separators_and_uses_last_answer(case, seps, rnd):
    """R3. Separators/whitespace inside <answer> are ignored; the last <answer> wins;
    no answer => 0.
    """
    puzzle, sol, _ = case
    chars = list(sol)
    for s in seps:
        chars.insert(rnd.randint(0, len(chars)), s)
    noisy = "".join(chars)
    assert sudoku_reward(wrap(noisy), puzzle, sol) == 1.0
    assert sudoku_reward(wrap("1111111111111111") + wrap(sol), puzzle, sol) == 1.0
    assert sudoku_reward("the answer is " + sol, puzzle, sol) == 0.0


# --------------------------------------------------------------- countdown
OPS = ["+", "-", "*", "/"]


@st.composite
def expression_trees(draw, numbers):
    """Random binary tree using each number exactly once; returns (text, Fraction|None)."""
    items = [(str(n), Fraction(n)) for n in numbers]
    order = draw(st.permutations(range(len(items))))
    items = [items[i] for i in order]
    while len(items) > 1:
        i = draw(st.integers(0, len(items) - 2))
        (ta, va), (tb, vb) = items[i], items[i + 1]
        op = draw(st.sampled_from(OPS))
        if va is None or vb is None:
            v = None
        elif op == "+":
            v = va + vb
        elif op == "-":
            v = va - vb
        elif op == "*":
            v = va * vb
        else:
            v = va / vb if vb != 0 else None
        items[i:i + 2] = [(f"({ta}{op}{tb})", v)]
    return items[0]


numbers_s = st.lists(st.integers(1, 100), min_size=3, max_size=4)


@given(numbers_s, st.data(), st.lists(st.integers(0, 40), max_size=6))
@example([1, 2, 3], None, [])
def test_R4_countdown_correct_expressions_score_one(numbers, data, space_at):
    """R4. For expression trees using each number exactly once, with target equal to the
    exact Fraction value of the tree, reward == 1; inserting spaces changes nothing.
    Oracle: fractions.Fraction evaluation of the generated tree.
    """
    if data is None:
        text, value = "(1+(2*3))", Fraction(7)
    else:
        text, value = data.draw(expression_trees(numbers))
    assume(value is not None)
    assert countdown_reward(wrap(text), numbers if data else [1, 2, 3], value) == 1.0
    spaced = list(text)
    for p in sorted(space_at, reverse=True):
        p = min(p, len(spaced))
        # only between tokens: a space inside a number would change the expression
        if 0 < p < len(spaced) and spaced[p - 1].isdigit() and spaced[p].isdigit():
            continue
        spaced.insert(p, " ")
    assert countdown_reward(wrap("".join(spaced)), numbers if data else [1, 2, 3], value) == 1.0


@given(numbers_s, st.data())
def test_R5_countdown_wrong_numbers_get_format_score_only(numbers, data):
    """R5. Using a wrong multiset of numbers scores exactly 0.1 (format score), even when
    the value matches; no <answer> scores 0.
    """
    text, value = data.draw(expression_trees(numbers + [7]))  # one extra number
    assume(value is not None)
    assert countdown_reward(wrap(text), numbers, value) == 0.1
    assert countdown_reward(text, numbers, value) == 0.0


@pytest.mark.parametrize("payload", [
    "__import__('os').system('echo pwned')",
    "1/0+2+3",
    "(1+2)**3",
    "[x for x in ().__class__.__bases__]",
    "1 + 2 + 3 =6",
    "",
])
def test_R5b_countdown_never_executes_arbitrary_code(payload):
    """R5b. Non-arithmetic or erroring answers never execute and never score 1."""
    assert countdown_reward(wrap(payload), [1, 2, 3], 6) <= 0.1


# --------------------------------------------------------------- gsm8k / math
@given(st.integers(-10**7, 10**7), st.booleans(), st.booleans(), st.sampled_from(["", ".0", ".00"]))
@example(1234567, True, True, ".00")
def test_R6_boxed_number_round_trip(n, commas, dollar, trailing):
    """R6. Formatting n with commas / $ / trailing zeros inside \\boxed{} and extracting
    gives back n (reward 1); a different number scores 0; the last \\boxed{} wins.
    Oracle: the generated number.
    """
    s = (f"{n:,}" if commas else str(n)) + trailing
    if dollar:
        s = "\\$" + s
    text = f"Some reasoning \\boxed{{{n + 1}}} then final \\boxed{{{s}}}."
    assert gsm8k_reward(text, str(n)) == 1.0
    assert gsm8k_reward(f"\\boxed{{{n + 1}}}", str(n)) == 0.0


@given(st.text(alphabet="abcx+-*/^ 0123456789{}\\", max_size=12))
def test_R6b_extract_boxed_handles_nested_braces(inner):
    """R6b. extract_boxed returns the balanced content of the last \\boxed{...}."""
    assume(inner.count("{") == inner.count("}"))
    depth = 0
    for ch in inner:
        depth += ch == "{"
        depth -= ch == "}"
        assume(depth >= 0)
    assert extract_boxed(f"x \\boxed{{1}} y \\boxed{{{inner}}} z") == inner


def test_R6c_math_reward_on_latex():
    """R6c. Equivalent LaTeX answers score 1, different ones 0 (examples)."""
    assert math_reward("so \\boxed{\\frac{1}{2}}", "\\frac{1}{2}") == 1.0
    assert math_reward("so \\boxed{\\frac{1}{3}}", "\\frac{1}{2}") == 0.0
    assert math_reward("no box here", "\\frac{1}{2}") == 0.0


# --------------------------------------------------------------- fuzz
@given(st.text(max_size=300), sudoku_cases(), numbers_s, st.integers(-100, 1000))
def test_R1_rewards_bounded_and_total_on_garbage(text, case, numbers, target):
    """R1. All rewards lie in [0, 1] and never raise, for arbitrary completion text
    (also wrapped in <answer> tags).  Oracle: range.
    """
    puzzle, sol, _ = case
    for t in (text, wrap(text)):
        for r in (sudoku_reward(t, puzzle, sol), countdown_reward(t, numbers, target),
                  gsm8k_reward(t, str(target)), math_reward(t, str(target))):
            assert 0.0 <= r <= 1.0
