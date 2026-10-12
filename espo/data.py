"""Task datasets and prompts (Sudoku, Countdown, GSM8K, MATH), following d1 / ESPO."""
from __future__ import annotations

import random
from fractions import Fraction
from typing import Optional

REASONING_SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

**Rules:**
- Fill empty cells with digits 1-4.
- Each row must contain digits 1-4 exactly once.
- Each column must contain digits 1-4 exactly once.
- Each 2x2 box must contain digits 1-4 exactly once.

**Example:**
Puzzle: 0401002010030310
This puzzle grid looks like this:
0 4 | 0 1
0 0 | 2 0
----+----
1 0 | 0 3
0 3 | 1 0

Solution: 2431312412434312
The solved grid looks like this:
2 4 | 3 1
3 1 | 2 4
----+----
1 2 | 4 3
4 3 | 1 2

**Important:** Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""


def sudoku_prompt(puzzle: str) -> list[dict]:
    return [{"role": "user",
             "content": f"{SUDOKU_SYSTEM_PROMPT}\n\nNow, solve the following Sudoku puzzle: {puzzle}\n"}]


def countdown_prompt(numbers, target) -> list[dict]:
    return [{"role": "user", "content": (
        f"{REASONING_SYSTEM_PROMPT}\nUsing only the numbers {list(numbers)}, create an arithmetic "
        f"expression that evaluates to exactly {target}. You must use all numbers from the list, and "
        "each number must be used exactly once. You may use the operations +, -, *, and / as needed. "
        "After reasoning, provide only your final expression inside <answer></answer> tags without "
        "including an equals sign or the target number. For example, if the numbers are [2, 3, 4] and "
        "the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>")}]


def boxed_prompt(question: str) -> list[dict]:
    return [{"role": "user",
             "content": f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."}]


# ----------------------------------------------------------------- sudoku generation
_BASE = [[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]]


def _count_solutions(cells: list[int], limit: int = 2) -> int:
    """Backtracking count of completions of a 4x4 grid (0 = empty), capped at ``limit``."""
    try:
        i = cells.index(0)
    except ValueError:
        return 1
    r, c = divmod(i, 4)
    br, bc = 2 * (r // 2), 2 * (c // 2)
    used = {cells[r * 4 + k] for k in range(4)} | {cells[k * 4 + c] for k in range(4)}
    used |= {cells[(br + dr) * 4 + bc + dc] for dr in (0, 1) for dc in (0, 1)}
    total = 0
    for v in (1, 2, 3, 4):
        if v not in used:
            cells[i] = v
            total += _count_solutions(cells, limit - total)
            cells[i] = 0
            if total >= limit:
                break
    return total


def random_sudoku_solution(rng: random.Random) -> str:
    """Uniform-ish valid 4x4 grid via symmetry transforms of a base grid."""
    digits = [1, 2, 3, 4]
    rng.shuffle(digits)
    bands = [0, 1]
    rng.shuffle(bands)
    rows = [b * 2 + k for b in bands for k in rng.sample([0, 1], 2)]
    stacks = [0, 1]
    rng.shuffle(stacks)
    cols = [s * 2 + k for s in stacks for k in rng.sample([0, 1], 2)]
    grid = [[digits[_BASE[r][c] - 1] for c in cols] for r in rows]
    if rng.random() < 0.5:
        grid = [list(x) for x in zip(*grid)]
    return "".join(str(v) for row in grid for v in row)


def make_sudoku_puzzle(solution: str, rng: random.Random, min_empty: int = 4,
                       max_empty: Optional[int] = None) -> str:
    """Blank cells of ``solution`` in random order while the solution stays unique."""
    cells = [int(c) for c in solution]
    order = list(range(16))
    rng.shuffle(order)
    target = rng.randint(min_empty, max_empty or 16)
    empty = 0
    for i in order:
        if empty >= target:
            break
        keep = cells[i]
        cells[i] = 0
        if _count_solutions(cells[:]) == 1:
            empty += 1
        else:
            cells[i] = keep
    if empty == 0:  # always possible to blank one cell
        cells[order[0]] = 0
    return "".join(str(c) for c in cells)


def sudoku_splits(n_train: int, n_test: int, seed: int = 0):
    """Disjoint train/test lists of {"puzzle", "solution"} rows."""
    rng = random.Random(seed)
    seen: set[str] = set()
    rows: list[dict] = []
    attempts = 0
    while len(rows) < n_train + n_test:
        attempts += 1
        if attempts > 1000 * (n_train + n_test):
            raise RuntimeError("could not generate enough distinct puzzles")
        sol = random_sudoku_solution(rng)
        puz = make_sudoku_puzzle(sol, rng)
        if puz in seen:
            continue
        seen.add(puz)
        rows.append({"puzzle": puz, "solution": sol})
    return rows[:n_train], rows[n_train:]


# ----------------------------------------------------------------- countdown generation
def random_countdown(rng: random.Random, n_numbers: int = 3, lo: int = 1, hi: int = 100):
    """Numbers and a positive-integer target reachable by a random expression tree."""
    while True:
        nums = [rng.randint(lo, hi) for _ in range(n_numbers)]
        vals = [Fraction(n) for n in nums]
        rng.shuffle(vals)
        while len(vals) > 1:
            i = rng.randrange(len(vals) - 1)
            a, b = vals[i], vals[i + 1]
            op = rng.choice("+-*/")
            if op == "/" and b == 0:
                op = "+"
            v = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else a + b}[op]
            vals[i:i + 2] = [v]
        v = vals[0]
        if v.denominator == 1 and 0 < v <= 1000:
            return nums, int(v)


# ----------------------------------------------------------------- loading
def load_task_rows(task: str, split: str = "train", seed: int = 0, n_synthetic: int = 20000,
                   sudoku_csv: Optional[str] = None, countdown_source: str = "hf") -> list[dict]:
    """Rows with a "messages" prompt plus the fields the task's reward needs."""
    if task == "sudoku":
        if sudoku_csv:
            return load_rows_from_file("sudoku", sudoku_csv)
        train, test = sudoku_splits(n_synthetic, 256, seed=seed)
        rows = train if split == "train" else test
        return [dict(r, messages=sudoku_prompt(r["puzzle"])) for r in rows]
    if task == "countdown":
        if countdown_source == "hf":
            from datasets import load_dataset

            ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
            ds = ds.filter(lambda x: len(x["nums"]) == 3)
            items = [(list(x["nums"]), int(x["target"])) for x in ds]
        else:
            rng = random.Random(seed + (0 if split == "train" else 1))
            items = [random_countdown(rng) for _ in range(n_synthetic if split == "train" else 256)]
        return [{"numbers": n, "target": t, "messages": countdown_prompt(n, t)} for n, t in items]
    if task == "gsm8k":
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split=split)
        return [{"answer": x["answer"].split("####")[-1].strip(), "messages": boxed_prompt(x["question"])}
                for x in ds]
    if task == "math":
        from datasets import load_dataset

        if split == "train":
            ds = _load_math_train()
        else:
            ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [{"answer": x["solution"] if "solution" in x else x["answer"],
                 "messages": boxed_prompt(x["problem"])} for x in ds]
    raise ValueError(f"unknown task {task!r}")


def _load_math_train():
    from datasets import concatenate_datasets, load_dataset

    subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]
    return concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", s, split="train")
                                 for s in subjects])


def load_rows_from_file(task: str, path: str) -> list[dict]:
    """Rows from the d1/ESPO dataset files: sudoku CSV (Puzzle,Solution), countdown JSONL
    ({"input": "a,b,c", "output": "t"}), or generic JSONL with the task's fields."""
    import csv
    import json

    if path.endswith(".csv"):
        with open(path) as f:
            rows = [{"puzzle": r["Puzzle"], "solution": r["Solution"]} for r in csv.DictReader(f)]
    else:
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if task == "countdown":
            rows = [{"numbers": [int(v) for v in r["input"].split(",")], "target": int(r["output"])}
                    if "input" in r else r for r in rows]
    prompt = {"sudoku": lambda r: sudoku_prompt(r["puzzle"]),
              "countdown": lambda r: countdown_prompt(r["numbers"], r["target"]),
              "gsm8k": lambda r: boxed_prompt(r["question"]),
              "math": lambda r: boxed_prompt(r["problem"])}[task]
    return [dict(r, messages=r.get("messages") or prompt(r)) for r in rows]
