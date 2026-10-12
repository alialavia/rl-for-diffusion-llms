"""Verifiable rewards for Sudoku, Countdown, GSM8K and MATH (d1/ESPO conventions)."""
from __future__ import annotations

import ast
import re
from collections import Counter
from fractions import Fraction
from typing import Optional

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_answer_tag(text: str) -> Optional[str]:
    """Content of the last ``<answer>...</answer>`` block, or None."""
    matches = _ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else None


# ----------------------------------------------------------------- sudoku
def sudoku_reward(completion: str, puzzle: str, solution: str) -> float:
    """Fraction of the puzzle's empty cells filled correctly (d1 reward)."""
    ans = extract_answer_tag(completion)
    if ans is None:
        return 0.0
    digits = "".join(c for c in ans if c.isdigit())
    empty = [i for i, c in enumerate(puzzle) if c == "0"]
    if not empty or len(digits) < 16:
        return 0.0
    correct = sum(digits[i] == solution[i] for i in empty)
    return correct / len(empty)


# ----------------------------------------------------------------- countdown
_ALLOWED_EXPR = re.compile(r"^[\d+\-*/().\s]+$")
_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
           ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b}


def safe_eval_arith(expr: str) -> Optional[Fraction]:
    """Exact value of an arithmetic expression over + - * / and integers, else None.

    Parses with ``ast`` and walks only whitelisted nodes; never calls ``eval``.
    """
    if not expr or not _ALLOWED_EXPR.match(expr) or len(expr) > 200:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError):
        return None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return Fraction(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Div) and b == 0:
                raise ZeroDivisionError
            return _BINOPS[type(node.op)](a, b)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError("disallowed node")

    try:
        return ev(tree)
    except (ZeroDivisionError, ValueError, RecursionError):
        return None


def countdown_reward(completion: str, numbers, target, format_score: float = 0.1,
                     score: float = 1.0) -> float:
    """1.0 if the <answer> expression uses exactly ``numbers`` and equals ``target``;
    ``format_score`` if an answer is present but wrong; 0 if there is no answer."""
    ans = extract_answer_tag(completion)
    if ans is None:
        return 0.0
    used = [int(n) for n in re.findall(r"\d+", ans)]
    if Counter(used) != Counter(int(n) for n in numbers):
        return format_score
    value = safe_eval_arith(ans)
    if value is None:
        return format_score
    if isinstance(target, float):
        ok = abs(float(value) - target) < 1e-5
    else:
        ok = value == Fraction(target)
    return score if ok else format_score


# ----------------------------------------------------------------- gsm8k / math
def extract_boxed(text: str) -> Optional[str]:
    """Balanced content of the last ``\\boxed{...}`` (handles nested braces)."""
    start = text.rfind("\\boxed{")
    while start != -1:
        i = start + len("\\boxed{")
        depth = 1
        j = i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[i:j]
            j += 1
        start = text.rfind("\\boxed{", 0, start)  # unbalanced: try an earlier one
    return None


def normalize_number(s: str) -> Optional[str]:
    """Canonical decimal string for a number written with commas / $ / units, or None."""
    s = s.strip().replace("\\$", "").replace("$", "").replace(",", "").replace("\\!", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s).replace("\\%", "").replace("%", "").strip()
    if not re.fullmatch(r"-?\d+(\.\d+)?", s):
        return None
    try:
        v = Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None
    return str(v.numerator) if v.denominator == 1 else str(float(v))


def gsm8k_reward(completion: str, answer: str) -> float:
    """1.0 if the last \\boxed{} (or <answer>) holds the reference number."""
    pred = extract_boxed(completion)
    if pred is None:
        pred = extract_answer_tag(completion)
    if pred is None:
        return 0.0
    a, p = normalize_number(str(answer)), normalize_number(pred)
    return 1.0 if a is not None and a == p else 0.0


def _strip_latex(s: str) -> str:
    s = s.strip().strip("$").replace(" ", "").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("^\\circ", "")
    s = s.replace("^{\\circ}", "").rstrip(".")
    return s


def math_reward(completion: str, answer: str) -> float:
    """1.0 if the last \\boxed{} matches the reference (math_verify if installed)."""
    pred = extract_boxed(completion)
    if pred is None:
        return 0.0
    ref = extract_boxed(answer) or answer
    try:  # optional, more permissive equivalence checking
        from math_verify import parse, verify

        if verify(parse("$" + ref + "$"), parse("$" + pred + "$")):
            return 1.0
    except Exception:
        pass
    np_, nr = normalize_number(pred), normalize_number(ref)
    if np_ is not None and np_ == nr:
        return 1.0
    return 1.0 if _strip_latex(pred) == _strip_latex(ref) else 0.0
