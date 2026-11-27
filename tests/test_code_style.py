"""House rules for the code base: short files and short functions."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_LINES = 400
MAX_FUNCTION_LINES = 50
PY_FILES = sorted(p for d in ("src", "experiments", "tests", "slides") for p in (ROOT / d).glob("*.py"))
JS_FILES = sorted((ROOT / "slides").glob("*.js"))


@pytest.mark.parametrize("path", PY_FILES + JS_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_file_is_at_most_400_lines(path):
    n = len(path.read_text().splitlines())
    assert n <= MAX_FILE_LINES, f"{path.name} has {n} lines; split it"


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_functions_are_at_most_50_lines(path):
    tree = ast.parse(path.read_text())
    too_long = [f"{n.name} ({n.end_lineno - n.lineno + 1} lines)" for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
                and n.end_lineno - n.lineno + 1 > MAX_FUNCTION_LINES]
    assert not too_long, f"{path.name}: {', '.join(too_long)}"
