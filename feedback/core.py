"""Run behavioral checks against a student's code and report what to fix and why.

Checks call the student's functions and classes on controlled inputs and compare the behavior with
what the assignment specifies. When the behavior matches a known mistake, the report names it.
Checks never contain or reveal a solution.
"""

from __future__ import annotations

import ast
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


class Fail(Exception):
    """Raised by a check with a message explaining the mistake."""

    def __init__(self, message: str, mistake: str = "unspecified"):
        super().__init__(message)
        self.mistake = mistake


@dataclass(frozen=True)
class Check:
    id: str
    title: str
    requires: tuple[str, ...]
    run: Callable[[dict], None]
    learn: str = ""


@dataclass
class Result:
    check: Check
    status: str  # "pass", "fail", "error", "skip"
    message: str = ""
    mistake: str = ""


@dataclass
class Report:
    assignment: str
    results: list[Result] = field(default_factory=list)

    @property
    def failed(self) -> list[Result]:
        return [r for r in self.results if r.status in ("fail", "error")]

    def mistakes(self) -> set[str]:
        return {r.mistake for r in self.failed if r.mistake}

    def text(self) -> str:
        marks = {"pass": "PASS", "fail": "FIX ", "error": "ERR ", "skip": "SKIP"}
        lines = [f"Feedback for {self.assignment}", ""]
        for r in self.results:
            lines.append(f"[{marks[r.status]}] {r.check.title}")
            if r.status != "pass":
                for part in r.message.strip().splitlines():
                    lines.append(f"       {part}")
                if r.check.learn and r.status in ("fail", "error"):
                    lines.append(f"       Learn why: {r.check.learn}")
        passed = sum(r.status == "pass" for r in self.results)
        lines += ["", f"{passed}/{len(self.results)} checks passed."]
        return "\n".join(lines)


def run(assignment: str, checks: list[Check], ns: dict) -> Report:
    report = Report(assignment)
    for c in checks:
        missing = [name for name in c.requires if name not in ns]
        if missing:
            report.results.append(Result(c, "skip", f"Define {', '.join(missing)} to run this check."))
            continue
        try:
            c.run(ns)
            report.results.append(Result(c, "pass"))
        except Fail as f:
            report.results.append(Result(c, "fail", str(f), f.mistake))
        except Exception as e:  # the student's code raised; show where
            tb = traceback.extract_tb(e.__traceback__)
            where = next((f"line {fr.lineno} in {fr.name}" for fr in reversed(tb)
                          if fr.filename.startswith("<student")), "")
            report.results.append(Result(c, "error", f"Your code raised {type(e).__name__}: {e}"
                                         + (f" ({where})" if where else ""), "crash"))
    return report


_KEEP = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _definitions_only(source: str) -> str:
    """Keep imports, functions, classes and constant assignments; drop training loops and plots."""
    tree = ast.parse(source)
    body = []
    for node in tree.body:
        if isinstance(node, _KEEP):
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Constant):
            body.append(node)
    tree.body = body
    return ast.unparse(tree)


def load(path: str | Path) -> dict:
    """Load a student's .py file or .ipynb notebook into a namespace, definitions only."""
    path = Path(path)
    if path.suffix == ".ipynb":
        cells = json.loads(path.read_text())["cells"]
        source = "\n\n".join("".join(c["source"]) for c in cells if c.get("cell_type") == "code")
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("!", "%")))
    else:
        source = path.read_text()
    ns: dict = {"__name__": "student"}
    exec(compile(_definitions_only(source), f"<student:{path.name}>", "exec"), ns)
    return ns
