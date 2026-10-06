"""Behavioral feedback for deep learning assignments."""

from importlib import import_module
from pathlib import Path

from feedback.core import Check, Fail, Report, Result, load, run

ASSIGNMENTS = ("vae", "diffusion")


def checks_for(assignment: str) -> list[Check]:
    if assignment not in ASSIGNMENTS:
        raise ValueError(f"Unknown assignment {assignment!r}; choose from {', '.join(ASSIGNMENTS)}.")
    return import_module(f"feedback.assignments.{assignment}").CHECKS


def check(assignment: str, source: str | Path | dict) -> Report:
    """Check a student's file (.py or .ipynb) or an already-loaded namespace."""
    ns = source if isinstance(source, dict) else load(source)
    return run(assignment, checks_for(assignment), ns)


__all__ = ["ASSIGNMENTS", "Check", "Fail", "Report", "Result", "check", "checks_for", "load", "run"]
