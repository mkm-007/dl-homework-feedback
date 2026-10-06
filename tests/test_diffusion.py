import subprocess
import sys
from pathlib import Path

import pytest
import torch

from bugs import diffusion as catalog
from feedback import check

ROOT = Path(__file__).resolve().parents[1]


def report_for(overrides, seed=0):
    torch.manual_seed(seed)
    return check("diffusion", catalog.namespace(overrides))


def test_reference_passes_every_check():
    report = report_for({})
    assert all(r.status == "pass" for r in report.results), report.text()


@pytest.mark.parametrize("name", catalog.MUTATIONS)
def test_mutation_is_diagnosed(name):
    overrides, expect = catalog.MUTATIONS[name]
    report = report_for(overrides)
    assert expect in report.mistakes(), report.text()


@pytest.mark.parametrize("name", catalog.CORRECT)
def test_correct_variant_passes(name):
    report = report_for(catalog.CORRECT[name])
    assert not report.failed, report.text()


def test_starter_reports_errors_for_every_check():
    report = check("diffusion", ROOT / "assignments/diffusion/starter.py")
    assert all(r.status == "error" for r in report.results), report.text()


def test_command_line_on_reference():
    out = subprocess.run([sys.executable, "-m", "feedback", "diffusion", str(ROOT / "reference/diffusion.py")],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and "8/8 checks passed" in out.stdout, out.stdout
