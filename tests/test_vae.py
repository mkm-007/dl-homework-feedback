import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from bugs import vae as catalog
from feedback import check, checks_for, load
from reference import vae as ref

ROOT = Path(__file__).resolve().parents[1]


def report_for(overrides, seed=0):
    torch.manual_seed(seed)
    return check("vae", catalog.namespace(overrides))


def test_reference_passes_every_check():
    report = report_for({})
    assert not report.failed, report.text()
    assert all(r.status == "pass" for r in report.results)


@pytest.mark.parametrize("name", catalog.MUTATIONS)
def test_mutation_is_diagnosed(name):
    overrides, expect = catalog.MUTATIONS[name]
    report = report_for(overrides)
    assert expect in report.mistakes(), report.text()


@pytest.mark.parametrize("name", catalog.CORRECT)
def test_correct_variant_passes(name):
    report = report_for(catalog.CORRECT[name])
    assert not report.failed, report.text()


def test_missing_names_are_skipped_not_crashed():
    ns = catalog.namespace()
    del ns["sample"], ns["kl_divergence"]
    report = check("vae", ns)
    skipped = {r.check.id for r in report.results if r.status == "skip"}
    assert skipped == {"kl", "sample"}
    assert "Define kl_divergence" in report.text()


def test_starter_reports_errors_for_every_check():
    report = check("vae", ROOT / "assignments/vae/starter.py")
    assert all(r.status == "error" for r in report.results)
    assert "NotImplementedError" in report.text()


def test_notebook_loads_definitions_only(tmp_path):
    source = inspect.getsource(ref)
    cells = [
        {"cell_type": "markdown", "source": ["# My VAE"]},
        {"cell_type": "code", "source": ["!pip install torch\n", "%matplotlib inline\n"]},
        {"cell_type": "code", "source": source.splitlines(keepends=True)},
        {"cell_type": "code", "source": ["raise SystemExit('training ran')\n", "for epoch in range(100):\n",
                                         "    pass\n", "model = Encoder()\n"]},
    ]
    path = tmp_path / "hw.ipynb"
    path.write_text(json.dumps({"cells": cells, "nbformat": 4, "nbformat_minor": 5, "metadata": {}}))
    ns = load(path)
    assert "model" not in ns and ns["Z_DIM"] == 2
    torch.manual_seed(0)
    assert not check("vae", path).failed


def test_feedback_never_contains_reference_code():
    lines = {line.strip() for line in inspect.getsource(ref).splitlines()
             if len(line.strip()) > 25 and not line.strip().startswith(("def ", "class ", '"""'))}
    for name, (overrides, _) in catalog.MUTATIONS.items():
        text = report_for(overrides).text()
        assert not [line for line in lines if line in text], name


def test_every_check_is_exercised_by_some_mutation():
    hit = set()
    for overrides, _ in catalog.MUTATIONS.values():
        hit |= {r.check.id for r in report_for(overrides).failed}
    assert hit >= {c.id for c in checks_for("vae")} - {"shapes", "training"}


def test_command_line_exit_codes():
    ok = subprocess.run([sys.executable, "-m", "feedback", "vae", str(ROOT / "reference/vae.py")],
                        cwd=ROOT, capture_output=True, text=True)
    assert ok.returncode == 0 and "9/9 checks passed" in ok.stdout
    bad = subprocess.run([sys.executable, "-m", "feedback", "vae", str(ROOT / "assignments/vae/starter.py")],
                         cwd=ROOT, capture_output=True, text=True)
    assert bad.returncode == 1
