"""Measure how well the checks detect and diagnose known mistakes without flagging correct code.

    python -m evaluation.run_eval vae [--seeds 20]

Each case is checked under several random seeds, because weight initialization and sampling
change what the student's code outputs.
"""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path

import torch

from feedback import ASSIGNMENTS, checks_for, run


def evaluate(assignment: str, seeds: int) -> dict:
    catalog = import_module(f"bugs.{assignment}")
    checks = checks_for(assignment)
    cases = ([(name, "mutation", overrides, expect) for name, (overrides, expect) in catalog.MUTATIONS.items()]
             + [(name, "held_out", overrides, "") for name, overrides in catalog.HELD_OUT.items()]
             + [(name, "correct", overrides, "") for name, overrides in {"reference": {}, **catalog.CORRECT}.items()])
    rows = []
    for name, kind, overrides, expect in cases:
        for seed in range(seeds):
            torch.manual_seed(1000 + seed)
            report = run(assignment, checks, catalog.namespace(overrides))
            mistakes = report.mistakes()
            rows.append({"case": name, "kind": kind, "seed": seed, "expect": expect,
                         "detected": bool(report.failed), "diagnosed": bool(expect) and expect in mistakes,
                         "reported": sorted(mistakes)})

    def rate(kind, key):
        rs = [r for r in rows if r["kind"] == kind]
        return sum(r[key] for r in rs) / len(rs) if rs else float("nan")

    summary = {
        "assignment": assignment,
        "seeds": seeds,
        "mutations": len(catalog.MUTATIONS),
        "held_out": len(catalog.HELD_OUT),
        "correct_variants": len(catalog.CORRECT) + 1,
        "detection_rate": rate("mutation", "detected"),
        "diagnosis_accuracy": rate("mutation", "diagnosed"),
        "held_out_detection_rate": rate("held_out", "detected"),
        "false_positive_rate": rate("correct", "detected"),
    }
    return {"summary": summary, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("assignment", choices=ASSIGNMENTS)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    result = evaluate(args.assignment, args.seeds)
    Path(f"evaluation/results_{args.assignment}.json").write_text(json.dumps(result, indent=2))

    by_case: dict[tuple, list] = {}
    for r in result["rows"]:
        by_case.setdefault((r["kind"], r["case"]), []).append(r)
    print(f"{'case':28} {'kind':9} {'detected':>9} {'diagnosed':>10}  reported")
    for (kind, case), rs in by_case.items():
        n = len(rs)
        det = f"{sum(r['detected'] for r in rs)}/{n}"
        diag = f"{sum(r['diagnosed'] for r in rs)}/{n}" if kind == "mutation" else "-"
        reported = sorted({m for r in rs for m in r["reported"]})
        print(f"{case:28} {kind:9} {det:>9} {diag:>10}  {', '.join(reported)}")
    s = result["summary"]
    print(f"\n{s['assignment']}: {s['mutations']} known mistakes, {s['held_out']} held-out mistakes, "
          f"{s['correct_variants']} correct implementations, {s['seeds']} seeds each")
    print(f"detection rate            {s['detection_rate']:.1%}")
    print(f"diagnosis accuracy        {s['diagnosis_accuracy']:.1%}")
    print(f"held-out detection rate   {s['held_out_detection_rate']:.1%}")
    print(f"false positive rate       {s['false_positive_rate']:.1%}")


if __name__ == "__main__":
    main()
