import argparse
import sys

from feedback import ASSIGNMENTS, check

parser = argparse.ArgumentParser(prog="python -m feedback", description="Check a deep learning assignment.")
parser.add_argument("assignment", choices=ASSIGNMENTS)
parser.add_argument("path", help="your .py file or .ipynb notebook")
args = parser.parse_args()

report = check(args.assignment, args.path)
print(report.text())
sys.exit(1 if report.failed else 0)
