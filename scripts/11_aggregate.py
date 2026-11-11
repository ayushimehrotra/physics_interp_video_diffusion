#!/usr/bin/env python3
"""Stage 11 - leaderboard and the standard figures from results/ (spec §8).

    /root/envs/physics_venv/bin/python scripts/11_aggregate.py [--results results]

Reads every results/<model>/<family>/<level>.json; writes results/leaderboard.{json,md} (blank where a level has
not been run; columns in physlaw/analysis/aggregate.py) and results/figures/fig*.png (each figure only when its
inputs exist; list in physlaw/analysis/figures.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from physlaw.analysis import aggregate, figures  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(REPO / "results"))
    a = ap.parse_args()
    res = Path(a.results)
    rows = aggregate.run(res)["rows"]
    print((res / "leaderboard.md").read_text())
    figures.draw(rows, res, res / "figures")


if __name__ == "__main__":
    main()
