"""Command-line entry point for the three experiment phases.

Usage::

    python experiments/run_experiments.py --phase 1   # train the agents; evaluate them and buy-and-hold
    python experiments/run_experiments.py --phase 2   # weighted and equal-weight ensembles
    python experiments/run_experiments.py --phase 3   # final tables and figures for all six methods

Phase 1 accepts ``--tickers``, ``--seeds``, ``--algos`` and ``--steps``
(defaults in ``src/config.py``). Phases 2 and 3 always use the stocks and
seeds of the last phase 1 run (``results/run_manifest.json``) and stop if
the settings, price files or code changed since then. Outputs go under
``--results-dir`` (default ``results/``); phase 1 clears the previous run there.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Allow ``from src import ...`` when run as ``python experiments/run_experiments.py``.
sys.path.insert(0, str(ROOT))

from src.phases import run_phase1, run_phase2  # noqa: E402
from src.report import run_phase3  # noqa: E402
from src.runs import ALGORITHMS  # noqa: E402

PHASE1_ONLY = ("algos", "steps")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--algos", choices=ALGORITHMS, nargs="+", help="phase 1 only (default: all three)")
    parser.add_argument("--tickers", nargs="+", help="phase 1: stocks (default: DataConfig.tickers)")
    parser.add_argument("--seeds", type=int, nargs="+", help="phase 1: seeds (default: TrainConfig.seeds)")
    parser.add_argument("--steps", type=int, help="phase 1: env steps per run (default: TrainConfig)")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    if args.phase != 1:
        ignored = [f"--{name}" for name in PHASE1_ONLY if getattr(args, name) is not None]
        if ignored:
            parser.error(f"{', '.join(ignored)} only apply to --phase 1")
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.algos is None:
        args.algos = list(ALGORITHMS)
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    {1: run_phase1, 2: run_phase2, 3: run_phase3}[args.phase](args)


if __name__ == "__main__":
    main()
