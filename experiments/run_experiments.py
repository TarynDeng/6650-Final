"""Entry point for training and evaluation runs.

Planned usage (not implemented yet)::

    python experiments/run_experiments.py --phase 1 --algo qlearning --ticker XXX --seeds 0 1 2 3 4
    python experiments/run_experiments.py --phase 2   # build ensembles from validation results
    python experiments/run_experiments.py --phase 3   # out-of-sample evaluation of all methods

Phases (``docs/proposal.md`` Section 6):
    1. Train Q-learning, SARSA, and DQN under identical data splits, actions,
       costs, initial capital, reward, and seed logging.
    2. Measure validation performance and fix the weighted / equal-weight
       ensemble weights.
    3. Evaluate all methods plus buy-and-hold on the test windows.

Outputs (metrics, learning curves, model files, figures) go under
``results/``, which is git-ignored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``from src import ...`` when run as ``python experiments/run_experiments.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ALGORITHMS = ("qlearning", "sarsa", "dqn")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--algo", choices=ALGORITHMS, help="algorithm for phase 1")
    parser.add_argument("--ticker", help="stock ticker (tentative list TBD)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser.parse_args(argv)


def run_phase1(args):
    """TODO: train one algorithm per seed and save models + learning curves."""
    raise NotImplementedError


def run_phase2(args):
    """TODO: compute validation terminal returns and ensemble weights."""
    raise NotImplementedError


def run_phase3(args):
    """TODO: evaluate all methods and buy-and-hold on the test windows."""
    raise NotImplementedError


def main(argv=None):
    args = parse_args(argv)
    {1: run_phase1, 2: run_phase2, 3: run_phase3}[args.phase](args)


if __name__ == "__main__":
    main()
