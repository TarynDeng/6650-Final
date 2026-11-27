"""Run bookkeeping: the manifest, and checks that keep phases from mixing runs.

Phase 1 writes ``results/run_manifest.json`` when it finishes. Phases 2 and 3
read it and stop if the stocks, seeds, settings, price files, or code in
``src/`` changed since phase 1.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

from .config import ExperimentConfig
from .data import RAW_DIR, REPO_ROOT, file_sha256

MANIFEST = "run_manifest.json"
ALGORITHMS = ("qlearning", "sarsa", "dqn")
RESULT_ITEMS = {"models", "logs", "tables", "figures", MANIFEST, "README.md", ".gitkeep", ".DS_Store"}


def git_commit() -> str:
    """Short commit hash, with ``-dirty`` if files outside ``results/`` have uncommitted changes.

    ``results/`` is excluded because a run rewrites its own committed tables.
    """
    def git(*cmd):
        return subprocess.check_output(["git", *cmd], cwd=REPO_ROOT, text=True).strip()

    try:
        dirty = git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)results")
        return git("rev-parse", "--short", "HEAD") + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def package_versions() -> dict:
    import gymnasium
    import matplotlib
    import numpy
    import pandas
    import torch

    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "pandas": pandas.__version__, "gymnasium": gymnasium.__version__,
            "torch": torch.__version__, "matplotlib": matplotlib.__version__}


def config_dict(cfg: ExperimentConfig) -> dict:
    """The config as plain JSON types, so it compares equal to the saved manifest."""
    return json.loads(json.dumps(dataclasses.asdict(cfg), default=str))


def data_hashes(tickers) -> dict:
    """SHA-256 of the price files actually on disk."""
    return {t: file_sha256(RAW_DIR / f"{t}.csv") for t in tickers}


def code_hash() -> str:
    """One SHA-256 over every ``src/*.py`` file, so code edits between phases are caught."""
    h = hashlib.sha256()
    for path in sorted((REPO_ROOT / "src").glob("*.py")):
        h.update(path.name.encode() + path.read_bytes())
    return h.hexdigest()


def write_manifest(out: Path, tickers, seeds, algos, steps: int, cfg: ExperimentConfig) -> None:
    manifest = {"tickers": tickers, "seeds": seeds, "algos": algos, "total_env_steps": steps,
                "git_commit": git_commit(), "code_sha256": code_hash(),
                "packages": package_versions(), "data_sha256": data_hashes(tickers),
                "runs": [f"{a}_{t}_seed{s}" for t in tickers for s in seeds for a in algos],
                "config": config_dict(cfg)}
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2))


def read_manifest(out: Path) -> dict:
    path = out / MANIFEST
    if not path.exists():
        sys.exit(f"{path} not found: run --phase 1 first.")
    return json.loads(path.read_text())


def check_results_dir(out: Path) -> None:
    """Refuse to clear a folder that does not look like a results folder."""
    if out.exists():
        unexpected = sorted(p.name for p in out.iterdir() if p.name not in RESULT_ITEMS)
        if unexpected:
            sys.exit(f"{out} contains {unexpected[:5]}, so it does not look like a results folder. "
                     "Refusing to clear it; choose an empty or results folder.")


def _changed_settings(manifest: dict, cfg: ExperimentConfig) -> list[str]:
    now = config_dict(cfg)
    return [f"{group}.{key}" for group in now for key in now[group]
            if manifest["config"].get(group, {}).get(key) != now[group][key]]


def phase1_settings(args, out: Path, cfg: ExperimentConfig):
    """Stocks and seeds of the phase 1 run in ``out``, after checking nothing changed.

    Phases 2 and 3 must use exactly the same stocks, seeds, config, data and
    code as phase 1, or they stop with an error.
    """
    m = read_manifest(out)
    if set(m["algos"]) != set(ALGORITHMS):
        sys.exit(f"Phase 1 trained only {m['algos']}; the ensembles need all of {ALGORITHMS}.")
    for name, given in (("tickers", args.tickers), ("seeds", args.seeds)):
        if given is not None and sorted(given) != sorted(m[name]):
            sys.exit(f"--{name} {given} differs from phase 1 ({m[name]}). Omit it, or re-run phase 1.")
    changed = _changed_settings(m, cfg)
    if changed:
        sys.exit(f"src/config.py changed since phase 1 ({', '.join(changed)}). Re-run phase 1.")
    if data_hashes(m["tickers"]) != m["data_sha256"]:
        sys.exit("Price files in data/raw/ changed since phase 1. Re-run phase 1.")
    if m.get("code_sha256") != code_hash():
        sys.exit("Code in src/ changed since phase 1. Re-run phase 1.")
    return m["tickers"], m["seeds"], m
