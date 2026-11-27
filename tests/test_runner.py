"""The experiment phases end to end, and the checks that keep runs from mixing."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import src.phases as phases
import src.report as report
import src.runs as runs
from src.config import ExperimentConfig
from tests.helpers import synthetic_splits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import run_experiments as cli  # noqa: E402


def args(**kw):
    base = {"tickers": None, "seeds": None, "steps": None, "algos": list(runs.ALGORITHMS),
            "results_dir": None}
    return SimpleNamespace(**(base | kw))


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    """Phases 1-3 on synthetic prices (named AAPL), 2 seeds, 300 steps."""
    fake = lambda ticker, cfg=None: synthetic_splits(seed=1)  # noqa: E731
    monkeypatch.setattr(phases, "load_ticker", fake)
    monkeypatch.setattr(report, "load_ticker", fake)
    out = tmp_path / "results"
    phases.run_phase1(args(tickers=["AAPL"], seeds=[0, 1], steps=300, results_dir=out))
    phases.run_phase2(args(results_dir=out))
    report.run_phase3(args(results_dir=out))
    return out


def test_phases_end_to_end(tiny_run):
    t = tiny_run / "tables"
    summary = pd.read_csv(t / "phase3_summary_by_method.csv", index_col="method")
    assert list(summary.index) == report.METHOD_ORDER
    assert (summary.loc["qlearning", ["n_windows", "n_episodes"]] == [3, 6]).all()
    weights = pd.read_csv(t / "phase2_weights.csv")
    w = weights[[f"w_{a}" for a in runs.ALGORITHMS]].sum(axis=1)
    assert (abs(w - 1) < 1e-9).all()
    sens = pd.read_csv(t / "phase2_tiebreak_sensitivity.csv")
    assert set(sens["tie_break"]) == {"top_weight_then_hold", "hold", "largest_action"}
    behavior = pd.read_csv(t / "phase3_trading_behavior.csv", index_col="method")
    assert behavior.loc["buy_and_hold", "avg_position"] > 0.95
    assert (tiny_run / "figures" / "phase3_learning_curves.png").exists()


def test_phase3_stops_when_a_seed_is_missing(tiny_run):
    p2 = tiny_run / "tables" / "phase2_test.csv"
    t = pd.read_csv(p2)
    t[t["seed"] != 1].to_csv(p2, index=False)
    with pytest.raises(SystemExit, match="expected"):
        report.run_phase3(args(results_dir=tiny_run))


def write_manifest(out, **overrides):
    cfg = ExperimentConfig()
    m = {"tickers": ["AAPL"], "seeds": [0, 1], "algos": list(runs.ALGORITHMS),
         "data_sha256": runs.data_hashes(["AAPL"]), "code_sha256": runs.code_hash(), "runs": [],
         "config": runs.config_dict(cfg)}
    m.update(overrides)
    out.mkdir(parents=True, exist_ok=True)
    (out / runs.MANIFEST).write_text(json.dumps(m))
    return m


def test_settings_accept_matching_run(tmp_path):
    write_manifest(tmp_path)
    tickers, seeds, _ = runs.phase1_settings(args(seeds=[1, 0]), tmp_path, ExperimentConfig())
    assert tickers == ["AAPL"] and seeds == [0, 1]


@pytest.mark.parametrize("override, message", [
    ({"data_sha256": {"AAPL": "0" * 64}}, "Price files"),
    ({"code_sha256": "0" * 64}, "Code in src/"),
    ({"algos": ["dqn"]}, "ensembles need"),
])
def test_settings_reject_changed_run(tmp_path, override, message):
    write_manifest(tmp_path, **override)
    with pytest.raises(SystemExit, match=message):
        runs.phase1_settings(args(), tmp_path, ExperimentConfig())


def test_settings_reject_changed_config_and_other_seeds(tmp_path):
    m = write_manifest(tmp_path)
    m["config"]["env"]["fee_rate"] = 0.005
    (tmp_path / runs.MANIFEST).write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="env.fee_rate"):
        runs.phase1_settings(args(), tmp_path, ExperimentConfig())
    write_manifest(tmp_path)
    with pytest.raises(SystemExit, match="--seeds"):
        runs.phase1_settings(args(seeds=[0]), tmp_path, ExperimentConfig())


def test_refuses_to_clear_a_non_results_folder(tmp_path):
    (tmp_path / "main.tex").write_text("paper")
    with pytest.raises(SystemExit, match="does not look like a results folder"):
        runs.check_results_dir(tmp_path)
    runs.check_results_dir(tmp_path / "new")  # a new folder is fine


def test_phase1_keeps_old_results_when_data_fails(tmp_path, monkeypatch):
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "phase1_test.csv").write_text("old results")

    def boom(*a, **k):
        raise ValueError("bad data")

    monkeypatch.setattr(phases, "load_ticker", boom)
    with pytest.raises(ValueError):
        phases.run_phase1(args(tickers=["AAPL"], seeds=[0], steps=100, results_dir=tmp_path))
    assert (tmp_path / "tables" / "phase1_test.csv").read_text() == "old results"


def test_cli_rejects_phase1_options_on_later_phases_and_bad_steps():
    with pytest.raises(SystemExit):
        cli.parse_args(["--phase", "2", "--steps", "100"])
    with pytest.raises(SystemExit):
        cli.parse_args(["--phase", "1", "--steps", "0"])
    assert cli.parse_args(["--phase", "1"]).algos == list(runs.ALGORITHMS)
