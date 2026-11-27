"""Guards in the experiment runner that keep runs from being mixed or lost."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_experiments", ROOT / "experiments" / "run_experiments.py")
rx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rx)


def write_manifest(out, **overrides):
    cfg = rx.ExperimentConfig()
    m = {"tickers": ["AAPL"], "seeds": [0, 1], "algos": list(rx.ALGORITHMS),
         "data_sha256": rx.data_hashes(["AAPL"]), "runs": [], "config": rx.config_dict(cfg)}
    m.update(overrides)
    out.mkdir(parents=True, exist_ok=True)
    (out / rx.MANIFEST).write_text(json.dumps(m))
    return m


def args(**kw):
    return SimpleNamespace(tickers=kw.get("tickers"), seeds=kw.get("seeds"))


def test_settings_accept_matching_run(tmp_path):
    write_manifest(tmp_path)
    tickers, seeds, _ = rx._settings(args(seeds=[1, 0]), tmp_path, rx.ExperimentConfig())
    assert tickers == ["AAPL"] and seeds == [0, 1]


def test_settings_reject_changed_config(tmp_path):
    m = write_manifest(tmp_path)
    m["config"]["env"]["fee_rate"] = 0.005
    (tmp_path / rx.MANIFEST).write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="env.fee_rate"):
        rx._settings(args(), tmp_path, rx.ExperimentConfig())


def test_settings_reject_changed_data_and_other_seeds(tmp_path):
    write_manifest(tmp_path, data_sha256={"AAPL": "0" * 64})
    with pytest.raises(SystemExit, match="Price files"):
        rx._settings(args(), tmp_path, rx.ExperimentConfig())
    write_manifest(tmp_path)
    with pytest.raises(SystemExit, match="--seeds"):
        rx._settings(args(seeds=[0]), tmp_path, rx.ExperimentConfig())


def test_settings_need_all_algorithms(tmp_path):
    write_manifest(tmp_path, algos=["dqn"])
    with pytest.raises(SystemExit, match="ensembles need"):
        rx._settings(args(), tmp_path, rx.ExperimentConfig())


def test_refuses_to_clear_a_non_results_folder(tmp_path):
    (tmp_path / "main.tex").write_text("paper")
    with pytest.raises(SystemExit, match="does not look like a results folder"):
        rx.check_results_dir(tmp_path)
    rx.check_results_dir(tmp_path / "new")  # a new folder is fine


def test_phase1_keeps_old_results_when_data_fails(tmp_path, monkeypatch):
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "phase1_test.csv").write_text("old results")

    def boom(*a, **k):
        raise ValueError("bad data")

    monkeypatch.setattr(rx, "load_ticker", boom)
    a = SimpleNamespace(tickers=["AAPL"], seeds=[0], steps=100, algos=list(rx.ALGORITHMS),
                        results_dir=tmp_path)
    with pytest.raises(ValueError):
        rx.run_phase1(a)
    assert (tmp_path / "tables" / "phase1_test.csv").read_text() == "old results"
