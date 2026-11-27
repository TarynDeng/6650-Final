"""Metrics and summary statistics on known inputs."""

import pandas as pd
import pytest

from src.evaluation import loss_category, paired_comparison, summarize, terminal_loss, terminal_return


def test_terminal_metrics_and_loss_category_boundaries():
    assert terminal_return([100, 125]) == pytest.approx(0.25)
    assert terminal_loss([100, 120]) == 0.0
    assert terminal_loss([100, 70]) == pytest.approx(0.3)
    cases = {0.0: 1, 1e-9: 2, 0.2: 2, 0.2000001: 3, 0.4: 3, 0.4000001: 4, 0.9: 4}
    assert {x: loss_category(x) for x in cases} == cases


def fake_results():
    """Two stocks x three years; 'rl' is 0.01 below 'bh' on average in each stock-year."""
    rows = []
    for t, base_t in (("A", 0.1), ("B", -0.1), ("C", 0.05)):
        for w, base_w in (("2023", 0.0), ("2024", 0.05), ("2025", -0.02)):
            base = base_t + base_w
            rows.append({"method": "bh", "ticker": t, "window": w, "seed": -1,
                         "terminal_return": base, "loss_category": 1 if base >= 0 else 2})
            for s in range(3):
                r = base - 0.02 + 0.01 * s
                rows.append({"method": "rl", "ticker": t, "window": w, "seed": s,
                             "terminal_return": r, "loss_category": 1 if r >= 0 else 2})
    return pd.DataFrame(rows)


def test_summary_uses_stock_years_as_samples():
    s = summarize(fake_results())
    assert s.loc["rl", "n_windows"] == 9 and s.loc["rl", "n_episodes"] == 27
    assert s.loc["bh", "n_windows"] == 9 and s.loc["bh", "n_episodes"] == 9
    assert s.loc["rl", "mean"] == pytest.approx(s.loc["bh", "mean"] - 0.01)
    assert s.loc["rl", "seed_std"] == pytest.approx(0.01)
    rl = fake_results().query("method == 'rl'")
    assert s.loc["rl", "share_cat2"] == pytest.approx((rl["loss_category"] == 2).mean())


def test_paired_comparison_and_year_range():
    r = fake_results()
    r.loc[(r.method == "rl") & (r.window == "2024"), "terminal_return"] += 0.03
    c = paired_comparison(r, "rl", "bh")
    assert c["mean_diff"] == pytest.approx(0.0)  # -0.01, +0.02, -0.01 per year
    assert c["year_min"] == pytest.approx(-0.01) and c["year_max"] == pytest.approx(0.02)
    assert c["a_wins"] == 3 and c["n_windows"] == 9 and c["n_years"] == 3
    assert c["ci_low"] < 0 < c["ci_high"]
