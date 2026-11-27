"""Shared test fixtures: synthetic prices, so tests never need the network."""

import numpy as np
import pandas as pd

from src.data import compute_market_features


def make_features(n: int = 80, seed: int = 0) -> pd.DataFrame:
    """``n`` days of features from a random-walk price series."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n + 25)))
    dates = pd.bdate_range("2020-01-01", periods=n + 25)
    return compute_market_features(pd.DataFrame({"Close": close}, index=dates)).iloc[:n]


def synthetic_splits(seed: int = 0) -> dict:
    """Train / val / three test windows with the shapes the real data has (but shorter)."""
    f = make_features(n=420, seed=seed)
    return {"train": f.iloc[:300], "val": f.iloc[300:330],
            "test": [f.iloc[330:360], f.iloc[360:390], f.iloc[390:420]]}
