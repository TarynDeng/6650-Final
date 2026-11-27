# Data

Daily prices for the default stocks are **committed** in `raw/`, so every teammate runs on exactly the same numbers.

```
data/
├── README.md
└── raw/
    ├── AAPL.csv  JPM.csv  XOM.csv   # Date, Open, High, Low, Close, Volume
    └── SHA256SUMS                   # checksum of each CSV
```

| Item | Decision |
|---|---|
| Source | Yahoo Finance via `yfinance` 1.8.0 |
| Tickers | AAPL, JPM, XOM (placeholder default, see the main README) |
| Date range | 2014-10-01 to 2025-12-31 (2014 only warms up the 20-day features) |
| Price adjustment | Split- and dividend-adjusted (`auto_adjust=True`) |
| Download date | 2026-11-26 |

**Why commit the files?** Yahoo Finance re-adjusts all past prices whenever a stock pays a new dividend. Downloading the same dates later would give slightly different prices, and therefore different results.

**Checks.** `src/data.py` refuses a file whose checksum does not match `SHA256SUMS`, and refuses to re-download a committed stock whose file is missing. A CSV that is not listed in `SHA256SUMS` is used with a warning. Files with missing or non-positive prices, duplicate or unordered dates, gaps longer than a week, or dates that do not cover the experiment are rejected.

**Adding a stock.** Add the ticker to `DataConfig.tickers` in `src/config.py` and run phase 1; the file is downloaded on first use. Then commit it and refresh the checksums:

```bash
cd data/raw && shasum -a 256 *.csv > SHA256SUMS
```

Do not edit the CSV files by hand.
