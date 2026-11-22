# Data

Market data are **not committed** to git (see `.gitignore`). Only this file is tracked. Anyone can rebuild the data by running the download / preprocessing code in `src/environment.py`.

## Layout

```
data/
├── README.md        # this file (tracked)
├── raw/             # downloaded daily market data, unmodified
│   └── {TICKER}.csv
└── processed/       # features and splits derived from raw/
    ├── {TICKER}_features.csv
    └── {TICKER}_splits.json   # date boundaries, discretization bins, normalization stats
```

## Conventions

- **Raw files** are exactly what the data source returned: one CSV per ticker, daily bars, indexed by trading date. Do not edit them by hand.
- **Processed files** are fully reproducible from `raw/`. Anything fit on data (normalization statistics, discretization bins) is fit on the **training split only** and saved next to the features so validation and test runs reuse it unchanged.
- Dates are ISO format (`YYYY-MM-DD`). Prices are adjusted or unadjusted as decided by the team; record the choice here.

## To decide (see Open Decisions in the main README)

- Data source (e.g. yfinance) and whether prices are split / dividend adjusted
- Final ticker list (about 3, tentative) and date range (about 2015–2025, tentative)

Record the final choices here:

| Item | Decision |
|------|----------|
| Data source | TBD |
| Tickers | TBD |
| Date range | TBD |
| Price adjustment | TBD |
| Download date | TBD |
