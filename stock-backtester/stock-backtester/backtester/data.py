"""Price loading: yfinance, a local CSV, or synthetic data for demos and tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path("data")


def synthetic_prices(n: int = 1500, seed: int = 0, start: str = "2018-01-01") -> pd.Series:
    """Geometric random walk. Useful for tests and for checking the pipeline.

    A random walk has no exploitable pattern, so any strategy should look
    roughly like noise on it. Never report results from this data as real.
    """
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=0.0003, scale=0.01, size=n)
    index = pd.bdate_range(start=start, periods=n)
    return pd.Series(100 * np.exp(np.cumsum(log_returns)), index=index, name="Close")


def _from_csv(path: str | Path) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    column = "Close" if "Close" in df.columns else df.columns[0]
    return df[column].astype(float).sort_index().dropna().rename("Close")


def load_prices(
    ticker: str = "^NSEI",
    start: str = "2015-01-01",
    end: str | None = None,
    csv_path: str | Path | None = None,
) -> pd.Series:
    """Return a daily adjusted close series.

    Order of preference: explicit CSV, cached CSV in ./data, then yfinance
    (downloaded once and cached so reruns are reproducible).
    """
    if csv_path is not None:
        return _from_csv(csv_path)

    cache_file = CACHE_DIR / f"{ticker.replace('^', '').replace('.', '_')}_{start}_{end or 'latest'}.csv"
    if cache_file.exists():
        return _from_csv(cache_file)

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "yfinance is not installed. Run `pip install -r requirements.txt`, "
            "or pass --csv path/to/prices.csv, or use --synthetic."
        ) from exc

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker and date range.")
    if isinstance(raw.columns, pd.MultiIndex):  # newer yfinance returns (field, ticker)
        raw.columns = raw.columns.droplevel(1)

    close = raw["Close"].astype(float).dropna().rename("Close")
    CACHE_DIR.mkdir(exist_ok=True)
    close.to_csv(cache_file)
    return close
