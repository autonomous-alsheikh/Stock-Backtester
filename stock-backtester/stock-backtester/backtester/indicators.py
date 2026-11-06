"""Technical indicators. Every value at time t uses only prices up to and including t."""

from __future__ import annotations

import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return close.rolling(window=window, min_periods=window).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing), scaled 0-100."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    # No losses in the window means RSI is 100 by definition.
    out = out.where(avg_loss != 0, 100.0)
    return out.where(avg_gain.notna() & avg_loss.notna())
