"""Strategies. Each returns a 0/1 signal computed at the close of day t.

The engine applies the signal on the NEXT day, so a signal never earns the
return of the bar it was computed from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import rsi, sma


def buy_and_hold(close: pd.Series) -> pd.Series:
    return pd.Series(1.0, index=close.index, name="buy_and_hold")


def sma_crossover(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """Long when the fast average is above the slow average, otherwise flat."""
    if fast >= slow:
        raise ValueError("fast window must be smaller than slow window")
    fast_ma, slow_ma = sma(close, fast), sma(close, slow)
    signal = (fast_ma > slow_ma).astype(float)
    signal[slow_ma.isna()] = 0.0
    return signal.rename(f"sma_{fast}_{slow}")


def rsi_mean_reversion(
    close: pd.Series, period: int = 14, low: float = 30.0, high: float = 70.0
) -> pd.Series:
    """Buy when RSI drops below `low`, sell when it rises above `high`."""
    values = rsi(close, period).to_numpy()
    out = np.zeros(len(values))
    position = 0.0
    for i, value in enumerate(values):
        if np.isnan(value):
            position = 0.0
        elif value < low:
            position = 1.0
        elif value > high:
            position = 0.0
        out[i] = position
    return pd.Series(out, index=close.index, name=f"rsi_{period}_{int(low)}_{int(high)}")
