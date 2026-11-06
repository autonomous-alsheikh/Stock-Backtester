"""Vectorised backtest engine."""

from __future__ import annotations

import pandas as pd


def backtest(close: pd.Series, signal: pd.Series, cost_bps: float = 10.0) -> pd.DataFrame:
    """Run a long/flat backtest.

    - The signal computed at the close of day t is held during day t+1
      (`signal.shift(1)`), which is what prevents look-ahead bias.
    - `cost_bps` is charged on every change in position (per unit traded).
    """
    signal = signal.reindex(close.index).fillna(0.0)
    asset_return = close.pct_change().fillna(0.0)
    position = signal.shift(1).fillna(0.0)
    trades = position.diff().abs().fillna(position.abs())
    costs = trades * (cost_bps / 10_000)
    strategy_return = position * asset_return - costs
    return pd.DataFrame(
        {
            "asset_return": asset_return,
            "position": position,
            "trades": trades,
            "strategy_return": strategy_return,
            "equity": (1 + strategy_return).cumprod(),
        }
    )
