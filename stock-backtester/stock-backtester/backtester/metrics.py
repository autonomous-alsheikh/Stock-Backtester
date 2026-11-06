"""Performance metrics (daily data, 252 trading days per year, zero risk-free rate)."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0


def cagr(equity: pd.Series) -> float:
    years = len(equity) / TRADING_DAYS
    if years <= 0 or equity.iloc[-1] <= 0:
        return float("nan")
    return float(equity.iloc[-1] ** (1 / years) - 1)


def annual_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series) -> float:
    std = returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(returns.mean() / std * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough fall, as a negative fraction."""
    return float((equity / equity.cummax() - 1).min())


def summarize(result: pd.DataFrame) -> dict[str, float]:
    """Summarise one `backtest()` result."""
    equity = result["equity"]
    returns = result["strategy_return"]
    return {
        "total_return_%": 100 * total_return(equity),
        "cagr_%": 100 * cagr(equity),
        "volatility_%": 100 * annual_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown_%": 100 * max_drawdown(equity),
        "trades": float(result["trades"].sum()),
        "time_in_market_%": 100 * float((result["position"] > 0).mean()),
    }
