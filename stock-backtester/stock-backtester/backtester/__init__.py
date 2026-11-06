"""Small, readable stock strategy backtester with a look-ahead-safe ML signal."""

from .data import load_prices, synthetic_prices
from .engine import backtest
from .indicators import rsi, sma
from .metrics import summarize
from .strategies import buy_and_hold, rsi_mean_reversion, sma_crossover

__all__ = [
    "load_prices",
    "synthetic_prices",
    "backtest",
    "rsi",
    "sma",
    "summarize",
    "buy_and_hold",
    "rsi_mean_reversion",
    "sma_crossover",
]
