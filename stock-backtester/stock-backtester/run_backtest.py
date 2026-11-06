"""Command-line entry point.

Examples:
    python run_backtest.py                       # NIFTY 50 via yfinance
    python run_backtest.py --ticker RELIANCE.NS --start 2012-01-01
    python run_backtest.py --csv my_prices.csv
    python run_backtest.py --synthetic           # pipeline check only, not real results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtester import (
    backtest,
    buy_and_hold,
    load_prices,
    rsi_mean_reversion,
    sma_crossover,
    summarize,
    synthetic_prices,
)
from backtester.ml import evaluate_models

RESULTS_DIR = Path("results")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest simple strategies and an ML direction signal.")
    p.add_argument("--ticker", default="^NSEI", help="Yahoo Finance ticker (default: ^NSEI, NIFTY 50)")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--csv", default=None, help="CSV with a date index and a Close column")
    p.add_argument("--synthetic", action="store_true", help="Use random-walk data (pipeline check only)")
    p.add_argument("--cost-bps", type=float, default=10.0, help="Cost per unit traded, in basis points")
    p.add_argument("--train-fraction", type=float, default=0.7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    close = synthetic_prices() if args.synthetic else load_prices(args.ticker, args.start, args.end, args.csv)
    label = "SYNTHETIC random walk (not real data)" if args.synthetic else (args.csv or args.ticker)
    print(f"Data: {label} | {close.index[0].date()} to {close.index[-1].date()} | {len(close)} days")

    # --- Rule-based strategies on the full history -------------------------------------
    signals = {
        "buy_and_hold": buy_and_hold(close),
        "sma_20_50": sma_crossover(close, 20, 50),
        "rsi_14_30_70": rsi_mean_reversion(close, 14, 30, 70),
    }
    results = {name: backtest(close, sig, args.cost_bps) for name, sig in signals.items()}
    table = pd.DataFrame({name: summarize(res) for name, res in results.items()}).T.round(2)
    print("\nFull-period results (costs included):")
    print(table.to_string())

    # --- ML signal, evaluated on the held-out test period only -------------------------
    ml = evaluate_models(close, args.train_fraction)
    test_start, test_end = ml["test_period"]
    test_close = close.loc[test_start:test_end]
    print(f"\nML evaluation. Train: {ml['train_period'][0]} to {ml['train_period'][1]}"
          f" | Test: {test_start} to {test_end}")
    print(f"Majority-class baseline accuracy on test set: {100 * ml['baseline_accuracy']:.1f}%")

    test_results = {"buy_and_hold (test period)": backtest(test_close, buy_and_hold(test_close), args.cost_bps)}
    for name, info in ml["models"].items():
        print(f"  {name}: accuracy {100 * info['accuracy']:.1f}%")
        test_results[f"ml_{name}"] = backtest(test_close, info["signal"], args.cost_bps)
    test_table = pd.DataFrame({n: summarize(r) for n, r in test_results.items()}).T.round(2)
    print("\nTest-period results (costs included):")
    print(test_table.to_string())

    # --- Save outputs ------------------------------------------------------------------
    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "full_period_metrics.csv")
    test_table.to_csv(RESULTS_DIR / "test_period_metrics.csv")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    for name, res in results.items():
        axes[0].plot(res.index, res["equity"], label=name)
    axes[0].set_title("Full period: growth of 1 unit")
    axes[0].legend()
    for name, res in test_results.items():
        axes[1].plot(res.index, res["equity"], label=name)
    axes[1].set_title("Held-out test period: growth of 1 unit")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "equity_curves.png", dpi=150)
    print(f"\nSaved metrics and chart to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
