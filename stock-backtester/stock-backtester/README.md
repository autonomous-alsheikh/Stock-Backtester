# Stock Strategy Backtester and Signal Validator

A small Python project that backtests rule-based trading strategies (SMA crossover, RSI mean reversion) against buy-and-hold, and tests whether a machine learning model can predict next-day direction better than guessing, using a strict time-based train/test split.

The goal is not to find a money-making strategy. It is to measure honestly whether a signal has any edge, which means avoiding look-ahead bias, including transaction costs, and comparing against simple baselines.

## What it does

- Loads daily prices from Yahoo Finance (cached locally), a CSV file, or synthetic data for testing
- Computes indicators: SMA and RSI (Wilder smoothing)
- Backtests long/flat strategies with transaction costs
- Reports total return, CAGR, volatility, Sharpe ratio, max drawdown, trade count and time in market
- Trains logistic regression and random forest models to predict next-day direction, fitted on an early period and tested on a later one
- Compares model accuracy with a majority-class baseline, then backtests the ML signal on the held-out period only

## How look-ahead bias is avoided

| Risk | How it is handled |
| --- | --- |
| Trading on information you did not have yet | A signal computed at the close of day t is applied on day t+1 (`signal.shift(1)` in `engine.py`) |
| Features that peek at the future | Every feature uses only prices up to day t. A test truncates the data and checks earlier features do not change |
| Leaky train/test split | Split is by time (first 70% train, last 30% test), never shuffled. The scaler and model are fitted on training data only |
| Ignoring costs | A configurable cost (default 10 bps per unit traded) is charged on every position change |
| Judging a model by accuracy alone | Accuracy is shown next to a majority-class baseline |

## Project structure

```
stock-backtester/
├── backtester/
│   ├── data.py         # yfinance / CSV / synthetic data
│   ├── indicators.py   # SMA, RSI
│   ├── strategies.py   # buy-and-hold, SMA crossover, RSI mean reversion
│   ├── engine.py       # backtest engine (one-day signal lag, costs)
│   ├── metrics.py      # return, CAGR, Sharpe, drawdown
│   └── ml.py           # features, target, time split, models
├── tests/test_core.py  # 11 tests, including a no-look-ahead check
├── run_backtest.py     # command-line entry point
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/YOUR-HANDLE/stock-backtester.git
cd stock-backtester
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python run_backtest.py                                  # NIFTY 50 (^NSEI), from 2015
python run_backtest.py --ticker RELIANCE.NS --start 2012-01-01
python run_backtest.py --csv my_prices.csv              # CSV with a date index and a Close column
python run_backtest.py --cost-bps 20 --train-fraction 0.8
python run_backtest.py --synthetic                      # pipeline check on random data
python -m pytest                                        # run the tests
```

Results are written to `results/` (metric tables as CSV and an equity curve chart).

## Results

> **TODO: run the project on real data and paste your own numbers here.** Replace the placeholders below, and keep the wording honest even if the results are unimpressive.

Data: `^NSEI`, `YYYY-MM-DD` to `YYYY-MM-DD`, cost 10 bps.

| Strategy | Total return % | Sharpe | Max drawdown % | Trades |
| --- | --- | --- | --- | --- |
| Buy and hold | | | | |
| SMA 20/50 | | | | |
| RSI 14 (30/70) | | | | |

ML on the held-out test period (`YYYY-MM-DD` to `YYYY-MM-DD`):

| Model | Accuracy | Baseline accuracy | Total return % | Sharpe |
| --- | --- | --- | --- | --- |
| Logistic regression | | | | |
| Random forest | | | | |

**What I found:** (2 to 3 sentences in your own words. For example: did any strategy beat buy-and-hold after costs? Did the ML models beat the baseline? How many trades did they make, and what do costs do to that?)

![Equity curves](results/equity_curves.png)

## Limitations

- Long/flat only, one asset at a time, no position sizing or shorting
- Daily closing prices; no slippage model beyond a flat cost
- A single train/test split. A walk-forward evaluation would be more robust
- Past performance on one index says little about future performance
- This is an educational project, not investment advice

## Ideas for extension

- Walk-forward validation with rolling retraining
- More features (volume, MACD, volatility regimes) and hyperparameter tuning on a validation set
- Multiple stocks and a simple portfolio
- A Streamlit app to explore parameters interactively

## License

MIT
