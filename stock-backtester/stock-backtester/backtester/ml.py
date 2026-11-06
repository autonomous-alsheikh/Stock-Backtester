"""Next-day direction classifier with a strict time-based split.

Rules this module follows to avoid fooling itself:
1. Features at row t use only prices up to the close of day t.
2. The target at row t is whether the close on day t+1 is higher.
3. Train and test sets are split by time, never shuffled.
4. The scaler and model are fitted on the training period only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .indicators import rsi, sma


def build_features(close: pd.Series) -> pd.DataFrame:
    log_ret = np.log(close).diff()
    feats = pd.DataFrame(index=close.index)
    for lag in (1, 2, 3, 5):
        feats[f"ret_lag{lag}"] = log_ret.shift(lag - 1)  # lag1 = today's return
    feats["rsi14"] = rsi(close, 14) / 100.0
    feats["dist_sma20"] = close / sma(close, 20) - 1
    feats["dist_sma50"] = close / sma(close, 50) - 1
    feats["vol10"] = log_ret.rolling(10).std()
    return feats


def build_target(close: pd.Series) -> pd.Series:
    """1 if tomorrow's close is above today's close. The last row has no target."""
    target = (close.shift(-1) > close).astype(float)
    target[close.shift(-1).isna()] = np.nan
    return target.rename("target")


def time_split(index: pd.Index, train_fraction: float = 0.7) -> tuple[pd.Index, pd.Index]:
    if not 0.1 <= train_fraction <= 0.9:
        raise ValueError("train_fraction should be between 0.1 and 0.9")
    cut = int(len(index) * train_fraction)
    return index[:cut], index[cut:]


def make_models(seed: int = 42) -> dict[str, object]:
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=20, random_state=seed, n_jobs=-1
        ),
    }


def evaluate_models(close: pd.Series, train_fraction: float = 0.7, seed: int = 42) -> dict:
    """Fit on the early period, predict on the later period.

    Returns accuracy next to the majority-class baseline (always guess "up" or
    "down", whichever was more common in training). A model that cannot beat
    that baseline has learned nothing useful.
    """
    data = build_features(close).join(build_target(close)).dropna()
    train_idx, test_idx = time_split(data.index, train_fraction)
    feature_cols = [c for c in data.columns if c != "target"]
    x_train, y_train = data.loc[train_idx, feature_cols], data.loc[train_idx, "target"]
    x_test, y_test = data.loc[test_idx, feature_cols], data.loc[test_idx, "target"]

    majority = float(y_train.mean() >= 0.5)
    baseline = float(accuracy_score(y_test, np.full(len(y_test), majority)))

    results: dict = {
        "train_period": (str(train_idx[0].date()), str(train_idx[-1].date())),
        "test_period": (str(test_idx[0].date()), str(test_idx[-1].date())),
        "baseline_accuracy": baseline,
        "models": {},
    }
    for name, model in make_models(seed).items():
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_test)[:, 1]
        pred = (proba > 0.5).astype(float)
        signal = pd.Series(pred, index=test_idx, name=f"ml_{name}")
        results["models"][name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "signal": signal,
        }
    return results
