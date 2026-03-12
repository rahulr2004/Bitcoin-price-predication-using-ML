"""
preprocessing.py
----------------
Scales features and builds sliding-window sequences for LSTM.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

FEATURE_COLS = [
    "Close", "Open", "High", "Low", "Volume",
    "RSI", "EMA_20", "MACD", "MACD_signal", "MACD_diff",
    "BB_upper", "BB_lower", "ATR", "OBV",
]

WINDOW_SIZE = 60


def preprocess_data(
    df: pd.DataFrame,
    window: int = WINDOW_SIZE,
    test_split: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Scale and window the data for LSTM training.

    Returns:
        X_train, X_test, y_train, y_test, scaler
        X shape: (samples, window, n_features)
        y shape: (samples,)  — next-day scaled Close price
    """
    cols   = [c for c in FEATURE_COLS if c in df.columns]
    data   = df[cols].values.astype(np.float32)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window : i])
        y.append(scaled[i][0])   # Close is col 0

    X, y = np.array(X), np.array(y)
    split = int(len(X) * (1 - test_split))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"[preprocessing] Train {X_train.shape}  Test {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler


def inverse_close(
    scaler: MinMaxScaler,
    scaled_values: np.ndarray,
    n_features: int,
) -> np.ndarray:
    """Inverse-transform scaled Close predictions back to USD."""
    dummy          = np.zeros((len(scaled_values), n_features), dtype=np.float32)
    dummy[:, 0]    = scaled_values.flatten()
    return scaler.inverse_transform(dummy)[:, 0]
