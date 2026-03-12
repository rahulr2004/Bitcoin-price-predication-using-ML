"""
indicators.py
-------------
Adds technical analysis indicators to a price DataFrame.
"""

import pandas as pd
import ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and attach technical indicators.

    Indicators:
        RSI, EMA_20, MACD, MACD_signal, MACD_diff,
        BB_upper, BB_lower, ATR, OBV

    Args:
        df: DataFrame with [Open, High, Low, Close, Volume]

    Returns:
        df with indicator columns appended; NaN warm-up rows dropped.
    """
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()

    # Momentum
    df["RSI"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    # Trend
    df["EMA_20"]      = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
    macd              = ta.trend.MACD(close=close)
    df["MACD"]        = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_diff"]   = macd.macd_diff()

    # Volatility
    bb             = ta.volatility.BollingerBands(close=close, window=20)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df["ATR"]      = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close
    ).average_true_range()

    # Volume
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(
        close=close, volume=vol
    ).on_balance_volume()

    df.dropna(inplace=True)
    return df
