"""
binance_data.py
---------------
Provides live Bitcoin price data from Binance's public REST API.

No API key required for public ticker endpoints.

Usage:
    from src.binance_data import get_live_price, get_klines

    price = get_live_price()               # e.g. 67423.55
    df    = get_klines(interval="1h", n=24) # last 24 hourly candles
"""

import requests
import pandas as pd
from typing import Optional


from .data_loader import get_usd_inr_rate

BASE_URL = "https://api.binance.com/api/v3"


def get_live_price(symbol: str = "BTCUSDT", currency: str = "USD") -> Optional[float]:
    """
    Fetch the current spot price from Binance's public REST API.

    No API key is required for this endpoint.

    Args:
        symbol: Binance trading pair (default: BTCUSDT)
        currency: "USD" or "INR"

    Returns:
        Current price as float, or None if the request fails.
    """
    url = f"{BASE_URL}/ticker/price?symbol={symbol}"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        price = float(resp.json()["price"])
        
        if currency.upper() == "INR":
            rate = get_usd_inr_rate()
            price *= rate
            print(f"[binance_data] Live {symbol} price: ₹{price:,.2f} INR")
        else:
            print(f"[binance_data] Live {symbol} price: ${price:,.2f} USD")
            
        return price
    except Exception as e:
        print(f"[binance_data] Could not fetch live price: {e}")
        return None


def get_24h_stats(symbol: str = "BTCUSDT", currency: str = "USD") -> Optional[dict]:
    """
    Fetch 24-hour ticker statistics for a symbol.

    Returns dict with keys: priceChange, priceChangePercent,
    lastPrice, highPrice, lowPrice, volume, etc.
    """
    url = f"{BASE_URL}/ticker/24hr?symbol={symbol}"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        stats = {
            "symbol":            data["symbol"],
            "last_price":        float(data["lastPrice"]),
            "price_change":      float(data["priceChange"]),
            "price_change_pct":  float(data["priceChangePercent"]),
            "high_24h":          float(data["highPrice"]),
            "low_24h":           float(data["lowPrice"]),
            "volume_24h":        float(data["volume"]),
            "quote_volume_24h":  float(data["quoteVolume"]),
        }
        
        if currency.upper() == "INR":
            rate = get_usd_inr_rate()
            for key in ["last_price", "price_change", "high_24h", "low_24h"]:
                stats[key] *= rate
                
        return stats
    except Exception as e:
        print(f"[binance_data] Could not fetch 24h stats: {e}")
        return None


def get_klines(
    symbol:   str = "BTCUSDT",
    interval: str = "1h",
    n:        int = 100,
    currency: str = "USD"
) -> Optional[pd.DataFrame]:
    """
    Fetch recent OHLCV candlestick data from Binance.

    Args:
        symbol:   Binance pair (default: BTCUSDT)
        interval: Kline interval — 1m, 5m, 15m, 1h, 4h, 1d, 1w, etc.
        n:        Number of candles to fetch (max 1000)
        currency: "USD" or "INR"

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume]
        indexed by UTC datetime, or None on failure.
    """
    url = f"{BASE_URL}/klines"
    params = {"symbol": symbol, "interval": interval, "limit": n}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json()

        df = pd.DataFrame(raw, columns=[
            "open_time", "Open", "High", "Low", "Close", "Volume",
            "close_time", "quote_vol", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df.set_index("open_time", inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        
        if currency.upper() == "INR":
            rate = get_usd_inr_rate()
            df[["Open", "High", "Low", "Close"]] *= rate
            
        print(f"[binance_data] Fetched {len(df)} {interval} candles for {symbol}")
        return df

    except Exception as e:
        print(f"[binance_data] Could not fetch klines: {e}")
        return None


if __name__ == "__main__":
    # Quick sanity check
    price = get_live_price(currency="INR")
    stats = get_24h_stats(currency="INR")
    if stats:
        print(f"24h change : {stats['price_change_pct']:+.2f}%")
        print(f"24h high   : ₹{stats['high_24h']:,.2f}")
        print(f"24h low    : ₹{stats['low_24h']:,.2f}")
    df = get_klines(interval="1h", n=5, currency="INR")
    if df is not None:
        print(df)
