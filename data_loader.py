"""
data_loader.py
--------------
Downloads historical Bitcoin (BTC-USD) OHLCV data from Yahoo Finance
and optionally fetches the live price from Binance (no API key needed
for public ticker endpoints).
"""

import os
import yfinance as yf
import pandas as pd
import requests


# ── Helpers ───────────────────────────────────────────────────

def get_usd_inr_rate() -> float:
    """
    Fetch the live USD to INR exchange rate using yfinance.
    """
    try:
        df = yf.download("INR=X", period="1d", progress=False)
        if not df.empty:
            rate = float(df["Close"].iloc[-1].item() if isinstance(df["Close"], pd.DataFrame) else df["Close"].iloc[-1])
            return rate
    except Exception as e:
        print(f"[data_loader] Could not fetch USD-INR rate: {e}")
    # Fallback rate if yfinance fails
    return 83.0


# ── Historical data ───────────────────────────────────────────

def load_data(
    ticker: str = "BTC-USD",
    start: str  = "2017-01-01",
    save: bool  = True,
    currency: str = "USD"
) -> pd.DataFrame:
    """
    Download Bitcoin historical OHLCV data.

    Args:
        ticker: Yahoo Finance ticker (ignored if currency="INR", we force "BTC-INR"). Default: BTC-USD
        start:  Start date  YYYY-MM-DD
        save:   Cache to ./data/ as CSV
        currency: "USD" or "INR"

    Returns:
        pd.DataFrame with columns [Open, High, Low, Close, Volume]
    """
    if currency.upper() == "INR":
        ticker = "BTC-INR"

    cache_path = os.path.join("data", f"{ticker.replace('-', '_')}.csv")

    if os.path.exists(cache_path):
        print(f"[data_loader] Loading cached data from {cache_path}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    print(f"[data_loader] Downloading {ticker} from Yahoo Finance …")
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(
            f"No data returned for '{ticker}'. Check your internet connection."
        )

    # yfinance >= 0.2.x may return a MultiIndex (Price, Ticker) — flatten it
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if save:
        os.makedirs("data", exist_ok=True)
        df.to_csv(cache_path)
        print(f"[data_loader] Saved → {cache_path}")

    return df


# ── Live price (Binance public API — no key required) ─────────

def get_live_price(symbol: str = "BTCUSDT", currency: str = "USD") -> float:
    """
    Fetch the current spot price from Binance's public REST API.

    No API key is required for this endpoint.

    Args:
        symbol: Binance trading pair (default: BTCUSDT)
        currency: "USD" or "INR"

    Returns:
        Current price as float, or None if the request fails.
    """
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        price = float(resp.json()["price"])
        
        if currency.upper() == "INR":
            rate = get_usd_inr_rate()
            price = price * rate
            print(f"[data_loader] Live {symbol} price: ₹{price:,.2f} INR (via {rate:.2f} rate)")
        else:
            print(f"[data_loader] Live {symbol} price: ${price:,.2f} USD")
            
        return price
    except Exception as e:
        print(f"[data_loader] Could not fetch live price: {e}")
        return None
