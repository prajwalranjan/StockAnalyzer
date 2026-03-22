"""
data_fetcher.py — Fetches stock prices and historical data.
Uses yfinance (Yahoo Finance) — completely free, no API key needed.

For Indian stocks, Yahoo Finance uses the format: SYMBOL.NS (NSE) or SYMBOL.BO (BSE)
Example: RELIANCE.NS, INFY.NS, TCS.NS
"""

import yfinance as yf
import pandas as pd


# A curated list of Nifty 500 stocks in the ₹50–₹2000 price range
# Good for swing trading with ₹20,000 capital
# You can expand this list later
STOCK_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "LT.NS", "WIPRO.NS",
    "HCLTECH.NS", "SUNPHARMA.NS", "TATAMOTORS.NS", "MARUTI.NS", "BAJFINANCE.NS",
    "TITAN.NS", "ADANIENT.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS", "NESTLEIND.NS",
    "DMART.NS", "PIDILITIND.NS", "MUTHOOTFIN.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS",
    "BANDHANBNK.NS", "CHOLAFIN.NS", "MANAPPURAM.NS", "PERSISTENT.NS", "LTIM.NS",
    "TATAPOWER.NS", "ADANIPORTS.NS", "ONGC.NS", "BPCL.NS", "IOC.NS",
    "GAIL.NS", "COALINDIA.NS", "HINDALCO.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "VEDL.NS", "SAIL.NS", "NMDC.NS", "NTPC.NS", "POWERGRID.NS",
    "HAVELLS.NS", "VOLTAS.NS", "WHIRLPOOL.NS", "CROMPTON.NS", "POLYCAB.NS",
]


def get_current_price(symbol: str) -> float:
    """
    Gets the latest price for a stock.
    Returns 0 if something goes wrong (e.g. no internet).
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d", interval="1m")
        if not data.empty:
            return float(data["Close"].iloc[-1])
        return 0.0
    except Exception as e:
        print(f"⚠️  Could not fetch price for {symbol}: {e}")
        return 0.0


def get_historical_data(symbol: str, period: str = "6mo") -> pd.DataFrame:
    """
    Gets historical daily price data for a stock.
    
    period options: "1mo", "3mo", "6mo", "1y", "2y"
    
    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    Each row = one trading day.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print(f"⚠️  Could not fetch history for {symbol}: {e}")
        return pd.DataFrame()


def get_multiple_prices(symbols: list) -> dict:
    """
    Efficiently fetches current prices for multiple stocks at once.
    Returns a dict like: {"RELIANCE.NS": 2450.5, "TCS.NS": 3800.2, ...}
    """
    prices = {}
    try:
        data = yf.download(symbols, period="2d", progress=False)["Close"]
        for symbol in symbols:
            if symbol in data.columns:
                prices[symbol] = float(data[symbol].dropna().iloc[-1])
    except Exception as e:
        print(f"⚠️  Batch price fetch failed: {e}")
        # Fall back to fetching one by one
        for symbol in symbols:
            prices[symbol] = get_current_price(symbol)
    return prices
