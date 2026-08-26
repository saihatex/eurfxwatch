"""
Download daily EURUSD OHLCV from Yahoo Finance.
Saves to data/raw/EURUSD_OHLC.csv
"""
import os, sys
import pandas as pd

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'EURUSD_OHLC.csv')

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed — running: pip install yfinance")
    os.system(f"{sys.executable} -m pip install yfinance -q")
    import yfinance as yf

print("Downloading EURUSD=X daily OHLCV from Yahoo Finance (2003–present)...")
ticker = yf.Ticker("EURUSD=X")
df = ticker.history(start="2003-01-01", interval="1d", auto_adjust=True)

df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
df.index = df.index.tz_localize(None)
df.index.name = 'Date'
df = df.sort_index()
df.to_csv(OUT_PATH)

print(f"Saved {len(df)} rows to {OUT_PATH}")
print(f"Date range: {df.index[0].date()} – {df.index[-1].date()}")
print(df.tail(5).to_string())
