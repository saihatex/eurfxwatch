"""
Download Intraday 1H and 15M EUR/USD OHLC Data from Yahoo Finance (EURUSD=X)
==========================================================================
Saves:
  - data/raw/EURUSD_1H.csv
  - data/raw/EURUSD_15M.csv
"""

import os
import sys
import pandas as pd

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_1H   = os.path.join(BASE_DIR, 'data', 'raw', 'EURUSD_1H.csv')
OUT_15M  = os.path.join(BASE_DIR, 'data', 'raw', 'EURUSD_15M.csv')

try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    os.system(f"{sys.executable} -m pip install yfinance -q")
    import yfinance as yf

print("="*80)
print("     DOWNLOADING INTRADAY 1H & 15M EUR/USD DATA")
print("="*80)

ticker = yf.Ticker("EURUSD=X")

# 1. Download 1H data (max historical window available ~ 730 days)
print("[yfinance] Downloading 1H EURUSD data...")
df_1h = ticker.history(period="730d", interval="1h", auto_adjust=True)
if len(df_1h) > 0:
    df_1h = df_1h[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_1h.index = df_1h.index.tz_localize(None)
    df_1h.index.name = 'Datetime'
    df_1h.to_csv(OUT_1H)
    print(f"  -> Saved {len(df_1h)} 1H bars ({df_1h.index[0]} to {df_1h.index[-1]})")

# 2. Download 15M data (max historical window available ~ 60 days)
print("[yfinance] Downloading 15M EURUSD data...")
df_15m = ticker.history(period="60d", interval="15m", auto_adjust=True)
if len(df_15m) > 0:
    df_15m = df_15m[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_15m.index = df_15m.index.tz_localize(None)
    df_15m.index.name = 'Datetime'
    df_15m.to_csv(OUT_15M)
    print(f"  -> Saved {len(df_15m)} 15M bars ({df_15m.index[0]} to {df_15m.index[-1]})")

print("\n[INTRADAY DOWNLOAD COMPLETE]")
