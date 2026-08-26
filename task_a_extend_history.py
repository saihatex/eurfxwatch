"""
TASK A: Fix Issue #4 — Extend Data History
===========================================
1. Download EURUSD=X daily data since 2010 → data/raw/EURUSD_1D.csv
2. Aggregate 1H → 4H bars → data/raw/EURUSD_4H.csv
3. Run FVG scanner on 4H data → regenerate data/processed/fvg_4h_database.csv
"""

import os, sys
import pandas as pd
import yfinance as yf

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_1H  = os.path.join(BASE_DIR, 'data', 'raw', 'EURUSD_1H.csv')
OUT_1D   = os.path.join(BASE_DIR, 'data', 'raw', 'EURUSD_1D.csv')
OUT_4H   = os.path.join(BASE_DIR, 'data', 'raw', 'EURUSD_4H.csv')
OUT_FVG  = os.path.join(BASE_DIR, 'data', 'processed', 'fvg_4h_database.csv')

print("=" * 70)
print("  TASK A — EXTEND DATA HISTORY")
print("=" * 70)

# ─── Step 1: Download 1D data ────────────────────────────────────────────────
print("\n[Step 1] Downloading EURUSD=X daily bars since 2010...")
df_1d = yf.download('EURUSD=X', start='2010-01-01', interval='1d',
                     auto_adjust=True, progress=False)
# yfinance may return MultiIndex columns; flatten if needed
if isinstance(df_1d.columns, pd.MultiIndex):
    df_1d.columns = [c[0] for c in df_1d.columns]

df_1d.index.name = 'Date'
df_1d = df_1d[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
df_1d.to_csv(OUT_1D)
print(f"  Saved {len(df_1d)} daily bars → {OUT_1D}")
print(f"  Date range: {df_1d.index[0].date()} → {df_1d.index[-1].date()}")

# ─── Step 2: Aggregate 1H → 4H bars ─────────────────────────────────────────
print("\n[Step 2] Aggregating 1H bars into 4H bars...")
df_1h = pd.read_csv(PATH_1H, index_col=0, parse_dates=True).sort_index()
print(f"  Loaded {len(df_1h)} 1H bars  ({df_1h.index[0]} → {df_1h.index[-1]})")

df_4h = df_1h.resample('4h').agg({
    'Open':   'first',
    'High':   'max',
    'Low':    'min',
    'Close':  'last',
    'Volume': 'sum'
}).dropna(subset=['Open', 'Close'])

df_4h.index.name = 'Datetime'
df_4h.to_csv(OUT_4H)
print(f"  Saved {len(df_4h)} 4H bars → {OUT_4H}")
print(f"  Date range: {df_4h.index[0]} → {df_4h.index[-1]}")

# ─── Step 3: FVG Scanner on 4H data ─────────────────────────────────────────
print("\n[Step 3] Running FVG scanner on 4H data...")

lo4    = df_4h['Low'].to_numpy()
hi4    = df_4h['High'].to_numpy()
cl4    = df_4h['Close'].to_numpy()
op4    = df_4h['Open'].to_numpy()
dates4 = df_4h.index

fvgs_4h = []
for i in range(1, len(df_4h) - 2):
    dt = dates4[i]
    # Bearish 4H FVG: prev bar low > next bar high → gap between them
    if lo4[i-1] > hi4[i+1]:
        fvgs_4h.append({
            'Datetime':          dt,
            'FVG_Type':          'BEARISH',
            'FVG_Lo':            round(hi4[i+1], 5),
            'FVG_Hi':            round(lo4[i-1], 5),
            'FVG_Mid':           round((hi4[i+1] + lo4[i-1]) / 2.0, 5),
            'Width_Pips':        round((lo4[i-1] - hi4[i+1]) * 10000, 1),
            'Displacement_Pips': round(abs(cl4[i] - op4[i]) * 10000, 1),
        })
    # Bullish 4H FVG: prev bar high < next bar low → gap between them
    if hi4[i-1] < lo4[i+1]:
        fvgs_4h.append({
            'Datetime':          dt,
            'FVG_Type':          'BULLISH',
            'FVG_Lo':            round(hi4[i-1], 5),
            'FVG_Hi':            round(lo4[i+1], 5),
            'FVG_Mid':           round((hi4[i-1] + lo4[i+1]) / 2.0, 5),
            'Width_Pips':        round((lo4[i+1] - hi4[i-1]) * 10000, 1),
            'Displacement_Pips': round(abs(cl4[i] - op4[i]) * 10000, 1),
        })

fvg_4h_df = pd.DataFrame(fvgs_4h)
fvg_4h_df.to_csv(OUT_FVG, index=False)

bear_n = (fvg_4h_df['FVG_Type'] == 'BEARISH').sum()
bull_n = (fvg_4h_df['FVG_Type'] == 'BULLISH').sum()
print(f"  Total 4H FVGs found: {len(fvg_4h_df)}")
print(f"    Bearish: {bear_n}   Bullish: {bull_n}")
print(f"  FVG date range: {fvg_4h_df['Datetime'].iloc[0]} → {fvg_4h_df['Datetime'].iloc[-1]}")
print(f"  Saved → {OUT_FVG}")

print("\n[TASK A COMPLETE]")
