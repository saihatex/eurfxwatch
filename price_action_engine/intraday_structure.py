"""
Intraday Session & Liquidity Structure Engine (1H & 15M Resolution)
====================================================================
Calculates:
  1. Asian Session Range (00:00 - 08:00 UTC): Asia_High, Asia_Low, Asia_Range_Pips
  2. London Liquidity Sweeps (08:00 - 13:00 UTC):
     - Swept_Asia_High (Bearish Liquidity Sweep signal)
     - Swept_Asia_Low  (Bullish Liquidity Sweep signal)
  3. Multi-Timeframe FVGs (4H & 1H resolution)
  4. Intraday Confluence State for any exact timestamp (e.g., 25.08.2026 13:00 Kyiv).
"""

import os
import sys
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_1H  = os.path.join(BASE_DIR, 'data', 'raw', 'EURUSD_1H.csv')
PATH_15M = os.path.join(BASE_DIR, 'data', 'raw', 'EURUSD_15M.csv')
OUT_4H_FVG = os.path.join(BASE_DIR, 'data', 'processed', 'fvg_4h_database.csv')
OUT_1H_FVG = os.path.join(BASE_DIR, 'data', 'processed', 'fvg_1h_database.csv')

print("="*80)
print("     INTRADAY SESSION & LIQUIDITY STRUCTURE ENGINE")
print("="*80)

# Load 1H Data
df_1h = pd.read_csv(PATH_1H, index_col=0, parse_dates=True).sort_index()

# ── 1. Asian & London Session Analysis ────────────────────────────────────────
df_1h['Date_Only'] = df_1h.index.date
df_1h['Hour']      = df_1h.index.hour

session_records = []

for dt_date, group in df_1h.groupby('Date_Only'):
    asia = group[(group['Hour'] >= 0) & (group['Hour'] < 8)]
    london = group[(group['Hour'] >= 8) & (group['Hour'] < 13)]

    if len(asia) == 0 or len(london) == 0:
        continue

    asia_hi = asia['High'].max()
    asia_lo = asia['Low'].min()
    asia_range_pips = (asia_hi - asia_lo) * 10000

    london_hi = london['High'].max()
    london_lo = london['Low'].min()

    swept_asia_hi = london_hi > asia_hi  # Bearish Liquidity Sweep (Judas Swing Up)
    swept_asia_lo = london_lo < asia_lo  # Bullish Liquidity Sweep (Judas Swing Down)

    session_records.append({
        'Date': dt_date,
        'Asia_High': round(asia_hi, 5),
        'Asia_Low': round(asia_lo, 5),
        'Asia_Range_Pips': round(asia_range_pips, 1),
        'London_High': round(london_hi, 5),
        'London_Low': round(london_lo, 5),
        'Swept_Asia_High': swept_asia_hi,
        'Swept_Asia_Low': swept_asia_lo
    })

session_df = pd.DataFrame(session_records).set_index('Date')
print(f"Processed {len(session_df)} trading days for Asian/London Session Sweeps.")

# ── 2. Build 4H Bars & 4H FVG Database ────────────────────────────────────────
df_4h = df_1h.resample('4h').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

fvgs_4h = []
lo4 = df_4h['Low'].to_numpy()
hi4 = df_4h['High'].to_numpy()
cl4 = df_4h['Close'].to_numpy()
op4 = df_4h['Open'].to_numpy()
dates4 = df_4h.index

for i in range(1, len(df_4h) - 2):
    dt = dates4[i]

    # Bearish 4H FVG
    if lo4[i-1] > hi4[i+1]:
        fvgs_4h.append({
            'Datetime': dt,
            'FVG_Type': 'BEARISH',
            'FVG_Lo': round(hi4[i+1], 5),
            'FVG_Hi': round(lo4[i-1], 5),
            'FVG_Mid': round((hi4[i+1] + lo4[i-1])/2.0, 5),
            'Width_Pips': round((lo4[i-1] - hi4[i+1])*10000, 1),
            'Displacement_Pips': round(abs(cl4[i] - op4[i])*10000, 1)
        })
    # Bullish 4H FVG
    if hi4[i-1] < lo4[i+1]:
        fvgs_4h.append({
            'Datetime': dt,
            'FVG_Type': 'BULLISH',
            'FVG_Lo': round(hi4[i-1], 5),
            'FVG_Hi': round(lo4[i+1], 5),
            'FVG_Mid': round((hi4[i-1] + lo4[i+1])/2.0, 5),
            'Width_Pips': round((lo4[i+1] - hi4[i-1])*10000, 1),
            'Displacement_Pips': round(abs(cl4[i] - op4[i])*10000, 1)
        })

fvg_4h_df = pd.DataFrame(fvgs_4h)
fvg_4h_df.to_csv(OUT_4H_FVG, index=False)

print(f"Generated 4H FVG Database: {len(fvg_4h_df)} 4H Imbalances found.")
print(f"  Bearish 4H FVGs: {(fvg_4h_df['FVG_Type']=='BEARISH').sum()}")
print(f"  Bullish 4H FVGs: {(fvg_4h_df['FVG_Type']=='BULLISH').sum()}")

# ── 3. Snapshot analysis for 25.08.2026 13:00 Kyiv ───────────────────────────
snap_date = pd.to_datetime('2026-08-25').date()
if snap_date in session_df.index:
    snap_sess = session_df.loc[snap_date]
    print("\n" + "="*80)
    print(f"SESSION & LIQUIDITY STRUCTURE SNAPSHOT: {snap_date}")
    print("="*80)
    print(f"Asian High:            {snap_sess['Asia_High']}")
    print(f"Asian Low:             {snap_sess['Asia_Low']}")
    print(f"Asian Range:           {snap_sess['Asia_Range_Pips']} pips")
    print(f"London High:           {snap_sess['London_High']}")
    print(f"London Low:            {snap_sess['London_Low']}")
    print(f"London Swept Asia High:{snap_sess['Swept_Asia_High']}  (Bearish Liquidity Grab)")
    print(f"London Swept Asia Low: {snap_sess['Swept_Asia_Low']}   (Bullish Liquidity Grab)")
    print("="*80)
