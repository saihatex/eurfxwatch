"""
Price Action Engine — Layer 3: FVG & Bar Anatomy Feature Builder (Corrected Forward Tracking)
========================================================================================
FVG is formed at bar i+1 when bar i+1 closes, establishing the gap zone:
  - Bearish FVG: [High_{i+1}, Low_{i-1}]
  - Bullish FVG: [High_{i-1}, Low_{i+1}]

Forward touch, fill, reject, and break are evaluated strictly starting from bar i+2 onwards
(k >= 2, i.e., future price action AFTER FVG pattern completion) to avoid trivial self-overlap.
"""

import os
import sys
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OHLC_PATH  = os.path.join(BASE_DIR, 'data', 'raw', 'EURUSD_OHLC.csv')
OUT_PA     = os.path.join(BASE_DIR, 'data', 'processed', 'price_action_features.csv')
OUT_FVG    = os.path.join(BASE_DIR, 'data', 'processed', 'fvg_database.csv')

# ── 1. Load OHLC ──────────────────────────────────────────────────────────────
ohlc = pd.read_csv(OHLC_PATH, index_col=0, parse_dates=True).sort_index()
ohlc = ohlc[['Open', 'High', 'Low', 'Close']].dropna()

# ── 2. ATR-14 ─────────────────────────────────────────────────────────────────
tr = pd.concat([
    ohlc['High'] - ohlc['Low'],
    (ohlc['High'] - ohlc['Close'].shift(1)).abs(),
    (ohlc['Low']  - ohlc['Close'].shift(1)).abs(),
], axis=1).max(axis=1)

atr = tr.ewm(span=14, adjust=False).mean()

# ── 3. Bar Anatomy ────────────────────────────────────────────────────────────
O, H, L, C = ohlc['Open'], ohlc['High'], ohlc['Low'], ohlc['Close']

pa = pd.DataFrame(index=ohlc.index)
pa['Open']  = O
pa['High']  = H
pa['Low']   = L
pa['Close'] = C
pa['ATR']   = atr

pa['Direction']     = np.sign(C - O).astype(int)
pa['Body_ATR']      = (C - O).abs() / atr
pa['Range_ATR']     = (H - L) / atr
pa['UpperWick_ATR'] = (H - np.maximum(O, C)) / atr
pa['LowerWick_ATR'] = (np.minimum(O, C) - L) / atr
pa['CloseLocation'] = np.where((H - L) > 1e-8, (C - L) / (H - L), 0.5)

streak = np.zeros(len(pa), dtype=int)
for i in range(1, len(pa)):
    if pa['Direction'].iloc[i] == pa['Direction'].iloc[i-1] and pa['Direction'].iloc[i] != 0:
        streak[i] = streak[i-1] + 1
    else:
        streak[i] = 0
pa['DirStreak'] = streak

pa['Range_Expansion'] = (H - L) / ((H - L).rolling(5).mean())

for h in [1, 2, 3, 5, 10]:
    pa[f'fwd_{h}D'] = np.log(pa['Close'].shift(-h) / pa['Close']) * 100

pa.to_csv(OUT_PA)

# ── 4. FVG Detection & Forward Tracking (Starting at Bar i+2) ─────────────
lo = L.to_numpy()
hi = H.to_numpy()
cl = C.to_numpy()
op = O.to_numpy()
dates = pa.index

fvgs = []

for i in range(1, len(pa) - 2):
    dt = dates[i]
    atr_val = pa['ATR'].iloc[i]

    # Bearish FVG (displacement bar i, confirmed at bar i+1)
    if lo[i-1] > hi[i+1]:
        gap_lo = hi[i+1]
        gap_hi = lo[i-1]
        gap_mid = (gap_lo + gap_hi) / 2.0
        width = gap_hi - gap_lo

        # Track future price action starting from bar i+2 (k=2) up to 20 bars
        touch_1d = touch_3d = touch_5d = touch_10d = False
        fill_50_10d = fill_100_10d = reject_10d = break_10d = False

        for k in range(2, 22):
            if i + k >= len(pa):
                break
            b_lo, b_hi, b_cl = lo[i+k], hi[i+k], cl[i+k]

            # For Bearish FVG, price touches gap if High >= gap_lo
            overlaps = b_hi >= gap_lo

            if overlaps:
                days_ahead = k - 1  # bar i+2 is 1 day ahead of completion at i+1
                if days_ahead <= 1:  touch_1d  = True
                if days_ahead <= 3:  touch_3d  = True
                if days_ahead <= 5:  touch_5d  = True
                if days_ahead <= 10: touch_10d = True

                if b_hi >= gap_mid and days_ahead <= 10:
                    fill_50_10d = True
                if b_hi >= gap_hi and days_ahead <= 10:
                    fill_100_10d = True

                if b_cl > gap_hi and days_ahead <= 10:
                    break_10d = True
                elif b_hi >= gap_lo and b_cl < gap_lo and days_ahead <= 10:
                    reject_10d = True

        fvgs.append({
            'FVG_Date'    : dt,
            'FVG_Type'    : 'BEARISH',
            'FVG_Lo'      : round(gap_lo, 6),
            'FVG_Hi'      : round(gap_hi, 6),
            'FVG_Mid'     : round(gap_mid, 6),
            'Width_pct'   : round(width / cl[i] * 100, 4),
            'Width_ATR'   : round(width / atr_val, 4) if atr_val > 0 else np.nan,
            'Displacement': round(abs(cl[i] - op[i]) / atr_val, 4) if atr_val > 0 else np.nan,
            'Bar_Range'   : round((hi[i] - lo[i]) / atr_val, 4) if atr_val > 0 else np.nan,
            'Close_Loc'   : round(pa['CloseLocation'].iloc[i], 4),
            'Touch_1D'    : touch_1d,
            'Touch_3D'    : touch_3d,
            'Touch_5D'    : touch_5d,
            'Touch_10D'   : touch_10d,
            'Fill_50_10D' : fill_50_10d,
            'Fill_100_10D': fill_100_10d,
            'Reject_10D'  : reject_10d,
            'Break_10D'   : break_10d
        })

    # Bullish FVG (displacement bar i, confirmed at bar i+1)
    if hi[i-1] < lo[i+1]:
        gap_lo = hi[i-1]
        gap_hi = lo[i+1]
        gap_mid = (gap_lo + gap_hi) / 2.0
        width = gap_hi - gap_lo

        touch_1d = touch_3d = touch_5d = touch_10d = False
        fill_50_10d = fill_100_10d = reject_10d = break_10d = False

        for k in range(2, 22):
            if i + k >= len(pa):
                break
            b_lo, b_hi, b_cl = lo[i+k], hi[i+k], cl[i+k]

            # For Bullish FVG, price touches gap if Low <= gap_hi
            overlaps = b_lo <= gap_hi

            if overlaps:
                days_ahead = k - 1
                if days_ahead <= 1:  touch_1d  = True
                if days_ahead <= 3:  touch_3d  = True
                if days_ahead <= 5:  touch_5d  = True
                if days_ahead <= 10: touch_10d = True

                if b_lo <= gap_mid and days_ahead <= 10:
                    fill_50_10d = True
                if b_lo <= gap_lo and days_ahead <= 10:
                    fill_100_10d = True

                if b_cl < gap_lo and days_ahead <= 10:
                    break_10d = True
                elif b_lo <= gap_hi and b_cl > gap_hi and days_ahead <= 10:
                    reject_10d = True

        fvgs.append({
            'FVG_Date'    : dt,
            'FVG_Type'    : 'BULLISH',
            'FVG_Lo'      : round(gap_lo, 6),
            'FVG_Hi'      : round(gap_hi, 6),
            'FVG_Mid'     : round(gap_mid, 6),
            'Width_pct'   : round(width / cl[i] * 100, 4),
            'Width_ATR'   : round(width / atr_val, 4) if atr_val > 0 else np.nan,
            'Displacement': round(abs(cl[i] - op[i]) / atr_val, 4) if atr_val > 0 else np.nan,
            'Bar_Range'   : round((hi[i] - lo[i]) / atr_val, 4) if atr_val > 0 else np.nan,
            'Close_Loc'   : round(pa['CloseLocation'].iloc[i], 4),
            'Touch_1D'    : touch_1d,
            'Touch_3D'    : touch_3d,
            'Touch_5D'    : touch_5d,
            'Touch_10D'   : touch_10d,
            'Fill_50_10D' : fill_50_10d,
            'Fill_100_10D': fill_100_10d,
            'Reject_10D'  : reject_10d,
            'Break_10D'   : break_10d
        })

fvg_df = pd.DataFrame(fvgs)
fvg_df.to_csv(OUT_FVG, index=False)

print("="*80)
print("  CORRECTED PRICE ACTION & FVG BUILD SUMMARY")
print("="*80)
print(f"PA Features saved: {len(pa)} rows")
print(f"FVG Database saved: {len(fvg_df)} total FVGs")
print(f"  BEARISH FVGs: {(fvg_df['FVG_Type'] == 'BEARISH').sum()}")
print(f"  BULLISH FVGs: {(fvg_df['FVG_Type'] == 'BULLISH').sum()}")
print("\nUnconditional Forward Touch Rates (Starting bar i+2):")
for ftype in ['BEARISH', 'BULLISH']:
    sub = fvg_df[fvg_df['FVG_Type'] == ftype]
    print(f"\n{ftype} FVGs (N={len(sub)}):")
    print(f"  Touch 1D:  {sub['Touch_1D'].mean()*100:.1f}%")
    print(f"  Touch 3D:  {sub['Touch_3D'].mean()*100:.1f}%")
    print(f"  Touch 5D:  {sub['Touch_5D'].mean()*100:.1f}%")
    print(f"  Touch 10D: {sub['Touch_10D'].mean()*100:.1f}%")
    print(f"  Fill 50% 10D:  {sub['Fill_50_10D'].mean()*100:.1f}%")
    print(f"  Fill 100% 10D: {sub['Fill_100_10D'].mean()*100:.1f}%")
    print(f"  Reject 10D:    {sub['Reject_10D'].mean()*100:.1f}%")
