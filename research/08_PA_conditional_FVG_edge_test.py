"""
FVG Conditional Edge Test v1
==============================
Core question: Does conditioning on regime (X1, X2, X4) and FVG
structural features (Distance/ATR, Displacement, Compression) produce
a meaningful Δ_edge = P(conditional) − P(base = 78.3%) ?

Test structure:
  (A) Univariate dose-response: each variable alone vs base rate
  (B) Regime conditioning: split by X1/X2 buckets
  (C) Distance/ATR dose-response (primary hypothesis)
  (D) Combined conditioning: Direction + Distance + Regime
  (E) OOS split (Train 2010-2019 | Val 2020-2022 | OOS 2023-2026)

All joins are date-aligned (FVG date = displacement bar date = i).
X1/X2 factors at day i come from aligned_raw_data.csv.
"""

import os, sys
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FVG_PATH  = os.path.join(BASE_DIR, 'data', 'processed', 'fvg_database.csv')
PA_PATH   = os.path.join(BASE_DIR, 'data', 'processed', 'price_action_features.csv')
AL_PATH   = os.path.join(BASE_DIR, 'data', 'processed', 'aligned_raw_data.csv')

BASE_RATE_3D = 0.655   # true unconditional 3D forward touch rate (bar i+2 onwards)

# ── 1. Load datasets ──────────────────────────────────────────────────────────
fvg = pd.read_csv(FVG_PATH, parse_dates=['FVG_Date'])
pa  = pd.read_csv(PA_PATH,  index_col=0, parse_dates=True)
al  = pd.read_csv(AL_PATH,  index_col=0, parse_dates=True)

# ── 2. Compute X1 / X2 point-in-time (same loop as before) ───────────────────
al = al.sort_index().dropna(subset=['EURUSD', 'X4'])
al['log_ret']    = np.log(al['EURUSD'] / al['EURUSD'].shift(1))
al['5D_log_ret'] = np.log(al['EURUSD'] / al['EURUSD'].shift(5))
al['vol_20d']    = al['log_ret'].rolling(20).std()

arr_mom = al['5D_log_ret'].to_numpy()
arr_vol = al['vol_20d'].to_numpy()
N = len(al)
X1a, X2a = np.full(N, np.nan), np.full(N, np.nan)

win3y, min_h = 756, 252
for i in range(min_h, N):
    hm = arr_mom[5:i]; vm = arr_mom[i]
    if not np.isnan(vm) and len(hm):
        X1a[i] = (hm <= vm).mean()
    if i >= win3y:
        hv = arr_vol[i-win3y:i]; vv = arr_vol[i]
        if not np.isnan(vv):
            hv = hv[~np.isnan(hv)]
            if len(hv): X2a[i] = (hv <= vv).mean()

al['X1'] = X1a
al['X2'] = X2a

# ── 3. Build master FVG analysis table ───────────────────────────────────────
# Merge FVG with PA features and regime factors at FVG_Date

pa_sel = pa[['ATR', 'Body_ATR', 'Range_ATR', 'CloseLocation',
             'Range_Expansion', 'Direction', 'Close']].copy()
pa_sel.index.name = 'FVG_Date'

al_sel = al[['X1', 'X2', 'X4']].copy()
al_sel.index.name = 'FVG_Date'

# Compute Distance to FVG from day-after close (bar i+1)
# We use pa['Close'].shift(-1) as the close of bar i+1 relative to FVG bar i
pa['Close_next'] = pa['Close'].shift(-1)
pa['ATR_next']   = pa['ATR'].shift(-1)
pa_sel['Close_next'] = pa['Close_next']
pa_sel['ATR_next']   = pa['ATR_next']

fvg = fvg.set_index('FVG_Date')
merged = fvg.join(pa_sel, how='inner').join(al_sel, how='inner')
merged = merged.dropna(subset=['X1', 'X2', 'X4', 'ATR', 'Touch_3D'])
merged = merged[merged.index.year >= 2010]   # aligned with rest of study

# Distance from Close_next to nearest FVG edge (normalized by ATR_next)
# For BEARISH FVG: zone is above bar i+1 close (price needs to go UP to touch it)
#   distance = (FVG_Lo - Close_next) / ATR_next   [positive = above price]
# For BULLISH FVG: zone is below bar i+1 close
#   distance = (Close_next - FVG_Hi) / ATR_next   [positive = below price]

def compute_distance(row):
    c    = row['Close_next']
    atr  = row['ATR_next']
    if pd.isna(c) or pd.isna(atr) or atr <= 0:
        return np.nan
    if row['FVG_Type'] == 'BEARISH':
        return (row['FVG_Lo'] - c) / atr   # >0 means gap is above; <0 means price already in gap
    else:
        return (c - row['FVG_Hi']) / atr   # >0 means gap is below

merged['Dist_ATR'] = merged.apply(compute_distance, axis=1)

# Era labels
def era(yr):
    if yr <= 2019: return 'Train (2010-2019)'
    if yr <= 2022: return 'Val (2020-2022)'
    return 'OOS (2023-2026)'

merged['Era'] = merged.index.year.map(era)

# ── 4. HELPER ─────────────────────────────────────────────────────────────────
def edge_row(label, sub, col='Touch_3D', base=BASE_RATE_3D):
    n = len(sub)
    if n < 10:
        return {'Label': label, 'N': n, 'P_3D': np.nan, 'Δ_edge_pp': np.nan,
                'P_5D': np.nan, 'P_10D': np.nan}
    p3  = sub[col].mean()
    p5  = sub['Touch_5D'].mean()
    p10 = sub['Touch_10D'].mean()
    return {
        'Label'     : label,
        'N'         : n,
        'P_3D'      : round(p3*100, 1),
        'Δ_edge_pp' : round((p3 - base)*100, 1),
        'P_5D'      : round(p5*100, 1),
        'P_10D'     : round(p10*100, 1),
    }

def print_table(rows, title):
    df = pd.DataFrame(rows)
    print(f'\n{"═"*78}')
    print(f'  {title}')
    print(f'  BASE RATE 3D = {BASE_RATE_3D*100:.1f}%  |  Δ_edge = conditional − base')
    print(f'{"═"*78}')
    print(df.to_string(index=False))

# ── 5A. Direction alone ───────────────────────────────────────────────────────
rows_A = []
for ftype in ['BEARISH', 'BULLISH']:
    sub = merged[merged['FVG_Type'] == ftype]
    rows_A.append(edge_row(ftype, sub))
print_table(rows_A, '(A) DIRECTION ALONE')

# ── 5B. X1 regime dose-response ───────────────────────────────────────────────
rows_B = []
bins_x1 = [(0.0, 0.25, 'X1<25% (Bearish)'),
            (0.25, 0.50, 'X1 25-50%'),
            (0.50, 0.75, 'X1 50-75%'),
            (0.75, 1.01, 'X1>75% (Bullish)')]
for lo, hi, lbl in bins_x1:
    sub = merged[(merged['X1'] >= lo) & (merged['X1'] < hi)]
    rows_B.append(edge_row(lbl, sub))
print_table(rows_B, '(B) X1 MOMENTUM REGIME DOSE-RESPONSE  (all FVGs)')

# ── 5C. X2 volatility dose-response ──────────────────────────────────────────
rows_C = []
bins_x2 = [(0.0, 0.25, 'X2<25% (Low Vol)'),
            (0.25, 0.75, 'X2 25-75% (Normal)'),
            (0.75, 0.90, 'X2 75-90% (High Vol)'),
            (0.90, 1.01, 'X2>90% (Crisis)')]
for lo, hi, lbl in bins_x2:
    sub = merged[(merged['X2'] >= lo) & (merged['X2'] < hi)]
    rows_C.append(edge_row(lbl, sub))
print_table(rows_C, '(C) X2 VOLATILITY REGIME DOSE-RESPONSE  (all FVGs)')

# ── 5D. Distance/ATR dose-response (MAIN HYPOTHESIS) ─────────────────────────
rows_D = []
dist_bins = [(-99, 0.0,  'Dist<0 (already inside FVG)'),
             (0.0,  0.25, 'Dist 0–0.25 ATR'),
             (0.25, 0.50, 'Dist 0.25–0.5 ATR'),
             (0.50, 1.00, 'Dist 0.5–1.0 ATR'),
             (1.00, 2.00, 'Dist 1–2 ATR'),
             (2.00, 99,   'Dist >2 ATR (far away)')]
for lo, hi, lbl in dist_bins:
    sub = merged[(merged['Dist_ATR'] >= lo) & (merged['Dist_ATR'] < hi)]
    rows_D.append(edge_row(lbl, sub))
print_table(rows_D, '(D) DISTANCE/ATR DOSE-RESPONSE  (main hypothesis)')

# ── 5E. Displacement dose-response ───────────────────────────────────────────
rows_E = []
disp_bins = [(0.0, 0.5, 'Displacement <0.5 ATR (weak bar)'),
             (0.5, 1.0, 'Displacement 0.5–1.0 ATR'),
             (1.0, 2.0, 'Displacement 1.0–2.0 ATR'),
             (2.0, 99,  'Displacement >2.0 ATR (strong bar)')]
for lo, hi, lbl in disp_bins:
    sub = merged[(merged['Displacement'] >= lo) & (merged['Displacement'] < hi)]
    rows_E.append(edge_row(lbl, sub))
print_table(rows_E, '(E) DISPLACEMENT DOSE-RESPONSE')

# ── 5F. Compression (Range/ATR) ───────────────────────────────────────────────
rows_F = []
range_bins = [(0.0, 0.5, 'Range <0.5 ATR (compression)'),
              (0.5, 1.0, 'Range 0.5–1.0 ATR'),
              (1.0, 1.5, 'Range 1.0–1.5 ATR'),
              (1.5, 99,  'Range >1.5 ATR (expansion)')]
for lo, hi, lbl in range_bins:
    sub = merged[(merged['Range_ATR'] >= lo) & (merged['Range_ATR'] < hi)]
    rows_F.append(edge_row(lbl, sub))
print_table(rows_F, '(F) FVG BAR RANGE/ATR (COMPRESSION vs EXPANSION)')

# ── 5G. Key interaction: Direction × Distance bucket ─────────────────────────
rows_G = []
for ftype in ['BEARISH', 'BULLISH']:
    for lo, hi, dlbl in dist_bins:
        sub = merged[(merged['FVG_Type'] == ftype) &
                     (merged['Dist_ATR'] >= lo) & (merged['Dist_ATR'] < hi)]
        lbl = f'{ftype[:4]} | {dlbl}'
        rows_G.append(edge_row(lbl, sub))
print_table(rows_G, '(G) DIRECTION × DISTANCE INTERACTION')

# ── 5H. X1/X2 regime × Distance (combined) ───────────────────────────────────
rows_H = []
regimes = [
    ('Low-Vol + Bullish (X2<35%, X1>60%)',
     (merged['X2'] < 0.35) & (merged['X1'] > 0.60)),
    ('High-Vol + Bullish (X2>75%, X1>60%)',
     (merged['X2'] > 0.75) & (merged['X1'] > 0.60)),
    ('Low-Vol + Bearish (X2<35%, X1<40%)',
     (merged['X2'] < 0.35) & (merged['X1'] < 0.40)),
    ('High-Vol + Bearish (X2>75%, X1<40%)',
     (merged['X2'] > 0.75) & (merged['X1'] < 0.40)),
]
for lbl, mask in regimes:
    sub_near = merged[mask & (merged['Dist_ATR'] >= 0) & (merged['Dist_ATR'] < 1.0)]
    sub_far  = merged[mask & (merged['Dist_ATR'] >= 1.0)]
    rows_H.append(edge_row(f'{lbl} | Dist<1 ATR', sub_near))
    rows_H.append(edge_row(f'{lbl} | Dist≥1 ATR', sub_far))
print_table(rows_H, '(H) REGIME × DISTANCE COMBINED')

# ── 5I. Era stability (OOS holdout) ──────────────────────────────────────────
rows_I = []
for era_lbl in ['Train (2010-2019)', 'Val (2020-2022)', 'OOS (2023-2026)']:
    sub = merged[merged['Era'] == era_lbl]
    rows_I.append(edge_row(era_lbl + ' | ALL FVGs', sub))
    # Near-distance FVGs only
    sub_near = sub[(sub['Dist_ATR'] >= 0) & (sub['Dist_ATR'] < 1.0)]
    rows_I.append(edge_row(era_lbl + ' | Dist<1 ATR', sub_near))
    # Near + high displacement
    sub_hd = sub_near[sub_near['Displacement'] > 1.0]
    rows_I.append(edge_row(era_lbl + ' | Dist<1 + Disp>1', sub_hd))
print_table(rows_I, '(I) ERA STABILITY — OOS HOLDOUT')

# ── 6. TODAY snapshot for the SHORT @ 1.16837 ────────────────────────────────
print(f'\n{"═"*78}')
print('  APPLICATION: SHORT @ 1.16837 — Nearest BEARISH FVGs below entry')
print(f'{"═"*78}')

# Find BEARISH FVGs with FVG_Hi in range [1.155, 1.168] (plausible targets)
today_close = 1.16837
today_atr   = pa['ATR'].loc['2026-08-25'] if '2026-08-25' in pa.index else 0.007

recent_bear = merged[(merged['FVG_Type'] == 'BEARISH') &
                     (merged['FVG_Hi'] <= today_close) &
                     (merged['FVG_Hi'] >= 1.155) &
                     (merged.index <= pd.to_datetime('2026-08-25'))]

if len(recent_bear):
    recent_bear = recent_bear.sort_values('FVG_Hi', ascending=False).head(5)
    recent_bear['Dist_from_entry'] = (today_close - recent_bear['FVG_Hi']) / today_atr
    print('\nMost recent BEARISH FVGs below entry price (targets for SHORT):')
    cols_show = ['FVG_Lo', 'FVG_Hi', 'Width_ATR', 'Displacement',
                 'Touch_3D', 'Touch_5D', 'Fill_10D', 'Dist_from_entry']
    avail = [c for c in cols_show if c in recent_bear.columns]
    print(recent_bear[avail].round(4).to_string())
else:
    print('No historical BEARISH FVGs found in the 1.155-1.168 range in the database.')
    print('(FVG database records FVGs at the time they formed; live FVG must be manually specified.)')

print()
