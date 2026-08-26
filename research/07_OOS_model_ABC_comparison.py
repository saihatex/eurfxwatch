"""
STRICT OOS TEST: Model A vs B vs C
===================================
v1.0 (A)  : X1 + X2 + X4  [frozen, no changes]
v1.1B (B) : A  + X5 (pre-defined: Q1-Dovish or Q4-Hawkish adds SHORT weight)
v1.1C (C) : B  + X4×X5 interaction term

Rules frozen BEFORE looking at 2023-2026 OOS:
  X5_pctl < 0.25  → Dovish Extreme (Q1)  → +2 to SHORT score
  X5_pctl >= 0.75 → Hawkish Extreme (Q4) → +2 to SHORT score
  X5_pctl 0.25-0.75 → Neutral            → 0

OOS evaluation period: 2023-01-01 → 2026-12-31
In-sample calibration:  2010-01-01 → 2022-12-31  (percentile history only)

No threshold tuning is performed on OOS data.
"""

import os, sys
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE_DIR, 'data', 'processed', 'aligned_raw_data.csv')

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD & BUILD FEATURE TIME-SERIES (full history, zero lookahead)
# ═══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(PROCESSED, index_col=0, parse_dates=True).sort_index()
df = df.dropna(subset=['EURUSD', 'US2Y', 'DE2Y', 'X4'])

df['log_ret']    = np.log(df['EURUSD'] / df['EURUSD'].shift(1))
df['5D_log_ret'] = np.log(df['EURUSD'] / df['EURUSD'].shift(5))
df['vol_20d']    = df['log_ret'].rolling(20).std()
df['X5_raw']     = df['US2Y'] - df['US2Y'].shift(5)   # ΔUS2Y over 5 days

# forward log-returns (%)
for h in [1, 3, 5, 10]:
    df[f'fwd_{h}D'] = np.log(df['EURUSD'].shift(-h) / df['EURUSD']) * 100

df = df.dropna(subset=['5D_log_ret', 'vol_20d', 'X5_raw'])

arr_mom = df['5D_log_ret'].to_numpy()
arr_vol = df['vol_20d'].to_numpy()
arr_x5  = df['X5_raw'].to_numpy()
N       = len(df)

X1_arr = np.full(N, np.nan)
X2_arr = np.full(N, np.nan)
X5_arr = np.full(N, np.nan)

win3y, min_h = 756, 252

print("Computing point-in-time percentiles (full history)…")
for i in range(min_h, N):
    # X1 momentum percentile
    hist_m = arr_mom[5:i]
    if not np.isnan(arr_mom[i]) and len(hist_m):
        X1_arr[i] = (hist_m <= arr_mom[i]).mean()
    # X2 volatility percentile (3-year rolling window)
    if i >= win3y:
        hist_v = arr_vol[i - win3y : i]
        if not np.isnan(arr_vol[i]):
            hv = hist_v[~np.isnan(hist_v)]
            if len(hv):
                X2_arr[i] = (hv <= arr_vol[i]).mean()
    # X5 ΔUS2Y percentile (all history up to t-1)
    hist_x = arr_x5[:i]
    if not np.isnan(arr_x5[i]):
        hx = hist_x[~np.isnan(hist_x)]
        if len(hx):
            X5_arr[i] = (hx <= arr_x5[i]).mean()

df['X1'] = X1_arr
df['X2'] = X2_arr
df['X5_pctl'] = X5_arr

df = df.dropna(subset=['X1', 'X2', 'X5_pctl'])
print(f"Rows with all features: {len(df)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SCORING FUNCTIONS (pre-defined, frozen)
# ═══════════════════════════════════════════════════════════════════════════════

def score_A(x1, x2, x4):
    """Model A = v1.0 exactly."""
    s, l = 0, 0
    if x1 >= 0.90: s += 3
    if x2 >= 0.75: s += 2
    if x4 > +0.05: s += 3
    elif x4 > 0:   s += 1
    if x1 <= 0.10 and x2 >= 0.75: l += 4
    elif 0.50 <= x1 < 0.85:       l += 2
    if x4 < -0.05: l += 3
    elif x4 < 0:   l += 1
    return s, l

# PRE-DEFINED X5 rule (set before looking at OOS):
#   Q1 (x5_pctl < 0.25)  → Dovish Extreme → SHORT +2
#   Q4 (x5_pctl >= 0.75) → Hawkish Extreme → SHORT +2
#   Q2/Q3               → Neutral → 0
X5_DOVISH_THR  = 0.25
X5_HAWKISH_THR = 0.75
X5_ADDON       = 2

def x5_boost(x5_pctl):
    if x5_pctl < X5_DOVISH_THR or x5_pctl >= X5_HAWKISH_THR:
        return X5_ADDON
    return 0

# X4×X5 interaction (Model C):
#   X4>0 & Dovish Q1 → additional +1 to SHORT (empirically ~100% on train)
#   X4<=0 & Hawkish Q4 → additional +1 to SHORT
def x4x5_addon(x4, x5_pctl):
    if x4 > 0 and x5_pctl < X5_DOVISH_THR:
        return 1
    if x4 <= 0 and x5_pctl >= X5_HAWKISH_THR:
        return 1
    return 0

def decide(short_score, long_score):
    if short_score >= 5 and short_score > long_score:
        return 'SHORT'
    elif long_score >= 4 and long_score > short_score:
        return 'LONG'
    return 'NO TRADE'

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  WALK-FORWARD OVER FULL HISTORY  (signals generated daily, PIT)
# ═══════════════════════════════════════════════════════════════════════════════
records = []
for i, (dt, row) in enumerate(df.iterrows()):
    x1, x2, x4, x5p = row['X1'], row['X2'], row['X4'], row['X5_pctl']
    if any(np.isnan(v) for v in [x1, x2, x4, x5p]):
        continue

    sA, lA = score_A(x1, x2, x4)

    sB = sA + x5_boost(x5p)
    lB = lA

    sC = sB + x4x5_addon(x4, x5p)
    lC = lB

    sig_A = decide(sA, lA)
    sig_B = decide(sB, lB)
    sig_C = decide(sC, lC)

    records.append({
        'Date'   : dt,
        'Year'   : dt.year,
        'EURUSD' : row['EURUSD'],
        'X1'     : x1, 'X2': x2, 'X4': x4, 'X5_pctl': x5p,
        'X5_raw' : row['X5_raw'],
        'sig_A'  : sig_A,
        'sig_B'  : sig_B,
        'sig_C'  : sig_C,
        'R_1D'   : row.get('fwd_1D', np.nan),
        'R_3D'   : row.get('fwd_3D', np.nan),
        'R_5D'   : row.get('fwd_5D', np.nan),
        'R_10D'  : row.get('fwd_10D', np.nan),
    })

res = pd.DataFrame(records)

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  STATISTICS HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def stats_block(df_sub, sig_col, ret_col, direction='SHORT'):
    """
    Full performance stats for one (model, signal, horizon) slice.
    direction: 'SHORT' → win means R<0 | 'LONG' → win means R>0
    """
    sub = df_sub[df_sub[sig_col] == direction][ret_col].dropna()
    n   = len(sub)
    if n == 0:
        return dict(N=0, Mean=np.nan, Median=np.nan, TrimM=np.nan,
                    WinRate=np.nan, MaxStreak=np.nan, MaxDD=np.nan)
    win_flag = (sub < 0) if direction == 'SHORT' else (sub > 0)
    # max consecutive loss streak
    pnl = np.where(win_flag, sub.abs(), -sub.abs())   # +abs for wins, -abs for losses
    streak, max_streak = 0, 0
    for v in pnl:
        if v < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    # cumulative drawdown (trade-level)
    cum = np.cumsum(pnl)
    rolling_max = np.maximum.accumulate(cum)
    dd = cum - rolling_max
    max_dd = dd.min()

    return dict(
        N        = n,
        Mean     = round(sub.mean(), 3),
        Median   = round(sub.median(), 3),
        TrimM    = round(sp_stats.trim_mean(sub, 0.05), 3),
        WinRate  = round(win_flag.mean() * 100, 1),
        MaxLStr  = max_streak,
        MaxDD_pct= round(max_dd, 3),
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  PRINT COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════════
HORIZONS = ['R_1D', 'R_3D', 'R_5D', 'R_10D']

def print_model_comparison(df_slice, period_label):
    print(f"\n{'═'*95}")
    print(f"  PERIOD: {period_label}")
    print(f"{'═'*95}")
    header = f"{'Model':<10} {'Signal':<10} {'Horizon':<8} {'N':>5} {'Mean%':>7} {'Med%':>7} "
    header += f"{'TrimM%':>7} {'WinRate%':>9} {'MaxLStr':>8} {'MaxDD%':>8}"
    print(header)
    print('-'*95)

    for model, sig_col in [('A (v1.0)', 'sig_A'), ('B (+X5)', 'sig_B'), ('C (+X4xX5)', 'sig_C')]:
        for direction in ['SHORT', 'LONG']:
            for hz in HORIZONS:
                s = stats_block(df_slice, sig_col, hz, direction)
                if s['N'] == 0:
                    continue
                print(f"{model:<10} {direction:<10} {hz:<8} {s['N']:>5} "
                      f"{s['Mean']:>7.3f} {s['Median']:>7.3f} {s['TrimM']:>7.3f} "
                      f"{s['WinRate']:>9.1f} {s['MaxLStr']:>8} {s['MaxDD_pct']:>8.3f}")
        print()

# ── Full 2010-2022 in-sample (reference) ─────────────────────────────────────
res_is  = res[(res['Year'] >= 2010) & (res['Year'] <= 2022)]
print_model_comparison(res_is, 'IN-SAMPLE 2010–2022 (reference only)')

# ── OOS 2023-2026 ─────────────────────────────────────────────────────────────
res_oos = res[res['Year'] >= 2023]
print_model_comparison(res_oos, 'TRUE OOS 2023–2026')

# ── OOS year-by-year SHORT only ───────────────────────────────────────────────
print(f"\n{'═'*95}")
print("  OOS YEAR-BY-YEAR  (SHORT signals, 3D horizon)")
print(f"{'═'*95}")
print(f"{'Year':<6} {'Model':<12} {'N':>5} {'Mean%':>7} {'Med%':>7} {'WinRate%':>9}")
print('-'*50)
for yr in sorted(res_oos['Year'].unique()):
    yr_df = res_oos[res_oos['Year'] == yr]
    for model, sig_col in [('A (v1.0)', 'sig_A'), ('B (+X5)', 'sig_B'), ('C (+X4xX5)', 'sig_C')]:
        s = stats_block(yr_df, sig_col, 'R_3D', 'SHORT')
        if s['N'] > 0:
            print(f"{yr:<6} {model:<12} {s['N']:>5} {s['Mean']:>7.3f} "
                  f"{s['Median']:>7.3f} {s['WinRate']:>9.1f}")
    print()

# ── OOS: count of trades per model ───────────────────────────────────────────
print(f"\n{'═'*95}")
print("  OOS SIGNAL DISTRIBUTION (2023–2026)")
print(f"{'═'*95}")
for sig_col, label in [('sig_A','A (v1.0)'), ('sig_B','B (+X5)'), ('sig_C','C (+X4xX5)')]:
    print(f"\nModel {label}:")
    print(res_oos[sig_col].value_counts().to_string())

print("\n\nDONE.")
