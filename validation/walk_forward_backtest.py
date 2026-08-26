import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'aligned_raw_data.csv')

print("="*95)
print("     MASSIVE BLIND WALK-FORWARD POINT-IN-TIME BACKTEST (2010–2026)")
print("="*95)

# Load aligned dataset
df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).dropna(subset=['EURUSD', 'US2Y', 'DE2Y', 'X4'])

df['log_return'] = np.log(df['EURUSD'] / df['EURUSD'].shift(1))
df['5D_log_return'] = np.log(df['EURUSD'] / df['EURUSD'].shift(5))
df['vol_20d'] = df['log_return'].rolling(20).std()

# Forward returns for 1D, 3D, 5D, 10D
df['fwd_1D'] = np.log(df['EURUSD'].shift(-1) / df['EURUSD'])
df['fwd_3D'] = np.log(df['EURUSD'].shift(-3) / df['EURUSD'])
df['fwd_5D'] = np.log(df['EURUSD'].shift(-5) / df['EURUSD'])
df['fwd_10D'] = np.log(df['EURUSD'].shift(-10) / df['EURUSD'])

# Point-in-time calculation of X1 and X2 (strictly up to t-1)
arr_5d = df['5D_log_return'].to_numpy()
arr_vol = df['vol_20d'].to_numpy()

pctl_5d = np.full(len(df), np.nan)
pctl_vol = np.full(len(df), np.nan)

window_3y = 756
min_history = 252

print("Computing Point-in-Time Percentiles (0% Future Leak)...")
for i in range(min_history, len(df)):
    hist = arr_5d[5:i]
    val = arr_5d[i]
    if not np.isnan(val):
        pctl_5d[i] = (hist <= val).mean()

for i in range(window_3y, len(df)):
    hist = arr_vol[i-window_3y:i]
    val = arr_vol[i]
    if not np.isnan(val):
        valid = hist[~np.isnan(hist)]
        if len(valid) > 0:
            pctl_vol[i] = (valid <= val).mean()

df['X1'] = pctl_5d
df['X2'] = pctl_vol

# Filter start from 2010-01-01 for Walk-Forward Backtest
wf_df = df[df.index >= '2010-01-01'].copy()

results = []

print(f"Executing Walk-Forward Replay over {len(wf_df)} trading days...")

for idx in range(len(wf_df)):
    row = wf_df.iloc[idx]
    dt = wf_df.index[idx]
    dt_str = dt.strftime('%Y-%m-%d')

    x1 = row['X1']
    x2 = row['X2']
    x4 = row['X4']
    eur = row['EURUSD']

    if np.isnan(x1) or np.isnan(x2) or np.isnan(x4):
        continue

    # Layer 2 Scoring
    short_score = 0
    long_score = 0

    if x1 >= 0.90: short_score += 3
    if x2 >= 0.75: short_score += 2
    if x4 > +0.05: short_score += 3
    elif x4 > 0: short_score += 1

    if x1 <= 0.10 and x2 >= 0.75: long_score += 4
    elif 0.50 <= x1 < 0.85: long_score += 2
    if x4 < -0.05: long_score += 3
    elif x4 < 0: long_score += 1

    # Signal Synthesis
    if short_score >= 5 and short_score > long_score:
        signal = "SHORT BIAS"
    elif long_score >= 4 and long_score > short_score:
        signal = "LONG BIAS"
    else:
        signal = "NO TRADE"

    # Forward Realized Returns
    r1d = row['fwd_1D'] * 100 if not np.isnan(row['fwd_1D']) else np.nan
    r3d = row['fwd_3D'] * 100 if not np.isnan(row['fwd_3D']) else np.nan
    r5d = row['fwd_5D'] * 100 if not np.isnan(row['fwd_5D']) else np.nan
    r10d = row['fwd_10D'] * 100 if not np.isnan(row['fwd_10D']) else np.nan

    # Partition Regime / Era
    if dt.year <= 2019:
        era = "Train (2010-2019)"
    elif dt.year <= 2022:
        era = "Validation (2020-2022)"
    else:
        era = "True OOS (2023-2026)"

    results.append({
        'Date': dt_str,
        'Era': era,
        'EURUSD': eur,
        'X1': x1,
        'X2': x2,
        'X4': x4,
        'Signal': signal,
        'R_1D': r1d,
        'R_3D': r3d,
        'R_5D': r5d,
        'R_10D': r10d
    })

res_df = pd.DataFrame(results)

print("\n" + "="*95)
print("SUMMARY PERFORMANCE BY ERA & SIGNAL DIRECTION")
print("="*95)

def analyze_subset(df_sub, label):
    print(f"\n--- {label} ---")
    for sig in ['SHORT BIAS', 'LONG BIAS', 'NO TRADE']:
        sub = df_sub[df_sub['Signal'] == sig]
        n = len(sub)
        if n == 0:
            continue
        
        r1 = sub['R_1D'].dropna()
        r3 = sub['R_3D'].dropna()
        r5 = sub['R_5D'].dropna()
        r10 = sub['R_10D'].dropna()

        if sig == "SHORT BIAS":
            win3 = (r3 < 0).mean() * 100
            win5 = (r5 < 0).mean() * 100
        elif sig == "LONG BIAS":
            win3 = (r3 > 0).mean() * 100
            win5 = (r5 > 0).mean() * 100
        else:
            win3 = (r3 > 0).mean() * 100
            win5 = (r5 > 0).mean() * 100

        print(f"[{sig:<10}] N={n:<4} | 3D Mean: {r3.mean():+.2f}% | 3D Med: {r3.median():+.2f}% | 3D Win Rate: {win3:.1f}% | 5D Mean: {r5.mean():+.2f}% | 5D Win Rate: {win5:.1f}%")

analyze_subset(res_df, "OVERALL WALK-FORWARD BACKTEST (2010–2026)")
analyze_subset(res_df[res_df['Era'] == "Train (2010-2019)"], "TRAIN PERIOD (2010–2019)")
analyze_subset(res_df[res_df['Era'] == "Validation (2020-2022)"], "VALIDATION PERIOD (2020–2022)")
analyze_subset(res_df[res_df['Era'] == "True OOS (2023-2026)"], "TRUE OUT-OF-SAMPLE (2023–2026)")

# Confusion Matrix for Directional Signals
print("\n" + "="*95)
print("CONFUSION MATRIX & EXPECTANCY STATS")
print("="*95)

trig_df = res_df[res_df['Signal'] != 'NO TRADE'].dropna(subset=['R_3D'])
short_df = trig_df[trig_df['Signal'] == 'SHORT BIAS']
long_df = trig_df[trig_df['Signal'] == 'LONG BIAS']

tp_short = (short_df['R_3D'] < 0).sum()
fp_short = (short_df['R_3D'] > 0).sum()

tp_long = (long_df['R_3D'] > 0).sum()
fp_long = (long_df['R_3D'] < 0).sum()

print(f"SHORT SIGNALS: True Positives (Dropped) = {tp_short} | False Positives (Rose) = {fp_short} | Win Rate = {tp_short/len(short_df)*100:.1f}%")
print(f"LONG SIGNALS:  True Positives (Rose)    = {tp_long} | False Positives (Dropped) = {fp_long} | Win Rate = {tp_long/len(long_df)*100:.1f}%")

# Save JSON artifact
res_json_path = os.path.join(BASE_DIR, 'validation', 'walk_forward_backtest_results.json')
res_df.to_json(res_json_path, orient='records', date_format='iso')
print(f"\n[WALK-FORWARD RESULTS PERSISTED] -> {res_json_path}")
