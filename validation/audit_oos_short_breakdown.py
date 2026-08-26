import os
import sys
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'aligned_raw_data.csv')

print("="*95)
print("     STEP 3A: DECONSTRUCTING OOS SHORT FAILURE MODE (2023–2026, N=59)")
print("="*95)

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).dropna(subset=['EURUSD', 'US2Y', 'DE2Y', 'X4'])

df['log_return'] = np.log(df['EURUSD'] / df['EURUSD'].shift(1))
df['5D_log_return'] = np.log(df['EURUSD'] / df['EURUSD'].shift(5))
df['vol_20d'] = df['log_return'].rolling(20).std()

df['fwd_1D'] = np.log(df['EURUSD'].shift(-1) / df['EURUSD']) * 100
df['fwd_3D'] = np.log(df['EURUSD'].shift(-3) / df['EURUSD']) * 100
df['fwd_5D'] = np.log(df['EURUSD'].shift(-5) / df['EURUSD']) * 100

arr_5d = df['5D_log_return'].to_numpy()
arr_vol = df['vol_20d'].to_numpy()

pctl_5d = np.full(len(df), np.nan)
pctl_vol = np.full(len(df), np.nan)

window_3y = 756
min_history = 252

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

# Filter 2023–2026 OOS period
oos_df = df[df.index >= '2023-01-01'].copy()

short_signals = []

for idx in range(len(oos_df)):
    row = oos_df.iloc[idx]
    x1, x2, x4 = row['X1'], row['X2'], row['X4']
    
    if np.isnan(x1) or np.isnan(x2) or np.isnan(x4):
        continue

    short_score = 0
    if x1 >= 0.90: short_score += 3
    if x2 >= 0.75: short_score += 2
    if x4 > +0.05: short_score += 3
    elif x4 > 0: short_score += 1

    long_score = 0
    if x1 <= 0.10 and x2 >= 0.75: long_score += 4
    elif 0.50 <= x1 < 0.85: long_score += 2
    if x4 < -0.05: long_score += 3
    elif x4 < 0: long_score += 1

    if short_score >= 5 and short_score > long_score:
        dt_str = oos_df.index[idx].strftime('%Y-%m-%d')
        year = oos_df.index[idx].year
        short_signals.append({
            'Date': dt_str,
            'Year': year,
            'EURUSD': row['EURUSD'],
            'X1': x1,
            'X2': x2,
            'X4': x4,
            'R_3D': row['fwd_3D'],
            'R_5D': row['fwd_5D']
        })

s_df = pd.DataFrame(short_signals)
print(f"\nTotal OOS SHORT Signals Identified (2023–2026): N = {len(s_df)}\n")

def print_audit_table(sub_df, title):
    print(f"\n--- {title} ---")
    n = len(sub_df)
    if n == 0:
        print("N = 0")
        return
    r3 = sub_df['R_3D'].dropna()
    r5 = sub_df['R_5D'].dropna()
    win3 = (r3 < 0).mean() * 100
    win5 = (r5 < 0).mean() * 100
    print(f"N = {n:<3} | 3D Mean: {r3.mean():+.2f}% | 3D Med: {r3.median():+.2f}% | 3D Win Rate: {win3:.1f}% | 5D Mean: {r5.mean():+.2f}% | 5D Win Rate: {win5:.1f}%")

# 1. Year-by-Year Breakdown
print_audit_table(s_df[s_df['Year'] == 2023], "2023 Breakdown")
print_audit_table(s_df[s_df['Year'] == 2024], "2024 Breakdown")
print_audit_table(s_df[s_df['Year'] == 2025], "2025 Breakdown")
print_audit_table(s_df[s_df['Year'] == 2026], "2026 Breakdown")

# 2. X4 Rates Breakdown (USD-supportive vs EUR-supportive)
print_audit_table(s_df[s_df['X4'] > 0], "Rates Shift USD-Supportive (X4 > 0)")
print_audit_table(s_df[s_df['X4'] <= 0], "Rates Shift EUR-Supportive (X4 <= 0)")
print_audit_table(s_df[s_df['X4'] > +0.05], "Rates Shift Strong USD-Supportive (X4 > +0.05)")

# 3. X1 Momentum Extreme Breakdown
print_audit_table(s_df[s_df['X1'] >= 0.95], "Extreme Momentum (X1 >= 95%)")
print_audit_table(s_df[(s_df['X1'] >= 0.90) & (s_df['X1'] < 0.95)], "Moderate Momentum (90% <= X1 < 95%)")

# 4. X2 Volatility Breakdown
print_audit_table(s_df[s_df['X2'] >= 0.90], "Crisis Volatility (X2 >= 90%)")
print_audit_table(s_df[(s_df['X2'] >= 0.75) & (s_df['X2'] < 0.90)], "High Volatility (75% <= X2 < 90%)")
