import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'aligned_raw_data.csv')

print("="*90)
print("     RESEARCH STAGE 5 — FACTOR X4 DOSE-RESPONSE & MONOTONICITY TEST")
print("="*90)

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).dropna()

# Log returns & volatility
df['log_return'] = np.log(df['EURUSD'] / df['EURUSD'].shift(1))
df['5D_log_return'] = np.log(df['EURUSD'] / df['EURUSD'].shift(5))
df['vol_20d'] = df['log_return'].rolling(20).std()

arr_5d = df['5D_log_return'].to_numpy()
arr_vol = df['vol_20d'].to_numpy()

strict_5d_pctl = np.full(len(df), np.nan)
rolling_3y_vol_pctl = np.full(len(df), np.nan)

window_3y = 756
min_history = 252

# Strictly up to t-1
for i in range(min_history, len(df)):
    hist_5d = arr_5d[5:i]
    val_5d = arr_5d[i]
    if not np.isnan(val_5d):
        strict_5d_pctl[i] = (hist_5d <= val_5d).mean()

for i in range(window_3y, len(df)):
    hist_vol = arr_vol[i-window_3y:i]
    val_vol = arr_vol[i]
    if not np.isnan(val_vol):
        valid = hist_vol[~np.isnan(hist_vol)]
        if len(valid) > 0:
            rolling_3y_vol_pctl[i] = (valid <= val_vol).mean()

df['5D_pctl'] = strict_5d_pctl
df['vol_pctl_3y'] = rolling_3y_vol_pctl
df['forward_3D'] = np.log(df['EURUSD'].shift(-3) / df['EURUSD'])

# Baseline First Trigger mask (X1, X2 untouched)
target_mask = (df['5D_pctl'] >= 0.90) & (df['vol_pctl_3y'] >= 0.75)
signal_indices = np.where(target_mask)[0]

first_triggers = []
last_idx = -999
for idx in signal_indices:
    if idx - last_idx >= 5:
        first_triggers.append(idx)
        last_idx = idx

df_trig = df.iloc[first_triggers].copy()

# Sort X4 continuous into 4 Quartiles:
# Q1: Highest X4 (Most USD-supportive rate shift / Maximum Divergence)
# Q4: Lowest X4 (Most EUR-supportive rate shift / Maximum Convergence)
df_trig['X4_bin'] = pd.qcut(df_trig['X4'], q=4, labels=['Q4 (EUR-supportive)', 'Q3 (Moderate EUR)', 'Q2 (Moderate USD)', 'Q1 (USD-supportive)'])

# Reorder categories for display Q1 -> Q2 -> Q3 -> Q4
cat_order = ['Q1 (USD-supportive)', 'Q2 (Moderate USD)', 'Q3 (Moderate EUR)', 'Q4 (EUR-supportive)']

res = []
for label in cat_order:
    sub = df_trig[df_trig['X4_bin'] == label]
    fwd = sub['forward_3D'].dropna()
    x4_mean = sub['X4'].mean()
    res.append({
        'Quartile': label,
        'N': len(fwd),
        'Avg X4 (Spread 5D Chg)': f"{x4_mean:+.4f}",
        'Mean 3D (%)': f"{fwd.mean()*100:+.3f}%",
        'Median 3D (%)': f"{fwd.median()*100:+.3f}%",
        'TrimMean 5%': f"{stats.trim_mean(fwd, 0.05)*100:+.3f}%",
        'P(R_3D < 0)': f"{(fwd<0).mean()*100:.1f}%",
        'Std (%)': f"{fwd.std()*100:.3f}%",
        'q05 (%)': f"{np.percentile(fwd, 5)*100:+.3f}%",
        'q95 (%)': f"{np.percentile(fwd, 95)*100:+.3f}%"
    })

print(f"Total aligned dataset points: {len(df)}")
print(f"Base First Triggers analyzed: {len(df_trig)}")
print("\nFACTOR X4 DOSE-RESPONSE MATRIX (Q1 = Max USD-supportive, Q4 = Max EUR-supportive):\n")
res_df = pd.DataFrame(res)
print(res_df.to_string(index=False))

# Check Monotonicity
means = [float(x.replace('%','')) for x in res_df['Mean 3D (%)']]
win_rates = [float(x.replace('%','')) for x in res_df['P(R_3D < 0)']]

print("\n" + "="*90)
print("MONOTONICITY & INDEPENDENT INFORMATION EVALUATION:")
print("="*90)
print(f"Mean 3D Return Progression (Q1 -> Q4): {means}")
print(f"Win Rate P(R_3D < 0) Progression (Q1 -> Q4): {win_rates}")

is_mean_monotonic = (means[0] <= means[1] <= means[2] <= means[3]) or (means[0] >= means[1] >= means[2] >= means[3])
is_win_monotonic = (win_rates[0] >= win_rates[1] >= win_rates[2] >= win_rates[3]) or (win_rates[0] <= win_rates[1] <= win_rates[2] <= win_rates[3])

if is_mean_monotonic or is_win_monotonic:
    print("\nCONCLUSION: Clear monotonic dose-response detected! Factor X4 adds independent predictive information.")
else:
    print("\nCONCLUSION: Non-monotonic response across quartiles. Factor X4 does NOT exhibit a clean linear dose-response.")
