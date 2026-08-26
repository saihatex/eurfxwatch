import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("="*80)
print("     RESEARCH STAGE 4 — CONTINUOUS FACTOR X3 (DXY DIVERGENCE)")
print("="*80)

data = yf.download(['EURUSD=X', 'DX-Y.NYB'], start='2000-01-01', progress=False)
eur = data['Close']['EURUSD=X'].dropna()
dxy = data['Close']['DX-Y.NYB'].dropna()

df = pd.DataFrame({'EUR': eur, 'DXY': dxy}).dropna()

df['log_return'] = np.log(df['EUR'] / df['EUR'].shift(1))
df['5D_log_return'] = np.log(df['EUR'] / df['EUR'].shift(5))
df['dxy_5D_log_return'] = np.log(df['DXY'] / df['DXY'].shift(5))
df['vol_20d'] = df['log_return'].rolling(20).std()

df['X3_divergence'] = df['5D_log_return'] + df['dxy_5D_log_return']

arr_5d = df['5D_log_return'].to_numpy()
arr_vol = df['vol_20d'].to_numpy()

strict_5d_pctl = np.full(len(df), np.nan)
rolling_3y_vol_pctl = np.full(len(df), np.nan)

window_3y = 756
min_history = 252

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
df['forward_3D'] = np.log(df['EUR'].shift(-3) / df['EUR'])

target_mask = (df['5D_pctl'] >= 0.90) & (df['vol_pctl_3y'] >= 0.75)
signal_indices = np.where(target_mask)[0]

first_triggers = []
last_idx = -999
for idx in signal_indices:
    if idx - last_idx >= 5:
        first_triggers.append(idx)
        last_idx = idx

df_trig = df.iloc[first_triggers].copy()
df_trig['X3_quartile'] = pd.qcut(df_trig['X3_divergence'], q=4, labels=['Q1 (Broad USD Drop)', 'Q2 (Moderate Div)', 'Q3 (High Div)', 'Q4 (Extreme EUR Spike)'])

res = []
for label in ['Q1 (Broad USD Drop)', 'Q2 (Moderate Div)', 'Q3 (High Div)', 'Q4 (Extreme EUR Spike)']:
    sub = df_trig[df_trig['X3_quartile'] == label]
    fwd = sub['forward_3D'].dropna()
    div_mean = sub['X3_divergence'].mean() * 100
    res.append({
        'X3 Quartile': label,
        'N': len(fwd),
        'Avg X3 Div (%)': f"{div_mean:+.3f}%",
        'Mean 3D (%)': f"{fwd.mean()*100:+.3f}%",
        'TrimMean 5%': f"{stats.trim_mean(fwd, 0.05)*100:+.3f}%",
        'Median 3D (%)': f"{fwd.median()*100:+.3f}%",
        'P(<0)': f"{(fwd<0).mean()*100:.1f}%",
        'Std (%)': f"{fwd.std()*100:.3f}%"
    })

print(pd.DataFrame(res).to_string(index=False))
