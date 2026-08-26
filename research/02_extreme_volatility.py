import os
import sys
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'EURUSD.csv')

print("="*80)
print("     RESEARCH STAGE 2 — EXTREME VOLATILITY GATE")
print("="*80)

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).dropna()
df['log_return'] = np.log(df['EURUSD'] / df['EURUSD'].shift(1))
df['vol_20d'] = df['log_return'].rolling(20).std()

arr_vol = df['vol_20d'].to_numpy()
rolling_3y_vol_pctl = np.full(len(arr_vol), np.nan)

window_3y = 756

for i in range(window_3y, len(df)):
    hist_vol = arr_vol[i-window_3y:i]
    val_vol = arr_vol[i]
    if not np.isnan(val_vol):
        valid = hist_vol[~np.isnan(hist_vol)]
        if len(valid) > 0:
            rolling_3y_vol_pctl[i] = (valid <= val_vol).mean()

df['vol_pctl_3y'] = rolling_3y_vol_pctl
df_high_vol = df[df['vol_pctl_3y'] >= 0.75].dropna()

print(f"Total days analyzed: {len(df.dropna())}")
print(f"Days with 3Y Volatility Percentile >= 75%: {len(df_high_vol)} ({len(df_high_vol)/len(df.dropna())*100:.1f}%)")
print(df_high_vol[['EURUSD', 'vol_20d', 'vol_pctl_3y']].tail())
