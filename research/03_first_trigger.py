import os
import sys
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'EURUSD.csv')

print("="*80)
print("     RESEARCH STAGE 3 — FIRST TRIGGER SIGNAL ISOLATION")
print("="*80)

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).dropna()
df['log_return'] = np.log(df['EURUSD'] / df['EURUSD'].shift(1))
df['5D_log_return'] = np.log(df['EURUSD'] / df['EURUSD'].shift(5))
df['vol_20d'] = df['log_return'].rolling(20).std()

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
df['forward_3D'] = np.log(df['EURUSD'].shift(-3) / df['EURUSD'])

target_mask = (df['5D_pctl'] >= 0.90) & (df['vol_pctl_3y'] >= 0.75)
signal_indices = np.where(target_mask)[0]

first_triggers = []
last_idx = -999
for idx in signal_indices:
    if idx - last_idx >= 5:
        first_triggers.append(idx)
        last_idx = idx

df_trig = df.iloc[first_triggers].copy()
fwd = df_trig['forward_3D'].dropna()

print(f"Total Triggers (Raw): {len(signal_indices)}")
print(f"First Triggers Isolated: {len(df_trig)}")
print(f"Mean 3D Forward Return: {fwd.mean()*100:+.3f}%")
print(f"Win Rate (Short < 0): {(fwd < 0).mean()*100:.1f}%")
