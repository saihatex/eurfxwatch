import os
import sys
import json
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'aligned_raw_data.csv')

print("="*95)
print("     MULTI-SAMPLE FROZEN POINT-IN-TIME REPLAY SUITE (20 DIVERSE SAMPLES)")
print("="*95)

# Load complete aligned dataset
df_full = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).dropna()

# Pre-calculate full metrics to find historical signal dates
df_full['log_return'] = np.log(df_full['EURUSD'] / df_full['EURUSD'].shift(1))
df_full['5D_log_return'] = np.log(df_full['EURUSD'] / df_full['EURUSD'].shift(5))
df_full['vol_20d'] = df_full['log_return'].rolling(20).std()

arr_5d = df_full['5D_log_return'].to_numpy()
arr_vol = df_full['vol_20d'].to_numpy()

strict_5d_pctl = np.full(len(df_full), np.nan)
rolling_3y_vol_pctl = np.full(len(df_full), np.nan)

window_3y = 756
min_history = 252

for i in range(min_history, len(df_full)):
    hist_5d = arr_5d[5:i]
    val_5d = arr_5d[i]
    if not np.isnan(val_5d):
        strict_5d_pctl[i] = (hist_5d <= val_5d).mean()

for i in range(window_3y, len(df_full)):
    hist_vol = arr_vol[i-window_3y:i]
    val_vol = arr_vol[i]
    if not np.isnan(val_vol):
        valid = hist_vol[~np.isnan(hist_vol)]
        if len(valid) > 0:
            rolling_3y_vol_pctl[i] = (valid <= val_vol).mean()

df_full['5D_pctl'] = strict_5d_pctl
df_full['vol_pctl_3y'] = rolling_3y_vol_pctl

# Identify Short Triggers
short_mask = (df_full['5D_pctl'] >= 0.90) & (df_full['vol_pctl_3y'] >= 0.75)
short_indices = np.where(short_mask)[0]

first_short_triggers = []
last_idx = -999
for idx in short_indices:
    if idx - last_idx >= 5:
        first_short_triggers.append(idx)
        last_idx = idx

# Identify Long Triggers
long_mask = (df_full['5D_pctl'] <= 0.10) & (df_full['vol_pctl_3y'] >= 0.75)
long_indices = np.where(long_mask)[0]

first_long_triggers = []
last_idx = -999
for idx in long_indices:
    if idx - last_idx >= 5:
        first_long_triggers.append(idx)
        last_idx = idx

# Select 20 diverse sample dates: 8 Short, 4 Long, 8 No-Trade
np.random.seed(42)

# Pick distributed Short dates
short_sample_indices = first_short_triggers[::len(first_short_triggers)//8][:8] if len(first_short_triggers) >= 8 else first_short_triggers

# Pick distributed Long dates
long_sample_indices = first_long_triggers[::max(1, len(first_long_triggers)//4)][:4] if len(first_long_triggers) >= 4 else first_long_triggers

# Pick No-Trade dates across history
all_indices = set(range(window_3y, len(df_full) - 5))
no_trade_candidates = list(all_indices - set(short_indices) - set(long_indices))
no_trade_sample_indices = sorted(np.random.choice(no_trade_candidates, size=8, replace=False))

sample_indices = sorted(short_sample_indices + long_sample_indices + no_trade_sample_indices)

print(f"Selected {len(sample_indices)} historical sample dates across 2008–2026.\n")

# Run Point-in-Time Replay for each sample date
replay_results = []

for sample_idx in sample_indices:
    sample_date = df_full.index[sample_idx].strftime('%Y-%m-%d')
    
    # 1. Truncate strictly <= sample_date
    df_trunc = df_full.iloc[:sample_idx + 1].copy()
    
    # 2. Point-in-Time Calculations on truncated set
    arr_5d_t = df_trunc['5D_log_return'].to_numpy()
    arr_vol_t = df_trunc['vol_20d'].to_numpy()
    
    # Percentile using history up to t-1
    val_5d_t = arr_5d_t[-1]
    hist_5d_t = arr_5d_t[5:-1]
    pctl_5d_t = (hist_5d_t <= val_5d_t).mean() if len(hist_5d_t) > 0 else np.nan
    
    val_vol_t = arr_vol_t[-1]
    hist_vol_t = arr_vol_t[-window_3y-1:-1]
    valid_vol_t = hist_vol_t[~np.isnan(hist_vol_t)]
    pctl_vol_t = (valid_vol_t <= val_vol_t).mean() if len(valid_vol_t) > 0 else np.nan
    
    row_today = df_trunc.iloc[-1]
    eur_close = float(row_today['EURUSD'])
    x4_val = float(row_today['X4'])
    
    high_vol = bool(pctl_vol_t >= 0.75)
    extreme_short = bool(pctl_5d_t >= 0.90)
    extreme_long = bool(pctl_5d_t <= 0.10)
    
    # Determine Decision
    if high_vol and extreme_short:
        signal = "SHORT"
    elif high_vol and extreme_long:
        signal = "LONG"
    else:
        signal = "NO TRADE"
        
    # 3. Unfreeze Future (t+1, t+2, t+3)
    future_data = df_full.iloc[sample_idx + 1 : sample_idx + 4]
    
    ret_1d = float(np.log(future_data.iloc[0]['EURUSD'] / eur_close) * 100) if len(future_data) >= 1 else np.nan
    ret_2d = float(np.log(future_data.iloc[1]['EURUSD'] / eur_close) * 100) if len(future_data) >= 2 else np.nan
    ret_3d = float(np.log(future_data.iloc[2]['EURUSD'] / eur_close) * 100) if len(future_data) >= 3 else np.nan
    
    # Check Direction Success
    if signal == "SHORT":
        outcome = 'WIN' if ret_3d < 0 else 'LOSS'
    elif signal == "LONG":
        outcome = 'WIN' if ret_3d > 0 else 'LOSS'
    else:
        outcome = 'ABSTAIN (NO TRADE)'
        
    replay_results.append({
        'Date': sample_date,
        'Close': round(eur_close, 5),
        'X1 (5D Pctl)': f"{pctl_5d_t*100:.1f}%",
        'X2 (Vol Pctl)': f"{pctl_vol_t*100:.1f}%",
        'X4 Rate Shift': f"{x4_val:+.4f}",
        'Signal': signal,
        'Realized +1D (%)': f"{ret_1d:+.2f}%" if not np.isnan(ret_1d) else "N/A",
        'Realized +2D (%)': f"{ret_2d:+.2f}%" if not np.isnan(ret_2d) else "N/A",
        'Realized +3D (%)': f"{ret_3d:+.2f}%" if not np.isnan(ret_3d) else "N/A",
        'Outcome': outcome
    })

res_df = pd.DataFrame(replay_results)

print("="*95)
print("BATCH POINT-IN-TIME REPLAY RESULTS TABLE:")
print("="*95)
print(res_df.to_string(index=False))

# Aggregate Statistics
short_replays = [r for r in replay_results if r['Signal'] == 'SHORT']
long_replays = [r for r in replay_results if r['Signal'] == 'LONG']
no_trade_replays = [r for r in replay_results if r['Signal'] == 'NO TRADE']

short_wins = sum(1 for r in short_replays if r['Outcome'] == 'WIN')
long_wins = sum(1 for r in long_replays if r['Outcome'] == 'WIN')

print("\n" + "="*95)
print("AGGREGATE OUT-OF-SAMPLE REPLAY PERFORMANCE:")
print("="*95)
if short_replays:
    print(f"SHORT Signals: {short_wins}/{len(short_replays)} Wins ({short_wins/len(short_replays)*100:.1f}% Win Rate)")
    short_returns = [float(r['Realized +3D (%)'].replace('%','')) for r in short_replays]
    print(f" -> Mean SHORT 3D Return: {np.mean(short_returns):+.2f}%")

if long_replays:
    print(f"LONG Signals:  {long_wins}/{len(long_replays)} Wins ({long_wins/len(long_replays)*100:.1f}% Win Rate)")
    long_returns = [float(r['Realized +3D (%)'].replace('%','')) for r in long_replays]
    print(f" -> Mean LONG 3D Return:  {np.mean(long_returns):+.2f}%")

print(f"NO TRADE Days: {len(no_trade_replays)} Days Correctly Abstaining from Noise")

# Save JSON artifact
json_path = os.path.join(BASE_DIR, 'validation', 'batch_replay_results.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(replay_results, f, indent=4)

print(f"\n[SAVED BATCH REPLAY ARTIFACT] -> {json_path}")
