import os
import sys
import json
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'aligned_raw_data.csv')

TEST_DATE = "2026-08-18"

print("="*75)
print(f"POINT-IN-TIME REPLAY (FROZEN PREDICTION MODE)")
print(f"CUT-OFF DATE: {TEST_DATE}")
print("="*75)

# Step 1 — Cutoff data strictly <= TEST_DATE
df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).dropna()

if TEST_DATE not in df.index.strftime('%Y-%m-%d'):
    print(f"[ERROR] Date {TEST_DATE} not found in dataset.")
    sys.exit(1)

cutoff_mask = df.index <= TEST_DATE
df_cutoff = df.loc[cutoff_mask].copy()

print(f"Data truncated to strictly <= {TEST_DATE}. History length: {len(df_cutoff)} trading days.")

# Step 2 — Compute features on truncated dataset (no future leak)
df_cutoff['log_return'] = np.log(df_cutoff['EURUSD'] / df_cutoff['EURUSD'].shift(1))
df_cutoff['5D_log_return'] = np.log(df_cutoff['EURUSD'] / df_cutoff['EURUSD'].shift(5))
df_cutoff['vol_20d'] = df_cutoff['log_return'].rolling(20).std()

arr_5d = df_cutoff['5D_log_return'].to_numpy()
arr_vol = df_cutoff['vol_20d'].to_numpy()

strict_5d_pctl = np.full(len(df_cutoff), np.nan)
rolling_3y_vol_pctl = np.full(len(df_cutoff), np.nan)

window_3y = 756
min_history = 252

# Strictly up to t-1
for i in range(min_history, len(df_cutoff)):
    hist_5d = arr_5d[5:i]
    val_5d = arr_5d[i]
    if not np.isnan(val_5d):
        strict_5d_pctl[i] = (hist_5d <= val_5d).mean()

for i in range(window_3y, len(df_cutoff)):
    hist_vol = arr_vol[i-window_3y:i]
    val_vol = arr_vol[i]
    if not np.isnan(val_vol):
        valid = hist_vol[~np.isnan(hist_vol)]
        if len(valid) > 0:
            rolling_3y_vol_pctl[i] = (valid <= val_vol).mean()

df_cutoff['5D_pctl'] = strict_5d_pctl
df_cutoff['vol_pctl_3y'] = rolling_3y_vol_pctl

# Identify First Triggers up to TEST_DATE
target_mask = (df_cutoff['5D_pctl'] >= 0.90) & (df_cutoff['vol_pctl_3y'] >= 0.75)
signal_indices = np.where(target_mask)[0]

first_triggers = []
last_idx = -999
for idx in signal_indices:
    if idx - last_idx >= 5:
        first_triggers.append(idx)
        last_idx = idx

is_today_trigger = (len(df_cutoff) - 1) in signal_indices
is_today_first_trigger = (len(df_cutoff) - 1) in first_triggers

# Current State on TEST_DATE
today_row = df_cutoff.iloc[-1]
eur_close = float(today_row['EURUSD'])
ret_5d = float(today_row['5D_log_return'])
pctl_5d = float(today_row['5D_pctl'])
pctl_vol = float(today_row['vol_pctl_3y'])
us2y_rate = float(today_row['US2Y'])
de2y_rate = float(today_row['DE2Y'])
spread_val = float(today_row['US_DE_Spread'])
x4_val = float(today_row['X4'])

high_vol = bool(pctl_vol >= 0.75)
extreme_mom = bool(pctl_5d >= 0.90)

# Determine Signal Decision
if is_today_first_trigger and high_vol and extreme_mom:
    signal_type = "SHORT EUR/USD"
    expected_3d = -0.0143  # Historical average for trigger
    median_3d = -0.0046
    p_neg = 0.80
    p_pos = 0.20
elif is_today_first_trigger and high_vol and pctl_5d <= 0.10:
    signal_type = "LONG EUR/USD"
    expected_3d = +0.0120
    median_3d = +0.0050
    p_neg = 0.25
    p_pos = 0.75
else:
    signal_type = "NO TRADE (Normal Regime)"
    expected_3d = 0.0
    median_3d = 0.0
    p_neg = 0.50
    p_pos = 0.50

# Output Display
print("\n" + "="*75)
print(f"POINT-IN-TIME REPLAY RESULT")
print(f"CUT-OFF: {TEST_DATE}")
print("="*75)

print(f"\nEUR/USD Close:               {eur_close:.5f}")
print(f"5D Return:                   {ret_5d*100:+.3f}%")
print(f"5D Percentile (F_t-1):       {pctl_5d*100:.1f}%")
print(f"Volatility 3Y Percentile:    {pctl_vol*100:.1f}%")
print(f"US 2Y Yield:                 {us2y_rate:.2f}%")
print(f"DE 2Y Yield (ECB AAA):       {de2y_rate:.3f}%")
print(f"Rate Spread (US2Y - DE2Y):   {spread_val:+.4f}")
print(f"X4 Rate Diff Change (5D):    {x4_val:+.4f}")

print("\n" + "-"*75)
print("MODEL STATE & FLAGS")
print("-"*75)
print(f"First Trigger:               {'YES' if is_today_first_trigger else 'NO'}")
print(f"High Volatility (>= 75%):    {'YES' if high_vol else 'NO'}")
print(f"Extreme Momentum (>= 90%):   {'YES' if extreme_mom else 'NO'}")

print("\n" + "="*75)
print(f"MODEL SIGNAL:                >>> {signal_type} <<<")
print("="*75)
print(f"Expected 3D Return (Forward): {expected_3d*100:+.2f}%")
print(f"Median 3D Return:            {median_3d*100:+.2f}%")
print(f"P(Negative Return / Short):  {p_neg*100:.1f}%")
print(f"P(Positive Return / Long):   {p_pos*100:.1f}%")
print("="*75)

# Step 3 — Freeze Prediction into JSON artifact
prediction_payload = {
    "cutoff_date": TEST_DATE,
    "eurusd_close": eur_close,
    "5d_return_pct": round(ret_5d * 100, 3),
    "5d_percentile": round(pctl_5d * 100, 1),
    "vol_3y_percentile": round(pctl_vol * 100, 1),
    "us2y_rate": us2y_rate,
    "de2y_rate": de2y_rate,
    "rate_spread": round(spread_val, 4),
    "x4_spread_5d_change": round(x4_val, 4),
    "flags": {
        "first_trigger": is_today_first_trigger,
        "high_volatility": high_vol,
        "extreme_momentum": extreme_mom
    },
    "model_signal": signal_type,
    "expected_3d_return_pct": round(expected_3d * 100, 2),
    "median_3d_return_pct": round(median_3d * 100, 2),
    "p_negative_pct": round(p_neg * 100, 1),
    "p_positive_pct": round(p_pos * 100, 1)
}

pred_file = os.path.join(BASE_DIR, 'validation', 'prediction_2026-08-18.json')
with open(pred_file, 'w', encoding='utf-8') as f:
    json.dump(prediction_payload, f, indent=4)

print(f"\n[FROZEN PREDICTION SAVED] -> {pred_file}")
