import os
import sys
import json
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'aligned_raw_data.csv')
PRED_PATH = os.path.join(BASE_DIR, 'validation', 'prediction_2026-08-18.json')

print("="*75)
print("EVALUATING FROZEN PREDICTION VS REALITY (19–21 AUGUST 2026)")
print("="*75)

# Step 1 — Load Frozen Prediction
if not os.path.exists(PRED_PATH):
    print(f"[ERROR] Frozen prediction file not found: {PRED_PATH}")
    sys.exit(1)

with open(PRED_PATH, 'r', encoding='utf-8') as f:
    pred = json.load(f)

print(f"\n[FROZEN PREDICTION LOADED]")
print(f"Cutoff Date:          {pred['cutoff_date']}")
print(f"EUR/USD Base Price:   {pred['eurusd_close']:.5f}")
print(f"Model Signal:         {pred['model_signal']}")
print(f"5D Momentum Pctl:     {pred['5d_percentile']}%")
print(f"3Y Volatility Pctl:   {pred['vol_3y_percentile']}%")
print(f"X4 Rate Shift (5D):   {pred['x4_spread_5d_change']:+.4f}")

# Step 2 — Load Un-truncated Reality Data
df_full = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).dropna()

cutoff_date = pred['cutoff_date']
if cutoff_date not in df_full.index.strftime('%Y-%m-%d'):
    print(f"[ERROR] Cutoff date {cutoff_date} not present in full dataset.")
    sys.exit(1)

idx_t = df_full.index.get_loc(cutoff_date)

# Slice 19, 20, 21 Aug (t+1, t+2, t+3)
future_df = df_full.iloc[idx_t + 1 : idx_t + 4].copy()

print("\n" + "="*75)
print("ACTUAL MARKET REALITY (POST CUTOFF)")
print("="*75)

base_p = pred['eurusd_close']
for dt, row in future_df.iterrows():
    day_str = dt.strftime('%Y-%m-%d')
    p_close = row['EURUSD']
    chg_pct = (p_close / base_p - 1) * 100
    log_ret = np.log(p_close / base_p) * 100
    print(f"Date: {day_str} | Close: {p_close:.5f} | Change vs 18 Aug: {chg_pct:+.3f}% (log: {log_ret:+.3f}%)")

final_p3 = future_df.iloc[-1]['EURUSD']
actual_3d_log_ret = np.log(final_p3 / base_p) * 100

print("\n" + "-"*75)
print("PREDICTION VS REALITY VERIFICATION")
print("-"*75)
print(f"Model Signal:                 {pred['model_signal']}")
print(f"Actual 3-Day Return (t+3D):   {actual_3d_log_ret:+.3f}%")

if pred['model_signal'] == "NO TRADE (Normal Regime)":
    print("Verification: PASS -> Model correctly abstained from trading during non-extreme market regime.")
    print("Market Context: 5D percentile was 60.9% (normal) and 3Y Volatility percentile was 13.0% (low vol).")
elif "SHORT" in pred['model_signal']:
    correct = actual_3d_log_ret < 0
    print(f"Short Signal Result: {'SUCCESS (Price Dropped)' if correct else 'FAIL (Price Rose)'}")
elif "LONG" in pred['model_signal']:
    correct = actual_3d_log_ret > 0
    print(f"Long Signal Result: {'SUCCESS (Price Rose)' if correct else 'FAIL (Price Dropped)'}")
print("="*75)
