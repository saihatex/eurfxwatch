import os
import sys
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'aligned_raw_data.csv')

print("="*80)
print("     STAGE 6 — POINT-IN-TIME REPLAY & AUDIT FRAMEWORK")
print("="*80)

# Target date for point-in-time replay audit
TARGET_DATE = "2026-08-18"

print(f"Executing Point-in-Time Audit for Target Date: {TARGET_DATE}\n")

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).dropna()

if TARGET_DATE not in df.index.strftime('%Y-%m-%d'):
    print(f"[ERROR] Target date {TARGET_DATE} not present in dataset.")
    sys.exit(1)

# Truncate dataset strictly up to TARGET_DATE (no future leak)
cutoff_idx = df.index.get_loc(TARGET_DATE)
df_historical = df.iloc[:cutoff_idx + 1].copy()

# Audit historical prices
row_t = df_historical.loc[TARGET_DATE]
p_t = row_t['EURUSD']
p_t5 = df_historical.iloc[-6]['EURUSD']
ret_5d = np.log(p_t / p_t5)

# Calculate 5D percentile using historical window up to t-1
arr_5d = np.log(df_historical['EURUSD'] / df_historical['EURUSD'].shift(5)).to_numpy()
hist_5d = arr_5d[5:-1] # strictly up to t-1
val_5d = arr_5d[-1]
strict_5d_pctl = (hist_5d <= val_5d).mean()

# Calculate 20D vol 3Y percentile using window up to t-1
arr_ret = np.log(df_historical['EURUSD'] / df_historical['EURUSD'].shift(1)).to_numpy()
vol_series = pd.Series(arr_ret).rolling(20).std().to_numpy()

window_3y = 756
hist_vol = vol_series[-window_3y-1:-1]
val_vol = vol_series[-1]
strict_vol_pctl = (hist_vol <= val_vol).mean()

print(f"--- POINT-IN-TIME AUDIT SUMMARY FOR {TARGET_DATE} ---")
print(f"1. EURUSD Closing Price (P_t):           {p_t:.5f}")
print(f"2. EURUSD 5D Ago (P_t-5):                 {p_t5:.5f}")
print(f"3. Calculated 5D Return:                  {ret_5d*100:+.3f}%")
print(f"4. 5D Momentum Percentile (F_t-1):        {strict_5d_pctl*100:.1f}%")
print(f"5. 20D Rolling Volatility:                {val_vol*100:.3f}%")
print(f"6. 3Y Volatility Percentile (F_t-1):      {strict_vol_pctl*100:.1f}%")
print(f"7. US2Y Rate:                             {row_t['US2Y']:.2f}%")
print(f"8. DE2Y Rate (ECB AAA 2Y):                {row_t['DE2Y']:.3f}%")
print(f"9. US-DE Rate Spread:                     {row_t['US_DE_Spread']:+.4f}")
print(f"10. Factor X4 (Spread 5D Chg):            {row_t['X4']:+.4f}")

is_signal = (strict_5d_pctl >= 0.90) and (strict_vol_pctl >= 0.75)
print("-" * 60)
print(f"Point-in-Time Signal Result: {'>>> ACTIVE SHORT SIGNAL <<<' if is_signal else 'NO SIGNAL (Normal Regime)'}")
print("-" * 60)
