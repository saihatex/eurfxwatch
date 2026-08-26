import os
import sys
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'EURUSD.csv')

print("="*80)
print("     RESEARCH STAGE 1 — BASE DISTRIBUTION STATISTICS")
print("="*80)

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).dropna()
df['log_return'] = np.log(df['EURUSD'] / df['EURUSD'].shift(1))
df['5D_log_return'] = np.log(df['EURUSD'] / df['EURUSD'].shift(5))

df = df.dropna()

print(f"Data range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
print(f"Total observations: {len(df)}")
print("\nDaily Return Stats:")
print(df['log_return'].describe())
print(f"Skewness: {df['log_return'].skew():.4f}")
print(f"Kurtosis: {df['log_return'].kurtosis():.4f}")

print("\n5D Log Return Stats:")
print(df['5D_log_return'].describe())
print(f"Skewness: {df['5D_log_return'].skew():.4f}")
print(f"Kurtosis: {df['5D_log_return'].kurtosis():.4f}")
