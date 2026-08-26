import os
import sys
import pandas as pd
import io
import urllib.request
import ssl

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')

print("="*80)
print("     DOWNLOADING X5 DATA: FED REPRICING PROXY (FRED)")
print("="*80)
print()
print("Strategy: X5 = Market Expectation of Fed in ~6M - Current Fed Funds Rate")
print("  = DTB6 (6-Month T-Bill, secondary market) - DFF (Effective Fed Funds Rate)")
print("  This is a clean, daily, real-time, point-in-time series from FRED.")
print("  Negative = Market prices cuts ahead (Dovish repricing)")
print("  Positive = Market prices hikes ahead or holds (Hawkish repricing)")
print()

ctx = ssl._create_unverified_context()

def fetch_fred(series_id, col_name):
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
    print(f"[FRED] Fetching {series_id} ({col_name})...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, context=ctx)
    df = pd.read_csv(io.BytesIO(resp.read()), index_col=0, parse_dates=True)
    df.columns = [col_name]
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    df = df.dropna()
    print(f"  -> {len(df)} rows | {df.index[0].date()} to {df.index[-1].date()}")
    return df

# Download both series
df_dtb6 = fetch_fred('DTB6', 'DTB6')   # 6-Month T-Bill (market expectation proxy)
df_dff   = fetch_fred('DFF',  'DFF')   # Effective Fed Funds Rate (where Fed is today)

# Save raw files
df_dtb6.to_csv(os.path.join(RAW_DIR, 'DTB6.csv'))
df_dff.to_csv(os.path.join(RAW_DIR, 'DFF.csv'))

# Build X5 raw series
df_x5 = pd.DataFrame({
    'DTB6': df_dtb6['DTB6'],
    'DFF':  df_dff['DFF']
}).dropna().sort_index()

# X5_level: forward-looking premium over current Fed rate
# Positive = market prices higher rates ahead (Hawkish)
# Negative = market prices cuts ahead (Dovish)
df_x5['X5_level'] = df_x5['DTB6'] - df_x5['DFF']

# X5_change: 5-day change in that premium (the REPRICING signal)
df_x5['X5_change'] = df_x5['X5_level'] - df_x5['X5_level'].shift(5)

df_x5.to_csv(os.path.join(RAW_DIR, 'X5_fed_repricing.csv'))

print()
print("="*80)
print("X5 RAW DATASET SAVED: data/raw/X5_fed_repricing.csv")
print("="*80)
print(f"Date range: {df_x5.index[0].date()} to {df_x5.index[-1].date()}")
print(f"Total rows: {len(df_x5)}")
print()
print("Last 5 rows:")
print(df_x5.dropna().tail())
print()
print("X5_level distribution (DTB6 - DFF):")
print(df_x5['X5_level'].describe().round(4))
print()
print("X5_change distribution (5D repricing):")
print(df_x5['X5_change'].dropna().describe().round(4))
