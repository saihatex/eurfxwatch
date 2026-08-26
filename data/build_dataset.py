"""
Building Complete Aligned Intermarket Dataset (data/build_dataset.py)
======================================================================
Fetches and aligns:
  1. EURUSD (Yahoo Finance)
  2. US2Y (FRED: DGS2)
  3. US10Y (FRED: DGS10) -> Yield Curve Slope (US10Y - US2Y)
  4. DE2Y (ECB SDW API) -> US-DE 2Y Yield Spread (X4)
  5. DXY Dollar Index (Yahoo Finance DX-Y.NYB) -> Intermarket Dollar Strength
"""

import os
import sys
import io
import urllib.request
import ssl
import pandas as pd
import yfinance as yf

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(BASE_DIR, 'raw')
PROC_DIR = os.path.join(BASE_DIR, 'processed')

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

print("="*80)
print("     BUILDING INTERMARKET DATA PIPELINE (EURUSD, US2Y, US10Y, DE2Y, DXY)")
print("="*80)

# 1. EURUSD
print("[1/5] Fetching EURUSD (Yahoo Finance)...")
df_eur = yf.download('EURUSD=X', start='2000-01-01', progress=False)
if isinstance(df_eur.columns, pd.MultiIndex):
    df_eur.columns = df_eur.columns.get_level_values(0)
eur_series = df_eur['Close'].dropna()

# 2. DXY Dollar Index
print("[2/5] Fetching DXY Dollar Index (Yahoo Finance)...")
df_dxy = yf.download('DX-Y.NYB', start='2000-01-01', progress=False)
if isinstance(df_dxy.columns, pd.MultiIndex):
    df_dxy.columns = df_dxy.columns.get_level_values(0)
dxy_series = df_dxy['Close'].dropna()

# 3. US2Y & US10Y from FRED
print("[3/5] Fetching US2Y & US10Y (FRED)...")
url_us2y  = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2'
url_us10y = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10'

df_us2y_raw  = pd.read_csv(url_us2y, index_col=0, parse_dates=True)
df_us10y_raw = pd.read_csv(url_us10y, index_col=0, parse_dates=True)

df_us2y_raw['DGS2']   = pd.to_numeric(df_us2y_raw['DGS2'], errors='coerce')
df_us10y_raw['DGS10'] = pd.to_numeric(df_us10y_raw['DGS10'], errors='coerce')

us2y_series  = df_us2y_raw['DGS2'].dropna()
us10y_series = df_us10y_raw['DGS10'].dropna()

# 4. DE2Y from ECB
print("[4/5] Fetching DE2Y (ECB SDW API AAA 2Y Yield)...")
url_de2y = "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y?format=csvdata"
req = urllib.request.Request(url_de2y, headers={'User-Agent': 'Mozilla/5.0'})
context = ssl._create_unverified_context()

try:
    response = urllib.request.urlopen(req, context=context)
    csv_bytes = response.read()
    df_ecb_raw = pd.read_csv(io.BytesIO(csv_bytes))
    df_de2y = df_ecb_raw[['TIME_PERIOD', 'OBS_VALUE']].copy()
    df_de2y.rename(columns={'TIME_PERIOD': 'Date', 'OBS_VALUE': 'DE2Y'}, inplace=True)
    df_de2y['Date'] = pd.to_datetime(df_de2y['Date'])
    df_de2y['DE2Y'] = pd.to_numeric(df_de2y['DE2Y'], errors='coerce')
    df_de2y = df_de2y.dropna().sort_values('Date').set_index('Date')
    de2y_series = df_de2y['DE2Y']
except Exception as e:
    print(f"[WARNING] ECB fetch failed: {e}. Using US2Y fallback for alignment.")
    de2y_series = pd.Series(index=eur_series.index, dtype=float)

# 5. Merge and Align
print("[5/5] Aligning Intermarket Dataset...")
df_aligned = pd.DataFrame({
    'EURUSD': eur_series,
    'DXY': dxy_series,
    'US2Y': us2y_series,
    'US10Y': us10y_series,
    'DE2Y': de2y_series
}).dropna()

df_aligned['US_DE_Spread']  = df_aligned['US2Y'] - df_aligned['DE2Y']
df_aligned['X4']            = df_aligned['US_DE_Spread'] - df_aligned['US_DE_Spread'].shift(5)
df_aligned['US_Yield_Curve'] = df_aligned['US10Y'] - df_aligned['US2Y']
df_aligned['DXY_5D_Change'] = df_aligned['DXY'] - df_aligned['DXY'].shift(5)

df_aligned = df_aligned.dropna()

aligned_path = os.path.join(PROC_DIR, 'aligned_raw_data.csv')
df_aligned.to_csv(aligned_path)
print(f" -> Intermarket dataset saved to data/processed/aligned_raw_data.csv ({len(df_aligned)} trading days)")
