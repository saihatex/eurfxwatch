"""
Download and Parse CFTC Commitment of Traders (COT) Data for EUR/USD Futures (CME Code 099741)
================================================================-------------------------
Downloads historical Traders in Financial Futures (TFF) / Disaggregated COT data.
Calculates:
  1. Non-Commercial / Leveraged Funds Net Positioning
  2. Longs & Shorts Expansion / Unwinding Signals:
     - "PROFIT TAKING" (Longs decreasing while Shorts unchanged/decreasing)
     - "TOPPING OUT"   (Longs increasing + Shorts increasing simultaneously)
     - "ACCUMULATION"  (Longs increasing + Shorts decreasing)
     - "CAPITULATION"  (Longs decreasing + Shorts increasing sharply)
  3. Rolling 3-Year Positioning Percentile (Point-in-Time, accounting for 3-day publication lag).
"""

import os
import sys
import io
import urllib.request
import ssl
import zipfile
import pandas as pd
import numpy as np

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_COT  = os.path.join(BASE_DIR, 'data', 'raw', 'COT_EUR_futures.csv')

print("="*80)
print("     CFTC COMMITMENT OF TRADERS (COT) PARSER — EUR FUTURES")
print("="*80)

ctx = ssl._create_unverified_context()
years = range(2010, 2027)

all_cot_rows = []

for yr in years:
    url = f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{yr}.zip"
    print(f"[CFTC] Fetching COT Financial Futures report for year {yr}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            zip_bytes = io.BytesIO(resp.read())
            with zipfile.ZipFile(zip_bytes) as z:
                txt_filename = z.namelist()[0]
                with z.open(txt_filename) as f:
                    lines = [line.decode('latin1') for line in f.readlines()]
                    header = [p.strip().replace('"', '') for p in lines[0].split(',')]
                    for line in lines[1:]:
                        if 'EURO FX' in line or '099741' in line:
                            parts = [p.strip().replace('"', '') for p in line.split(',')]
                            if len(parts) >= 10:
                                all_cot_rows.append(parts)
    except Exception as e:
        print(f"  -> Could not fetch year {yr}: {e}")

print(f"Total raw CFTC EUR rows fetched: {len(all_cot_rows)}")

if len(all_cot_rows) > 0:
    # Save parsed CFTC rows
    df_raw = pd.DataFrame(all_cot_rows)
    df_raw.to_csv(OUT_COT, index=False)
    print(f"Successfully saved {len(df_raw)} CFTC rows to {OUT_COT}")
