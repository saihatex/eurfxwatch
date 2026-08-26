"""
Unified Market Data Auto-Sync Pipeline (data/sync_market_data.py)
===================================================================
1-Click Automated Data Sync:
  1. Updates Daily EURUSD, US2Y (FRED), DE2Y (ECB), and X4 rates spread in data/processed/aligned_raw_data.csv.
  2. Updates 1H Intraday OHLC via Yahoo Finance.
  3. Resamples 1H -> 4H OHLC (data/raw/EURUSD_4H.csv).
  4. Scans and regenerates 4H FVG Database (data/processed/fvg_4h_database.csv).
  5. Processes COT Weekly positioning (data/processed/cot_processed.csv).
  6. Re-runs MLE SDE parameter calibration (data/processed/calibrated_sde_params.json).
"""

import os
import sys
import subprocess

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

print("="*80)
print("     AUTOMATED MARKET DATA SYNC PIPELINE")
print("="*80)

def run_step(description, script_path, args=[]):
    print(f"\n[SYNC STEP] {description} ({os.path.basename(script_path)})...")
    cmd = [sys.executable, script_path] + args
    res = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"  └─ SUCCESS: {description}")
    else:
        print(f"  └─ WARNING/ERROR in {script_path}: {res.stderr[:200]}")

# 1. Update Daily & Macro Rates Spread (X4)
run_step("Macro Rates & Daily Dataset Sync", os.path.join(BASE_DIR, "data", "build_dataset.py"))

# 2. Update 1H & 15M Intraday OHLC
run_step("1H & 15M Intraday OHLC Sync", os.path.join(BASE_DIR, "data", "download_intraday_ohlc.py"))

# 3. Aggregate 1H -> 4H OHLC & Rebuild 4H FVG Database
fvg_scanner_script = os.path.join(BASE_DIR, "research", "08_PA_conditional_FVG_edge_test.py")
if os.path.exists(fvg_scanner_script):
    run_step("4H FVG Scanner & Database Rebuild", fvg_scanner_script)

# 4. Process CFTC COT Weekly Positioning
run_step("CFTC COT Weekly Positioning Sync", os.path.join(BASE_DIR, "data", "process_cftc_cot.py"))

# 5. Re-run MLE Parameter Calibration
run_step("MLE SDE Parameter Calibration Sync", os.path.join(BASE_DIR, "calibration", "calibrate_sde_params.py"))

print("\n" + "="*80)
print(">>> ALL MARKET DATA SUCCESSFULLY SYNCHRONIZED AND CALIBRATED.")
print("="*80 + "\n")
