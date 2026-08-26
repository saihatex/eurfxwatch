"""
Evaluate LONG trade from 1.16747 with reference stop 1.16300
"""
import os, sys
import pandas as pd

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from price_target_engine.engine_v2 import TargetProbabilityEngineV2

engine = TargetProbabilityEngineV2()

ENTRY = 1.16747
STOP  = 1.16300
TARGETS = [1.16900, 1.17100, 1.17250, 1.17500]

res = engine.evaluate_targets(
    entry_price=ENTRY,
    targets=TARGETS,
    stop_level=STOP,
    direction="LONG",
    cutoff_date="2026-08-25",
    x1_window=(0.60, 0.90),
    x2_window=(0.00, 0.35)
)

print("="*90)
print("     EUR/USD LONG EVALUATION — ENTRY 1.16747 (REF STOP: 1.16300)")
print("="*90)
print(f"Sample N = {res['n_analogs']} historical analogues")
print(res['targets'].to_string(index=False))
print("="*90)
