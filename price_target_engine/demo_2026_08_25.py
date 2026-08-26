"""
Demo: Price Target Engine applied to the live SHORT trade
=========================================================
Entry:   SHORT @ 1.16837  (25.08.2026, 13:00 Kyiv)
Targets: 1.1660, 1.1640, 1.1620, 1.1600, 1.1550
Stop:    1.1720 (implicit reference)

Regime at entry (from frozen decision engine output):
  X1 = 78.6%  (Bullish momentum, moderate)
  X2 = 19.7%  (Low volatility)
  X4 = +0.034 (Mild USD-supportive)
  FINAL: NO TRADE (v1.0)

We run Target Engine under two conditioning regimes:
  (a) UNCONDITIONAL — full history up to 25.08.2026
  (b) REGIME-MATCHED — analogues with X1 ∈ [0.60, 0.90], X2 ∈ [0.00, 0.35]
      (i.e., moderate momentum + low volatility, like today)
"""

import os, sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from price_target_engine.engine import PriceTargetEngine

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ── Trade parameters ──────────────────────────────────────────────────────────
ENTRY       = 1.16837
CUTOFF_DATE = '2026-08-25'
DIRECTION   = 'SHORT'

TARGETS_DOWN = [1.1660, 1.1640, 1.1620, 1.1600, 1.1550]
STOP_LEVEL   = 1.1720   # reference upside level

# Regime at entry (from frozen 25.08.2026 13:00 snapshot)
X1_AT_ENTRY  = 0.786   # 78.6%
X2_AT_ENTRY  = 0.197   # 19.7%

# Regime window for analogue matching (±15 percentile points)
X1_RANGE  = (0.60, 0.90)   # moderate bullish impulse regime
X2_RANGE  = (0.00, 0.35)   # low volatility regime

print('='*85)
print('  PRICE TARGET ENGINE — LIVE TRADE DEMO')
print('  SHORT @ 1.16837  |  25.08.2026 13:00 Kyiv  |  Broker: Forex.com')
print('='*85)

engine = PriceTargetEngine()

# Print distances to each target
print('\nTarget levels (pip distances from entry):')
for L in TARGETS_DOWN:
    pips = round((ENTRY - L) * 10000, 1)
    pct  = round((ENTRY - L) / ENTRY * 100, 3)
    print(f'  {L:.4f}  →  {pips:+.1f} pips  ({pct:+.3f}%)')
print(f'  {STOP_LEVEL:.4f}  →  {round((ENTRY - STOP_LEVEL)*10000,1):+.1f} pips  (reference stop)')

# ─── (a) UNCONDITIONAL table ──────────────────────────────────────────────────
print('\n' + '─'*85)
print('(A) UNCONDITIONAL TOUCH PROBABILITY TABLE  (full history up to 25.08.2026)')
print('    No regime filter — base rate from all trading days')
print('─'*85)

tbl_a = engine.touch_probability_table(
    entry_price  = ENTRY,
    targets      = TARGETS_DOWN,
    direction    = DIRECTION,
    cutoff_date  = CUTOFF_DATE,
)
print(tbl_a.to_string(index=False))

# ─── (b) REGIME-MATCHED table ─────────────────────────────────────────────────
print('\n' + '─'*85)
print(f'(B) REGIME-MATCHED TOUCH PROBABILITY TABLE  (analogues: X1∈{X1_RANGE}, X2∈{X2_RANGE})')
print( '    Low-volatility + moderate-bullish-impulse regime — today\'s actual regime')
print('─'*85)

tbl_b = engine.touch_probability_table(
    entry_price  = ENTRY,
    targets      = TARGETS_DOWN,
    direction    = DIRECTION,
    cutoff_date  = CUTOFF_DATE,
    x1_range     = X1_RANGE,
    x2_range     = X2_RANGE,
)
print(tbl_b.to_string(index=False))

# ─── UPSIDE reference (stop level) ────────────────────────────────────────────
print('\n' + '─'*85)
print('(C) UPSIDE REFERENCE — P(price reaches STOP level before end of horizon)')
print('    i.e., P(max_{1..H} P_{t+k} ≥ 1.1720 | today\'s regime) — ADVERSE for SHORT')
print('─'*85)

tbl_stop_uncond = engine.touch_probability_table(
    entry_price  = ENTRY,
    targets      = [STOP_LEVEL],
    direction    = 'LONG',
    cutoff_date  = CUTOFF_DATE,
)
tbl_stop_regime = engine.touch_probability_table(
    entry_price  = ENTRY,
    targets      = [STOP_LEVEL],
    direction    = 'LONG',
    cutoff_date  = CUTOFF_DATE,
    x1_range     = X1_RANGE,
    x2_range     = X2_RANGE,
)
print('Unconditional:')
print(tbl_stop_uncond.to_string(index=False))
print('Regime-matched:')
print(tbl_stop_regime.to_string(index=False))

# ─── MAE / MFE (SHORT position, regime-matched) ───────────────────────────────
print('\n' + '─'*85)
print('(D) MAE / MFE DISTRIBUTIONS  (SHORT, regime-matched, 10D window)')
print('    MAE: how far price went AGAINST the short before end of 10D')
print('    MFE: how far price went IN FAVOR  of the short before end of 10D')
print('─'*85)

stats_uncond = engine.mae_mfe_stats('SHORT', CUTOFF_DATE)
stats_regime = engine.mae_mfe_stats('SHORT', CUTOFF_DATE, X1_RANGE, X2_RANGE)

for label, s in [('Unconditional', stats_uncond), ('Regime-matched', stats_regime)]:
    print(f'\n{label}  (N={s["N_analogues"]})')
    print(f"  MAE 10D (adverse move against short): "
          f"median {s['MAE_10D_%']['median']:+.2f}%  "
          f"p75 {s['MAE_10D_%']['p75']:+.2f}%  "
          f"p90 {s['MAE_10D_%']['p90']:+.2f}%")
    print(f"  MFE 10D (favorable move for short):   "
          f"median {s['MFE_10D_%']['median']:+.2f}%  "
          f"p75 {s['MFE_10D_%']['p75']:+.2f}%  "
          f"p90 {s['MFE_10D_%']['p90']:+.2f}%")

# ─── Synthesis ────────────────────────────────────────────────────────────────
print('\n' + '='*85)
print('SYNTHESIS: TRADE EXPECTANCY (regime-matched, 3D horizon)')
print('='*85)

# Fetch regime-matched 3D touch for TP and stop
tp_3d_raw  = tbl_b[tbl_b['Target'] == 1.1640]['P_touch_3D'].values
sp_3d_raw  = tbl_stop_regime[tbl_stop_regime['Target'] == STOP_LEVEL]['P_touch_3D'].values

tp_3d  = float(tp_3d_raw[0]) / 100  if len(tp_3d_raw)  else 0.0
sp_3d  = float(sp_3d_raw[0])  / 100 if len(sp_3d_raw)  else 0.0

r_tp   = (ENTRY - 1.1640) / ENTRY * 100   # reward if TP hit (%)
r_sl   = (STOP_LEVEL - ENTRY) / ENTRY * 100  # risk if stop hit (%)

ev = tp_3d * r_tp - sp_3d * abs(r_sl)

print(f'\n  Entry:          {ENTRY:.5f}')
print(f'  Target (TP):    1.1640  →  reward {r_tp:+.2f}%')
print(f'  Stop (ref):     {STOP_LEVEL:.4f}  →  risk  {r_sl:+.2f}%')
print(f'\n  P(TP hit within 3D, regime-matched):   {tp_3d*100:.1f}%')
print(f'  P(Stop hit within 3D, regime-matched): {sp_3d*100:.1f}%')
print(f'\n  Expected Value (3D, simplified):  {ev:+.4f}%')
print()
print('  NOTE: This is a simplified EV using 3D touch probs only.')
print('  It does not account for path dependency, partial exits, or')
print('  simultaneous TP+SL touches in the same window.')
print('='*85)
