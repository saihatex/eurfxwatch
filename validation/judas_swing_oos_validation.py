"""
TASK B: Fix Issue #3 — OOS Walk-Forward Judas Swing Validation
===============================================================
1. Load EURUSD_1H.csv, split IS (70%) / OOS (30%) chronologically.
2. For each OOS day where London (08-13 UTC) swept Asian High (00-08 UTC):
   - Measure forward returns at +1H, +2H, +4H, +8H after 13:00 UTC.
   - Run scipy.stats.ttest_1samp(returns, 0) for each horizon.
3. Print result table.
4. Save to data/processed/judas_swing_oos_results.json
"""

import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_1H   = os.path.join(BASE_DIR, 'data', 'raw', 'EURUSD_1H.csv')
OUT_JSON  = os.path.join(BASE_DIR, 'data', 'processed', 'judas_swing_oos_results.json')

print("=" * 70)
print("  TASK B — OOS JUDAS SWING VALIDATION")
print("=" * 70)

# ─── Load 1H data ────────────────────────────────────────────────────────────
df = pd.read_csv(PATH_1H, index_col=0, parse_dates=True).sort_index()
print(f"\nLoaded {len(df)} 1H bars: {df.index[0]} → {df.index[-1]}")

# ─── IS / OOS split ──────────────────────────────────────────────────────────
split_idx = int(len(df) * 0.70)
df_is  = df.iloc[:split_idx]
df_oos = df.iloc[split_idx:]
print(f"IS  ({split_idx} bars): {df_is.index[0].date()} → {df_is.index[-1].date()}")
print(f"OOS ({len(df_oos)} bars): {df_oos.index[0].date()} → {df_oos.index[-1].date()}")

# ─── Identify sweep days in OOS ──────────────────────────────────────────────
df_oos = df_oos.copy()
df_oos['Date_Only'] = df_oos.index.normalize()  # midnight of each day
df_oos['Hour']      = df_oos.index.hour

horizons = [1, 2, 4, 8]  # hours after 13:00 UTC reference

events = []   # list of dicts with returns for each sweep event

unique_dates = df_oos['Date_Only'].unique()
for dt in unique_dates:
    day_bars = df_oos[df_oos['Date_Only'] == dt]

    asia_bars   = day_bars[day_bars['Hour'].between(0, 7)]
    london_bars = day_bars[day_bars['Hour'].between(8, 12)]

    if len(asia_bars) == 0 or len(london_bars) == 0:
        continue

    asia_high    = asia_bars['High'].max()
    london_high  = london_bars['High'].max()

    swept = london_high > asia_high
    if not swept:
        continue

    # Reference close: last 1H bar closing before / at 13:00 UTC
    ref_bar_ts = dt + pd.Timedelta(hours=12)  # 12:00 bar, closes at 13:00
    if ref_bar_ts not in df_oos.index:
        # try nearest bar ending at 13:00
        candidates = day_bars[day_bars['Hour'] == 12]
        if len(candidates) == 0:
            continue
        ref_bar_ts = candidates.index[-1]

    ref_close = df_oos.loc[ref_bar_ts, 'Close']

    row = {'date': str(dt.date()), 'asia_high': asia_high,
           'london_high': london_high, 'ref_close': ref_close}

    for h in horizons:
        fwd_ts = ref_bar_ts + pd.Timedelta(hours=h)
        if fwd_ts in df_oos.index:
            fwd_close = df_oos.loc[fwd_ts, 'Close']
            # Return from ref_close (bearish = negative return)
            ret = (fwd_close - ref_close) / ref_close
            row[f'ret_{h}h'] = ret
        else:
            row[f'ret_{h}h'] = np.nan

    events.append(row)

ev_df = pd.DataFrame(events)
print(f"\nTotal OOS Judas Swing (London swept Asian High) events: {len(ev_df)}")

# ─── Statistical tests per horizon ───────────────────────────────────────────
print("\n" + "─" * 70)
print(f"  {'Horizon':<10} {'N':>5} {'Mean Ret':>10} {'Std':>10} {'t-stat':>10} {'p-value':>10} {'Win% Bear':>10}")
print("─" * 70)

results = {}
for h in horizons:
    col = f'ret_{h}h'
    rets = ev_df[col].dropna().values
    n = len(rets)
    if n < 5:
        print(f"  +{h}H        {n:>5}  {'—':>10} {'—':>10} {'—':>10} {'—':>10} {'—':>10}")
        results[f'+{h}H'] = {'N': n, 'mean_ret': None, 'std': None,
                               't_stat': None, 'p_value': None, 'win_rate_bear_pct': None}
        continue

    mean_r = np.mean(rets)
    std_r  = np.std(rets, ddof=1)
    t_stat, p_val = stats.ttest_1samp(rets, 0.0)
    win_rate = (rets < 0).mean() * 100  # bearish = negative return

    print(f"  +{h}H        {n:>5}  {mean_r:>10.5f} {std_r:>10.5f} {t_stat:>10.3f} {p_val:>10.4f} {win_rate:>9.1f}%")

    results[f'+{h}H'] = {
        'N': int(n),
        'mean_ret': round(float(mean_r), 6),
        'std': round(float(std_r), 6),
        't_stat': round(float(t_stat), 4),
        'p_value': round(float(p_val), 4),
        'win_rate_bear_pct': round(float(win_rate), 2),
    }

print("─" * 70)

# ─── Verdict ─────────────────────────────────────────────────────────────────
p_values = [v['p_value'] for v in results.values() if v['p_value'] is not None]
significant = any(p < 0.05 for p in p_values)
verdict = "SIGNIFICANT — Judas Swing edge detected (p < 0.05)" if significant \
          else "NO EDGE — None of the horizons reach p < 0.05"

print(f"\n  VERDICT: {verdict}")

# ─── Save results ─────────────────────────────────────────────────────────────
output = {
    'is_period': {'start': str(df_is.index[0].date()), 'end': str(df_is.index[-1].date()),
                  'n_bars': split_idx},
    'oos_period': {'start': str(df_oos.index[0].date()), 'end': str(df_oos.index[-1].date()),
                   'n_bars': len(df_oos)},
    'sweep_events_total': len(ev_df),
    'horizons': results,
    'verdict': verdict,
}
with open(OUT_JSON, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n  Results saved → {OUT_JSON}")
print("\n[TASK B COMPLETE]")
