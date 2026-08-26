import os, sys
import numpy as np
import pandas as pd
from scipy import stats

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE_DIR, 'data', 'processed', 'aligned_raw_data.csv')

print('='*90)
print('  X5 PROXY TEST  –  US2Y 5‑day change as Fed Repricing Signal')
print('='*90)

# ── 1. Load aligned data (EURUSD, US2Y, DE2Y, X4 already there) ──────────────
df = pd.read_csv(PROCESSED, index_col=0, parse_dates=True)
df = df.sort_index()
df = df.dropna(subset=['EURUSD', 'US2Y'])

# ── 2. Build core time‑series features ───────────────────────────────────────
df['log_ret']       = np.log(df['EURUSD'] / df['EURUSD'].shift(1))
df['5D_log_ret']    = np.log(df['EURUSD'] / df['EURUSD'].shift(5))
df['vol_20d']       = df['log_ret'].rolling(20).std()

# X5 proxy: 5‑day change in US 2Y yield  (Fed Repricing signal)
df['X5_change']     = df['US2Y'] - df['US2Y'].shift(5)

# Forward returns (log %)
for h in [1, 3, 5, 10]:
    df[f'fwd_{h}D'] = np.log(df['EURUSD'].shift(-h) / df['EURUSD']) * 100

df = df.dropna(subset=['5D_log_ret', 'vol_20d', 'X5_change'])

# ── 3. Point‑in‑time percentile loops (zero lookahead) ───────────────────────
arr_mom = df['5D_log_ret'].to_numpy()
arr_vol = df['vol_20d'].to_numpy()
arr_x5  = df['X5_change'].to_numpy()
N       = len(df)

X1_pctl = np.full(N, np.nan)
X2_pctl = np.full(N, np.nan)
X5_pctl = np.full(N, np.nan)

win3y   = 756
min_h   = 252

for i in range(min_h, N):
    # X1 — 5‑day momentum percentile (vs all history up to t‑1)
    hist_m = arr_mom[5:i]
    if not np.isnan(arr_mom[i]) and len(hist_m):
        X1_pctl[i] = (hist_m <= arr_mom[i]).mean()

    # X2 — 20D vol vs 3‑year rolling window
    if i >= win3y:
        hist_v = arr_vol[i - win3y : i]
        if not np.isnan(arr_vol[i]):
            valid = hist_v[~np.isnan(hist_v)]
            if len(valid):
                X2_pctl[i] = (valid <= arr_vol[i]).mean()

    # X5 — US2Y 5‑day change percentile (vs all history up to t‑1)
    hist_x = arr_x5[:i]
    if not np.isnan(arr_x5[i]):
        valid_x = hist_x[~np.isnan(hist_x)]
        if len(valid_x):
            X5_pctl[i] = (valid_x <= arr_x5[i]).mean()

df['X1'] = X1_pctl
df['X2'] = X2_pctl
df['X5_pctl'] = X5_pctl

print(f"\nRows with all percentiles ready: {df.dropna(subset=['X1','X2','X5_pctl']).shape[0]}")

# ── 4. SHORT trigger identification (same v1.0 gate) ─────────────────────────
full = df.dropna(subset=['X1', 'X2', 'X5_pctl', 'fwd_3D']).copy()

short_mask = (full['X1'] >= 0.90) & (full['X2'] >= 0.75)
candidate_idxs = np.where(short_mask.values)[0]

first_idxs = []
last = -999
for i in candidate_idxs:
    if i - last >= 5:
        first_idxs.append(i)
        last = i

triggers = full.iloc[first_idxs].copy()
print(f"First‑trigger SHORT episodes found: N = {len(triggers)}")
print(f"Date range: {triggers.index[0].date()} – {triggers.index[-1].date()}\n")

# ── 5. X5 Dose‑Response table ─────────────────────────────────────────────────
# Re‑compute quartile labels based on X5_pctl
triggers = triggers.copy()
triggers['X5_Q'] = pd.qcut(
    triggers['X5_pctl'], q=4,
    labels=['Q1 Dovish (ΔUS2Y↓)', 'Q2 Mild‑Dovish', 'Q3 Mild‑Hawkish', 'Q4 Hawkish (ΔUS2Y↑)'],
    duplicates='drop'
)

print('='*90)
print('X5 PROXY DOSE‑RESPONSE  (First‑Trigger SHORT episodes)')
print('Q1 = weakest ΔUS2Y (Fed cutting / dovish repricing)')
print('Q4 = strongest ΔUS2Y (Fed hiking / hawkish repricing)')
print('='*90)

rows = []
for label in ['Q1 Dovish (ΔUS2Y↓)', 'Q2 Mild‑Dovish', 'Q3 Mild‑Hawkish', 'Q4 Hawkish (ΔUS2Y↑)']:
    sub = triggers[triggers['X5_Q'] == label]
    r3 = sub['fwd_3D'].dropna()
    r5 = sub['fwd_5D'].dropna()
    dx5 = sub['X5_change'].mean()
    rows.append(dict(
        Q=label, N=len(r3),
        avg_dUS2Y=f'{dx5:+.4f}',
        mean_3D=f'{r3.mean():+.3f}%',
        med_3D =f'{r3.median():+.3f}%',
        P_win_3D=f'{(r3<0).mean()*100:.1f}%',
        mean_5D=f'{r5.mean():+.3f}%',
        P_win_5D=f'{(r5<0).mean()*100:.1f}%',
    ))

print(pd.DataFrame(rows).to_string(index=False))

# ── 6. X4 × X5 Interaction ────────────────────────────────────────────────────
print('\n' + '='*90)
print('X4 × X5 INTERACTION  (SHORT triggers, full history)')
print('Hypothesis: X4>0 & X5 Hawkish = strongest SHORT  |  X4>0 & X5 Dovish = trap')
print('='*90)

combos = [
    ('X4>0  & Hawkish Q4', (triggers['X4'] > 0) & (triggers['X5_Q'] == 'Q4 Hawkish (ΔUS2Y↑)')),
    ('X4>0  & Dovish  Q1', (triggers['X4'] > 0) & (triggers['X5_Q'] == 'Q1 Dovish (ΔUS2Y↓)')),
    ('X4≤0  & Hawkish Q4', (triggers['X4'] <= 0) & (triggers['X5_Q'] == 'Q4 Hawkish (ΔUS2Y↑)')),
    ('X4≤0  & Dovish  Q1', (triggers['X4'] <= 0) & (triggers['X5_Q'] == 'Q1 Dovish (ΔUS2Y↓)')),
]

int_rows = []
for name, mask in combos:
    sub = triggers[mask]
    r3 = sub['fwd_3D'].dropna()
    r5 = sub['fwd_5D'].dropna()
    int_rows.append(dict(
        Condition=name, N=len(r3),
        mean_3D=f'{r3.mean():+.3f}%' if len(r3) else 'N/A',
        P_win_3D=f'{(r3<0).mean()*100:.1f}%' if len(r3) else 'N/A',
        mean_5D=f'{r5.mean():+.3f}%' if len(r5) else 'N/A',
        P_win_5D=f'{(r5<0).mean()*100:.1f}%' if len(r5) else 'N/A',
    ))

print(pd.DataFrame(int_rows).to_string(index=False))

# ── 7. OOS‑only (2023‑2026) ───────────────────────────────────────────────────
print('\n' + '='*90)
print('OOS (2023‑2026) BREAKDOWN BY X5 QUARTILE')
print('='*90)

oos = triggers[triggers.index.year >= 2023]
print(f'OOS SHORT triggers (2023‑2026) with X5 proxy: N = {len(oos)}')

oos_rows = []
for label in ['Q1 Dovish (ΔUS2Y↓)', 'Q2 Mild‑Dovish', 'Q3 Mild‑Hawkish', 'Q4 Hawkish (ΔUS2Y↑)']:
    sub = oos[oos['X5_Q'] == label]
    r3 = sub['fwd_3D'].dropna()
    oos_rows.append(dict(
        Q=label, N=len(r3),
        mean_3D=f'{r3.mean():+.3f}%' if len(r3) else 'N/A',
        P_win_3D=f'{(r3<0).mean()*100:.1f}%' if len(r3) else 'N/A',
    ))
print(pd.DataFrame(oos_rows).to_string(index=False))

# ── 8. Monotonicity verdict ───────────────────────────────────────────────────
means  = [float(r['mean_3D'].replace('%','')) for r in rows]
wins   = [float(r['P_win_3D'].replace('%','')) for r in rows]

print('\n' + '='*90)
print('MONOTONICITY VERDICT')
print(f"3D Mean   progression Q1→Q4: {means}")
print(f"3D WinRate progression Q1→Q4: {wins}")

mono_mean = (all(means[i] <= means[i+1] for i in range(3)) or
             all(means[i] >= means[i+1] for i in range(3)))
mono_win  = (all(wins[i] <= wins[i+1] for i in range(3)) or
             all(wins[i] >= wins[i+1] for i in range(3)))

if mono_mean or mono_win:
    print('\n>>> VERDICT: MONOTONIC dose‑response detected.')
    print('    X5 (Fed Repricing proxy) carries independent directional information.')
    print('    Proceed to v1.1 integration test.\n')
else:
    print('\n>>> VERDICT: NON‑MONOTONIC — X5 alone does not show clean dose‑response.')
    print('    Check OOS sub‑table and X4×X5 interaction before integrating.\n')
