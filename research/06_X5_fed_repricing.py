import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'processed', 'aligned_raw_data.csv')
X5_PATH    = os.path.join(BASE_DIR, 'data', 'raw', 'X5_fed_repricing.csv')

print("="*95)
print("     X5 STANDALONE DOSE-RESPONSE TEST — FED REPRICING vs EUR/USD FORWARD RETURNS")
print("="*95)

# ── 1. Load aligned data + X5 ────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).dropna(subset=['EURUSD','US2Y','DE2Y','X4'])
x5 = pd.read_csv(X5_PATH,  index_col=0, parse_dates=True)[['X5_level','X5_change']].dropna()

df = df.join(x5, how='inner')

# ── 2. Compute X1 / X2 strictly point-in-time (no future leak) ──────────────
df['log_return']    = np.log(df['EURUSD'] / df['EURUSD'].shift(1))
df['5D_log_return'] = np.log(df['EURUSD'] / df['EURUSD'].shift(5))
df['vol_20d']       = df['log_return'].rolling(20).std()

arr_5d  = df['5D_log_return'].to_numpy()
arr_vol = df['vol_20d'].to_numpy()

pctl_5d  = np.full(len(df), np.nan)
pctl_vol = np.full(len(df), np.nan)

window_3y   = 756
min_history = 252

for i in range(min_history, len(df)):
    hist = arr_5d[5:i];  val = arr_5d[i]
    if not np.isnan(val):
        pctl_5d[i] = (hist <= val).mean()

for i in range(window_3y, len(df)):
    hist = arr_vol[i-window_3y:i]; val = arr_vol[i]
    if not np.isnan(val):
        valid = hist[~np.isnan(hist)]
        if len(valid) > 0:
            pctl_vol[i] = (valid <= val).mean()

df['X1'] = pctl_5d
df['X2'] = pctl_vol

# Forward returns
df['fwd_1D']  = np.log(df['EURUSD'].shift(-1)  / df['EURUSD']) * 100
df['fwd_3D']  = np.log(df['EURUSD'].shift(-3)  / df['EURUSD']) * 100
df['fwd_5D']  = np.log(df['EURUSD'].shift(-5)  / df['EURUSD']) * 100
df['fwd_10D'] = np.log(df['EURUSD'].shift(-10) / df['EURUSD']) * 100

df.dropna(subset=['X1','X2','X5_change'], inplace=True)

# ── 3. Compute X5 point-in-time percentile (rolling, strictly history up to t-1) ──
arr_x5 = df['X5_change'].to_numpy()
pctl_x5 = np.full(len(df), np.nan)

for i in range(min_history, len(df)):
    hist = arr_x5[:i]; val = arr_x5[i]
    if not np.isnan(val):
        pctl_x5[i] = (hist <= val).mean()

df['X5_pctl'] = pctl_x5
df.dropna(subset=['X5_pctl'], inplace=True)

# ── 4. Identify First Trigger Episodes (same gate as v1.0) ──────────────────
short_mask = (df['X1'] >= 0.90) & (df['X2'] >= 0.75)
signal_idxs = np.where(short_mask)[0]

first_triggers = []
last_idx = -999
for idx in signal_idxs:
    if idx - last_idx >= 5:
        first_triggers.append(idx)
        last_idx = idx

df_trig = df.iloc[first_triggers].copy()
df_trig = df_trig.dropna(subset=['fwd_3D','X5_pctl'])

print(f"\nFirst Trigger Episodes with X5 coverage: N = {len(df_trig)}")
print(f"Date range: {df_trig.index[0].date()} – {df_trig.index[-1].date()}")

# ── 5. X5 Standalone Dose-Response (4 Quartiles of X5_change) ───────────────
df_trig['X5_Q'] = pd.qcut(
    df_trig['X5_pctl'], q=4,
    labels=['Q1 (Dovish / Cut Pricing)', 'Q2 (Mild Dovish)', 'Q3 (Mild Hawkish)', 'Q4 (Hawkish / Hike Pricing)']
)

print("\n" + "="*95)
print("X5 STANDALONE DOSE-RESPONSE MATRIX (SHORT Triggers only, N first-trigger episodes)")
print("Q1 = Most Dovish Fed Repricing (cuts expected) | Q4 = Most Hawkish (hikes/holds)")
print("="*95)

rows = []
for label in ['Q1 (Dovish / Cut Pricing)', 'Q2 (Mild Dovish)', 'Q3 (Mild Hawkish)', 'Q4 (Hawkish / Hike Pricing)']:
    sub = df_trig[df_trig['X5_Q'] == label]
    r1  = sub['fwd_1D'].dropna()
    r3  = sub['fwd_3D'].dropna()
    r5  = sub['fwd_5D'].dropna()
    x5m = sub['X5_change'].mean()
    rows.append({
        'X5 Quartile':       label,
        'N':                 len(r3),
        'Avg X5 Chg':        f"{x5m:+.4f}",
        'Mean 1D (%)':       f"{r1.mean()*1:+.3f}%" if len(r1) else "N/A",
        'Mean 3D (%)':       f"{r3.mean():+.3f}%",
        'Median 3D (%)':     f"{r3.median():+.3f}%",
        'TrimMean 5% 3D':    f"{stats.trim_mean(r3,0.05):+.3f}%",
        'P(R3D<0)':          f"{(r3<0).mean()*100:.1f}%",
        'Mean 5D (%)':       f"{r5.mean():+.3f}%",
        'P(R5D<0)':          f"{(r5<0).mean()*100:.1f}%",
        'q05 3D (%)':        f"{np.percentile(r3,5):+.3f}%",
        'q95 3D (%)':        f"{np.percentile(r3,95):+.3f}%",
    })

print(pd.DataFrame(rows).to_string(index=False))

# ── 6. X4 × X5 Interaction Test ─────────────────────────────────────────────
print("\n" + "="*95)
print("X4 × X5 INTERACTION TEST (SHORT Triggers)")
print("Hypothesis: X4>0 & X5>0 (Hawkish) = strongest SHORT; X4>0 & X5<0 (Dovish) = trap")
print("="*95)

combos = [
    ("X4 > 0  &  X5 Hawkish (Q4)",  (df_trig['X4'] >  0) & (df_trig['X5_Q'] == 'Q4 (Hawkish / Hike Pricing)')),
    ("X4 > 0  &  X5 Dovish  (Q1)",  (df_trig['X4'] >  0) & (df_trig['X5_Q'] == 'Q1 (Dovish / Cut Pricing)')),
    ("X4 <= 0 &  X5 Hawkish (Q4)",  (df_trig['X4'] <= 0) & (df_trig['X5_Q'] == 'Q4 (Hawkish / Hike Pricing)')),
    ("X4 <= 0 &  X5 Dovish  (Q1)",  (df_trig['X4'] <= 0) & (df_trig['X5_Q'] == 'Q1 (Dovish / Cut Pricing)')),
]

int_rows = []
for label, mask in combos:
    sub = df_trig[mask].dropna(subset=['fwd_3D'])
    r3 = sub['fwd_3D']
    r5 = sub['fwd_5D'].dropna()
    int_rows.append({
        'Condition':         label,
        'N':                 len(r3),
        'Mean 3D (%)':       f"{r3.mean():+.3f}%" if len(r3) else "N/A",
        'Median 3D (%)':     f"{r3.median():+.3f}%" if len(r3) else "N/A",
        'P(R3D<0)':          f"{(r3<0).mean()*100:.1f}%" if len(r3) else "N/A",
        'Mean 5D (%)':       f"{r5.mean():+.3f}%" if len(r5) else "N/A",
        'P(R5D<0)':          f"{(r5<0).mean()*100:.1f}%" if len(r5) else "N/A",
    })

print(pd.DataFrame(int_rows).to_string(index=False))

# ── 7. OOS-only breakdown (2023-2026) with X5 ───────────────────────────────
print("\n" + "="*95)
print("OOS (2023-2026) BREAKDOWN: Do SHORT triggers look different by X5 regime?")
print("="*95)

oos_trig = df_trig[df_trig.index.year >= 2023].dropna(subset=['fwd_3D'])
print(f"OOS SHORT triggers with X5 data: N = {len(oos_trig)}")

oos_rows = []
for label in ['Q1 (Dovish / Cut Pricing)', 'Q2 (Mild Dovish)', 'Q3 (Mild Hawkish)', 'Q4 (Hawkish / Hike Pricing)']:
    sub = oos_trig[oos_trig['X5_Q'] == label]
    r3  = sub['fwd_3D'].dropna()
    oos_rows.append({
        'X5 Quartile':   label,
        'N':             len(r3),
        'Mean 3D (%)':   f"{r3.mean():+.3f}%" if len(r3) else "N/A",
        'Median 3D (%)': f"{r3.median():+.3f}%" if len(r3) else "N/A",
        'P(R3D<0)':      f"{(r3<0).mean()*100:.1f}%" if len(r3) else "N/A",
    })

print(pd.DataFrame(oos_rows).to_string(index=False))

# ── 8. Monotonicity check ────────────────────────────────────────────────────
means = [float(r['Mean 3D (%)'].replace('%','')) for r in rows]
wins  = [float(r['P(R3D<0)'].replace('%','')) for r in rows]

print("\n" + "="*95)
print("MONOTONICITY VERDICT")
print("="*95)
print(f"Mean 3D progression (Q1 Dovish → Q4 Hawkish): {means}")
print(f"Win Rate progression (Q1 → Q4):                {wins}")

is_mean_mono = all(means[i] <= means[i+1] for i in range(len(means)-1)) or \
               all(means[i] >= means[i+1] for i in range(len(means)-1))
is_win_mono  = all(wins[i] >= wins[i+1] for i in range(len(wins)-1)) or \
               all(wins[i] <= wins[i+1] for i in range(len(wins)-1))

if is_mean_mono or is_win_mono:
    print("\nVERDICT: MONOTONIC dose-response detected — X5 carries independent predictive information.")
else:
    print("\nVERDICT: NON-MONOTONIC — X5 does not show clean dose-response in this specification.")
    print("         Check OOS sub-table and X4×X5 interaction for more nuanced signal.")
