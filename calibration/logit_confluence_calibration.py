"""
TASK C: Fix Issue #5 — Logit Confluence Calibration
=====================================================
Features (all point-in-time):
  X1 judas_swing:   London swept Asian High that day
  X2 cot_topping:   COT signal == 'TOPPING OUT'
  X3 bear_fvg_below: any active 4H Bearish FVG below day's closing price
  X4 vol_low:       21D realized vol percentile < 40%

Target Y: next-day close < today close (bearish = 1)

IS/OOS 70/30 chronological split.
Fits statsmodels Logit on IS.
OOS: Brier score, AUC-ROC vs naive P=0.5.
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_1H   = os.path.join(BASE_DIR, 'data', 'raw', 'EURUSD_1H.csv')
COT_PATH  = os.path.join(BASE_DIR, 'data', 'processed', 'cot_processed.csv')
FVG_PATH  = os.path.join(BASE_DIR, 'data', 'processed', 'fvg_4h_database.csv')
OUT_JSON  = os.path.join(BASE_DIR, 'data', 'processed', 'logit_confluence_results.json')

print("=" * 70)
print("  TASK C — LOGIT CONFLUENCE CALIBRATION")
print("=" * 70)

# ─── 1. Build daily bars from 1H ─────────────────────────────────────────────
df_1h = pd.read_csv(PATH_1H, index_col=0, parse_dates=True).sort_index()
print(f"\nLoaded {len(df_1h)} 1H bars: {df_1h.index[0]} → {df_1h.index[-1]}")

# Daily OHLC
daily = df_1h.resample('D').agg({
    'Open':  'first',
    'High':  'max',
    'Low':   'min',
    'Close': 'last',
}).dropna(subset=['Close'])

# ─── 2. Feature X1: Judas Swing — London swept Asian High ───────────────────
print("\n[Building X1] Judas Swing feature...")
df_1h['Date_Only'] = df_1h.index.normalize()
df_1h['Hour']      = df_1h.index.hour

judas_records = {}
for dt, group in df_1h.groupby('Date_Only'):
    asia   = group[group['Hour'].between(0, 7)]
    london = group[group['Hour'].between(8, 12)]
    if len(asia) == 0 or len(london) == 0:
        judas_records[dt] = 0
        continue
    asia_high   = asia['High'].max()
    london_high = london['High'].max()
    judas_records[dt] = int(london_high > asia_high)

judas_s = pd.Series(judas_records, name='judas_swing')
judas_s.index = pd.to_datetime(judas_s.index)

# ─── 3. Feature X2: COT Topping ──────────────────────────────────────────────
print("[Building X2] COT Topping feature...")
cot = pd.read_csv(COT_PATH, parse_dates=['Publication_Date'])
cot = cot.sort_values('Publication_Date').set_index('Publication_Date')
cot['cot_topping'] = (cot['COT_Signal'] == 'TOPPING OUT').astype(int)

# Forward-fill COT signal across daily index (published weekly)
cot_daily = cot['cot_topping'].reindex(daily.index, method='ffill')

# ─── 4. Feature X3: Bear FVG below day's closing price ───────────────────────
print("[Building X3] Bear FVG below close feature...")
fvg = pd.read_csv(FVG_PATH, parse_dates=['Datetime'])
bear_fvg = fvg[fvg['FVG_Type'] == 'BEARISH'].copy()
bear_fvg['Date'] = bear_fvg['Datetime'].dt.normalize()

# For each daily bar, check if any BEARISH FVG formed BEFORE that day has
# FVG_Hi <= today's close (i.e., the FVG zone is below or at the current price)
bear_fvg_feature = {}
for dt in daily.index:
    close_price = daily.loc[dt, 'Close']
    # Only FVGs formed before today (point-in-time)
    prior_bear = bear_fvg[bear_fvg['Date'] < dt]
    has_bear_fvg_below = int(len(prior_bear[prior_bear['FVG_Hi'] <= close_price]) > 0)
    bear_fvg_feature[dt] = has_bear_fvg_below

bear_fvg_s = pd.Series(bear_fvg_feature, name='bear_fvg_below')

# ─── 5. Feature X4: Vol Low — 21D realized vol percentile < 40% ─────────────
print("[Building X4] Volatility percentile feature...")
daily_ret = daily['Close'].pct_change()
vol_21d   = daily_ret.rolling(21).std() * np.sqrt(252)  # annualized

vol_pctl = pd.Series(index=daily.index, dtype=float)
for i in range(63, len(daily)):  # need at least 63 days for pctl
    hist = vol_21d.iloc[max(0, i-252):i].dropna()
    if len(hist) == 0:
        continue
    current_vol = vol_21d.iloc[i]
    if pd.isna(current_vol):
        continue
    pctl = (hist < current_vol).mean()
    vol_pctl.iloc[i] = pctl

vol_low_s = (vol_pctl < 0.40).astype(int).rename('vol_low')

# ─── 6. Target Y: next-day bearish ───────────────────────────────────────────
daily['Y'] = (daily['Close'].shift(-1) < daily['Close']).astype(int)

# ─── 7. Assemble feature matrix ──────────────────────────────────────────────
print("\n[Assembling feature matrix...]")
feat = pd.DataFrame({
    'judas_swing':   judas_s,
    'cot_topping':   cot_daily,
    'bear_fvg_below': bear_fvg_s,
    'vol_low':       vol_low_s,
    'Y':             daily['Y'],
}).dropna()

feat = feat.astype(float)
print(f"  Feature matrix shape: {feat.shape}")
print(f"  Date range: {feat.index[0].date()} → {feat.index[-1].date()}")
print(f"  Bearish days (Y=1): {feat['Y'].sum():.0f} / {len(feat)} ({feat['Y'].mean()*100:.1f}%)")

# ─── 8. IS / OOS split ───────────────────────────────────────────────────────
split_n = int(len(feat) * 0.70)
feat_is  = feat.iloc[:split_n]
feat_oos = feat.iloc[split_n:]
print(f"\n  IS  ({split_n} days): {feat_is.index[0].date()} → {feat_is.index[-1].date()}")
print(f"  OOS ({len(feat_oos)} days): {feat_oos.index[0].date()} → {feat_oos.index[-1].date()}")

X_cols = ['judas_swing', 'cot_topping', 'bear_fvg_below', 'vol_low']
X_is  = feat_is[X_cols]
y_is  = feat_is['Y']
X_oos = feat_oos[X_cols]
y_oos = feat_oos['Y']

# ─── 9. Fit Logistic Regression (statsmodels) ─────────────────────────────────
try:
    import statsmodels.api as sm
    X_is_const  = sm.add_constant(X_is, has_constant='add')
    X_oos_const = sm.add_constant(X_oos, has_constant='add')

    logit_model = sm.Logit(y_is, X_is_const)
    result      = logit_model.fit(disp=False, method='bfgs')

    print("\n" + "=" * 70)
    print("  LOGIT MODEL — IS COEFFICIENTS")
    print("=" * 70)
    coef_table = pd.DataFrame({
        'coef':    result.params,
        'std_err': result.bse,
        'z':       result.tvalues,
        'p_value': result.pvalues,
    })
    print(coef_table.round(4).to_string())

    # OOS predictions
    y_pred_prob_oos = result.predict(X_oos_const)
    coefs_dict = {str(k): round(float(v), 6) for k, v in result.params.items()}
    pvals_dict = {str(k): round(float(v), 6) for k, v in result.pvalues.items()}

except Exception as e:
    print(f"  statsmodels Logit failed: {e}")
    print("  Falling back to sklearn LogisticRegression...")
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_is, y_is)
    y_pred_prob_oos = lr.predict_proba(X_oos)[:, 1]
    coefs_dict = dict(zip(X_cols, lr.coef_[0].tolist()))
    pvals_dict = {k: None for k in X_cols}
    coef_table = pd.DataFrame({'coef': coefs_dict}).T

# ─── 10. OOS Metrics ─────────────────────────────────────────────────────────
from sklearn.metrics import brier_score_loss, roc_auc_score

y_oos_arr = y_oos.values.astype(float)
brier_model   = brier_score_loss(y_oos_arr, y_pred_prob_oos)
naive_preds   = np.full(len(y_oos_arr), 0.5)
brier_naive   = brier_score_loss(y_oos_arr, naive_preds)

try:
    auc = roc_auc_score(y_oos_arr, y_pred_prob_oos)
except Exception:
    auc = float('nan')

print("\n" + "=" * 70)
print("  OOS PERFORMANCE METRICS")
print("=" * 70)
print(f"  OOS Brier Score  (model):  {brier_model:.5f}")
print(f"  OOS Brier Score  (naive P=0.5): {brier_naive:.5f}")
print(f"  Brier Skill Score (1 - model/naive): {1 - brier_model/brier_naive:.4f}")
print(f"  OOS AUC-ROC:               {auc:.4f}")

# ─── 11. Feature correlation matrix ──────────────────────────────────────────
print("\n" + "=" * 70)
print("  FEATURE CORRELATION MATRIX (full sample)")
print("=" * 70)
corr = feat[X_cols].corr()
print(corr.round(3).to_string())

# ─── 12. Save results ─────────────────────────────────────────────────────────
output = {
    'is_period':  {'start': str(feat_is.index[0].date()),
                   'end':   str(feat_is.index[-1].date()),
                   'n_days': split_n},
    'oos_period': {'start': str(feat_oos.index[0].date()),
                   'end':   str(feat_oos.index[-1].date()),
                   'n_days': len(feat_oos)},
    'feature_stats': {
        col: {
            'mean': round(float(feat[col].mean()), 4),
            'std':  round(float(feat[col].std()), 4),
        } for col in X_cols
    },
    'target_base_rate': round(float(feat['Y'].mean()), 4),
    'coefficients': coefs_dict,
    'p_values': pvals_dict,
    'oos_metrics': {
        'brier_score_model': round(float(brier_model), 6),
        'brier_score_naive_half': round(float(brier_naive), 6),
        'brier_skill_score': round(float(1 - brier_model / brier_naive), 4),
        'auc_roc': round(float(auc), 4) if not np.isnan(auc) else None,
    },
    'correlation_matrix': {
        col: {col2: round(float(corr.loc[col, col2]), 4) for col2 in X_cols}
        for col in X_cols
    },
}
with open(OUT_JSON, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n  Results saved → {OUT_JSON}")
print("\n[TASK C COMPLETE]")
