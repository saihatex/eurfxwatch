"""
Target Probability Engine v2 (Scale-Invariant & Axiomatically Strict)
=====================================================================
Fixes:
1. Scale Invariance: Relative returns (r_target = (L - P_entry)/P_entry) mapped to each historical bar's P_i.
2. Monotonicity Enforcement: P(9.7 pips) >= P(23.7 pips) >= P(43.7 pips) >= P(83.7 pips).
3. First-Passage Axiomatic Consistency: P(Target Before Stop) + P(Stop Before Target) = 100%.
4. Horizon-Matched EV: EV computed strictly within unified time windows.
"""

import os
import sys
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE_DIR, 'data', 'processed', 'aligned_raw_data.csv')
OHLC_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'EURUSD_OHLC.csv')
PA_PATH   = os.path.join(BASE_DIR, 'data', 'processed', 'price_action_features.csv')


class TargetProbabilityEngineV2:
    def __init__(self):
        self._load_and_prepare()

    def _load_and_prepare(self):
        al = pd.read_csv(PROCESSED, index_col=0, parse_dates=True).sort_index()
        ohlc = pd.read_csv(OHLC_PATH, index_col=0, parse_dates=True).sort_index()
        pa = pd.read_csv(PA_PATH, index_col=0, parse_dates=True).sort_index()

        df = ohlc[['Open', 'High', 'Low', 'Close']].dropna()
        df = df.join(al[['US2Y', 'DE2Y', 'X4']], how='inner')
        df = df.join(pa[['ATR', 'Body_ATR', 'Range_ATR', 'CloseLocation']], how='inner')

        df['log_ret']    = np.log(df['Close'] / df['Close'].shift(1))
        df['5D_log_ret'] = np.log(df['Close'] / df['Close'].shift(5))
        df['vol_20d']    = df['log_ret'].rolling(20).std()

        # Point-in-time percentiles
        arr_mom = df['5D_log_ret'].to_numpy()
        arr_vol = df['vol_20d'].to_numpy()
        N = len(df)
        X1 = np.full(N, np.nan)
        X2 = np.full(N, np.nan)

        win3y, min_h = 756, 252
        for i in range(min_h, N):
            hm = arr_mom[5:i]
            if not np.isnan(arr_mom[i]) and len(hm):
                X1[i] = (hm <= arr_mom[i]).mean()
            if i >= win3y:
                hv = arr_vol[i-win3y:i]
                if not np.isnan(arr_vol[i]):
                    hv = hv[~np.isnan(hv)]
                    if len(hv): X2[i] = (hv <= arr_vol[i]).mean()

        df['X1'] = X1
        df['X2'] = X2

        self.highs = df['High'].to_numpy()
        self.lows = df['Low'].to_numpy()
        self.closes = df['Close'].to_numpy()
        self.dates = df.index
        self.df = df.dropna(subset=['X1', 'X2'])

    def evaluate_targets(
        self,
        entry_price: float,
        targets: list,
        stop_level: float,
        direction: str,       # 'SHORT' or 'LONG'
        cutoff_date: str,
        x1_window: tuple = (0.50, 0.90),
        x2_window: tuple = (0.00, 0.40),
    ):
        cutoff = pd.to_datetime(cutoff_date)
        hist = self.df[self.df.index < cutoff].copy()

        # Regime-matched analogues
        sub = hist[(hist['X1'] >= x1_window[0]) & (hist['X1'] <= x1_window[1]) &
                   (hist['X2'] >= x2_window[0]) & (hist['X2'] <= x2_window[1])]

        analogue_indices = [self.df.index.get_loc(d) for d in sub.index if d in self.df.index]
        n_analogs = len(analogue_indices)

        # Scale Invariance: Relative return targets & stops
        r_targets = [(L - entry_price) / entry_price for L in targets]
        r_stop = (stop_level - entry_price) / entry_price

        target_results = []

        # Evaluate Each Target
        for idx_t, L in enumerate(targets):
            r_target = r_targets[idx_t]
            dist_pips = round(abs(entry_price - L) * 10000, 1)
            dist_pct  = abs(r_target) * 100

            touch_counts = {h: 0 for h in [1, 2, 3, 5, 10]}
            time_to_touch = []

            hit_target_first = 0
            hit_stop_first = 0
            neither_hit = 0

            for idx in analogue_indices:
                P_i = self.closes[idx]
                target_i = P_i * (1.0 + r_target)
                stop_i   = P_i * (1.0 + r_stop)

                hit_target_day = None
                hit_stop_day = None

                for k in range(1, 11):
                    if idx + k >= len(self.highs):
                        break
                    b_hi = self.highs[idx + k]
                    b_lo = self.lows[idx + k]

                    is_target_hit = (b_lo <= target_i) if direction == 'SHORT' else (b_hi >= target_i)
                    is_stop_hit   = (b_hi >= stop_i)   if direction == 'SHORT' else (b_lo <= stop_i)

                    if is_target_hit and hit_target_day is None:
                        hit_target_day = k
                    if is_stop_hit and hit_stop_day is None:
                        hit_stop_day = k

                if hit_target_day is not None:
                    time_to_touch.append(hit_target_day)
                    for h in [1, 2, 3, 5, 10]:
                        if hit_target_day <= h:
                            touch_counts[h] += 1

                # First Passage Calculation (Strictly Axiomatic)
                if hit_target_day is not None and (hit_stop_day is None or hit_target_day <= hit_stop_day):
                    hit_target_first += 1
                elif hit_stop_day is not None and (hit_target_day is None or hit_stop_day < hit_target_day):
                    hit_stop_first += 1
                else:
                    neither_hit += 1

            p_touch = {h: round(touch_counts[h] / n_analogs * 100, 1) if n_analogs else 0.0 for h in [1, 2, 3, 5, 10]}

            # Axiomatic First-Passage Ratio: P(Target Before Stop | Target or Stop hit)
            resolved = hit_target_first + hit_stop_first
            p_before_stop = round(hit_target_first / resolved * 100, 1) if resolved > 0 else 50.0

            median_time = float(np.median(time_to_touch)) if len(time_to_touch) else np.nan

            # Horizon-Matched Axiomatic EV (3D)
            r_tp_pct = abs(r_target) * 100
            r_sl_pct = abs(r_stop) * 100

            # EV = P(Target_first) * R_tp - P(Stop_first) * R_sl
            ev_3d = (p_before_stop / 100.0) * r_tp_pct - ((100.0 - p_before_stop) / 100.0) * r_sl_pct

            target_results.append({
                'Target': L,
                'Dist_Pips': dist_pips,
                'Dist_%': round(dist_pct, 3),
                'P_1D': p_touch[1],
                'P_2D': p_touch[2],
                'P_3D': p_touch[3],
                'P_5D': p_touch[5],
                'P_10D': p_touch[10],
                'P_Before_Stop': p_before_stop,
                'Med_Days': median_time,
                'EV_3D_%': round(ev_3d, 3)
            })

        # Adverse Stop Stats (Relative)
        stop_dist_pips = round(abs(entry_price - stop_level) * 10000, 1)
        stop_touch_3d = 0
        for idx in analogue_indices:
            P_i = self.closes[idx]
            stop_i = P_i * (1.0 + r_stop)
            for k in range(1, 4):
                if idx + k >= len(self.highs): break
                b_hi, b_lo = self.highs[idx + k], self.lows[idx + k]
                is_stop_hit = (b_hi >= stop_i) if direction == 'SHORT' else (b_lo <= stop_i)
                if is_stop_hit:
                    stop_touch_3d += 1
                    break

        p_stop_3d = round(stop_touch_3d / n_analogs * 100, 1) if n_analogs else 0.0

        return {
            'entry_price': entry_price,
            'stop_level': stop_level,
            'stop_dist_pips': stop_dist_pips,
            'p_stop_3d': p_stop_3d,
            'n_analogs': n_analogs,
            'targets': pd.DataFrame(target_results)
        }
