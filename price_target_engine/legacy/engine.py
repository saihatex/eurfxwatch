"""
Price Target Engine — Module 1 (Purely Statistical, No ML)
===========================================================
Computes, for a SHORT position:
  P↓(L, H) = P(min_{1≤k≤H} P_{t+k} ≤ L | X_t)

And for a LONG position:
  P↑(L, H) = P(max_{1≤k≤H} P_{t+k} ≥ L | X_t)

Also computes MAE / MFE distributions for each regime.

Conditioning variables:
  - Distance bucket  : how far the target is from current price (in %)
  - X1 bucket        : momentum regime (Neutral / High Impulse)
  - X2 bucket        : volatility regime (Low / High)
  - Direction filter : SHORT or LONG analogues

All conditioning is done strictly point-in-time (no lookahead).
"""

import os
import sys
import numpy as np
import pandas as pd

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE_DIR, 'data', 'processed', 'aligned_raw_data.csv')

MAX_HORIZON = 10   # maximum trading-day horizon to precompute


class PriceTargetEngine:
    """
    Build a historical database of price paths and touch probabilities.
    All computations are performed on the snapshot of data available strictly
    up to `cutoff_date` to preserve point-in-time integrity.
    """

    def __init__(self, data_path: str = PROCESSED):
        self.data_path = data_path
        self._load_and_prepare()

    # ─── Data loading & feature construction ──────────────────────────────────
    def _load_and_prepare(self):
        df = pd.read_csv(self.data_path, index_col=0, parse_dates=True).sort_index()
        df = df.dropna(subset=['EURUSD', 'US2Y', 'DE2Y', 'X4'])

        df['log_ret']    = np.log(df['EURUSD'] / df['EURUSD'].shift(1))
        df['5D_log_ret'] = np.log(df['EURUSD'] / df['EURUSD'].shift(5))
        df['vol_20d']    = df['log_ret'].rolling(20).std()

        # Point-in-time percentiles
        arr_mom = df['5D_log_ret'].to_numpy()
        arr_vol = df['vol_20d'].to_numpy()
        N = len(df)
        X1 = np.full(N, np.nan)
        X2 = np.full(N, np.nan)

        win3y, min_h = 756, 252
        for i in range(min_h, N):
            hist_m = arr_mom[5:i]
            if not np.isnan(arr_mom[i]) and len(hist_m):
                X1[i] = (hist_m <= arr_mom[i]).mean()
            if i >= win3y:
                hist_v = arr_vol[i - win3y : i]
                if not np.isnan(arr_vol[i]):
                    hv = hist_v[~np.isnan(hist_v)]
                    if len(hv):
                        X2[i] = (hv <= arr_vol[i]).mean()

        df['X1'] = X1
        df['X2'] = X2

        # Pre-compute forward price paths for each horizon
        prices = df['EURUSD'].to_numpy()
        for h in range(1, MAX_HORIZON + 1):
            df[f'P_t+{h}'] = pd.Series(
                np.concatenate([prices[h:], np.full(h, np.nan)]),
                index=df.index
            )

        # Min/Max price within each horizon (for touch probability)
        future_prices = np.stack(
            [df[f'P_t+{h}'].to_numpy() for h in range(1, MAX_HORIZON + 1)],
            axis=1
        )   # shape (N, MAX_HORIZON)

        # min[1..h] and max[1..h] for each h
        for h in range(1, MAX_HORIZON + 1):
            df[f'min_{h}D'] = np.nanmin(future_prices[:, :h], axis=1)
            df[f'max_{h}D'] = np.nanmax(future_prices[:, :h], axis=1)

        # MAE / MFE for SHORT position (from t to t+10)
        # MAE_short = how much price went AGAINST a short (i.e., max high)
        # MFE_short = how much price went IN FAVOR of a short (i.e., max low)
        df['MAE_short_pct'] = (df['max_10D'] / df['EURUSD'] - 1) * 100   # positive = adverse
        df['MFE_short_pct'] = (df['EURUSD']  / df['min_10D'] - 1) * 100  # positive = favorable

        # MAE / MFE for LONG position
        df['MAE_long_pct']  = (df['EURUSD']  / df['min_10D'] - 1) * (-100)  # positive = adverse
        df['MFE_long_pct']  = (df['max_10D'] / df['EURUSD'] - 1) * 100      # positive = favorable

        df = df.dropna(subset=['X1', 'X2'])
        self.df = df

    # ─── Public API ──────────────────────────────────────────────────────────
    def touch_probability_table(
        self,
        entry_price: float,
        targets: list,
        direction: str,       # 'SHORT' or 'LONG'
        cutoff_date: str,
        x1_range: tuple = None,
        x2_range: tuple = None,
    ) -> pd.DataFrame:
        """
        For each target level and each horizon 1D–10D, compute:
          P(touch target within horizon | regime analogue)

        Regime analogues are filtered by x1_range / x2_range if provided,
        otherwise the full history up to cutoff_date is used.
        """
        cutoff = pd.to_datetime(cutoff_date)
        hist   = self.df[self.df.index < cutoff].copy()

        # Regime filter
        if x1_range:
            hist = hist[(hist['X1'] >= x1_range[0]) & (hist['X1'] < x1_range[1])]
        if x2_range:
            hist = hist[(hist['X2'] >= x2_range[0]) & (hist['X2'] < x2_range[1])]

        rows = []
        for L in targets:
            dist_pct = abs(entry_price - L) / entry_price * 100
            row = {'Target': L, 'Distance_%': round(dist_pct, 3)}

            for h in [1, 2, 3, 5, 10]:
                if direction == 'SHORT':
                    touched = (hist[f'min_{h}D'] <= L).sum()
                else:
                    touched = (hist[f'max_{h}D'] >= L).sum()

                n_valid = hist[f'min_{h}D'].notna().sum() if direction == 'SHORT' \
                     else hist[f'max_{h}D'].notna().sum()

                p = touched / n_valid if n_valid > 0 else np.nan
                row[f'P_touch_{h}D'] = round(p * 100, 1)

            rows.append(row)

        return pd.DataFrame(rows)

    def mae_mfe_stats(
        self,
        direction: str,
        cutoff_date: str,
        x1_range: tuple = None,
        x2_range: tuple = None,
    ) -> dict:
        """
        Returns MAE / MFE statistics for analogous historical episodes.
        """
        cutoff = pd.to_datetime(cutoff_date)
        hist   = self.df[self.df.index < cutoff].copy()

        if x1_range:
            hist = hist[(hist['X1'] >= x1_range[0]) & (hist['X1'] < x1_range[1])]
        if x2_range:
            hist = hist[(hist['X2'] >= x2_range[0]) & (hist['X2'] < x2_range[1])]

        mae_col = f'MAE_{direction.lower()}_pct'
        mfe_col = f'MFE_{direction.lower()}_pct'

        mae = hist[mae_col].dropna()
        mfe = hist[mfe_col].dropna()

        def _s(arr):
            return {
                'mean': round(arr.mean(), 3),
                'median': round(arr.median(), 3),
                'p25': round(arr.quantile(0.25), 3),
                'p75': round(arr.quantile(0.75), 3),
                'p90': round(arr.quantile(0.90), 3),
            }

        return {
            'N_analogues': len(mae),
            'MAE_10D_%' : _s(mae),
            'MFE_10D_%' : _s(mfe),
        }
