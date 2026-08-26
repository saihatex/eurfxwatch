import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

class EURUSDDecisionEngine:
    """
    3-Layer Quantitative Decision Engine for EUR/USD:
    - Layer 1: Regime Identification (Vol & Trend)
    - Layer 2: Directional Evidence Scoring (LONG vs SHORT thesis)
    - Layer 3: Conditional Historical Distribution & Expected Return
    """
    def __init__(self, data_path=None):
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, 'data', 'processed', 'aligned_raw_data.csv')
        self.data_path = data_path
        self._load_data()

    def _load_data(self):
        self.df = pd.read_csv(self.data_path, index_col=0, parse_dates=True).dropna(subset=['EURUSD', 'US2Y', 'DE2Y', 'X4'])
        self.df['log_return'] = np.log(self.df['EURUSD'] / self.df['EURUSD'].shift(1))
        self.df['5D_log_return'] = np.log(self.df['EURUSD'] / self.df['EURUSD'].shift(5))
        self.df['vol_20d'] = self.df['log_return'].rolling(20).std()

        # Compute point-in-time percentiles up to t-1
        arr_5d = self.df['5D_log_return'].to_numpy()
        arr_vol = self.df['vol_20d'].to_numpy()

        pctl_5d = np.full(len(self.df), np.nan)
        pctl_vol = np.full(len(self.df), np.nan)

        window_3y = 756
        min_history = 252

        for i in range(min_history, len(self.df)):
            hist = arr_5d[5:i]
            val = arr_5d[i]
            if not np.isnan(val):
                pctl_5d[i] = (hist <= val).mean()

        for i in range(window_3y, len(self.df)):
            hist = arr_vol[i-window_3y:i]
            val = arr_vol[i]
            if not np.isnan(val):
                valid = hist[~np.isnan(hist)]
                if len(valid) > 0:
                    pctl_vol[i] = (valid <= val).mean()

        self.df['X1_5D_pctl'] = pctl_5d
        self.df['X2_vol_pctl'] = pctl_vol
        self.df['forward_3D'] = np.log(self.df['EURUSD'].shift(-3) / self.df['EURUSD'])
        
        # Drop rows where feature percentiles could not be calculated
        self.df = self.df.dropna(subset=['X1_5D_pctl', 'X2_vol_pctl'])

    def evaluate(self, date_str):
        """
        Evaluates the market state strictly as of date_str (Point-in-Time).
        """
        if date_str not in self.df.index.strftime('%Y-%m-%d'):
            raise ValueError(f"Date {date_str} not in dataset.")

        cutoff_idx = self.df.index.get_loc(date_str)
        df_hist = self.df.iloc[:cutoff_idx + 1]
        today = df_hist.iloc[-1]

        eur_close = float(today['EURUSD'])
        x1_mom = float(today['X1_5D_pctl'])
        x2_vol = float(today['X2_vol_pctl'])
        us2y = float(today['US2Y'])
        de2y = float(today['DE2Y'])
        spread = float(today['US_DE_Spread'])
        x4_rates = float(today['X4'])

        # --- LAYER 1: REGIME IDENTIFICATION ---
        if x2_vol < 0.25:
            vol_regime = "LOW VOLATILITY"
        elif x2_vol < 0.75:
            vol_regime = "NORMAL VOLATILITY"
        elif x2_vol < 0.90:
            vol_regime = "HIGH VOLATILITY"
        else:
            vol_regime = "EXTREME / CRISIS VOLATILITY"

        if x1_mom >= 0.75:
            trend_regime = "EUR BULLISH IMPULSE"
        elif x1_mom <= 0.25:
            trend_regime = "EUR BEARISH IMPULSE"
        else:
            trend_regime = "NEUTRAL / SIDEWAYS"

        # --- LAYER 2: DIRECTIONAL EVIDENCE SCORING ---
        short_factors = []
        long_factors = []

        short_score = 0
        long_score = 0

        # SHORT Thesis Evidences
        if x1_mom >= 0.90:
            short_score += 3
            short_factors.append("Extreme Momentum Exhaustion (X1 >= 90%)")
        if x2_vol >= 0.75:
            short_score += 2
            short_factors.append("High Volatility Gate Passed (X2 >= 75%)")
        if x4_rates > +0.05:
            short_score += 3
            short_factors.append("Rates Divergence (X4 > +0.05, USD-Supportive)")
        elif x4_rates > 0:
            short_score += 1
            short_factors.append("Rates Shift Mild USD-Supportive (X4 > 0)")

        # LONG Thesis Evidences
        if x1_mom <= 0.10 and x2_vol >= 0.75:
            long_score += 4
            long_factors.append("Oversold Rebound Setup (X1 <= 10%, X2 >= 75%)")
        elif x1_mom >= 0.50 and x1_mom < 0.85:
            long_score += 2
            long_factors.append("Bullish Momentum Continuation (50% <= X1 < 85%)")
        
        if x4_rates < -0.05:
            long_score += 3
            long_factors.append("Rates Support EUR Continuation (X4 < -0.05)")
        elif x4_rates < 0:
            long_score += 1
            long_factors.append("Rates Mildly Support EUR (X4 < 0)")

        # --- LAYER 3: CONDITIONAL HISTORICAL DISTRIBUTION ---
        # Query historical days strictly up to cutoff-1
        hist_sample = df_hist.iloc[:-1].dropna(subset=['forward_3D'])
        
        # Short Condition Subset
        short_sub = hist_sample[
            (hist_sample['X1_5D_pctl'] >= 0.85) &
            (hist_sample['X2_vol_pctl'] >= 0.60)
        ]['forward_3D']

        # Long Condition Subset
        long_sub = hist_sample[
            (hist_sample['X1_5D_pctl'] >= 0.40) & (hist_sample['X1_5D_pctl'] <= 0.85) &
            (hist_sample['X4'] < 0)
        ]['forward_3D']

        short_exp = float(short_sub.mean() * 100) if len(short_sub) >= 5 else -0.50
        short_med = float(short_sub.median() * 100) if len(short_sub) >= 5 else -0.30
        short_p_neg = float((short_sub < 0).mean() * 100) if len(short_sub) >= 5 else 65.0

        long_exp = float(long_sub.mean() * 100) if len(long_sub) >= 5 else +0.40
        long_med = float(long_sub.median() * 100) if len(long_sub) >= 5 else +0.25
        long_p_pos = float((long_sub > 0).mean() * 100) if len(long_sub) >= 5 else 60.0

        # Final Decision Synthesis
        if short_score >= 5 and short_score > long_score:
            final_signal = "SHORT BIAS"
            confidence = "HIGH" if short_score >= 7 else "MEDIUM"
        elif long_score >= 4 and long_score > short_score:
            final_signal = "LONG BIAS"
            confidence = "HIGH" if long_score >= 6 else "MEDIUM"
        else:
            final_signal = "NO TRADE"
            confidence = "LOW"

        return {
            "date": date_str,
            "eur_close": eur_close,
            "regime": {
                "volatility": vol_regime,
                "trend": trend_regime,
                "x1_mom_pctl": round(x1_mom * 100, 1),
                "x2_vol_pctl": round(x2_vol * 100, 1),
                "us2y": us2y,
                "de2y": de2y,
                "spread": round(spread, 4),
                "x4_rates": round(x4_rates, 4)
            },
            "scores": {
                "short_score": short_score,
                "long_score": long_score,
                "short_factors": short_factors,
                "long_factors": long_factors
            },
            "distributions": {
                "short": {
                    "expected_3d": round(short_exp, 2),
                    "median_3d": round(short_med, 2),
                    "p_negative": round(short_p_neg, 1)
                },
                "long": {
                    "expected_3d": round(long_exp, 2),
                    "median_3d": round(long_med, 2),
                    "p_positive": round(long_p_pos, 1)
                }
            },
            "decision": {
                "final_signal": final_signal,
                "confidence": confidence
            }
        }
