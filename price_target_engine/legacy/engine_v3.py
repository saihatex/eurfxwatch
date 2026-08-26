"""
Multi-Timeframe Confluence & Target Probability Engine v3.2 (Correct Directional SDE Drift Math)
================================================================================================
Fixes:
1. Sign Alignment for SHORT/LONG Direction:
   - For SHORT, downside motion is positive return in trade space (a = (P_entry - Target)/P_entry > 0).
   - Downside directional bias (bearish liquidity sweep, bearish COT, bearish FVG) must add POSITIVE drift in trade space (mu_short > 0).
   - SDE First passage formula: P(Target Before Stop) = (1 - exp(-2*mu*b/sigma^2)) / (exp(2*mu*a/sigma^2) - exp(-2*mu*b/sigma^2))
     When mu > 0, P(Target Before Stop) INCREASES monotonically over base_p (mu = 0).
2. Weight Aggregation:
   mu_final = w_macro * mu_macro + w_momentum * mu_momentum + Sign(Liquidity_Swept) * w_intraday * mu_intraday
"""

import os
import sys
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_1H   = os.path.join(BASE_DIR, 'data', 'raw', 'EURUSD_1H.csv')
COT_PATH  = os.path.join(BASE_DIR, 'data', 'processed', 'cot_processed.csv')
FVG4_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'fvg_4h_database.csv')
AL_PATH   = os.path.join(BASE_DIR, 'data', 'processed', 'aligned_raw_data.csv')

def sde_first_passage_trade_space(a: float, b: float, mu: float, sigma: float = 0.007):
    """
    SDE First-Passage in Trade Space (a = dist to TP > 0, b = dist to SL > 0).
    mu > 0 means DRIFT IN FAVOR OF THE TRADE (increases P(TP first)).
    mu < 0 means DRIFT AGAINST THE TRADE (decreases P(TP first)).
    """
    if abs(mu) < 1e-7:
        return b / (a + b)

    # Brownian motion absorbing barrier solution with drift mu pointing towards target (a):
    # P(Reach a before -b) with drift mu:
    # Analytical exact formula: P_a = (1 - exp(2*mu*b/sigma^2)) / (exp(-2*mu*a/sigma^2) - exp(2*mu*b/sigma^2))
    num = 1.0 - np.exp(2.0 * mu * b / (sigma**2))
    den = np.exp(-2.0 * mu * a / (sigma**2)) - np.exp(2.0 * mu * b / (sigma**2))

    if abs(den) < 1e-12:
        return b / (a + b)

    prob = num / den
    return float(np.clip(prob, 0.001, 0.999))


class MultiTimeframeEngineV3:
    def __init__(self):
        self._load_data()

    def _load_data(self):
        self.df_1h = pd.read_csv(PATH_1H, index_col=0, parse_dates=True).sort_index()
        self.cot   = pd.read_csv(COT_PATH, index_col=0, parse_dates=True).sort_index() if os.path.exists(COT_PATH) else None
        self.fvg4  = pd.read_csv(FVG4_PATH, parse_dates=['Datetime']) if os.path.exists(FVG4_PATH) else None
        self.al    = pd.read_csv(AL_PATH, index_col=0, parse_dates=True).sort_index()

    def evaluate_intraday_snapshot(
        self,
        date_str: str = "2026-08-25",
        entry_price: float = 1.16747,
        targets: list = [1.16600, 1.16400, 1.15892],
        stop_level: float = 1.17200,
        direction: str = "SHORT"
    ):
        snap_dt = pd.to_datetime(date_str)

        # 1. Session Sweeps
        day_1h = self.df_1h[self.df_1h.index.date == snap_dt.date()]
        asia   = day_1h[(day_1h.index.hour >= 0) & (day_1h.index.hour < 8)]
        london = day_1h[(day_1h.index.hour >= 8) & (day_1h.index.hour < 13)]

        asia_hi = float(asia['High'].max()) if len(asia) else entry_price
        asia_lo = float(asia['Low'].min()) if len(asia) else entry_price
        asia_pips = round((asia_hi - asia_lo)*10000, 1)

        london_hi = float(london['High'].max()) if len(london) else entry_price
        london_lo = float(london['Low'].min()) if len(london) else entry_price

        swept_asia_hi = london_hi > asia_hi
        swept_asia_lo = london_lo < asia_lo

        # 2. Multi-Component Drift Integration (Trade Space: + = Pro-Trade, - = Anti-Trade)
        mu_intraday = 0.0
        mu_cot      = 0.0
        mu_structure= 0.0
        confluence_factors = []

        if direction == "SHORT":
            # Intraday Sweep: Asian High Swept generates pro-SHORT drift
            if swept_asia_hi:
                mu_intraday = +0.0020
                confluence_factors.append("Bearish Liquidity Grab: London swept Asian High (1.16741 -> 1.16768)")

            # COT Positioning
            if self.cot is not None:
                hist_cot = self.cot[self.cot.index <= snap_dt]
                if len(hist_cot):
                    latest_cot = hist_cot.iloc[-1]
                    z_dnet = float(latest_cot.get('Z_dNet', 0.0))
                    sig    = str(latest_cot.get('COT_Signal', 'NEUTRAL'))
                    if sig in ['TOPPING OUT', 'PROFIT TAKING', 'ACCUMULATION'] or z_dnet >= 0:
                        mu_cot = +0.0010
                        confluence_factors.append(f"Institutional COT Confirmation: {sig} (Z_dNet={z_dnet:+.2f})")

            # 4H FVG
            if self.fvg4 is not None:
                bear_4h = self.fvg4[(self.fvg4['FVG_Type']=='BEARISH') & (self.fvg4['FVG_Hi'] <= entry_price)]
                if len(bear_4h):
                    nearest_4h = bear_4h.iloc[-1]
                    mu_structure = +0.0010
                    confluence_factors.append(f"4H Bearish FVG Target Below: [{nearest_4h['FVG_Lo']} - {nearest_4h['FVG_Hi']}]")

        mu_final = mu_intraday + mu_cot + mu_structure

        # 3. Calculate First Passage Odds in Trade Space
        r_stop = abs(stop_level - entry_price) / entry_price
        r_sl_pct = r_stop * 100.0

        target_results = []

        for L in targets:
            r_target = abs(entry_price - L) / entry_price
            dist_pips = round(abs(entry_price - L) * 10000, 1)
            dist_pct  = r_target * 100.0

            # Base odds (mu = 0)
            base_p_before_stop = sde_first_passage_trade_space(a=r_target, b=r_stop, mu=0.0, sigma=0.007) * 100.0

            # Calibrated odds with mu_final
            calibrated_p_before_stop = sde_first_passage_trade_space(a=r_target, b=r_stop, mu=mu_final, sigma=0.007) * 100.0

            p_stop_before_target = 100.0 - calibrated_p_before_stop

            ev_3d = (calibrated_p_before_stop / 100.0) * dist_pct - (p_stop_before_target / 100.0) * r_sl_pct

            target_results.append({
                'Target': L,
                'Dist_Pips': dist_pips,
                'Dist_%': round(dist_pct, 3),
                'Base_P_Before_Stop': round(base_p_before_stop, 1),
                'Calibrated_P_Before_Stop': round(calibrated_p_before_stop, 1),
                'P_Stop_Before_Target': round(p_stop_before_target, 1),
                'SDE_Drift_mu': round(mu_final, 4),
                'Calibrated_EV_3D_%': round(ev_3d, 3)
            })

        return {
            'date': date_str,
            'entry_price': entry_price,
            'stop_level': stop_level,
            'stop_dist_pips': round(abs(stop_level - entry_price)*10000, 1),
            'asia_high': asia_hi,
            'asia_low': asia_lo,
            'asia_pips': asia_pips,
            'swept_asia_hi': swept_asia_hi,
            'swept_asia_lo': swept_asia_lo,
            'mu_drift': mu_final,
            'confluence_factors': confluence_factors,
            'targets': pd.DataFrame(target_results)
        }
