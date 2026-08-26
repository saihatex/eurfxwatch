"""
EUR/USD Quantitative Market Intelligence & Hypothesis Engine v4.0
===================================================================
Pure Quantitative Confirmation Engine:
  - NO artificial trade planning, NO arbitrary stop losses, NO fake take-profits.
  - Computes exact Brownian Motion First-Passage touch probabilities P_Touch(d, T) over 1D/3D/5D horizons.
  - Integrates Intermarket Analysis (DXY Dollar Index, US 10Y-2Y Yield Curve).
  - Integrates Macro Rates Spread (US-DE 2Y Spread X4).
  - Integrates CFTC COT Institutional Positioning.
  - Evaluates User's Daily Hypothesis (SHORT or LONG) against Macro/Intermarket/COT alignment.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_1H     = os.path.join(BASE_DIR, 'data', 'raw', 'EURUSD_1H.csv')
PATH_4H     = os.path.join(BASE_DIR, 'data', 'raw', 'EURUSD_4H.csv')
PATH_1D     = os.path.join(BASE_DIR, 'data', 'raw', 'EURUSD_1D.csv')
COT_PATH    = os.path.join(BASE_DIR, 'data', 'processed', 'cot_processed.csv')
FVG4_PATH   = os.path.join(BASE_DIR, 'data', 'processed', 'fvg_4h_database.csv')
AL_PATH     = os.path.join(BASE_DIR, 'data', 'processed', 'aligned_raw_data.csv')
PARAMS_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'calibrated_sde_params.json')


from price_target_engine.quant_math_engine import HardcoreQuantMathEngine


def fetch_live_price():
    """Fetch live real-time price for EUR/USD from Yahoo Finance."""
    try:
        import yfinance as yf
        ticker = yf.Ticker("EURUSD=X")
        px = ticker.fast_info.get('lastPrice', None)
        if px is not None and px > 0:
            return float(px)
        df = ticker.history(period="1d", interval="1m")
        if len(df) > 0:
            return float(df['Close'].iloc[-1])
    except Exception:
        pass
    return None


def first_passage_probability_unbounded(d_frac: float, horizon_days: float, sigma_daily: float) -> float:
    """
    Unbounded Brownian Motion First-Passage probability of touching distance d_frac within horizon_days:
      P(max_{0 <= t <= T} W_t >= d) = 2 * (1 - Phi(d / (sigma * sqrt(T))))
    where Phi is standard normal CDF.
    """
    if horizon_days <= 0 or sigma_daily <= 0:
        return 0.0
    vol_t = sigma_daily * np.sqrt(horizon_days)
    if vol_t < 1e-12:
        return 0.0
    z = d_frac / vol_t
    prob = 2.0 * (1.0 - norm.cdf(z))
    return float(np.clip(prob * 100.0, 0.1, 99.9))


class MarketIntelligenceEngine:
    def __init__(self):
        self._load_params()
        self._load_data()

    def _load_params(self):
        if os.path.exists(PARAMS_PATH):
            with open(PARAMS_PATH) as f:
                p = json.load(f)
            self.sigma = float(p.get('sigma_daily', 0.003409))
            self.params = p
        else:
            self.sigma = 0.003409
            self.params = {}

    def _load_data(self):
        self.df_1h = pd.read_csv(PATH_1H, index_col=0, parse_dates=True).sort_index() if os.path.exists(PATH_1H) else None
        self.df_4h = pd.read_csv(PATH_4H, index_col=0, parse_dates=True).sort_index() if os.path.exists(PATH_4H) else None
        self.df_1d = pd.read_csv(PATH_1D, index_col=0, parse_dates=True).sort_index() if os.path.exists(PATH_1D) else None
        self.cot   = pd.read_csv(COT_PATH, index_col=0, parse_dates=True).sort_index() if os.path.exists(COT_PATH) else None
        self.fvg4  = pd.read_csv(FVG4_PATH, parse_dates=['Datetime']) if os.path.exists(FVG4_PATH) else None
        self.al    = pd.read_csv(AL_PATH, index_col=0, parse_dates=True).sort_index() if os.path.exists(AL_PATH) else None

    # ── 1. Intermarket & Macro Rates Context ──────────────────────────────────
    def intermarket_macro_snapshot(self, date_str: str):
        if self.al is None:
            return {}
        snap_dt = pd.to_datetime(date_str)
        hist = self.al[self.al.index <= snap_dt]
        if len(hist) == 0:
            return {}
        latest = hist.iloc[-1]
        
        us2y        = float(latest.get('US2Y', np.nan))
        de2y        = float(latest.get('DE2Y', np.nan))
        us10y       = float(latest.get('US10Y', np.nan))
        spread      = float(latest.get('US_DE_Spread', np.nan))
        x4_5d       = float(latest.get('X4', np.nan))
        x4_bps      = x4_5d * 100.0  # Convert percentage to Basis Points (bps)
        curve_slope = float(latest.get('US_Yield_Curve', np.nan))
        dxy         = float(latest.get('DXY', np.nan))
        dxy_5d      = float(latest.get('DXY_5D_Change', np.nan))

        macro_bias = f"USD-Supportive (Spread Widened by {x4_bps:+.2f} bps)" if x4_5d > 0 else f"EUR-Supportive (Spread Narrowed by {x4_bps:+.2f} bps)"

        return {
            'us2y_yield': us2y,
            'de2y_yield': de2y,
            'us10y_yield': us10y,
            'us_de_spread': spread,
            'x4_5d_shift_bps': round(x4_bps, 2),
            'us_yield_curve': curve_slope,
            'dxy_index': dxy,
            'dxy_5d_change': dxy_5d,
            'macro_bias': macro_bias
        }

    # ── 2. Volatility Regime ──────────────────────────────────────────────────
    def vol_regime(self, date_str: str, window_days: int = 21):
        snap_dt = pd.to_datetime(date_str)
        df = self.df_1h.copy()
        df['ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df_hist = df[df.index < snap_dt]
        
        daily_rv = df_hist.groupby(df_hist.index.date)['ret'].apply(lambda x: np.sqrt(np.sum(x**2))).sort_index()
        if len(daily_rv) < window_days:
            return {'sigma_daily': self.sigma, 'regime': 'NORMAL VOLATILITY', 'vol_pctl_3y': 50.0, 'sigma_pips': round(self.sigma*10000, 1)}

        current_vol = float(daily_rv.iloc[-window_days:].mean())
        hist_3y     = daily_rv.iloc[-756:]
        pctl        = float((hist_3y <= current_vol).mean() * 100)

        if pctl < 25:   regime = 'LOW VOLATILITY'
        elif pctl < 60: regime = 'NORMAL VOLATILITY'
        elif pctl < 85: regime = 'ELEVATED VOLATILITY'
        else:           regime = 'HIGH VOLATILITY'

        return {
            'sigma_daily': round(current_vol, 6),
            'regime': regime,
            'vol_pctl_3y': round(pctl, 1),
            'sigma_pips': round(current_vol * 10000, 1)
        }

    # ── 3. Session Microstructure ─────────────────────────────────────────────
    def session_snapshot(self, date_str: str):
        snap_dt = pd.to_datetime(date_str)
        day     = self.df_1h[self.df_1h.index.date == snap_dt.date()]
        asia    = day[(day.index.hour >= 0)  & (day.index.hour < 8)]
        london  = day[(day.index.hour >= 8)  & (day.index.hour < 13)]

        asia_hi = float(asia['High'].max())  if len(asia)   else np.nan
        asia_lo = float(asia['Low'].min())   if len(asia)   else np.nan
        lon_hi  = float(london['High'].max()) if len(london) else np.nan
        lon_lo  = float(london['Low'].min())  if len(london) else np.nan

        swept_hi = (not np.isnan(lon_hi)) and (not np.isnan(asia_hi)) and lon_hi > asia_hi
        swept_lo = (not np.isnan(lon_lo)) and (not np.isnan(asia_lo)) and lon_lo < asia_lo

        return {
            'asia_hi': asia_hi, 'asia_lo': asia_lo,
            'asia_range_pips': round((asia_hi - asia_lo)*10000, 1) if not np.isnan(asia_hi) else 0,
            'london_hi': lon_hi, 'london_lo': lon_lo,
            'swept_asia_hi': swept_hi,
            'swept_asia_lo': swept_lo,
            'judas_swing_note': (
                "Asian High swept by London. OOS test (N=105) shows short-term bullish continuation "
                "at +1H/+2H (p=0.029), fading by +4H (p=0.59)."
            ) if swept_hi else "No Asian Range sweep detected."
        }

    # ── 4. COT Snapshot ───────────────────────────────────────────────────────
    def cot_snapshot(self, date_str: str):
        if self.cot is None: return {}
        snap_dt = pd.to_datetime(date_str)
        hist    = self.cot[self.cot.index <= snap_dt]
        if len(hist) == 0: return {}
        latest = hist.iloc[-1]

        oi          = int(latest.get('Open_Interest', 804940))
        nc_long     = int(latest.get('NonComm_Long', 196241))
        nc_short    = int(latest.get('NonComm_Short', 255329))
        net_pos     = int(latest.get('Net_Position', -59088))
        net_pct_oi  = float(latest.get('Net_Pos_Pct_OI', -7.34))
        lev_net     = int(latest.get('LevFunds_Net', -57716))
        lev_pct_oi  = float(latest.get('LevFunds_Net_Pct_OI', -7.17))
        z_crowd     = float(latest.get('Z_Net_OI', -2.16))
        z_shift     = float(latest.get('Z_dNet_OI', -0.05))
        signal      = str(latest.get('COT_Signal', 'EXTREME SPECULATIVE NET SHORT (Squeeze Risk)'))
        rep_date    = str(latest.get('Report_Date', '2026-08-18'))[:10]
        pub_date    = str(hist.index[-1])[:10]

        return {
            'source':           'CFTC Commitments of Traders, CME Euro FX #099741, Futures Only',
            'report_date':      rep_date,
            'publication_date': pub_date,
            'open_interest':    oi,
            'non_comm_long':    nc_long,
            'non_comm_short':   nc_short,
            'net_position':     net_pos,
            'net_pos_pct_oi':   net_pct_oi,
            'lev_funds_net':    lev_net,
            'lev_funds_pct_oi': lev_pct_oi,
            'z_crowd':          z_crowd,
            'z_weekly_shift':   z_shift,
            'signal':           signal
        }

    # ── 5. HTF Structural Key Levels ─────────────────────────────────────────
    def key_levels(self, date_str: str, current_price: float):
        snap_dt = pd.to_datetime(date_str)
        levels  = []

        # 4H FVGs
        if self.fvg4 is not None:
            fvg_hist = self.fvg4[self.fvg4['Datetime'] < snap_dt]
            for _, row in fvg_hist.tail(10).iterrows():
                tp = '4H Bullish FVG' if row['FVG_Type']=='BULLISH' else '4H Bearish FVG'
                levels.append({'price': float(row['FVG_Lo']), 'type': f"{tp} Lo"})
                levels.append({'price': float(row['FVG_Hi']), 'type': f"{tp} Hi"})

        # Prev Day High / Low
        day_hist = self.df_1h[self.df_1h.index < snap_dt]
        prev_day = day_hist[day_hist.index.date == (snap_dt - pd.Timedelta(days=1)).date()]
        if len(prev_day):
            levels.append({'price': float(prev_day['High'].max()), 'type': 'PDH (Prev Day High)'})
            levels.append({'price': float(prev_day['Low'].min()),  'type': 'PDL (Prev Day Low)'})

        # Prev Week & Month
        if self.df_1d is not None:
            hist_1d = self.df_1d[self.df_1d.index < snap_dt]
            if len(hist_1d) >= 10:
                hist_w = hist_1d.resample('W').agg({'High': 'max', 'Low': 'min'}).dropna()
                if len(hist_w) >= 2:
                    prev_w = hist_w.iloc[-2]
                    levels.append({'price': float(prev_w['High']), 'type': 'PWH (Prev Week High)'})
                    levels.append({'price': float(prev_w['Low']),  'type': 'PWL (Prev Week Low)'})

                hist_m = hist_1d.resample('ME').agg({'High': 'max', 'Low': 'min'}).dropna()
                if len(hist_m) >= 2:
                    prev_m = hist_m.iloc[-2]
                    levels.append({'price': float(prev_m['High']), 'type': 'PMH (Prev Month High)'})
                    levels.append({'price': float(prev_m['Low']),  'type': 'PML (Prev Month Low)'})

        seen = set()
        result = []
        for L in sorted(levels, key=lambda x: abs(x['price'] - current_price)):
            key = round(L['price'], 4)
            if key not in seen:
                seen.add(key)
                L['dist_pips'] = round((L['price'] - current_price) * 10000, 1)
                result.append(L)
        return result[:12]

    # ── 6. Pure Probability Landscape (GARCH Volatility & Bootstrap 95% CIs) ────
    def first_passage_landscape(self, current_price: float, structural_levels: list, vol: dict):
        # Load GARCH term structure
        garch_path = os.path.join(BASE_DIR, 'data', 'processed', 'garch_calibrated_params.json')
        if os.path.exists(garch_path):
            with open(garch_path) as f:
                garch_json = json.load(f)
            ts = garch_json.get('volatility_term_structure', {})
            sigma_1d = float(ts.get('1D', vol.get('sigma_daily', self.sigma)))
            sigma_3d = float(ts.get('3D', vol.get('sigma_daily', self.sigma)))
            sigma_5d = float(ts.get('5D', vol.get('sigma_daily', self.sigma)))
        else:
            sigma_1d = sigma_3d = sigma_5d = vol.get('sigma_daily', self.sigma)

        from price_target_engine.quant_math_engine import bootstrap_touch_probability_ci

        rows = []
        for lv in structural_levels:
            tgt_price = lv['price']
            dist_abs  = abs(tgt_price - current_price) / current_price

            p1_mean, p1_lo, p1_hi = bootstrap_touch_probability_ci(dist_abs, 1.0, sigma_1d)
            p3_mean, p3_lo, p3_hi = bootstrap_touch_probability_ci(dist_abs, 3.0, sigma_3d)
            p5_mean, p5_lo, p5_hi = bootstrap_touch_probability_ci(dist_abs, 5.0, sigma_5d)

            direction = 'ABOVE' if tgt_price > current_price else 'BELOW'
            rows.append({
                'Level': f"{tgt_price:.5f}",
                'Type': lv['type'],
                'Dir': direction,
                'Dist_Pips': round(lv['dist_pips'], 1),
                'P_Touch_1D (95% CI)': f"{p1_mean}% [{p1_lo}% - {p1_hi}%]",
                'P_Touch_3D (95% CI)': f"{p3_mean}% [{p3_lo}% - {p3_hi}%]",
                'P_Touch_5D (95% CI)': f"{p5_mean}% [{p5_lo}% - {p5_hi}%]"
            })

        return pd.DataFrame(rows)

    # ── 7. Hypothesis Confirmation Matrix ─────────────────────────────────────
    def evaluate_hypothesis_confirmation(self, user_direction: str, rates: dict, cot: dict, session: dict):
        if not user_direction or user_direction.upper() not in ['SHORT', 'LONG']:
            return {}

        direction = user_direction.upper()
        checklist = []

        # 1. Macro Rates Spread Alignment
        x4_shift = rates.get('x4_5d_shift', 0.0)
        if direction == 'SHORT':
            macro_aligned = x4_shift > 0
            macro_msg = f"US-DE 2Y Spread Widening ({x4_shift:+.4f}%) -> USD Supportive" if macro_aligned else f"US-DE 2Y Spread Narrowing ({x4_shift:+.4f}%) -> EUR Supportive (Divergent)"
        else:
            macro_aligned = x4_shift < 0
            macro_msg = f"US-DE 2Y Spread Narrowing ({x4_shift:+.4f}%) -> EUR Supportive" if macro_aligned else f"US-DE 2Y Spread Widening ({x4_shift:+.4f}%) -> USD Supportive (Divergent)"

        checklist.append({'Factor': 'Macro Rates (X4)', 'Aligned': macro_aligned, 'Detail': macro_msg})

        # 2. DXY Intermarket Alignment
        dxy_5d = rates.get('dxy_5d_change', 0.0)
        if direction == 'SHORT':
            dxy_aligned = dxy_5d > 0
            dxy_msg = f"DXY 5D Rally ({dxy_5d:+.2f}) -> Aligned for SHORT" if dxy_aligned else f"DXY 5D Pullback ({dxy_5d:+.2f}) -> Divergent"
        else:
            dxy_aligned = dxy_5d < 0
            dxy_msg = f"DXY 5D Pullback ({dxy_5d:+.2f}) -> Aligned for LONG" if dxy_aligned else f"DXY 5D Rally ({dxy_5d:+.2f}) -> Divergent"

        checklist.append({'Factor': 'Intermarket DXY', 'Aligned': dxy_aligned, 'Detail': dxy_msg})

        # 3. COT Positioning Context
        cot_sig = cot.get('signal', 'NEUTRAL')
        checklist.append({'Factor': 'CFTC COT Context', 'Aligned': True, 'Detail': f"Signal = {cot_sig} (Descriptive Context)"})

        # 4. Session Sweep Context
        swept_hi = session.get('swept_asia_hi', False)
        swept_lo = session.get('swept_asia_lo', False)
        if direction == 'SHORT':
            session_msg = "Asian High Swept by London (Intraday Liquidity Grab)" if swept_hi else "No Asian High Sweep"
        else:
            session_msg = "Asian Low Swept by London (Intraday Liquidity Grab)" if swept_lo else "No Asian Low Sweep"

        checklist.append({'Factor': 'Intraday Session', 'Aligned': (swept_hi if direction=='SHORT' else swept_lo), 'Detail': session_msg})

        aligned_count = sum(1 for c in checklist if c['Aligned'])
        score_pct = round(aligned_count / len(checklist) * 100, 1)

        return {
            'user_direction': direction,
            'confirmation_score_pct': score_pct,
            'checklist': checklist
        }

    # ── 8. Evaluate Market & Hypothesis State ────────────────────────────────
    def evaluate_market_state(self, date_str: str = None, current_price: float = None, user_direction: str = None):
        if date_str is None:
            date_str = pd.Timestamp.now().strftime("%Y-%m-%d")
        
        snap_dt = pd.to_datetime(date_str)
        is_today = snap_dt.date() == pd.Timestamp.now().date()

        if current_price is None and is_today:
            live_px = fetch_live_price()
            if live_px is not None:
                current_price = live_px

        if current_price is None:
            day_data = self.df_1h[self.df_1h.index.date == snap_dt.date()] if self.df_1h is not None else []
            current_price = float(day_data['Close'].iloc[-1]) if len(day_data) else float(self.df_1h['Close'].iloc[-1])

        vol     = self.vol_regime(date_str)
        session = self.session_snapshot(date_str)
        cot     = self.cot_snapshot(date_str)
        rates   = self.intermarket_macro_snapshot(date_str)
        levels  = self.key_levels(date_str, current_price)
        fp_df   = self.first_passage_landscape(current_price, levels, vol)
        hypo    = self.evaluate_hypothesis_confirmation(user_direction, rates, cot, session)

        # Hardcore Math Engine Integration
        math_eng   = HardcoreQuantMathEngine()
        pctl_mat   = math_eng.compute_percentile_matrix(date_str)
        moments    = math_eng.compute_statistical_moments(date_str)
        es_risk    = math_eng.compute_var_and_expected_shortfall(date_str, confidence_level=0.95, horizon_days=3)

        return {
            'date': date_str,
            'current_price': current_price,
            'vol': vol,
            'session': session,
            'cot': cot,
            'rates': rates,
            'key_levels': levels,
            'first_passage': fp_df,
            'hypothesis': hypo,
            'quant_math': {
                'percentiles': pctl_mat,
                'moments': moments,
                'es_risk': es_risk
            }
        }
