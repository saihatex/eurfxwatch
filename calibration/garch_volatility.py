"""
GARCH(1,1) Dynamic Volatility Forecasting & Calibration Module
================================================================
Fits GARCH(1,1) model via Maximum Likelihood Estimation (MLE):
  sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

Provides:
  - Dynamic daily volatility forecast sigma_t
  - Persistence parameter (alpha + beta)
  - Term structure forecast sigma(T) for T = 1D, 3D, 5D horizons
  - Volatility clustering regime detection
Saves: data/processed/garch_calibrated_params.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_1H    = os.path.join(BASE_DIR, 'data', 'raw', 'EURUSD_1H.csv')
PATH_1D    = os.path.join(BASE_DIR, 'data', 'raw', 'EURUSD_1D.csv')
OUT_PARAMS = os.path.join(BASE_DIR, 'data', 'processed', 'garch_calibrated_params.json')


def garch_log_likelihood(params, returns):
    omega, alpha, beta = params
    n = len(returns)

    # Constraint check: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
    if omega <= 1e-8 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.999:
        return 1e10

    var = np.zeros(n)
    var[0] = np.var(returns)

    for t in range(1, n):
        var[t] = omega + alpha * (returns[t-1]**2) + beta * var[t-1]
        if var[t] <= 1e-10:
            var[t] = 1e-10

    # Negative log-likelihood under normal innovation assumption
    log_lik = -0.5 * np.sum(np.log(2.0 * np.pi * var) + (returns**2) / var)
    return -log_lik


def fit_garch_1_1(returns: np.ndarray):
    """Fits GARCH(1,1) model via Scipy MLE."""
    var_init = np.var(returns)
    omega_init = var_init * 0.05
    alpha_init = 0.05
    beta_init  = 0.90

    res = minimize(
        garch_log_likelihood,
        x0=[omega_init, alpha_init, beta_init],
        args=(returns,),
        method='L-BFGS-B',
        bounds=[(1e-8, None), (1e-4, 0.30), (0.50, 0.98)]
    )

    omega, alpha, beta = res.x
    persistence = alpha + beta
    long_run_var = omega / (1.0 - persistence) if persistence < 1.0 else var_init
    long_run_vol = np.sqrt(long_run_var)

    # Filtered conditional variance series
    n = len(returns)
    cond_var = np.zeros(n)
    cond_var[0] = var_init
    for t in range(1, n):
        cond_var[t] = omega + alpha * (returns[t-1]**2) + beta * cond_var[t-1]

    current_vol = np.sqrt(cond_var[-1])

    return {
        'omega': float(omega),
        'alpha': float(alpha),
        'beta': float(beta),
        'persistence': float(persistence),
        'long_run_daily_vol': float(long_run_vol),
        'current_daily_vol': float(current_vol),
        'cond_var_series': cond_var
    }


def forecast_garch_term_structure(garch_res: dict, max_horizon_days: int = 5):
    """
    Forecasts multi-day cumulative volatility term structure sigma(T) under GARCH(1,1):
      E[sigma_{t+k}^2] = V_L + (alpha + beta)^k * (sigma_t^2 - V_L)
    """
    omega = garch_res['omega']
    alpha = garch_res['alpha']
    beta  = garch_res['beta']
    v_L   = garch_res['long_run_daily_vol']**2
    v_t   = garch_res['current_daily_vol']**2

    term_structure = {}
    cum_var = 0.0

    for k in range(1, max_horizon_days + 1):
        v_k = v_L + ((alpha + beta)**k) * (v_t - v_L)
        cum_var += v_k
        # Per-day effective volatility over T days horizon
        eff_daily_vol = np.sqrt(cum_var / k)
        term_structure[f'{k}D'] = float(eff_daily_vol)

    return term_structure


def run_garch_calibration():
    print("="*80)
    print("     GARCH(1,1) DYNAMIC VOLATILITY CALIBRATION")
    print("="*80)

    if not os.path.exists(PATH_1D):
        print(f"Error: {PATH_1D} not found!")
        return

    df_1d = pd.read_csv(PATH_1D, index_col=0, parse_dates=True).sort_index()
    close = df_1d['Close'].dropna().to_numpy()
    log_returns = np.diff(np.log(close))

    # Fit on last 1000 daily returns (~4 years)
    returns_fit = log_returns[-1000:]
    garch_fit = fit_garch_1_1(returns_fit)
    term_struct = forecast_garch_term_structure(garch_fit, max_horizon_days=5)

    print(f"\nGARCH(1,1) Estimated Parameters:")
    print(f"  omega (base variance): {garch_fit['omega']:.8f}")
    print(f"  alpha (ARCH shock):    {garch_fit['alpha']:.4f}")
    print(f"  beta  (GARCH memory):   {garch_fit['beta']:.4f}")
    print(f"  Persistence (α + β):    {garch_fit['persistence']:.4f}  (Half-life of vol shock: {np.log(0.5)/np.log(garch_fit['persistence']):.1f} days)")
    print(f"  Long-Run Daily Vol:    {garch_fit['long_run_daily_vol']:.6f} ({garch_fit['long_run_daily_vol']*10000:.1f} pips/day)")
    print(f"  Current Dynamic Vol:   {garch_fit['current_daily_vol']:.6f} ({garch_fit['current_daily_vol']*10000:.1f} pips/day)")

    print(f"\nGARCH Volatility Term Structure Forecast σ(T):")
    for horizon, vol_val in term_struct.items():
        print(f"  {horizon} Forecast Daily Vol: {vol_val:.6f} ({vol_val*10000:.1f} pips/day)")

    output_data = {
        'garch_params': {
            'omega': garch_fit['omega'],
            'alpha': garch_fit['alpha'],
            'beta': garch_fit['beta'],
            'persistence': garch_fit['persistence'],
            'long_run_daily_vol': garch_fit['long_run_daily_vol'],
            'current_daily_vol': garch_fit['current_daily_vol']
        },
        'volatility_term_structure': term_struct
    }

    with open(OUT_PARAMS, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved calibrated GARCH parameters to {OUT_PARAMS}")

if __name__ == '__main__':
    run_garch_calibration()
