# EUR/USD Quant Research Log

## Overview
Rigorous quantitative analysis of EUR/USD short-term momentum exhaustion, volatility filtering, cross-asset divergence (DXY), interest rate differentials (US 2Y vs DE 2Y), and point-in-time backtesting.

---

## Stage 1 — Base distribution
- **Goal:** Analyze basic return distributions, daily log-returns, and 5-day rolling log-returns ($5D\_return = \ln(P_t / P_{t-5})$).
- **Status:** COMPLETED

---

## Stage 2 — Volatility gate
- **Goal:** Filter out noise by requiring a high volatility environment (20D rolling std in top 25% over a 3-year / 756-day rolling window).
- **Status:** COMPLETED

---

## Stage 3 — First Trigger & Decision Engine v1.0
- **Goal:** Build 3-Layer Decision Engine (`decision_engine/engine.py`) for LONG vs SHORT vs NO TRADE.
- **Status:** COMPLETED

---

## Stage 3A — Deconstruct OOS SHORT Failure Mode (2023–2026, N=59)
- **Goal:** Perform forensic empirical breakdown of the 59 SHORT signals during 2023–2026 to pinpoint exact failure modes.
- **Status:** COMPLETED
- **Diagnostic Findings:**

### 1. Yearly Breakdown
| Year | N | 3D Mean (%) | 3D Med (%) | 3D Win Rate (%) | 5D Mean (%) | 5D Win Rate (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **2023** | 17 | -0.12% | -0.05% | **52.9%** | -0.05% | 47.1% | Functional |
| **2024** | 7 | +0.11% | +0.29% | **28.6%** | -0.01% | 57.1% | Breakdown |
| **2025** | 32 | +0.14% | +0.04% | **40.6%** | +0.22% | 43.8% | Breakdown |
| **2026** | 3 | +0.32% | +0.12% | **33.3%** | +0.79% | 0.0% | Breakdown |

### 2. Rates ($X_4$) & Volatility ($X_2$) Sub-Regime Breakdown
| Sub-Regime | N | 3D Mean (%) | 3D Win Rate (%) | 5D Mean (%) | 5D Win Rate (%) | Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Crisis Vol ($X_2 \ge 90\%$)** | 11 | **-0.18%** | **54.5%** | **-0.23%** | **63.6%** | **EXHAUSTION STILL WORKS** |
| **High Vol ($75\% \le X_2 < 90\%$)** | 43 | +0.11% | **34.9%** | +0.27% | **37.2%** | **FAILURE MODE** |
| **$X_4 > +0.05$ (Rates USD)** | 45 | +0.09% | **44.4%** | +0.09% | 51.1% | **DECOUPLED FROM YIELD** |

### Core Root Cause Analysis:
1. **The Fed Rate Cut Pivot Regime (2024–2026):** After 2023, US monetary policy entered the terminal rate / rate cut anticipation regime. Market momentum during EUR surges was driven directly by **US Macro Economic Surprises (CPI, NFP, ISM misses vs consensus)** rather than static 2Y yield differentials ($X_4$).
2. **Volatility Threshold Shift:** Under moderate high vol ($75\% \le X_2 < 90\%$), technical exhaustion failed (win rate 34.9%), whereas under extreme crisis vol ($X_2 \ge 90\%$), mean reversion preserved a 63.6% 5D win rate.
3. **Actionable Roadmap:** Requires $X_5 = \text{US Macro Surprise} = \frac{\text{Actual}_t - \text{Consensus}_t}{\sigma}$ to capture Fed repricing impulses before attempting price target/imbalance engines.
