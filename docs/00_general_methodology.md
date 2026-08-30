# General Methodology & Global Pipeline Decisions

## Microstructure & Trade Execution Risk Assumptions

### Zero-Volume and Trading Suspensions
- **Systematic Assumption:** Days with zero recorded trading volume (attributable to exchange-mandated trading halts, extreme illiquidity, or local holidays) are treated as non-tradable periods.
- **Backtesting Guardrails:**
  - Signals generated on zero-volume days cannot execute orders on that day.
  - Price series on zero-volume days maintain stale closing prices, preventing false volatility/return signals from triggering rebalancing events.
  - Portfolio construction algorithms enforce liquidity filters to prevent allocation to halted or zero-volume assets.

---

## Dynamic Universe & Survivorship Considerations
- **Unbalanced Panel Framework:** To mitigate survivorship and selection bias, the cross-sectional pipeline supports a dynamically varying constituent count over time rather than restricting analyses to assets present across the entire sample period.

---

## Dynamic Universe & Missingness Integrity

### Temporal Missingness Logic & IPO Handling
- **Missing Value Origin:** Missing values ($\text{NaN}$) in cross-sectional return panels originate strictly from listing date boundaries (IPOs) or initial calculation lookback windows.
- **Survivorship & Look-Ahead Guardrails:** Pre-IPO null values for assets listing mid-sample (e.g., ABNB, ZTS) are preserved as $\text{NaN}$ and strictly excluded from cross-sectional rankings at date $t$. Imputing pre-IPO values or back-filling prices introduces look-ahead bias and breaks unbalanced panel integrity.
- **Lookback NaN Integrity:** Moving-window calculations generate structured leading $\text{NaN}$s corresponding to the parameter length (e.g., $252$ days for 12-1 momentum; $21$ days for 1-month reversal). Downstream models must drop or handle leading initialization windows explicitly during alignment.

--- 

## Cross-Sectional Feature Preprocessing Requirements

### Skewness & Heavy-Tail Handling Protocols
- **Raw Storage Policy:** All raw factor calculations (e.g., Amihud illiquidity, raw volatility ratios) are stored in `data/processed/` in their natural mathematical scale to preserve auditability.
- **Downstream Preprocessing Pipeline:** Features exhibiting extreme right-skewness or multi-order-of-magnitude dispersion must undergo monotonic log-transformations ($\ln(X + \epsilon)$), cross-sectional Winsorization (e.g., at 1st and 99th percentiles), and cross-sectional $Z$-score standardization prior to training ML models (LightGBM / Elastic Net).

--- 

## Global Sample Start Truncation (Warm-Up Alignment)
- **Sample Truncation Standard:** To prevent variable cross-sectional sample sizes driven by lookback initialization, all feature panels are truncated globally at $T_{\text{init}} = 252$ trading days from sample start.
- **Panel Alignment Guardrail:** Cross-sectional signals, ML training matrices, and backtesting pipelines operate strictly on dates $t \ge T_{\text{init}}$, ensuring that all long-lookback features (such as 12-1 momentum and 252-day volatility metrics) have reached steady-state calculation integrity without back-filling or leading imputations.

---

## Cross-Sectional Feature Preprocessing & Selection Standards

### Quantitative Feature Screening Rules
Before feature combination or ML training, raw factor candidates must satisfy strict diagnostic criteria:

* **Information Coefficient Screening:** Retain features exhibiting statistically significant rank predictive power ($|t\text{-stat}| > 2.0$) over the targeted forecast horizon.
* **Redundancy & Multicollinearity Filtering:** Features exhibiting $\text{VIF} > 10$ undergo intra-family screening; the candidate with the highest Information Ratio ($\text{ICIR}$) is retained while redundant proxies are dropped.
* **Signal Standardization:** Retained factors undergo non-parametric log transformation (where scale dispersion demands), cross-sectional Winsorization at 1%/99% quantiles to bound tail leverage, and daily cross-sectional Z-score standardization ($\mu=0, \sigma=1$).