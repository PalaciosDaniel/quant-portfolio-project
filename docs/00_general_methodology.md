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

--- 

## Specific Ticker Exclusions & Corporate Action Handling
- **Manual Universe Exclusions**: Beyond volume and listing integrity filters, specific tickers are explicitly excluded from the global panel prior to feature engineering due to structural corporate actions:
  - **`SW` & `AMCR`**: Excluded due to severe historical data inconsistencies arising from corporate mergers and synthetic price/volume reconstruction errors.

---

## Forward Target Alignment & Missingness Guardrails

### Forward Horizon Truncation (`Lead NaNs`)
- **Mechanical End-of-Sample Missingness**: Constructing forward targets over an $h$-day horizon (e.g., $h=21$ trading days) introduces $h \times N$ structurally unobservable values at the end of the sample ($t > T_{\text{max}} - h$).
- **Handling Protocol**: Downstream training pipelines must strictly purge the final $h$ cross-sections from feature matrices $X$ and target vectors $y$ prior to model fitting to prevent artificial NaN drop errors or target leakage.

### Raw Scale Target Preservation Policy
- **Absolute Preservation Standard**: Feature transformations (cross-sectional Z-score, uniform rank projection, quantile Winsorization) are restricted strictly to input factors ($X$).
- **Portfolio Optimization Compatibility**: Target return series ($y$) are maintained in their raw percentage return scale across all training, validation, and backtesting pipelines to ensure economic interpretability during shrinkage-based portfolio construction and transaction cost modeling.

---

## Walk-Forward Validation & Combinatorial Purged Cross-Validation (CPCV)

### Temporal Train / Validation / Test Segmentation
- **Development Period (Train/Validation):** 2011-01-03 to 2024-11-27 (~14 years).
- **Transition Buffer (Isolation):** 2024-11-28 to mid-January 2025.
- **Out-of-Sample Backtest (Test):** From mid-January 2025 onward.

### CPCV Block Structure & Hyperparameter Optimization
- **Block Configuration:** The development period is split into **N=7** contiguous blocks of ~2 years each.
- **Pairing Scheme (φ=2):** Each iteration assigns 2 blocks to Validation and 5 to Train, producing **C(7,2) = 21 combinations** and 6 complete synthetic backtest paths, used to evaluate model stability (PBO, DSR).

### Purge & Embargo Protocol
- **Scope:** Trims apply exclusively to the Train set; Validation blocks are left intact.
- **Train → Validation (Purge):** The last **21 trading days** of Train are removed, matching the label horizon (h=21), eliminating forward-return overlap leakage.
- **Validation → Train (Embargo):** The first **11 trading days** of Train are removed to neutralize residual autocorrelation and market memory.
- **Block Continuity:**
  - **Train → Train:** concatenated continuously with no trimming — non-sequential tabular models treat samples as independent observations.
  - **Validation → Validation:** contiguous blocks are merged into a single evaluation block with no intermediate purge; non-contiguous blocks each enforce their own boundaries against Train.
- **Calendar Basis:** Purge/embargo windows and contiguous-segment detection are computed on the **trading-day calendar**, independent of the natural calendar.

#### Boundary Distinction Note: 
While global Forward Horizon Truncation drops the final $h=21$ cross-sections of the entire dataset to eliminate unobservable lead labels, the CPCV Purge applies the identical $h=21$ window locally across internal fold boundaries to prevent cross-fold overlap.

### Final Buffer & Walk-Forward Backtest
- **Final Buffer:** 21 calendar days of purge (December 2024, matching label lifespan) plus 15 calendar days (~11 trading days) of embargo (early January 2025) to guard against year-end distortions.
- **OOS Evaluation:** Runs from mid-January 2025 via **Expanding Window Walk-Forward**, progressively retraining on the full available history since 2011 and reapplying the purge/embargo protocol at each retraining milestone.

---

## OOS Evaluation Window, Data Buffering & Pipeline Isolation
- **Purging & Embargo Buffer:** Development training labels end in early December 2024 to accommodate the final 21-day forward return window. An explicit embargo buffer is enforced through mid-January 2025 (`2025-01-15`) to eliminate label overlap between development and OOS sets.
- **OOS Boundary:** Evaluation window is strictly pinned from `2025-01-15` to `2026-07-10` (~1.5 years / 373 trading sessions) to ensure experiment reproducibility and prevent dynamic window expansion upon pipeline re-execution.
- **Forward Label Buffer:** Price data ingestion extends **+37 calendar days (+21 trading days)** past `OOS_END` (through `2026-09-01`) to eliminate label truncation on terminal OOS test dates.
- **Static Universe Enforcement (`universe_496.json`):** Out-of-sample evaluations enforce static universe matching against the 496 validated development assets via a lightweight manifest, excluding post-2024 index additions to isolate factor signal decay from universe composition changes.
- **Vendor Data Drift & Reproducibility Protocol:** To protect downstream backtesting against retroactive open-source API changes (e.g., missingness caused by corporate actions like ticker deprecations/mergers), downstream inference strictly couples to frozen, fully-validated feature and target artifacts (`X_oos.parquet`, `y_oos.parquet`, $N=496$), treating live vendor queries purely as audit steps.

---

### Out-of-Sample Model Selection Criteria
- **Primary Metric:** Models are evaluated out-of-sample primarily via **Rank Information Coefficient (Rank IC)** and **Rank IC Information Ratio (IR)**. Standard point-prediction loss metrics ($\text{RMSE}$, $\text{MAE}$) are tracked solely to verify scale stability but do not govern model selection.
- **Acceptance Thresholds:** Production candidate signals must achieve an OOS Mean Rank IC $> 0.0200$ with an asymptotic $t$-statistic $> 2.0$ ($p < 0.05$) across non-overlapping execution dates.
- **Signal Parsimony & Regularization:** In the presence of high inter-model rank correlation ($\rho > 0.90$), models exhibiting lower temporal IC volatility ($\sigma_{\text{IC}}$) are selected to ensure portfolio optimization stability downstream.

---

### Cross-Sectional Decile Monotonicity Protocol
- **Tie Resolution in Discrete Signals:** Tree-based predictive models exhibiting discretized forecast spaces are evaluated via cross-sectional percentile ranking (`method="first"`) prior to decile assignment to ensure balanced bucket sizing.
- **Monotonicity Verification Standard:** Alpha signals are subjected to a two-stage Spearman rank monotonicity test across deciles:
  1. **Full Universe ($D_1\text{--}D_{10}$):** Tests global sorting capability.
  2. **Truncated Universe ($D_1\text{--}D_9$):** Isolates systemic sorting skill from top-decile ($D_{10}$) acceleration. A signal is verified as robust only if the truncated Spearman correlation preserves at least 80% of full-universe rank monotonicity.
