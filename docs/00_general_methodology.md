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

## Out-of-Sample Model Selection Criteria
- **Primary Metric:** Models are evaluated out-of-sample primarily via **Rank Information Coefficient (Rank IC)** and **Rank IC Information Ratio (IR)**. Standard point-prediction loss metrics ($\text{RMSE}$, $\text{MAE}$) are tracked solely to verify scale stability but do not govern model selection.
- **Acceptance Thresholds:** Production candidate signals must achieve an OOS Mean Rank IC $> 0.0200$ with an asymptotic $t$-statistic $> 2.0$ ($p < 0.05$) across non-overlapping execution dates.
- **Signal Parsimony & Regularization:** In the presence of high inter-model rank correlation ($\rho > 0.90$), models exhibiting lower temporal IC volatility ($\sigma_{\text{IC}}$) are selected to ensure portfolio optimization stability downstream.

---

## Cross-Sectional Decile Monotonicity Protocol
- **Tie Resolution in Discrete Signals:** Tree-based predictive models exhibiting discretized forecast spaces are evaluated via cross-sectional percentile ranking (`method="first"`) prior to decile assignment to ensure balanced bucket sizing.
- **Monotonicity Verification Standard:** Alpha signals are subjected to a two-stage Spearman rank monotonicity test across deciles:
  1. **Full Universe ($D_1\text{--}D_{10}$):** Tests global sorting capability.
  2. **Truncated Universe ($D_1\text{--}D_9$):** Isolates systemic sorting skill from top-decile ($D_{10}$) acceleration. A signal is verified as robust only if the truncated Spearman correlation preserves at least 80% of full-universe rank monotonicity.

---

## Portfolio Rebalancing Frequencies & Horizon Misalignment
- **Target Horizon vs. Rebalance Frequency**: Predictions target a 21-day forward return horizon ($\text{forward\_return\_21d}$). While the default baseline rebalance frequency is matched to the signal horizon ($\Delta t = 21\text{ sessions}$), rebalance frequencies are treated as a distinct decision variable from target generation.
- **Trade-off Mechanism**: Higher frequency rebalancing captures fresh rank changes but accelerates turnover and transaction costs. Lower frequency rebalancing reduces trading drag at the risk of alpha decay. Alternative rebalancing schedules ($\Delta t \in \{1, 5, 10, 21\}$ sessions) are tested to evaluate the net turnover-versus-decay curve.

---

## Model Space Exclusions
- **Elastic Net Omission Criteria**: Elastic Net ($\text{L}_1 / \text{L}_2$ penalization) is excluded from portfolio generation pipelines when feature pre-selection already enforces strict low inter-factor collinearity. In low-dimensional factor sets ($\le 3$ orthogonalized features), Elastic Net collapses to Ridge-equivalent allocations without producing distinct alpha signals.

---

## Exposure & Neutrality Definitions

### Long-Short Weight Allocation & Neutrality
- **Target Exposure Scaling:** For long-short strategies, long legs are assigned target exposure $E_{\text{long}} = +0.5$ and short legs $E_{\text{short}} = -0.5$.
- **Per-Side Equal Weighting:** On each rebalancing date $t$, position weights for long assets $N_{\text{long}, t}$ and short assets $N_{\text{short}, t}$ are computed independently:
  $$w_{i, t}^{\text{long}} = \frac{E_{\text{long}}}{N_{\text{long}, t}}, \quad w_{j, t}^{\text{short}} = \frac{E_{\text{short}}}{N_{\text{short}, t}}$$
- **Dynamic Rebalancing:** Asymmetries in cross-sectional asset availability across long and short quantiles are re-scaled daily to guarantee strict **net exposure neutrality** ($\sum w_i = 0$) and constant **gross exposure** ($\sum |w_i| = 1.0$).

---

## Rationale for Explicit Weight Caps (motivates Block 8 operational constraints)

Even simple, unconstrained heuristic weighting schemes are vulnerable to severe concentration from data anomalies. In the portfolio-construction backtest, both **Inverse Volatility** and **Risk Parity** (neither of which carries an explicit position cap) produced single-asset weight spikes far above their typical range — up to **22.83%** in one case — when a single asset's estimated volatility or covariance behaved idiosyncratically, without any reduction in the number of active positions. This confirms that explicit position caps are necessary for *any* weighting scheme, not only optimizer-based ones, and justifies applying them uniformly across all portfolios in the operational-constraints stage rather than relying on a scheme's own construction to self-limit concentration.

--- 

## Operational Constraints Framework (motivates Block 8 in the portfolio-construction notebook)

Not every constraint commonly used in institutional portfolio management is relevant to this project's experimental design. The framework applied uniformly to every portfolio (across models and universes) is organized into three categories:

**Selected (core framework), applied to every portfolio:**
- **Maximum position weight** ($w_i \le w_{\max} = 5\%$) — mandatory; extreme-concentration cases observed pre-constraint (up to a 45.92% single-asset weight and 0.2779 HHI) confirm this is needed to bound idiosyncratic risk and ensure real diversification.
- **Minimum effective position weight** ($w_i \ge w_{\min} = 0.5\%$, else $w_i = 0$) — a post-optimization cleanup rule rather than a hard optimizer constraint, removing numerical "dust" positions that are operationally negligible but costly to hold/execute.
- **Turnover control** — since portfolios rebalance every 21 days on dynamic ML signals across several weighting philosophies, turnover is the key variable connecting theoretical performance to net-of-cost economic viability.

**Discarded as redundant or irrelevant:**
- **Exposure constraints** — not added separately, since leverage and net/gross exposure are already fixed by the Long-Only / Long-Short construction mandate itself.
- **Maximum cardinality** — redundant with the quantile-based selection cutoffs (Top 10/20/30%).
- **Liquidity limits and tracking error** — not meaningful given the highly liquid S&P 500 universe and the absence of a benchmark-replication objective.

**Discarded as structurally complex (future work):**
- **Sector constraints** (would require a GICS-style taxonomy and per-industry limits) and **factor neutralization** (Fama-French/Barra, requiring continuous beta estimation) are both deferred to limitations/future work, since they would shift focus away from the project's central goal of evaluating the raw ML signal and allocation schemes.

### Data Traceability Protocol

To preserve experimental auditability, the original unconstrained portfolio allocations are retained in `portfolio_weights`. All downstream operational overlays (box constraints, min-weight cleanup, and turnover smoothing) generate a distinct, parallel data structure named `constrained_portfolio_weights`.

--- 

## Turnover Diagnostics & Hysteresis Justification  

Before introducing hysteresis buffer zones, total turnover across portfolios is decomposed into two distinct sources:
1. **Re-optimization / Continued Adjustment Turnover:** Weight changes among assets already present in the portfolio at $t-1$.
2. **Marginal Selection Turnover:** Turnover driven by complete entries and exits at the universe boundary. This is further split into marginal oscillations (within $\pm 3$ percentile points of the selection cutoff) vs. deep non-marginal entries/exits.

**Empirical Findings & Selective Application:**
- **Risk-Based Schemes (Inverse Volatility & Risk Parity):** Individual risk metrics and risk-contribution figures fluctuate gradually but noisily near the cutoff. Under strict selection rules (Top 10%), marginal oscillations account for **35% to 49%** of total turnover without reflecting genuine changes in conviction. Applying a buffer zone ($\theta_{\text{out}} = \theta_{\text{in}} - 0.03$) stabilizes universe membership and substantially reduces execution friction.
- **Optimized Schemes (Maximum Sharpe) & Signal Weighting:** Turnover is dominated by internal weight re-optimization driven by shifting active alpha forecasts ($\hat{\alpha}$) and covariance updates, or by signal re-ranking across the portfolio. Because buffer zones act exclusively on universe filtering rather than optimization weights, they provide negligible turnover relief here while adding unnecessary hyperparameter complexity.

Consequently, buffer zones are applied selectively only to Risk-Based Top 10% portfolios, preserving method simplicity elsewhere.

---

## Turnover Measurement Convention (extends existing entry — adds the drift-adjusted formula)

Two distinct turnover formulations are used across the project, deliberately:

- **Diagnostic turnover** (buffer-zone decision, Section 7.1): computed directly on theoretical weights between consecutive rebalances, without market price drift, to isolate signal/selection-rule noise from market movement.

- **Operational turnover** (Block 8 turnover cap, transaction-cost estimation, and net-return computation): incorporates **market drift**. At each rebalance $t_k$, turnover is measured against the weight vector the portfolio has actually drifted to since $t_{k-1}$ — not against the previous target weight:
  
  $$T_k = \frac12\sum_i \left| w_{i,t_k} - w_{i,t_k}^{\text{drifted}} \right|, \qquad w_{i,t_k}^{\text{drifted}} = \frac{w_{i,t_{k-1}}\left(1+R_{i,t_{k-1}\to t_k}\right)}{1+R_{p,t_{k-1}\to t_k}}$$
  
  where $R_{i,t_{k-1}\to t_k}$ is asset $i$'s compounded return over the holding period and $R_{p,t_{k-1}\to t_k} = \sum_i w_{i,t_{k-1}} R_{i,t_{k-1}\to t_k}$ is the portfolio's buy-and-hold return over that same interval. This captures the true operational rotation required to move the portfolio from its live market position to the new target vector — for both long-only and long-short constructions. Long-Only renormalizes to total exposure; each Long-Short leg renormalizes independently to preserve neutrality. This single definition governs both the Block 8 turnover cap and the transaction-cost model in the execution/backtesting notebook — it is not redefined per notebook.

---

## Turnover Smoothing — Design Choice

Turnover is capped via **post-hoc linear interpolation** between the prior realized weight vector and the new raw target, rather than via a transaction-cost penalty term inside the optimizer itself (consistent with DeMiguel et al., 2009; Fastrich et al., 2015). Reasons: (1) avoids non-convex/non-differentiable terms that could break optimizer convergence across walk-forward iterations, (2) preserves the model's optimal weight direction while only limiting transition speed, (3) keeps execution logic modular and applies uniformly across every weighting scheme without touching model estimation. The externally-facing turnover target is 30%, but the internal design cap is set to **25%** to leave headroom for the downstream box-constraint cleanup (min/max weight enforcement), which otherwise pushes effective turnover above the nominal cap.

### Design Headroom in Turnover Constraints (`MAX_TURNOVER_DESIGN`)

While the institutional turnover target is capped at **30% per 21-day rebalance**, the internal smoothing algorithm enforces a stricter design threshold of **`MAX_TURNOVER_DESIGN = 25%`**. This $5\%$ safety buffer absorbs the inevitable weight expansion (turnover dilation) caused by downstream post-processing steps (pruning micro-positions below $0.5\%$ and iteratively re-capping positions that exceed $5.0\%$). Setting the internal interpolation target to 25% ensures that the final executable portfolio strictly 
complies with the headline 30% turnover ceiling.

---

## Transaction Cost Assumptions

Transaction costs are modeled as a flat linear cost applied to traded volume per 21-day rebalance, evaluated under three scenarios: **conservative (10 bps)**, **base (15 bps)**, **stressed (20 bps)**. The base case (0.15%) is the standard assumption used for headline cost figures in this project.

**Future work (not implemented):** regime-dependent dynamic costs (wider bid-ask spreads during volatility spikes/sell-offs), non-linear price-impact models scaled by average daily volume (ADV), and asymmetric/fixed per-order costs for full position liquidation — noted as directions for more realistic microstructure modeling.

--- 

## Execution Lag Convention & Timing Framework

To strictly eliminate look-ahead bias, portfolio targets computed from information available through trading session $t_k$ are never assumed tradable or active at $t_k$. Execution takes effect exactly one trading session later at $t_k+1$ (close of $t_k$ / open of $t_k+1$):

$$\boldsymbol{w}_{t_{k}}^{\text{target}} = \mathcal{F}(\text{Information up to } t_k)$$

$$\boldsymbol{w}_{t_{k}^+}^{\text{executed}} = \boldsymbol{w}_{t_{k}}^{\text{target}} \quad \text{applied at close of } t_k \text{ (or open of } t_k + 1\text{)}$$

This lag convention is enforced uniformly across all backtesting routines, signal applications, and transaction cost calculations.

---

## Rebalancing Calendar & Intra-Period Buy-and-Hold Dynamics

- **Rebalancing Frequency:** Fixed at **21 trading sessions** ($\text{REBALANCING\_FREQUENCY} = 21$), strictly aligned with the 21-day forward synthetic prediction target horizon (`forward_return_21d`).
- **Signal Date ($t_k$):** The recorded rebalance timestamp in allocation datasets, marking the session close when prices and forecasts are finalized to build $\boldsymbol{w}_{t_{k}}^{\text{target}}$.
- **Intra-Period Holding Horizon ($t_k + 1 \to t_{k+1}$):** Executed holdings generate portfolio returns from $t_k + 1$ through the next rebalance date $t_{k+1}$.
- **Intra-Period Weight Drift:** Between rebalances, no intermediate trading occurs. Weights drift organically with relative asset performance:

$$w_{i, t+1} = \frac{w_{i, t} \cdot (1 + R_{i, t+1})}{1 + R_{p, t+1}}$$

where $R_{i, t+1}$ is asset $i$'s daily return and $R_{p, t+1} = \sum_{i} w_{i, t} R_{i, t+1}$ is the daily gross portfolio return. Asset weights reset to target only at scheduled rebalancing events ($t_{k+1}$).

---

## All-In Transaction Cost Model

To evaluate real-world economic viability without over-parameterizing unavailable intraday microstructure metrics (order-book depth, asset-level ADV), transaction costs are modeled via a consolidated linear drag proportional to rebalance turnover:

$$C_t = T_t \times c$$

where $T_t$ represents the drift-adjusted rebalance turnover and $c$ is the single all-in fee coefficient bundling three core operational frictions:
1. Explicit brokerage execution fees.
2. Implicit bid-ask spread crossing costs.
3. Market impact and execution slippage.

**Sensitivity Scenarios:**
- **Conservative Scenario:** $10\text{ bps}$ ($c = 0.0010$).
- **Base Scenario (Reference Case):** $15\text{ bps}$ ($c = 0.0015$) — headline benchmark for net performance figures.
- **Stressed Scenario:** $20\text{ bps}$ ($c = 0.0020$).