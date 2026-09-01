# 04. Factor Validation — Theoretical Documentation

## Factor Initialization & Lookback Warm-Up Rules

Factor calculation lookbacks vary across style metrics based on underlying statistical and financial definitions:

* **Momentum (12-1) & Rolling Volatility:** Require a strict 252-day historical price sequence ($t-252$ to $t-21$ for momentum; $t-252$ to $t$ for volatility).
* **Short-Term Reversal:** Uses a 21-day lookback window driven by its fixed 21-day lag requirement ($t-21$).
* **Directional Volatilities (Upside/Downside):** Set a 252-day window but permit evaluation with a minimum threshold of 21 signed-return days ($R_t > 0$ or $R_t < 0$) to prevent excessive missingness during persistent single-direction market trends.
* **Amihud Illiquidity:** Uses a 21-day rolling window requiring a minimum threshold of 15 valid non-zero volume trading sessions.

### Methodological Design & Minimum Window Rationale

The trade-off between warm-up latency and estimation robustness was explicitly evaluated when setting these rules:

* **Strict Full-Window Requirement (Rolling Volatility):** While a minimum threshold of 30 or 60 days could theoretically be allowed to produce early volatility estimates, a strict 252-day non-null window is enforced. Given multi-year price availability across the investment universe and zero internal NaNs for this factor, forfeiting the initial 251 trading days is an acceptable trade-off to guarantee that every active volatility factor value is statistically robust.
* **Relaxed Thresholds (Amihud & Directional Volatilities):** Conversely, minimum sample rules (15 days for Amihud; 21 signed-return days for directional volatilities) are essential to prevent structural data dropouts, especially for directional metrics where persistent market trends drastically reduce the count of positive or negative sessions within a standard window.

---

## Asset Exclusion Criteria: Synthetic & Restructured Series

Assets exhibiting synthetic historical reconstructions following M&A, spinoffs, or corporate restructurings (e.g., `AMCR`, `SW`) display zero-volume trading halts and artificial price flatlines. Including these assets distorts cross-sectional statistical properties, introduces false liquidity spikes, and skews volatility factor ranks. Asset removal at the preprocessing layer is mandatory.

---

## Empirical Distribution & Outlier Diagnostics Rationale

### Cross-Sectional Dispersion Dynamics: $\text{IQR}_t$ vs. $\sigma_t$

Evaluating the temporal evolution of cross-sectional dispersion is essential to verify factor persistence and diagnose market-wide shock transmission mechanisms:

* **Factor Discrimination Power:** Monitoring the cross-sectional Interquartile Range ($\text{IQR}_t$) over time ensures that factor spread does not collapse to zero ($\text{IQR}_t > 0$), confirming that the metric retains continuous sorting and stock-ranking capability across all market regimes.
* **Parametric vs. Non-Parametric Dispersion Sensitivity:** Standard deviation ($\sigma_t$) measures total dispersion but is highly sensitive to extreme cross-sectional outliers. Conversely, $\text{IQR}_t$ captures the dispersion of the core 50% of the distribution. A divergent spike where $\sigma_t$ expands significantly faster than $\text{IQR}_t$ signals that market stress is concentrated in distribution tails (isolated extreme asset movements). Synchronous expansions in both metrics indicate systemic, universe-wide structural shifts.

### Non-Parametric Outlier Identification via Tukey Fences

Applying parametric outlier thresholds ($\mu \pm 3\sigma$) to cross-sectional financial factors is methodologically invalid due to widespread heavy tails (kurtosis up to 13+ in volatilities and extreme values in raw momentum). We adopt Tukey's interquartile rule:


$$\text{Outlier Bounds} = [\text{Q1} - 1.5 \times \text{IQR}, \, \text{Q3} + 1.5 \times \text{IQR}]$$


This non-parametric approach robustly quantifies extreme values without assuming underlying Gaussianity.

### Log-Transformation of Micro-Scale Factors

Raw Amihud illiquidity measures price impact per unit of dollar volume:


$$\text{Illiquidity}_{i,t} = \frac{\vert{}R_{i,t}\vert{}}{\text{Volume}_{i,t} \times \text{Price}_{i,t}}$$


Because low-volume sessions push the denominator toward zero, the raw metric spans multiple orders of magnitude with extreme positive skewness ($>80$). A monotonic base-10 log transformation is mandatory:


$$\text{Amihud}_{\text{log}, i, t} = \log_{10}(\text{Illiquidity}_{i,t} + \epsilon)$$


This compresses multi-order dispersion into a unimodal distribution suited for linear models and tree-based splits.

### Leverage Effect in Directional Volatility Diagnostics

Empirical findings confirm asymmetric variance dynamics (the leverage effect): market drops trigger larger volatility jumps than equivalent market rises. Downside volatility displays higher cross-sectional kurtosis (13.84 vs. 9.34) and outlier rates (5.60% vs. 4.36%) than upside volatility, justifying their separate inclusion as distinct risk factors.

--- 

## Temporal Memory & Signal Decay Diagnostics

### Spearman Rank Decay Dynamics ($k$-Lag Stability)
To evaluate cross-sectional ranking stability across investment horizons $k \in \{1, 2, \dots, 63\}$, daily Spearman rank correlations are computed between $t$ and $t-k$:
$$\rho_s(k) = \text{Corr}\left(\text{Rank}(F_{t}), \, \text{Rank}(F_{t-k})\right)$$
- **High-Persistence Signals ($\rho_s(21) > 0.85$):** Rolling volatility, Amihud illiquidity, and 12-1 momentum maintain cross-sectional sorting power across multi-week horizons. Portfolio rebalancing for these metrics can operate at lower frequencies without suffering significant signal decay.
- **Fast-Decaying Signals ($\rho_s(5) < 0.50$):** Short-term reversal loses cross-sectional memory rapidly ($\rho_s(21) \approx 0$). Exploitability of fast signals is strictly governed by transaction cost drag relative to decay speed.

---

## Multicollinearity & Factor Redundancy Diagnostics

### Cross-Sectional Average Spearman Correlation Structure
Cross-sectional inter-factor dependence is audited via the average daily Spearman rank correlation matrix $\bar{\rho}_{i,j}$:
$$\bar{\rho}_{i,j} = \frac{1}{T} \sum_{t=1}^{T} \text{Corr}\left(\text{Rank}(F_{i,t}), \, \text{Rank}(F_{j,t})\right)$$
- **Why Spearman Rank Correlation:** Portfolio sorting and long-short quantile strategies rely on relative stock ordering rather than absolute linear magnitudes. Spearman correlation evaluates monotonicity while remaining completely robust to heavy-tailed non-Gaussian distribution features and extreme outliers. Averaging cross-sectional matrices daily isolates stable factor relationships from time-varying market regimes.

### Hierarchical Factor Clustering & Dendrogram Topology
To move beyond pairwise visual inspection, inter-factor relationships are mapped into a non-parametric metric distance space:
$$D(i, j) = 1 - |\bar{\rho}_{i,j}|$$
Using average linkage hierarchical clustering on $D(i,j)$, the factor universe is partitioned into an empirical dendrogram topology.
- **Why Hierarchical Clustering:** Transforming correlation into a metric distance allows unsupervised, quantitative grouping of factors. This visual topology identifies functional redundancy (isolating distinct risk, liquidity, momentum, and reversal sub-spaces) and prevents over-representing a single latent risk dimension in signal generation pipelines.

### Variance Inflation Factor ($\text{VIF}$) Protocols
Cross-sectional multicollinearity is audited daily using the Variance Inflation Factor for factor $j$:
$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
where $R_j^2$ is the coefficient of determination obtained by regressing factor $j$ against all remaining factors in the cross-section at time $t$.
- **Why VIF Diagnostics:** Severe multicollinearity artificially inflates estimator variance in multi-factor alpha models, destabilizing factor weights and inducing extreme turnover. Tracking $\text{VIF}_t$ dynamically diagnoses structural overlap.
- **Threshold Policy:** Metrics displaying persistent $\text{VIF} > 10$ (observed in `rolling_volatility`, `upside_volatility`, and `downside_volatility`) indicate severe redundancy and are flagged for feature selection, dropping, or dimensionality reduction prior to linear model fitting.


---

## Preliminary Predictive Power & Factor Selection Architecture

### Information Coefficient (IC) & Predictive Evaluation Protocols

The Information Coefficient (IC) framework serves as the definitive quantitative gatekeeper prior to model fitting, evaluating a factor's capacity to accurately forecast the cross-sectional ranking order of 21-day forward returns ($R_{i, t+1:t+21}$).

Predictive evaluation relies on daily cross-sectional Spearman rank correlations ($\text{IC}_t$) between factor values $F_{i,t}$ and forward returns:


$$\text{IC}_t = \text{Corr}\left(\text{Rank}(F_{t}), \, \text{Rank}(R_{t+1:t+21})\right)$$


This non-parametric approach captures monotonic predictive relationships without imposing linear constraints or suffering sensitivity to cross-sectional outliers.

#### Multi-Metric Diagnostic Rationale

To robustly characterize the full distribution of daily predictive power, seven complementary metrics are tracked:

* **Mean IC:** Quantifies the average predictive power and overall directional sign (positive vs. inverse alpha) of the factor signal across the full sample.
* **Median IC:** Serves as a non-parametric baseline for signal strength, ensuring that average performance is not artificially driven by localized outlier regimes.
* **Std IC ($\sigma_{\text{IC}}$):** Measures the temporal volatility of predictive power; lower values indicate stable, low-noise forecasting power across market cycles.
* **Information Ratio ($\text{ICIR} = \bar{\text{IC}} / \sigma_{\text{IC}}$):** Evaluates risk-adjusted predictive efficiency by measuring return forecasting power per unit of IC volatility.
* **Hit Rate ($\% \text{IC} > 0$):** Measures temporal consistency by calculating the proportion of trading sessions where the factor correctly assigns positive directional rank correlation.
* **Statistical Significance ($t\text{-stat}$ & $p\text{-value}$):** Evaluates whether historical alpha capacity is statistically distinguishable from zero:

$$\text{t-statistic} = \frac{\bar{\text{IC}}}{\sigma_{\text{IC}} / \sqrt{N}}$$



where $N$ represents the effective sample size of daily cross-sectional correlation observations.

#### Cumulative IC ($\sum \text{IC}_t$) & Regime Stability Analysis

Beyond static summary statistics, inspecting the Cumulative IC trajectory provides visual confirmation of signal consistency:


$$\text{Cumulative IC}_T = \sum_{t=1}^{T} \text{IC}_t$$


A monotonic positive trajectory confirms structural signal persistence, whereas trend reversals or flatlines diagnose drawdown sensitivity and performance breakdown across changing market regimes (e.g., volatility spikes or liquidity shocks).

### Multi-Criteria Feature Selection Filter

Factors are selected for downstream modeling based on a strict multi-stage quantitative screening policy:

1. **Predictive Validity:** Require $\text{ICIR} \ge 0.05$ and two-tailed statistical significance ($\vert{}t\text{-stat}\vert{} > 2.0$, $p < 0.05$).
2. **Collinearity Resolution:** When factor pairs exhibit $\text{VIF} > 10$ or mutual rank correlation $\bar{\rho} > 0.85$ (e.g., the volatility family), select the single metric maximizing the $\text{ICIR}$ metric (`upside_volatility`).
3. **Horizon Alignment:** Drop features whose rank decay dynamic fails to match the targeted monthly rebalancing window (`short_term_reversal` exhibits signal decay and inversion over 21 trading days).

--- 

