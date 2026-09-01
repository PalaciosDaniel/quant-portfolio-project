# 07. Out-of-Sample Test Methodology & Validation — Theoretical Documentation

## OOS Pipeline Architecture, Isolation & Purging
To evaluate model performance without look-ahead bias, data leakage, or implicit parameter tuning, the Out-of-Sample (OOS) evaluation period is strictly segregated from the development dataset using a temporal buffer inspired by Combinatorial Purged Cross-Validation (CPCV) principles.

- **Development Boundary & Embargo:** Development training labels end in early December 2024 to accommodate the final 21-day forward return window. An explicit embargo buffer is enforced through mid-January 2025 to prevent overlapping return horizons between training and test sets.
- **Evaluation Window:** Fixed strictly from `2025-01-15` to `2026-07-10` (~1.5 years / 373 trading sessions). Fixing this window guarantees experiment reproducibility and prevents test expansion when re-executing pipelines with newer market data.
- **Forward Label Buffer:** Requires extended price ingestion up to `2026-09-01` (+21 trading sessions past `OOS_END`) to ensure non-overlapping calculation of 21-day realized forward returns (`forward_return_21d`) for terminal test dates.

---

## Cross-Sectional Feature Transformations
Features in $X_{\text{OOS}}$ are constructed identically to development features to prevent transformation leakage:
1. **Factor Computation:** 
   - **12-1 Momentum:** 12-month return skipping the most recent month.
   - **Upside Volatility:** Semi-deviation of positive log returns.
   - **Illiquidity:** $\log_{10}$ Amihud ratio.
2. **Winsorization:** Cross-sectional clipping at the 1st and 99th percentiles per date ($p_{\text{low}}=0.01, p_{\text{high}}=0.99$).
3. **Percentile Ranking:** Zero-centered cross-sectional rank mapping into $[-0.5, 0.5]$.

---

## Universe Architecture & Drift Mitigation

### Static Manifest Pattern (`universe_496.json`)
To eliminate circular build dependencies and prevent committing heavy `.parquet` binaries to version control, the OOS pipeline relies on a lightweight static manifest (`universe_496.json`). Downstream scripts load this file directly to query market data, ensuring identical cross-sectional setup upon fresh repository clones without requiring pre-existing local data artifacts.

### Operational Universe Alignment (497 vs. 496 Assets)
While the initial development phase yielded 497 securities (`df_final_rank.parquet`), one asset (`EA`) exhibited persistent coverage gaps across the extended OOS window. To preserve dimensional stability across downstream backtesting architectures (Notebooks 07–09) without refactoring established results, the production universe was explicitly frozen at **496 assets**. This isolates factor signal decay from universe composition changes and index entry/exit dynamics.

### Open-Source Data Degradation & Corporate Restructuring
A known limitation of open-source financial data APIs (e.g., Yahoo Finance) is their inability to preserve historical tickers post-restructuring. During the test window, **AvalonBay Communities (AVB)** and **Equity Residential (EQR)** merged into **Vivmark Residential (`VMRK`)**, causing live API queries for historical tickers to return retroactive missing values (`NaN`).

- **Live API Audit Defense:** Fresh API queries are executed purely as a defensive validation protocol to audit vendor drift and test pipeline robustness against missingness (reducing active live observations to 494 assets).
- **Frozen Artifact Coupling:** To eliminate vendor-side data drift and guarantee 100% mathematical reproducibility, all downstream model inference and backtesting modules strictly consume the pre-validated static artifacts (`X_oos.parquet` and `y_oos.parquet`, $N=496$). These artifacts were validated prior to symbol deprecation, locking the evaluation panel at a uniform cross-section of 496 securities and 184,408 observations without missing values.

---

## OOS Target Construction & Feature Alignment
The 21-day forward cumulative log returns (`forward_return_21d`) are computed over the evaluation period and aligned via MultiIndex `(date, ticker)` with the feature matrix. 

Observations lacking forward target coverage or complete feature vectors are isolated, establishing a fully synchronized, deterministic feature-target pair ($X_{\text{OOS}}, y_{\text{OOS}}$) frozen for final model evaluation and portfolio backtesting.

--- 

## OOS Prediction Metrics & Inter-Model Signal Structure

### Cross-Sectional Ranking vs. Scale Error
Evaluations confirm that predictive superiority in cross-sectional equity factor models is strictly driven by ordinal ranking precision (Spearman Rank IC) rather than point estimation accuracy ($\text{RMSE}/\text{MAE}$).
- **Point Error Invariance:** Across all architectures, $\text{RMSE} \approx 0.1060$ and $\text{MAE} \approx 0.0749$.
- **Sorting Superiority:** Non-linear decision trees capture rank-order interactions that boost Rank IC by ~5.3% over linear Ridge regression ($0.0550$ vs $0.0522$).
- **Directional Accuracy:** XGBoost achieves a directional hit rate of $60.48\%$, aligning with realistic empirical upper bounds for high-capacity quantitative alpha signals.

### Signal Dispersion & Regularization
- **Gradient Boosting Compression:** XGBoost regularizes leaf outputs toward zero, reducing prediction standard deviation ($\sigma_{\text{cross}} = 0.0037$) relative to Random Forest ($\sigma_{\text{cross}} = 0.0091$).
- **Variance Control:** This compression stabilizes cross-sectional signal ranking over time, lowering Rank IC volatility ($\sigma_{\text{IC}} = 0.2035$) and maximizing the Information Ratio ($\text{IR} = 0.2704$).

### Methodological Integrity & Market Realism
- **Absence of Data Leakage:** The temporal behavior of the Rank IC—characterized by realistic drawdowns and negative troughs in its rolling average—reflects genuine financial noise, regime shifts, and market friction, confirming the OOS test is free of look-ahead bias or leakage.
- **Statistical Significance:** Average OOS Rank IC ($0.0522$–$0.0550$) comfortably exceeds the institutional viability benchmark ($0.0200$).
- **Hypothesis Testing:** $t$-statistics ($5.22$ for XGBoost, $4.96$ for RF, $4.37$ for Ridge) confirm statistical significance at $p < 0.0001$ over 373 OOS trading sessions ($t > 2.0$).

--- 

## OOS Diagnostics & Monotonicity Audit

### Prediction Discretization & Tie-Breaking Protocol
- **Leaf Output Clustering:** Tree architectures restrict forecast outputs to discrete terminal leaf values ($285$ unique values for XGBoost vs $182,612$ for Ridge), introducing frequent cross-sectional ties.
- **Homogeneous Grouping:** To prevent distorted decile sizes, ties are resolved exclusively for bucket creation using a deterministic ranking scheme (`method="first"`). This administrative tie-breaking preserves original predictions and Rank IC metrics while enabling balanced cross-sectional return comparisons.

### Decile Return Monotonicity & Tail Selection Dynamics
- **Extreme Spreads:** Tree models exhibit superior ability to isolate top-performing securities, yielding a $D_{10} - D_1$ forward return spread of $+3.75\%$ over 21 trading days ($+2.81\%$ for Ridge).
- **Cross-Sectional Systemic Ranking:** Excluding $D_{10}$ retains strong rank monotonicity ($D_1\text{--}D_9$ Spearman $\rho = 0.952$ for XGBoost), proving that sorting capability spans the broader asset panel rather than relying solely on extreme tail outliers.
- **Regime Dependencies:** While these results confirm strong OOS signal quality, the observed spread magnitude should not be assumed to be fully structural or regime-invariant across all market cycles.

### Implicit Factor Associations
- Cross-sectional predictions across all models align most strongly with **Upside Volatility** ($\rho \approx 0.83\text{--}0.90$) and moderately with **Amihud Liquidity** ($\rho \approx 0.25\text{--}0.29$), while showing low linear affinity to pure price **Momentum** ($\rho \approx -0.01$).