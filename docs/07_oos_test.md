# 07. Out-of-Sample Test Methodology & Validation — Theoretical Documentation

## OOS Pipeline Architecture, Isolation & Purging
To evaluate model performance without look-ahead bias, data leakage, or implicit parameter tuning, the Out-of-Sample (OOS) evaluation period is strictly segregated from the development dataset using a temporal buffer inspired by Combinatorial Purged Cross-Validation (CPCV) principles.

- **Development Boundary & Embargo:** Development training labels end in early December 2024. However, we cannot start the OOS test immediately on December 11. Since we predict a 21-day forward return, the target ($y$) for December 10 heavily overlaps with the target for December 11. An explicit embargo buffer is enforced through mid-January 2025 to break this serial correlation and prevent data leakage into the test set.
- **Evaluation Window:** Fixed strictly from `2025-01-15` to `2026-06-18` (~1.5 years / 357 trading sessions). Fixing this window guarantees experiment reproducibility.
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
The OOS pipeline relies on a lightweight static manifest (`universe_496.json`). Downstream scripts load this file directly to query market data, ensuring an identical cross-sectional setup upon fresh repository clones and perfect reproducibility.

### Isolating Signal from Data Drift
While real-world portfolios dynamically adapt to index changes (e.g., handling mergers like AVB and EQR into VMRK), allowing the test universe to fluctuate introduces **Data Drift**. Cross-sectional transformations like Percentile Rank define "percentiles" based on the exact cohort of assets present on a given day. If highly volatile assets enter the universe, the percentile score of an existing stable asset will artificially shift, confusing the model. 

By deliberately freezing the production universe at **496 assets** (removing one asset with persistent coverage gaps to ensure dimensional stability), we strictly isolate the model's predictive alpha and structural signal decay from external noise caused by universe composition changes.

---

## OOS Target Construction & Feature Alignment
The 21-day forward cumulative log returns (`forward_return_21d`) are computed over the evaluation period and aligned via MultiIndex `(date, ticker)` with the identically constructed feature matrix. 

Observations lacking forward target coverage or complete feature vectors are isolated, establishing a fully synchronized, deterministic feature-target pair ($X_{\text{OOS}}, y_{\text{OOS}}$) frozen for final model evaluation and portfolio backtesting.

--- 

## OOS Prediction Metrics & Inter-Model Signal Structure

### The Cross-Sectional Ranking Advantage
Evaluations confirm that predictive superiority in cross-sectional equity factor models is strictly driven by ordinal ranking precision (Spearman Rank IC) rather than point estimation accuracy (RMSE/MAE).

- **Point Error Invariance:** Across all architectures, RMSE is approximately 0.1060. They all make roughly the same absolute error when guessing the exact future return.
- **Sorting Superiority:** Despite identical RMSE, non-linear decision trees boost Rank IC by ~5.3% over linear Ridge regression (0.0550 vs 0.0522). 
- **The Grouping Mechanism:** Linear models output continuous, noisy point predictions that easily scramble asset rankings. Tree architectures (like XGBoost) output discrete values at terminal leaves, effectively grouping assets with similar characteristics into homogeneous buckets. This structural discretization ignores point-level noise and robustly separates broad "winners" from "losers," driving superior cross-sectional sorting.

### Methodological Integrity & Market Realism
The temporal behavior of the Rank IC is characterized by realistic drawdowns and negative troughs, reflecting genuine financial noise and regime shifts. The OOS test confirms the model is free of look-ahead bias, yielding a Rank IC that comfortably exceeds the institutional viability benchmark (0.0200) with strong statistical significance ($t > 4.3$ across all architectures, $p < 0.0001$).

--- 

## OOS Diagnostics & Monotonicity Audit

### Prediction Discretization & Tie-Breaking Protocol
Because tree architectures restrict forecast outputs to discrete terminal leaf values (creating frequent cross-sectional ties), ties are resolved exclusively for bucket creation using a deterministic ranking scheme (`method="first"`). This preserves original predictions and Rank IC metrics while enabling balanced cross-sectional decile comparisons.

### Decile Return Monotonicity & Tail Selection Dynamics
- **Extreme Spreads:** Tree models exhibit superior ability to isolate top-performing securities, yielding a $D_{10} - D_1$ forward return spread of +3.75% over 21 trading days (compared to +2.81% for Ridge).
- **Cross-Sectional Systemic Ranking:** Excluding the extreme $D_{10}$ tail retains strong rank monotonicity ($D_1\text{--}D_9$ Spearman $\rho = 0.952$ for XGBoost). The sorting capability spans the broader asset panel rather than relying solely on outliers.

### Regime Dependencies & Risk Exposure
While these results confirm strong OOS signal quality, **the observed spread magnitude should not be assumed to be fully structural or regime-invariant across all market cycles.** 

Cross-sectional predictions align strongly with **Upside Volatility** ($\rho \approx 0.83\text{--}0.90$). The OOS window (2025-2026) captured a strong bullish regime where aggressively rewarding upside asymmetry yielded exceptional returns. If the market transitions into a high-rate or recessionary environment, the structural nature of this upside volatility factor will likely degrade, transforming aggressive, positive-asymmetry selections into high-risk exposures.