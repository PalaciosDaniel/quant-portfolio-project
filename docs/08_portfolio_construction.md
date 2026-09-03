# 08. Portfolio Construction — Theoretical Documentation

## Benchmark Portfolio Parameter Specifications

### Signal Horizon & Rebalancing Cadence
- **Target Horizon Coupling**: The factor models predict a 21-trading-day forward cumulative return (`forward_return_21d`). 
- **Baseline Rebalance Cadence**: The initial benchmark portfolios adopt a matching 21-session periodic rebalancing schedule.
- **Turnover Sensitivity**: Rebalance frequency is treated as an explicit parameter run; alternative cadences are evaluated downstream to assess the trade-off between alpha decay, portfolio turnover, and transaction costs.

### Baseline Signal Decile Selection
- **Extreme Ranking Deciles**: Primary baseline long and short positions are constructed using top and bottom cross-sectional rank percentiles.
  - **Long Portfolio Target**: Top $10\%$ highest-ranked securities ($D_{10}$).
  - **Short Portfolio Target**: Bottom $10\%$ lowest-ranked securities ($D_1$).
  - **Sensitivity Screening**: Alternative selection cutoffs ($20\%$, $30\%$) are evaluated as sensitivity runs to test strategy capacity versus alpha decay.

### Capital Allocation & Exposure Rules
- **Long-Only Portfolios**:
  - **Gross Exposure Target**: $100\%$ ($1.00$).
  - **Net Exposure Target**: $+100\%$ ($+1.00$).
- **Long-Short Portfolios**:
  - **Long Leg Exposure**: $+50\%$ ($+0.50$).
  - **Short Leg Exposure**: $-50\%$ ($-0.50$).
  - **Gross Exposure Target**: $100\%$ ($1.00$).
  - **Net Exposure Target**: $0\%$ ($0.00$, Dollar/Market Neutral).
  - **Cross-Model Standardized Limits**: Target exposures are held strictly constant across all candidate models to ensure homogenous performance attribution, with position counts scaling dynamically based on active cross-sectional universe availability at each rebalance date.

### Universe & Pre-OOS History Allocation
- **Risk Estimation Lookback**: Price coverage extends prior to the OOS prediction start date to supply sufficient historical lookback periods for empirical variance-covariance estimation, Ledoit-Wolf shrinkage, and factor loading estimations without truncating early OOS portfolio rebalance dates.

---

## Model Universe & Methodological Exclusions

### Explicit Exclusion of Elastic Net
- **Feature Pipeline Pre-Conditioning**: Elastic Net regularization ($L_1/L_2$ hybrid penalty) is primarily designed for automatic variable selection and stabilization under high-dimensional or collinear feature spaces.
- **Dimensionality Reduction Pre-Filtering**: The upstream feature selection pipeline already reduced the input space to three orthogonal factors with verified individual signal contributions.
- **Redundancy Elimination**: Because informational redundancy was explicitly resolved during factor filtering, Elastic Net offers no marginal theoretical value over Ridge or Ordinary Least Squares (OLS)—a decision confirmed by the identical empirical performance observed across linear candidate models.

---

## Signal Ranking & Quantile Allocation Protocols

### Dynamic Cross-Sectional Universe Partitioning
- **Quantile Bucket Formation**: Continuous cross-sectional prediction vectors are dynamically partitioned into ten equal-sized decile buckets ($D_1$ to $D_{10}$) on each rebalance date $t$, strictly conditioning on active universe availability ($N_t$).

### Deterministic Tie Resolution (`method="first"`)
- **Tree-Model Discretization**: Decision-tree architectures (e.g., XGBoost) generate step-wise step functions that cluster numerous stock predictions into identical discrete values (e.g., $>50\%$ of forecasts collapsing to a single scalar value).
- **Quantile Bucket Stabilization**: Applying standard average ranking (`method="average"`) on heavily clustered predictions causes severe daily decile sizing distortions. Enforcing deterministic tie-breaking via `method="first"` preserves balanced cross-sectional bucket allocations ($10\%$ of available universe $N_t$ per decile) across all rebalancing dates without altering underlying predictive rank order or Rank IC metrics.

---

## Benchmark Portfolio Formation & Allocation Rules

### Decoupling Alpha Selection from Capital Allocation
- **Baseline Design Architecture**: Before introducing signal-strength scaling, risk constraints, or mathematical optimization, benchmark portfolios are constructed using simple, transparent allocation rules to establish an unconstrained baseline.
- **Separation Principle**: Cross-sectional signal rankings strictly determine asset *selection*, while position *weighting* is governed independently by Equal-Weighting (EW).
- **Comparative Baseline Role**: These benchmarks serve as the reference against which Signal-Weighted, Risk-Based, and Mathematically Optimized portfolios are downstream evaluated to test for genuine value-add net of turnover and implementation costs.

### Benchmark Specifications & Implementation Matrix
- **Long-Only Top 10% Equal-Weight**:
  - **Constituent Selection**: Top decile ($D_{10}$).
  - **Weighting Protocol**: $w_{i,t} = 1 / |D_{10,t}|$ for each $i \in D_{10,t}$.
  - **Target Exposure**: $+100\%$ Net Exposure ($E_{\text{net}} = 1.00$), $100\%$ Gross Exposure ($E_{\text{gross}} = 1.00$).
- **Long-Short D10/D1 Market-Neutral Equal-Weight**:
  - **Constituent Selection**: Top decile ($D_{10}$, Long Leg) and Bottom decile ($D_1$, Short Leg).
  - **Weighting Protocol**: Equal-weighted within legs: $w_{i, \text{long}} = +0.50 / |D_{10,t}|$ and $w_{i, \text{short}} = -0.50 / |D_{1,t}|$.
  - **Target Exposure & Neutrality**: $+50\%$ Long leg / $-50\%$ Short leg, enforcing $0\%$ Net Exposure ($E_{\text{net}} = 0.00$, Market Neutral) and $100\%$ Gross Exposure ($E_{\text{gross}} = 1.00$) to isolate cross-sectional relative ranking capability from market directional bias.
- **Standardized Weights Structure**: All resulting portfolio weight matrices ($\mathbf{W}_{N \times T}$) across candidate models and benchmark variants are stored in a unified, standardized schema to facilitate downstream comparative backtesting and risk attribution.

---

## Mathematical Exposure Definitions

For any portfolio $p$ with weight vector $\mathbf{w}_t = [w_{1,t}, w_{2,t}, \dots, w_{N,t}]^T$ at date $t$:

$$\text{Long Exposure } (E_{L,t}) = \sum_{w_{i,t} > 0} w_{i,t}$$

$$\text{Short Exposure } (E_{S,t}) = \sum_{w_{i,t} < 0} w_{i,t}$$

$$\text{Gross Exposure } (E_{\text{gross},t}) = \sum_{i=1}^{N} |w_{i,t}| = E_{L,t} + |E_{S,t}|$$

$$\text{Net Exposure } (E_{\text{net},t}) = \sum_{i=1}^{N} w_{i,t} = E_{L,t} + E_{S,t}$$

---

## Portfolio Selection Sensitivity Framework

### Theoretical Rationale & Trade-offs
- **Signal Dilution vs. Concentration Risk:** Restricting asset selection to extreme quantiles (e.g., Top 10%) maximizes exposure to pure factor signal strength but increases portfolio variance and idiosyncratic risk due to low asset counts. Expanding the selection threshold (Top 20%, Top 30%) enhances diversification at the expense of diluting signal alpha.
- **Quantile Cutoff Selection:** Evaluating 10%, 20%, and 30% tail cutoffs allows empirical identification of the optimal point on the capacity/alpha frontier before signal decay dominates execution performance.

---

## Portfolio Weighting Methodologies: Signal Weighting

**Formulas.** For long-only:
$$w_{i,t} = \frac{r_{i,t}}{\sum_{j \in S_t} r_{j,t}}$$
For long-short, each side is weighted independently to preserve $E_L=+0.50$, $E_S=-0.50$: the long side uses $r_{i,t}$ directly; the short side uses the bearish intensity $1-r_{i,t}$, so lower-ranked assets get larger absolute short weight.

> **Note on target exposure:** this ±0.50 target is specific to Signal Weighting's long-short construction. `00_general_methodology.md` currently documents a *different* long-short target (±1.0, equal-weighted) for the baseline Long-Short Equal Weight portfolio — confirm whether this is an intentional per-method difference or should be reconciled (see conflict note below).

**Rank recomputation.** The prediction rank is recalculated within the selected subset (not inherited from the full universe): a global rank would compress D10 values into $0.90$–$1.00$, producing a negligible tilt versus Equal Weight. Recomputing within-subset redistributes values across $[1/N, 1.0]$, allowing for meaningful differentiation among selected positions.

**Alternative considered and discarded:** shifting the signal relative to the subgroup's min/max (e.g. $s_i = r_i - r_{\min}$ on the long side) to push the marginal asset's weight toward zero. Rejected because it forces that asset's weight to *exactly* zero, silently shrinking effective portfolio size and contradicting the diversification principle established earlier in the project. In-subset rank recomputation achieves the same penalization of marginal assets without ever reaching a literal zero weight ($\min = 1/N$ by construction).

---

## Portfolio Weighting Methodologies: Risk-Based Allocation — Shared Volatility Estimation

In the Risk-Based framework, machine learning model predictions are used exclusively to select the asset universe at each date, delegating weight assignment entirely to observed market risk metrics. This design tests whether explicit risk management improves the risk-return profile over the default benchmark (Equal Weight) without altering the model's underlying asset selection.

A 21-session ($\approx 1$ month) rolling volatility of daily returns is used for both Inverse Volatility and Risk Parity risk estimation:
$$\sigma_{i,t} = \text{Std}(r_{i,t-20}, \dots, r_{i,t-1})$$
Chosen for responsiveness to regime changes; uses strictly pre-decision data (no look-ahead). To compute metrics from day one of OOS (Jan 15, 2025), the 21 sessions immediately preceding OOS (from the pre-test buffer period) are used.

**Relationship to the volatility factor:** this 21-day window is distinct from the 252-day rolling volatility computed during feature engineering as a model predictor. The 252-day window targets medium/long-term risk for prediction; the 21-day window here targets a fast-reacting, contemporaneous risk estimate for weighting only. They are not meant to replicate each other. The 21-day estimation is shifted by one session (`shift(1)`) to avoid look-ahead bias in weight formation.

### Inverse Volatility

$$w_{i,t} = \frac{1/\sigma_{i,t}}{\sum_{j \in S_t} 1/\sigma_{j,t}}$$

Allocates capital inversely to individual recent volatility, favoring more stable assets while penalizing volatile ones. Uses **total** volatility rather than downside volatility to keep a neutral risk definition that doesn't introduce assumptions about return asymmetry (downside volatility is reserved for potential robustness checks).

### Risk Parity

Moves from individual asset risk to portfolio-level risk via the covariance matrix $\boldsymbol{\Sigma}_t$. Since the selected universe (50–150 assets) exceeds the 21-observation window, the sample covariance is unstable; **Ledoit-Wolf shrinkage** is applied:
$$\boldsymbol{\Sigma}_{\text{LW}} = (1-\lambda)\boldsymbol{\Sigma}_{\text{sample}} + \lambda\boldsymbol{\Sigma}_{\text{target}}$$
($\lambda$ estimated automatically via the Ledoit-Wolf procedure). Total risk contribution of asset $i$ given weights $\mathbf{w}$ and portfolio volatility $\sigma_p = \sqrt{\mathbf{w}^\top \boldsymbol{\Sigma}_{\text{LW}} \mathbf{w}}$:
$$RC_i = w_i \cdot \frac{(\boldsymbol{\Sigma}_{\text{LW}}\mathbf{w})_i}{\sigma_p}$$
**Objective:** equalize $RC_i$ across assets ($RC_1 = \dots = RC_N$), not equalize capital — a more volatile or more correlated asset receives a smaller weight than a calmer, less-correlated one.

---

## Portfolio Weighting Methodologies: Mathematical Optimization (Mean-Variance / Maximum Sharpe)

This framework directly combines predictive ML signals with the underlying risk structure of selected assets. Unlike heuristic or risk-based methods—where predictions only select assets or scale weights—here the signal is calibrated into an expected active return ($\hat{\alpha}$) and explicitly fed into a mean-variance optimization problem.

The process operates in two steps: first, relative cross-sectional ranks are calibrated into expected alpha; second, expected alpha is combined with the asset covariance matrix to solve for optimal weights. This explicit separation isolates two distinct sources of value: predictive signal quality (identifying opportunities) vs. portfolio construction efficiency (allocating capital under risk constraints).

#### Signal Calibration — Rank to Expected Alpha

Raw ML predictions are trained on continuous loss functions and excel at *ranking* assets, but their absolute scale is unreliable due to variance compression (shrinkage) typical of tree-based models under low signal-to-noise ratios. Calibration converts cross-sectional rank into expected alpha via a two-part decomposition:
$$r_{i,t}^{\text{rank}} \to \underbrace{\text{shape}(\text{pct}_{i,t})}_{\text{relative shape}} \times \underbrace{\text{scale}_t}_{\text{regime level}} = \hat\alpha_{i,t}$$

**Why split shape and scale:** the two need very different sample sizes to estimate reliably. Shape is a 10-point (decile) function needing a long history to be non-noisy; scale is a single scalar that converges on relatively few recent observations. Splitting them lets each use its own appropriate window instead of forcing one suboptimal horizon for both.

**Design:**
1. **OOF predictions + path averaging.** Historical predictions come only from Combinatorial Purged Cross-Validation (CPCV), so the model never saw them in training. Since CPCV evaluates a given date across multiple $\binom{N}{k}$ paths, repeated (date, ticker) predictions are averaged into $\bar p_{i,t}$ to reduce variance and ensure uniqueness.
2. **Cross-sectional percentile normalization:** $\text{pct}_{i,t} = \text{rank}(\bar p_{i,t})/N_t \in [0,1]$, isolating the signal from shifts in the model's global scale over time.
3. **21-day temporal purge:** only pairs $(\text{pct}_{i,t'}, r_{i,t':t'+21})$ with a fully realized return window ($t'+21 \le t$) are used, eliminating look-ahead bias both in-history and as OOS progresses.
   - *Alternative considered and discarded:* additionally exploiting the transition window between the end of validation and the start of OOS (final-model predictions on that stretch, already purged by the time OOS evaluation starts). Rejected because it would require rebuilding the full feature-generation pipeline (including winsorization and ranking) for that stretch — something never run past the validation cutoff — not just running inference on existing predictions. Given it represents only 1–2 months versus the ~4-month scale window, and Maximum Sharpe is one of several portfolio-construction methods rather than the project's central focus, the expected benefit didn't justify the engineering cost.
4. **Shape component — isotonic regression on relative returns:** valid pairs are grouped into 10 percentile deciles; for each, the mean return *relative* to the cross-sectional universe mean on that date is computed ($r_{i,t}^{\text{rel}} = r_{i,t} - \bar r_t^{\text{universe}}$), not the absolute return. This relative-return choice is what makes the shape/scale split possible: fitting shape on absolute returns would implicitly bake in each historical period's market regime level, contaminating shape with the exact effect scale is meant to isolate. An isotonic regression (Pool Adjacent Violators, monotonic non-decreasing) is fit over these 10 points using an expanding window over all purged history.
5. **Scale component — recent decile-spread multiple, with startup shrink:** $\text{spread}_t = \bar r^{\text{rel}}_{\text{decile 10}} - \bar r^{\text{rel}}_{\text{decile 1}}$, computed on a short rolling window of the last ~4× the prediction horizon (~85 sessions, ~4 months of purged history), normalized against the long-history spread used for shape: $\text{scale}_t = \text{spread}_t^{\text{recent}}/\text{spread}^{\text{history}}$.
   - Scale is highly sensitive to window composition at OOS startup, when the ~4-month window is still mostly OOF historical data (no transition window buffers this, per point 3). A shrink toward the neutral value ($\text{scale}_t = 1.0$) proportional to the fraction of the window still historical is applied:
     $$\text{scale}_t^{\text{adj}} = 1.0 + w_t^{\text{OOS}}\cdot(\text{scale}_t - 1.0), \quad w_t^{\text{OOS}} = \frac{\text{real OOS obs. in window}}{\text{total window size}}$$
   - $\text{scale}_t^{\text{adj}}$ starts near 1.0 at OOS start and converges to the unadjusted $\text{scale}_t$ once the window is fully real-OOS (after ~4 months).
6. **Combination and refresh frequency:** $\hat\alpha_{i,t} = \text{shape}(\text{pct}_{i,t}) \times \text{scale}_t^{\text{adj}}$. Since $\text{scale}_t^{\text{adj}} > 0$ by construction, decile monotonicity from shape is preserved. Both components refresh monthly (shape on the cumulative expanding window, scale on the short rolling window) to avoid reacting to short-term noise.

**Startup bias and its dissolution.** Neither component has enough genuine OOS data at OOS start, but the effect differs: shape's long history (CPCV over 14 years of train/validation) gives it a small, *pessimistic*-leaning startup bias (its submodels train on a $(K-1)/K$ fraction of the data available to the final model, so discriminative power starts slightly below production level); this dilutes slowly and is small from the start given the history/OOS size ratio. Scale is the component genuinely exposed at startup, since its short window is, early on, composed almost entirely of historical OOF observations. The shrink in point 5 directly addresses this asymmetry, letting scale approach its unadjusted value gradually as the window fills with real OOS data. *Robustness check option:* compare shape/scale (with and without shrink) fit on historical data versus fit exclusively on real OOS pairs once the OOS sample is large enough for both.

#### Validation & Empirical Diagnostics — Rank to Expected Alpha

The primary trigger for equal-weight fallback dates stems from the *momentum crash* experienced by the S&P 500 in early 2025 following tariff announcements. During this severe correction, quantitative signals suffered a widespread breakdown, prompting the calibration algorithm to set the scale parameter to zero to protect the portfolio. This resulted in an activation rate of $16.67\%$ (62 out of 372 dates) for Ridge and XGBoost. Random Forest exhibited a lower activation rate of $11.29\%$ (42 dates) due to a faster recovery, detecting market stabilization and exiting the negative bias one month earlier than alternative models.

This dynamic is reflected with exact fidelity in the **cross-sectional variance of alpha**, where days with near-zero variance ($< 1\text{e-}12$) align precisely with each model's fallback dates (62 for Ridge/XGBoost, 42 for Random Forest), confirming that MVO optimization scaling operates with complete consistency. Finally, **ranking concordance (Spearman $\rho$)** demonstrates strong cross-sectional consensus among active predictions, highlighted by a high correlation between XGBoost and Random Forest ($\rho = 0.9634$), while dates omitted from the correlation match with mathematical precision the periods when signals were neutralized by the fallback mechanism.

Regarding operational continuity, all pairwise model comparisons report an identical skip rate of 62 dates (16.67%). This uniform figure stems directly from the out-of-sample momentum reversal period. Correlation calculations require varying position weights and must skip rebalancing dates where either model in a pair generates flat, equal-weighted allocations due to zero-alpha fallbacks. Because Ridge and XGBoost experience 62 fallback dates while Random Forest experiences a subset of 42 dates during the same window, taking the union of flat dates for any model pair yields exactly 62 skipped evaluations. Accounting for these excluded regime shifts, the underlying non-zero weight rankings display near-zero instances of severe disagreement (percent of dates with correlation below 0.3 remains at 0.00% across nearly all configurations), confirming that active capital allocation remains highly cohesive when valid signals are present.

### Portfolio Construction — Mean-Variance Optimization

Once expected alpha estimates ($\hat{\boldsymbol{\alpha}}_t$) are obtained, they are combined with the dependency structure of the selected assets to determine portfolio weights via mean-variance optimization. The objective is to allocate capital to maximize expected active return per unit of risk:

$$\max_{\mathbf{w}} \frac{\mathbf{w}^\top \hat{\boldsymbol\alpha}_t}{\sqrt{\mathbf{w}^\top \boldsymbol\Sigma_t \mathbf{w}}}$$

Risk estimation reuses the Risk Parity methodology (21-day rolling window, strictly prior data, Ledoit-Wolf shrinkage). Constraints prevent one high-alpha asset from dominating the solution once its interaction with the rest of the portfolio (via volatility and correlation) is accounted for. Thus, an asset with a high alpha may receive a reduced weighting if its inclusion disproportionately increases total risk.

This approach sets Maximum Sharpe apart from previous methods: while Signal Weighting considers only prediction intensity, Inverse Volatility considers only individual risk, and Risk Parity seeks to equalize risk contribution, mean-variance optimization simultaneously weighs calibrated expected return and joint risk, ensuring efficient capital allocation.

**Why the max-weight constraint sits inside the optimizer here (unlike the rest of Block 6):** Signal Weighting, Inverse Volatility, and Risk Parity all use closed-form weighting formulas — any constraint on them can only be a post-hoc adjustment to an already-computed weight vector. Maximum Sharpe is the only scheme in this block solving an explicit optimization problem, making it the only one where the max-weight limit can (and, to preserve optimality, should) be built directly into the Quadratic Program (QP) as a linear inequality ($w_i \le w_{\max}$) rather than corrected afterward. Clipping-and-renormalizing a solution post hoc does not preserve optimality — the resulting Sharpe ratio can be meaningfully lower than the true constrained solution's, and relative proportions between assets get arbitrarily distorted by the clipping procedure itself.

**Why the min-weight floor is still post hoc, even for Maximum Sharpe:** enforcing $w_i \ge w_{\min}$ or $w_i = 0$ exactly would make the problem semicontinuous (cardinality-constrained), non-convex, and solvable only via mixed-integer programming. That complexity adds no methodological value here, so standard industry practice is used instead: optimize without the floor, then prune and renormalize residual weights below the threshold — the same rule applied to the other three schemes (detailed in the Block 8 operational-constraints framework).

**Why turnover stays out of the optimizer entirely:** (1) keeping turnover out of the optimizer preserves methodological comparability across all four allocation schemes — Sharpe shouldn't get special treatment on a constraint that isn't exclusive to it; (2) turnover is only meaningful evaluated on the *final* weight vector, i.e., after the min-weight cleanup, so it cannot be verified inside the optimization problem itself without conflicting with that later renormalization.

---

### Portfolio Integrity Validation

**Herfindahl-Hirschman Index (HHI) & Extreme Portfolio Cases**

Contrary to theoretical expectations, extreme market events do not necessarily reduce the count of active positions (which remains at 50 securities); rather, they severely distort capital allocation. This behavior illustrates how a drastic, idiosyncratic collapse in an asset's estimated volatility causes risk-based allocation schemes to assign it a disproportionate weight.

This empirical finding confirms that, even within simple heuristics, imposing explicit weight caps ($w_{\max}$) is strictly imperative to prevent severe concentration bias when facing data anomalies or temporary volatility collapses.

---

## Buffer-Zone Turnover Diagnostic & Decomposition (Section 7.1)

The turnover measured for the buffer-zone decision is computed directly on theoretical weights between consecutive rebalances ($w_t$ vs. $w_{t-1}$), **without** incorporating market price drift. This isolates pure signal/selection-rule noise from market movement, and is intentionally distinct from the drift-adjusted turnover formulation used in the operational-constraints stage (Section 8.3), whose purpose is to simulate real order execution.

Prior to deciding on the implementation of hysteresis bands, total turnover is decomposed into two distinct drivers:
1. **Re-optimization / Continued Adjustment Turnover:** Weight changes among assets already present in the portfolio at $t-1$.
2. **Marginal Selection Turnover:** Turnover driven by complete entries and exits at the universe boundary. This is further split into marginal oscillations (within $\pm 3$ percentile points of the selection cutoff) vs. deep non-marginal entries/exits.

**Empirical Evidence & Selective Trigger:**
- **Risk-Based Schemes (Inverse Volatility & Risk Parity):** Individual risk metrics and risk-contribution figures fluctuate gradually but noisily near the cutoff. Under strict selection rules (Top 10%), marginal oscillations account for **35% to 49%** of total turnover without reflecting genuine changes in active conviction.
- **Optimized Schemes (Maximum Sharpe) & Signal Weighting:** Turnover is overwhelmingly structural, driven by active alpha forecast ($\hat{\alpha}$) re-rankings and covariance updates across assets already selected. Because hysteresis bands act exclusively on universe filtering rather than optimization weights, they offer negligible turnover relief here.

Consequently, buffer zones are applied selectively only to Risk-Based Top 10% portfolios, avoiding unnecessary hyperparameter complexity elsewhere.

---

## Buffer-Zone (Hysteresis) Mechanism — Design & Execution Details (Section 7.2)

Two distinct thresholds replace the static percentile cutoff:
- **Entry Threshold ($\theta_{\text{in}}$):** Matches the original selection cutoff (e.g., 90th percentile for Top 10%). An asset not present in the portfolio at $t-1$ can only enter at time $t$ if its score strictly exceeds this threshold.
- **Exit Threshold ($\theta_{\text{out}} = \theta_{\text{in}} - b$):** An asset already held in the portfolio at $t-1$ remains included as long as its percentile rank stays above this looser threshold. Bandwidth is fixed at **$b = 0.03$** for all implementations.

**Implementation Specifications:**
- **Sequential Chronological Execution:** Because eligibility at time $t$ explicitly depends on the realized portfolio state at $t-1$, universe filtering cannot be vectorized across dates and requires chronological iteration to maintain portfolio state across consecutive rebalancing periods.
- **Orthogonality to Allocation Rules:** The buffer mechanism operates exclusively on universe selection. Once the eligible asset pool is finalized under hysteresis rules, portfolio weights are calculated using the exact Inverse Volatility or Risk Parity formulations defined in Block 6.
- **Quality Control:** Entry criteria for non-held assets stay strictly at $\theta_{\text{in}}$, ensuring the hysteresis band stabilizes existing positions without lowering the quality bar for new additions.

---

## Maximum Position Weight — Iterative Proportional Redistribution (Waterfall)

The 5.0% cap ($w_i \le w_{\max} = 0.05$) is enforced via an **iterative proportional redistribution ("waterfall")** heuristic overlay, executed independently per date and per side (long/short):
1. **Identify Breaches:** Detect all positions exceeding the 5.0% cap.
2. **Clip & Pool:** Truncate those positions to exactly 5.0% and pool the total excess capital.
3. **Proportional Redistribution:** Distribute the excess capital proportionally among remaining unclipped positions ($w_j < 5.0\%$).
4. **Convergence Iteration:** Repeat steps 1–3 iteratively until all asset weights satisfy $w_i \le 5.0\%$ and sum precisely to target exposure ($\sum w = 1.0$ for Long-Only; target exposure per leg for Long-Short).

This rule-based overlay was chosen over a hard optimizer-side constraint for closed-form schemes (Signal Weighting, Inverse Volatility, Risk Parity) since those do not solve an optimization problem to begin with. It also absorbs the renormalization-driven cap breach that occurs downstream in Maximum Sharpe after its minimum-weight cleanup, eliminating extreme concentration spikes without altering the relative ordinal ranking of assets.

---

## Minimum Effective Position Weight — Post-Optimization Cleanup

The 0.5% floor ($w_{\min} = 0.005$) is enforced as a post-optimization cleanup rule designed to eliminate numerical "dust" positions (orders of magnitude $10^{-18}$ to $10^{-20}$) that add operational execution friction without contributing meaningfully to risk/return:
- **Pruning & Renormalization:** Any position below 0.5% is zeroed out ($w_i = 0$), and its capital is redistributed proportionally among the remaining valid positions.
- **Exposure Neutrality:** Long-Only and Long-Short portfolios are processed independently. For Long-Short strategies, each leg is pruned and renormalized separately to preserve original gross/net exposure and market neutrality.

Hard cardinality constraints inside the optimizer were rejected, as they introduce integer programming complexity and convergence failures for no methodological benefit.

---

## Turnover Control & Smoothing Mechanics (Sections 8.3.3 & 8.3.4)

Turnover is measured on **drift-adjusted operational weights** between consecutive 21-day rebalancing events. To constrain portfolio transition speed across all models without discarding new ML signal directions, a **post-hoc linear interpolation** mechanism is applied.

When raw drift-adjusted turnover ($\text{Turnover}_t^{\text{raw}}$) exceeds the internal design threshold ($\text{MAX\_TURNOVER\_DESIGN} = 0.25$), a scalar attenuation factor $\gamma \in (0, 1)$ is derived:

$$\gamma = \frac{0.25}{\text{Turnover}_t^{\text{raw}}}$$

The executable constrained weight vector $w_t^{\text{constrained}}$ is constructed by linearly interpolating between the previously executed allocation $w_{t-1}^{\text{constrained}}$ and the new unconstrained target $w_t^{\text{raw}}$:

$$w_t^{\text{constrained}} = w_{t-1}^{\text{constrained}} + \gamma \cdot (w_t^{\text{raw}} - w_{t-1}^{\text{constrained}})$$

**Differential Portfolio Dynamics:**
- **Optimized & Risk-Based Schemes (Maximum Sharpe, Risk Parity, Inverse Volatility):** The interpolation algorithm intervenes dynamically and regularly to smooth high weight volatility driven by risk-matrix updates and alpha shifts.
- **Heuristic Schemes (Equal Weight & Signal Weighting):** The algorithm remains dormant during standard market regimes, operating strictly as a non-invasive risk dampener during extreme cross-sectional market shocks.

**Design Buffer Headroom:**
To prevent downstream box-constraint post-processing (pruning micro-positions below 0.5% and re-capping positions above 5.0%) from pushing effective turnover past the institutional ceiling, the internal smoothing target is set to **25%**. The resulting post-processing weight expansion dilutes turnover back up to the headline **30% operational ceiling**, ensuring strict dual compliance with position limits and turnover caps.

---

## Execution Timing & Lag Convention (Section 9.1)

Target weight vectors $\boldsymbol{w}_{t_{k}}^{\text{target}}$ generated from data available through $t_k$ take effect at $t_k+1$ (close of $t_k$ / open of $t_k+1$), not at $t_k$ itself:

$$\boldsymbol{w}_{t_{k}}^{\text{target}} = \mathcal{F}(\text{Data up to } t_k)$$

$$\boldsymbol{w}_{t_{k}^+}^{\text{executed}} = \boldsymbol{w}_{t_{k}}^{\text{target}} \quad \text{active from } t_k+1$$

This standard look-ahead bias avoidance rule applies project-wide across all backtests.

---

## Buy-and-Hold Intra-Period Dynamic

Between rebalancing dates ($t_k + 1 \to t_{k+1}$), no intermediate trades occur and weights drift dynamically with relative asset returns:

$$w_{i,t+1} = \frac{w_{i,t}\cdot(1+R_{i,t+1})}{1+R_{p,t+1}}$$

where $R_{p,t+1} = \sum_i w_{i,t} R_{i,t+1}$ represents gross daily portfolio return. Weights reset to target strictly on scheduled rebalancing dates; no intermediate rebalancing or drift correction is performed.

---

## Turnover Definition — Drift-Adjusted Execution Stage (Section 10.2)

At each rebalance event $t_k$, turnover is measured against the **drifted** weight vector $w_{i,t_k}^{\text{drifted}}$ that the portfolio actually reached through market price movements since $t_{k-1}$, rather than static past target weights:

$$T_k = \frac12\sum_{i=1}^N \left|w_{i,t_k} - w_{i,t_k}^{\text{drifted}}\right|$$

where drifted weights projected via compounded asset returns $R_{i, t_{k-1} \to t_k}$ are defined as:

$$w_{i,t_k}^{\text{drifted}} = \frac{w_{i,t_{k-1}}(1+R_{i,t_{k-1}\to t_k})}{1+R_{p,t_{k-1}\to t_k}}$$

For Long-Only portfolios, drifted weights re-normalize to total exposure; for Long-Short portfolios, each leg re-normalizes independently to preserve gross exposure and market neutrality.

---

## Consolidated Linear Transaction Cost Model (Section 10.1)

Transaction costs are evaluated via a linear drag applied directly to traded volume at rebalance events:

$$C_t = T_t \times c$$

The single coefficient $c$ consolidates brokerage commissions, bid-ask spread crossing, and volume-driven market impact slippage. Three sensitivity fee scenarios are evaluated:
- **Conservative:** $10\text{ bps}$ ($c = 0.0010$).
- **Base (Headline Benchmark):** $15\text{ bps}$ ($c = 0.0015$).
- **Stressed:** $20\text{ bps}$ ($c = 0.0020$).

---

## Calendar Mapping, Cost Deduction & Net Returns Computation (Section 10.2)

Net portfolio performance is derived in three distinct steps:
1. **Operational Calendar Mapping:** Each rebalance penalty $C_k = T_k \times c$ is mapped to its effective execution date ($t_k + 1$). On non-rebalancing intermediate trading sessions, imputed transaction cost is strictly zero ($C_t = 0.0$).
2. **Net Return Calculation:** Imputed costs are subtracted directly from daily gross return series:

$$R_{p,t}^{\text{net}} = R_{p,t}^{\text{gross}} - C_t$$

3. **Cumulative Compounding:** Net asset value (NAV) curves are computed via daily compounding:

$$\text{Cumulative Net Return}_T = \prod_{t=1}^T(1+R_{p,t}^{\text{net}}) - 1$$

---

## Primary Export Dataset: `net_portfolio_returns.parquet`

The resulting dataset consolidates **16,464 out-of-sample observations** (392 trading sessions $\times$ 42 strategy combinations).

**Schema & Column Structure:**
- `date`, `model`, `portfolio`: Temporal and strategy identifiers.
- `gross_return`: Unadjusted daily gross portfolio return.
- `transaction_cost_conservative`, `transaction_cost_base`, `transaction_cost_stressed`: Imputed per-rebalance operational drag series.
- `net_return_conservative`, `net_return_base`, `net_return_stressed`: Fee-adjusted daily net return series.
- `cumulative_net_return_conservative`, `cumulative_net_return_base`, `cumulative_net_return_stressed`: Compounded out-of-sample cumulative net performance series, serving as direct inputs to downstream evaluation routines.