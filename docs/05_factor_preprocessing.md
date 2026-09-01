# 05. Feature Preprocessing & Conditioning Specifications — Theoretical Documentation

## Theoretical Rationale: Cross-Sectional Winsorization vs. Pooled Winsorization
In cross-sectional equity modeling, feature clipping must be executed within daily cross-sections ($t$) rather than across the full panel aggregate ($T \times N$). 

### Justification:
1. **Regime Neutrality**: Financial time series exhibit time-varying volatility clusters. A pooled quantile estimate would overwhelmingly clip observations during high-volatility regimes (e.g., 2008 GFC, 2020 COVID shock) while failing to isolate true cross-sectional outliers during quiet market regimes.
2. **Elimination of Look-Ahead Bias**: Pooled statistics compute percentiles using future time steps ($t > \tau$). Daily cross-sectional clipping utilizes only information available at time $t$.
3. **Information Order Preservation**: Replaces values outside $[q_{0.01, t}, q_{0.99, t}]$ with the boundary value itself, maintaining cross-sectional ordinal ranking while placing a strict bound on extreme leverage in downstream ML algorithms (Elastic Net, LightGBM).

## Feature Transformations & Tail Mechanics

### Monotonic Log-Transformation of Amihud Illiquidity
Raw Amihud illiquidity is defined as:
$$I_{i,t} = \frac{|R_{i,t}|}{P_{i,t} \times V_{i,t}}$$
Because $I_{i,t} \ge 0$ is highly right-skewed with multi-order-of-magnitude dispersion, applying linear Winsorization directly to raw $I_{i,t}$ compresses low-to-medium illiquidity variations into a narrow band. We apply:
$$\text{Log10\_Amihud}_{i,t} = \log_{10}(I_{i,t}) \quad \forall I_{i,t} > 0$$
This normalizes the scale, linearizes relative differences in illiquidity, and conditions the distribution for symmetric quantile clipping.

### Empirical Clipping Properties
Across $N$ assets over daily cross-sections, the total percentage of clipped observations exceeds the nominal $2\%$ ($1\% + 1\%$) cross-sectional baseline, averaging $\approx 7.5\% - 8.4\%$. This phenomenon occurs because extreme assets (e.g., distressed micro-caps or hyper-momentum stocks) consistently violate percentile bounds across consecutive time steps $t$, accumulating repeatedly in the global counter.

--- 

## Cross-Sectional Standardization Mechanics

### Dual-Normalisation Strategy
To supply tailored inputs for distinct statistical machine learning frameworks, two cross-sectional normalization transformations are executed date-by-date across all $t$:

1. **Parametric Z-Score Standardisation**:
   $$z_{i,t} = \frac{x_{i,t} - \mu_t}{\sigma_t}$$
   * **Mathematical Role**: Enforces $\mu_t = 0, \sigma_t = 1$. Preserves exact metric distances, cross-sectional covariance structures, and linear proportions between asset signals.
   * **Downstream Target**: Linear models (Elastic Net, Ridge) and Information Coefficient ($\text{IC}$) rank-correlation diagnostics.

2. **Non-Parametric Uniform Rank Projection**:
   $$r_{i,t} = \text{Rank}(x_{i,t}) - 0.5 \quad \text{where } \text{Rank}(x_{i,t}) \in (0, 1]$$
   * **Mathematical Role**: Maps cross-sectional signals onto a uniform distribution centered on $[-0.5, 0.5]$. Completely eliminates residual heavy tails or non-Gaussianities without affecting cross-sectional ordinal sorting.
   * **Downstream Target**: Gradient Boosted Decision Trees (LightGBM), ensuring split-criterion invariance to non-linear monotonicity.

## Target Alignment & Raw Scale Preservation

### Forward Return Specification
The modeling target is defined as the 21-trading-day unadjusted forward return:
$$y_{i,t} = R_{i, t+1 \to t+21} = \frac{P_{i, t+21}}{P_{i, t}} - 1$$

### Non-Transformation Mandate
While explanatory factors undergo aggressive scaling (Winsorization, Z-score/Rank transformations), the target variable $y_{i,t}$ remains strictly in its original raw financial scale. Transforming or winsorizing targets distorts expected portfolio return estimates during portfolio optimization (e.g., mean-variance optimization with Ledoit-Wolf shrinkage) and invalidates realistic transaction cost calculations.

---

