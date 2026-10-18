# 09. Portfolio Performance & Sensitivity Analysis — Theoretical Documentation

After submitting the strategies to execution frictions in the previous notebook, the final performance evaluation must strictly focus on **net-of-cost returns**. Measuring generated value solely through cumulative return is insufficient in modern quantitative management: it is essential to contrast the obtained return against assumed volatility, drawdown severity, and capital recovery speed.

## Performance Metric Definitions

Three metric families structure this notebook's multidimensional evaluation to determine which architectures not only generate gross alpha but consolidate solid, resilient portfolios immune to operational erosion:

**Return & volatility**

* **Cumulative Return** — total return over the OOS horizon.
* **CAGR** — annualized growth rate, normalizing for horizon length.
* **Annualized Volatility** — std. dev. of daily returns × $\sqrt{252}$.

**Risk-adjusted efficiency**

* **Sharpe Ratio** — excess return per unit of total volatility.
* **Sortino Ratio** — excess return per unit of downside volatility only, penalizing negative returns exclusively.
* **Calmar Ratio** — CAGR ÷ Maximum Drawdown; return earned per unit of worst historical loss.

**Drawdown & stress behavior**

* **Maximum Drawdown (MDD)** — largest peak-to-trough percentage loss.
* **Average Drawdown** — mean loss across all drawdown episodes, so risk assessment doesn't hinge on one extreme event.
* **Underwater Duration** — longest consecutive run of sessions below the prior historical peak.

---

## Interpretation Theory: Why Results Differ Across Models, Weighting Schemes & Universes

**Regularization vs. Nonlinearity Capture.** Ridge's superior efficiency ratios trace to its L2 penalty, which reduces coefficient variance under multicollinearity between market factors — producing more conservative, stable return estimates and minimizing unnecessary rotation and allocation noise. XGBoost and Random Forest instead excel on absolute return by mapping complex nonlinear feature interactions; that predictive aggressiveness translates into more structural volatility in the resulting weights and deeper temporary drawdowns.

**The Information Ratio of the Signal (Signal Weighting).** Signal Weight's strong performance confirms the models' predicted return carries a monotonic alpha signal — magnitude, not just sign, indicates conviction. Scaling position size by signal intensity captures more of that alpha than passive Equal Weight, more than compensating for the added volatility.

**Out-of-Sample Failure of Mean-Variance Optimization.** Maximum Sharpe portfolios' modest relative performance (mean Sharpe 1.463 vs. 2.035 for Equal Weight) is a textbook illustration of MVO's vulnerability to OOS estimation error. Maximizing Sharpe in-sample assigns extreme weight to assets whose expected-return or covariance estimates carry the most noise; OOS, the resulting weight instability and its rebalancing cost penalize the risk-adjusted profile relative to more robust heuristics.

**Concentration vs. Diversification Trade-off.** CAGR degrades progressively from Top 10% to Top 30%, showing predictive power concentrates at the extremes of the ranking ("tail alpha"). Adding names dilutes the captured risk premium in exchange for lower total volatility.

**Symmetry & Hedging in Long-Short Portfolios.** Long-Short's contained drawdowns come from partially neutralizing systematic (beta) risk: shorting the worst-signal assets isolates idiosyncratic stock-selection return, trading the market risk premium for a more stable capital path during stress.

**Gross vs. Net Returns.** Comparing gross and net performance is an additional robustness filter. A high gross CAGR built on excessive turnover can lose much of its edge to transaction costs — net return is the economically relevant metric. This matters most for Maximum Sharpe, where high weight sensitivity generates substantial operational cost.

**Risk-Adjusted vs. Absolute Returns.** CAGR should never be read in isolation. Sharpe/Sortino price return per unit of risk; Calmar relates return to Maximum Drawdown; Underwater Duration distinguishes a deep-but-fast recovery from a strategy that stays submerged for long stretches. Therefore, final strategy selection must rely on the full suite of metrics, not just cumulative return.

---

## Sensitivity Framework & Frequency Grid

Selecting the rebalancing frequency represents one of the most critical trade-offs in quantitative portfolio management. While frequent reweighting quickly adapts positions to evolving predictive signals, it also spikes turnover and transaction costs. Conversely, stretched frequencies reduce operational friction at the cost of allowing a progressive misalignment of the portfolio from its target optimal weights.

During the buy-and-hold interval between rebalance dates, disparate asset returns cause **weight drift**—winning assets gain relative weight, while losers lose weight. This phenomenon generates two opposing effects:

1. **Target Portfolio Degradation:** Drift can distort the risk-return profile and dilute the quantitative model's capacity to capture the market inefficiencies detected at $T$.
2. **Real Turnover Measurement:** Rotation at the $T+1$ rebalance event is not calculated against the nominal weights assigned at $T$, but against the effectively *drifted weights* just before executing new orders. Ignoring this adjustment distorts both intermediate return simulation and exact transaction cost quantification.

To evaluate this impact, this framework extends the analytical universe by testing sensitivity across five alternative rebalancing horizons: **5 days** (weekly), **10 days** (biweekly), **21 days** (monthly, baseline), **42 days** (bimonthly), and **63 days** (quarterly). The objective is to identify the optimal equilibrium where signal freshness overcompensates for trading friction and the drift effect.

---

## Subperiod Performance & Temporal Stability Analysis

To validate the robustness of the findings, assessing the average global performance over the entire OOS horizon is insufficient. An outstanding cumulative return could be the deceptive result of a single exceptionally favorable period.

The OOS period splits into three roughly equal phases (Early / Middle / Late OOS). For each phase and each model × weighting-scheme combination, the core metrics are replicated:

$$\text{Subperiod metrics} = \{\text{CAGR}, \text{Sharpe}, \text{Sortino}, \text{Maximum Drawdown}, \text{Recovery Time}, \text{Turnover}\}$$

**Consistency Rate** — Alongside the phase breakdown, a direct quantitative persistence metric is introduced. It measures the share of subperiods in which a given strategy's net return beats its passive reference (same-model Equal Weight):

$$\text{Consistency \%} = \frac{\sum_{p=1}^{P} \mathbb{I}\left(\text{CAGR}_{\text{strategy}, p} > \text{CAGR}_{\text{benchmark}, p}\right)}{P} \times 100$$

This approach discriminates between architectures that offer genuine, persistent alpha across different market conditions versus those dependent on specific regimes.

---

## Benchmark Comparison, Alpha Attribution & Rolling Performance

Evaluating a quantitative strategy in isolation is methodologically insufficient. A high return does not automatically imply predictive skill; it may simply reflect passive exposure to systematic risk factors (Market Beta) or anomalous, concentrated behavior. To validate whether the system generates **genuine alpha**, this block structures a five-phase evaluation framework:

1. **Candidate Strategy Selection** to focus the attribution analysis.
2. **Causal Performance Decomposition** (Benchmarking) to isolate the ML algorithm, momentum factors, and portfolio weighting.
3. **Statistical Significance Validation** via Monte Carlo simulation.
4. **Active Management Metrics** (Tracking Error, Information Ratio).
5. **Continuous Temporal Consistency** (Rolling performance, drawdown profiles).

### Candidate Selection Methodology

Testing every combination through expensive tests would be computationally wasteful. Four candidates are selected instead to synthesize the efficient frontier of the study:

* **Highest Sharpe** (`long_only_top_10_signal_weight`, RF, 21d) — the point of maximum risk-to-return conversion; minimizes the impact of execution friction.
* **Highest CAGR** (`long_only_top_10_signal_weight`, XGBoost, 21d) — led gross and net capital generation; captures tail alpha at the cost of higher volatility.
* **Highest Calmar** (`long_short_equal_weight`, XGBoost, 21d) — focused on capital preservation and bounding drawdowns during stress.
* **Baseline ML** (`long_only_equal_weight`, RF, 21d) — the control isolating pure predictive skill from advanced portfolio construction techniques.

### Benchmark Suite: Definitions & Design Rationale

Four benchmarks isolate, in sequence, each pipeline stage: $\text{ML Prediction} \to \text{Signal Selection} \to \text{Portfolio Weighting} \to \text{Final Strategy}$.

**Benchmark A — Market Baseline.** $\text{S\&P 500 Buy \& Hold}$. Answers whether active ML-driven management adds enough value over passive market exposure.

**Benchmark B — Momentum (Top 10%) → Equal Weight.** Isolates the ML model's discriminatory power against a mechanical cross-sectional momentum rule.

**Benchmark C — Momentum (Top 10%) → Signal Weighting.** Holds the weighting scheme fixed at Signal Weighting and swaps the ranking signal for momentum, isolating whether the ML model's predicted magnitude carries more valuable information for position sizing than raw momentum. Testing an alternative optimizer here would confound signal and weighting-method effects simultaneously.

**Momentum lookback choice (21 days):** Using a 12-month momentum factor over an OOS window of only ~18 months would expose the comparison to heavy regime-specific bias. Because the ML models predict returns over a 21-day horizon, comparing them to a 12-month factor would conflate the evaluation of predictive skill with differing investment horizons.

**Constraint parity:** Benchmarks B and C use the identical operational constraint set as the ML portfolios (per-asset weight bounds, turnover limits) to guarantee any performance gap is strictly attributable to the predictive signal.

### Monte Carlo Random-Selection Test: Methodology

$N=500$ pseudo-random portfolios are simulated preserving the real strategy's operational constraints (same universe dimension $K$, same 21-day frequency, same weighting scheme, and same 15 bps friction).

For each simulation $j$, net CAGR is computed, forming the null distribution. The empirical p-value:

$$p_{\text{empirical}} = \frac{1}{N} \sum_{j=1}^{N} \mathbb{I}\left(\text{CAGR}_j^{\text{random}} \ge \text{CAGR}_{\text{ML}}\right)$$

A candidate above the 95th empirical percentile ($p < 0.05$) rejects the null of chance selection, confirming the model extracts a real informational pattern.

### Active Risk Metrics: Definitions

Daily active return: $R_{\text{active}, t} = R_{\text{strategy}, t} - R_{b, t}$.

* **Excess CAGR** = $\text{CAGR}_{\text{strategy}} - \text{CAGR}_{\text{benchmark}}$.
* **Tracking Error (TE)** — annualized std. dev. of daily active returns: $\text{TE} = \sigma(R_{\text{active}}) \times \sqrt{252}$. A high TE signals a portfolio structurally misaligned with the benchmark.
* **Information Ratio (IR)** — excess return per unit of active risk: $\text{IR} = \dfrac{(\bar{R}_{\text{strategy}} - \bar{R}_b) \times 252}{\sigma(R_{\text{active}}) \times \sqrt{252}}$. Unlike Sharpe, IR isolates skill in remunerating deviation from the benchmark. **IR > 0.50** = solid; **IR > 1.00** = exceptional.

### Circular Block Bootstrap

To test whether observed Sharpe advantages reflect genuine outperformance rather than temporal luck, a circular block bootstrap ($N=2000$, 21-day block size) resamples daily return paths while preserving the autocorrelation induced by the rebalancing cycle.

### CAPM Alpha/Beta Regression

CAPM regression against the S&P 500 utilizes Newey-West (HAC) robust standard errors with a 21-day lag. The 21-day rebalancing cycles induce memory and autocorrelation in daily returns; the HAC correction adjusts the covariance of estimators to prevent this noise from distorting t-statistics and p-values.

* **Annualized Alpha** — risk-adjusted excess return decoupled from beta.
* **Beta** — systematic sensitivity to the market.
* **Alpha t-stat / p-value** — significance of alpha ($\vert{}t\vert{} > 1.96$ or $p < 0.05$).
* **R²** — share of strategy variance strictly explained by the S&P 500.

### Rolling Sharpe Window Choice

Given an OOS horizon of ~1.5 years, a 126-session (6-month) rolling window balances statistical stability against temporal responsiveness. Including the S&P 500 establishes the real opportunity cost of capital, isolating whether performance reflects genuine predictive skill or simple market-beta exposure.

### Continuous Drawdown Profile: Definition

Rather than summarizing downside risk in a single static figure (Maximum Drawdown), the continuous loss trajectory from the previous equity peak ($W_t$) is mapped:

$$DD_t = \frac{W_t}{\max_{s \le t} W_s} - 1, \quad \forall t \in [1, T]$$

This time series precisely reveals drawdown depth, loss frequency, and recovery duration (underwater duration), enabling a direct comparison of candidate resilience versus the S&P 500 during market stress.