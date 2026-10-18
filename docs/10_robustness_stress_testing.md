# Notebook 10 — Robustness & Stress Testing — Theoretical Documentation

## Design Principle

The notebook is **adversarial**: it looks for the conditions under which the selected candidate (Random Forest, Top 10 Signal Weight, 21d) degrades, loses statistical validity, or disappears. Each test receives a qualitative verdict — **Passed, Passed with caveats, Mixed, Not passed** — and failures are reported even when structural (a long-only, high-beta strategy cannot be expected to survive a bear market). Verdicts are judgments, not thresholds; the only pre-specified quantitative criterion is deflated PSR > 0.95. Unless stated otherwise, figures refer to the OOS window **2025-01-16 → 2026-06-18 (357 trading days)**.

---

## Deflated Sharpe Ratio

The DSR (**Bailey & López de Prado, 2014**) corrects an observed Sharpe for the selection bias of a large specification search: with many model/universe/weighting combinations, the chance of finding a "winner" by luck grows quickly (p-hacking). It adjusts for non-normality (skewness, raw kurtosis), sample size, and the number of independent trials.

**Trial-count convention.** Structural hypotheses (unique model × weighting scheme at the 21-day base frequency, **N = 42**) are distinguished from execution variants (other rebalancing frequencies replaying the same signal). The **conservative** scenario counts the full grid (**N = 210**).

**Procedure.** (1) Cross-sectional Sharpe variance across trials. (2) Expected maximum Sharpe by chance given N (Euler–Mascheroni constant). (3) Probabilistic Sharpe Ratio of the observed Sharpe against it, using daily-return skewness and raw kurtosis. (4) Deflated PSR > 0.95 rejects the hypothesis of a search artifact.

---

## Training Window & Prediction Horizon Sensitivity

- Only the winning architecture is retrained (avoids tripling compute). Rebalancing is fixed at 21 days to isolate the horizon effect from the frequency effect (notebook 09 §5). Tail forward returns that are NaN are used only for evaluation, never for trading.
- **Comparability.** An independent retraining gives different absolute levels: (21d, expanding) reaches ~82% CAGR vs. ~89% for the final model; rank correlation ~0.98; top-decile overlap ~86% (boundary names near the 90th percentile move a concentrated CAGR by several points); ~44 names per date because constraints apply after selection.
- **Code organization.** All functions live in `src/portfolio/hyperparameter_sensitivity.py`; some duplicate logic elsewhere (turnover, drift, costs). This is deliberate: the section builds the full pipeline from factors to net returns, and existing validated code was left untouched.
- **Limitation.** Retraining variability is attributed to seeds and boundary names but **not measured**; repeating (21d, expanding) with several seeds would size it.

**Limitation:** the 82% vs. 89% gap is not negligible. We attribute it to training variability plus boundary changes, but have not measured it. Repeating (21d, expanding) with several seeds would show whether both values fall in the same range. Conclusions about rolling vs. expanding are reliable only if their difference clearly exceeds this gap.

---

## Universe (Sector) Sensitivity

**Motivation.** The OOS period coincides with a technology-led rally; confining the model to one sector at a time neutralizes sector rotation, so alpha must come from stock-picking.

**Design.**
- **Pre-registered primary cut:** Top 20 (two deciles), fixed before looking at performance, favoring diversification in confined samples and avoiding data-snooping; Top 10/30 are robustness checks.
- **Constraint rescaling:** the main portfolio selects ~50 names; a ~60-stock sector yields ~6, and a static 5% cap would leave up to 70% forced liquidity. Min/max weights are rescaled with the selected-name count, preserving the same relative slack versus 1/N and versus the maximum weight implied by the Gamma decay of the main portfolio. Turnover constraint with drift as in the main strategy.
- **Local active risk:** each portfolio is compared with the equal-weight portfolio of its own sector (not the S&P 500), so excess return reflects selection skill. Metrics: Sharpe differential, Tracking Error, Information Ratio.
- **Sector mapping:** Yahoo Finance sector labels (not official GICS), downloaded with incremental checkpointing, plus a manual patch for unresolved tickers.
- **Zero imputation:** sectors with no selected names get weight and count 0, not NaN; otherwise mean exposure would be measured "only when the model invests in it".

---

## Outlier Resistance

- **Best 5 days:** heavy tails can make a Sharpe depend on a few euphoric days. Zeroing the 5 best days (1.4% of the sample) for both strategy and index: strategy Sharpe 2.47 → **1.48** (−40%, CAGR 48.6%), S&P 500 1.02 → 0.15 (−86%). Three of the strategy's five best days are also among the S&P 500's five best — consistent with β ≈ 1.8.
- **Wealth index:** compounding makes the removal geometric; the plot shows whether outperformance is gradual or a single jump.
- **Best stock every day (cross-sectional):** guards against one "hero" stock carrying the rest. CAGR 89.75% → **0.26%**, Sharpe 2.47 → **0.01**. The test is **severe by construction** (removes the best outcome ex post, every day, from a ~44-position portfolio), **uncalibrated** (no reference portfolio) and **approximate** (gross contributions minus a cost proxy equal to 5% of the mean daily net return). It provides no evidence that profits are spread across holdings, but its absolute result cannot be interpreted without a reference.
- **Clarification:** These Sharpe ratios are CAGR ÷ volatility without a risk-free rate. The tests do not address sample length or regime.

---

## Advanced Attribution

- **Up/down capture:** arithmetic daily means (not CAGR) to avoid compounding distortions on one-sided partitions. Observed: **211.7%** up, **171.4%** down, spread 40.3 points. Rough check: β = 1.84 plus a constant daily alpha of ~0.16% implies ≈207% / ≈163%, so the spread looks like the alpha intercept rather than convexity.
- **Rolling CAPM (60 days):** plain OLS point estimates on overlapping windows (no Newey-West: too few degrees of freedom), not significance tests. Beta 1.55–2.26 (mean 1.84), alpha mean 39.8% (−4.8% to +84%), tracking error 21.1%.
- **Stress resampling:** 21-day blocks, worst third of S&P 500 blocks, 12 blocks per synthetic year, N = 2000, preserving cross-asset covariance. Result: S&P −33.5%, strategy **−38.8%** (95% interval −64.2% to −6.1%), beats the index in 26.7% of runs, profitable in 1.2%. A loss of only ~1.16× the index's despite β = 1.84 means selection absorbs part of the blow, but gives no protection. **Thin pool:** built from the OOS window only; 337 overlapping blocks ≈ 17 independent ones; the "stress" threshold is a +0.18% 21-day return, so the blocks are flat-to-down stretches of a bull market from a handful of episodes.
- **Active drawdown:** relative wealth $RW_t = W^{\text{strategy}}_t / W^{\text{S\&P}}_t$; drawdown from its running maximum (max **−18.15%**); underwater duration = longest run below that maximum (**59 days**). In absolute terms the retrained (21d, expanding) model shows ~−32% max drawdown and 86 days underwater; volatility is 33–35% in every configuration.