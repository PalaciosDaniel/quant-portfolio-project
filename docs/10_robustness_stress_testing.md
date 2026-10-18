# Notebook 10 — Robustness & Stress Testing — Theoretical Documentation

## Design Principle

The notebook is **adversarial**: it looks for the conditions under which the selected candidate (Random Forest, Top 10 Signal Weight, 21d) degrades, loses statistical validity, or disappears. Each test receives a qualitative verdict — **Passed, Passed with caveats, Mixed, Not passed** — and failures are reported even when structural (a long-only, high-beta strategy cannot be expected to survive a bear market). Verdicts are judgments, not thresholds; the only pre-specified quantitative criterion is deflated PSR > 0.95. Unless stated otherwise, figures refer to the OOS window **2025-01-16 → 2026-06-18 (357 trading days)**.

---

## Deflated Sharpe Ratio

The DSR (**Bailey & López de Prado, 2014**) corrects an observed Sharpe for the selection bias of a large specification search: with many model/universe/weighting combinations, the chance of finding a "winner" by luck grows quickly (p-hacking). It adjusts for non-normality (skewness, raw kurtosis), sample size, and the number of independent trials.

**Trial-count convention.** Structural hypotheses (unique model × weighting scheme at the 21-day base frequency, **N = 42**) are distinguished from execution variants (other rebalancing frequencies replaying the same signal). The **conservative** scenario counts the full grid (**N = 210**).

**Procedure.** (1) Cross-sectional Sharpe variance across trials. (2) Expected maximum Sharpe by chance given N (Euler–Mascheroni constant). (3) Probabilistic Sharpe Ratio of the observed Sharpe against it, using daily-return skewness and raw kurtosis. (4) Deflated PSR > 0.95 rejects the hypothesis of a search artifact.

**Reading the result.** Expected Sharpe by chance: **1.04** (N = 42) and **1.10** (N = 210); observed **2.47** over 357 observations (skewness 0.09, kurtosis 9.96). Deflated PSR: **0.953** and **0.945**. The observed Sharpe is far above luck, but the verdict flips at the 0.95 threshold depending on the trial count, so the reading is "probably not selection bias", not proof.

**Notes.**
- The Sharpe used throughout the project is CAGR ÷ volatility (risk-free = 0), and it is the value fed to the PSR. *[Confirm what `probabilistic_sharpe_ratio` expects; mean/std-based Sharpe would be slightly higher.]*
- The trial count is a judgement. *[State whether N includes only the notebook 09 grid or also earlier feature/hyperparameter searches.]*
- *[State whether the Sharpe variance is shared between scenarios or computed per scenario.]*
- *[Formulas for expected-max-Sharpe and PSR can be added from the paper.]*

---

## Training Window & Prediction Horizon Sensitivity

- Only the winning architecture is retrained (avoids tripling compute). Rebalancing is fixed at 21 days to isolate the horizon effect from the frequency effect (notebook 09 §5). Tail forward returns that are NaN are used only for evaluation, never for trading.
- **Comparability.** An independent retraining gives different absolute levels: (21d, expanding) reaches ~82% CAGR vs. ~89% for the final model; rank correlation ~0.98; top-decile overlap ~86% (boundary names near the 90th percentile move a concentrated CAGR by several points); ~44 names per date because constraints apply after selection.
- **How to read the results.** All six configurations remain strongly positive (net Sharpe **2.36–2.80**, CAGR **82–93%**, max drawdown **30–32%**). The Sharpe gaps between windows (**0.16–0.22**) are only slightly larger than the gap between the retrained (21d, expanding) model and the final model (**0.11**, retraining noise alone), so **the ranking of configurations is not established**. The final model's 21-day horizon scores lowest within each window, so it was not chosen on these results; changing it now would be selection on the OOS window.
- **Code organization.** All functions live in `src/portfolio/hyperparameter_sensitivity.py`; some duplicate logic elsewhere (turnover, drift, costs). This is deliberate: the section builds the full pipeline from factors to net returns, and existing validated code was left untouched.
- **Limitation.** Retraining variability is attributed to seeds and boundary names but **not measured**; repeating (21d, expanding) with several seeds would size it.

---

## Universe (Sector) Sensitivity

**Motivation.** The OOS period coincides with a technology-led rally; confining the model to one sector at a time neutralizes sector rotation, so alpha must come from stock-picking.

**Design.**
- **Pre-registered primary cut:** Top 20 (two deciles), fixed before looking at performance, favoring diversification in confined samples and avoiding data-snooping; Top 10/30 are robustness checks.
- **Constraint rescaling:** the main portfolio selects ~50 names; a ~60-stock sector yields ~6, and a static 5% cap would leave up to 70% forced liquidity. Min/max weights are rescaled with the selected-name count, preserving the same relative slack versus 1/N and versus the maximum weight implied by the Gamma decay of the main portfolio *[define, or link]*. Turnover constraint with drift as in the main strategy.
- **Local active risk:** each portfolio is compared with the equal-weight portfolio of its own sector (not the S&P 500), so excess return reflects selection skill. Metrics: Sharpe differential, Tracking Error, Information Ratio.
- **Sector mapping:** Yahoo Finance sector labels (not official GICS), downloaded with incremental checkpointing, plus a manual patch for unresolved tickers.
- **Zero imputation:** sectors with no selected names get weight and count 0, not NaN; otherwise mean exposure would be measured "only when the model invests in it".

**How to read the results.** The Top 20 beats its sector benchmark in all four sectors (excess CAGR **+15.5%** Healthcare to **+110.2%** Technology; Sharpe difference **+0.42 to +1.66**; IR **0.98–2.34**), so the signal ranks stocks within sectors. Caveats: dominated by Technology (CAGR 147%, Sharpe 3.06) while Healthcare reaches only 0.68 and Financial Services 1.14; max drawdowns are worse than each benchmark's (22–37% vs. 15–26%); four sectors and one cut (Top 20, not the final Top 10) show within-sector ranking power, **not sector neutrality of the final portfolio**.

---

## Outlier Resistance

- **Best 5 days:** heavy tails can make a Sharpe depend on a few euphoric days. Zeroing the 5 best days (1.4% of the sample) for both strategy and index: strategy Sharpe 2.47 → **1.48** (−40%, CAGR 48.6%), S&P 500 1.02 → 0.15 (−86%). Three of the strategy's five best days are also among the S&P 500's five best — consistent with β ≈ 1.8.
- **Wealth index:** compounding makes the removal geometric; the plot shows whether outperformance is gradual or a single jump.
- **Best stock every day (cross-sectional):** guards against one "hero" stock carrying the rest. CAGR 89.75% → **0.26%**, Sharpe 2.47 → **0.01**. The test is **severe by construction** (removes the best outcome ex post, every day, from a ~44-position portfolio), **uncalibrated** (no reference portfolio) and **approximate** (gross contributions minus a cost proxy equal to 5% of the mean daily net return). It provides no evidence that profits are spread across holdings, but its absolute result cannot be interpreted without a reference.
- These Sharpe ratios are CAGR ÷ volatility without a risk-free rate. The tests do not address sample length or regime.

---

## Advanced Attribution

- **Up/down capture:** arithmetic daily means (not CAGR) to avoid compounding distortions on one-sided partitions. Observed: **211.7%** up, **171.4%** down, spread 40.3 points. Rough check: β = 1.84 plus a constant daily alpha of ~0.16% implies ≈207% / ≈163%, so the spread looks like the alpha intercept rather than convexity. This is an interpretation, not a test.
- **Rolling CAPM (60 days):** plain OLS point estimates on overlapping windows (no Newey-West: too few degrees of freedom), not significance tests. Beta 1.55–2.26 (mean 1.84), alpha mean 39.8% (−4.8% to +84%), tracking error 21.1%.
- **Stress resampling:** 21-day blocks, worst third of S&P 500 blocks, 12 blocks per synthetic year, N = 2000, preserving cross-asset covariance. Result: S&P −33.5%, strategy **−38.8%** (95% interval −64.2% to −6.1%), beats the index in 26.7% of runs, profitable in 1.2%. A loss of only ~1.16× the index's despite β = 1.84 means selection absorbs part of the blow, but gives no protection. **Thin pool:** built from the OOS window only; 337 overlapping blocks ≈ 17 independent ones; the "stress" threshold is a +0.18% 21-day return, so the blocks are flat-to-down stretches of a bull market from a handful of episodes.
- **Active drawdown:** relative wealth $RW_t = W^{\text{strategy}}_t / W^{\text{S\&P}}_t$; drawdown from its running maximum (max **−18.15%**); underwater duration = longest run below that maximum (**59 days**). In absolute terms the retrained (21d, expanding) model shows ~−32% max drawdown and 86 days underwater; volatility is 33–35% in every configuration.

---

## Synthesis

Scorecard: **1 Passed** (training window/horizon), **5 Passed with caveats** (DSR, sector, best-5-days, rolling attribution, active drawdown), **1 Mixed** (capture), **2 Not passed** (cross-sectional outliers; stress resampling, expected for long-only high-beta).

**Supported:** ranking content in this sample; not dependent on one parameter choice or on technology alone; stable exposure.
**Not supported:** defensive/market-neutral behaviour; breadth across holdings; a ranking of configurations; extrapolating CAGR ≈ 90% / Sharpe ≈ 2.5 (357 days, single bullish regime, kurtosis ≈ 10).
**Overall:** consistent with a genuine cross-sectional signal rather than a data-mining artifact, but not proven; the packaging (high-beta, long-only, concentrated) loses heavily in a prolonged decline. Supports the signal as a research result, not the strategy as robust or deployable.

**Limitations.** One OOS window reused by every test (sensitivity, not independent validation; no bear market observed); retraining variability unmeasured; thin stress pool; DSR trial count is a judgement; approximations (cost proxy; Sharpe without risk-free rate).

**Next steps.** (1) Repeat (21d, expanding) with several seeds to size retraining noise. (2) Extend evaluation to earlier years including a bear market. (3) Evaluate a beta-hedged version to test whether alpha survives without market exposure. (4) Calibrate the cross-sectional outlier test against a reference portfolio (e.g., random Top 10).