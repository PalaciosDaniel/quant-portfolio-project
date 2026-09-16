# 03. Returns & Feature Engineering — Theoretical Documentation

## Base Return Transformations

### Theoretical Foundations
Transforming non-stationary asset prices ($P_t$) into stationary return series is required for quantitative modeling. We define three standard formulations:

1. **Log Returns ($\mathbf{r_t}$):**
   $$r_t = \ln(P_t) - \ln(P_{t-1})$$
   Used for continuous-time statistical modeling, stationarity guarantees, and additive volatility estimation over time. Under continuous compounding, $\mu_{\text{log}} \approx \mu_{\text{simple}} - \frac{1}{2}\sigma^2$.

2. **Simple Discrete Returns ($\mathbf{R_t}$):**
   $$R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$
   Cross-sectionally additive metrics required for actual strategy P&L computation, linear portfolio aggregation, and transaction cost modeling.

3. **Cumulative Compounded Returns ($\mathbf{C_t}$):**
   $$C_t = \prod_{k=1}^{t} (1 + R_k) - 1$$
   Used to reconstruct continuous performance curves, monitor maximum drawdown levels, and normalize performance from baseline $t=0$. Bounded below at $-1.0$ ($-100\%$).

--- 

Aquí tienes la redacción ajustada, integrando tus ideas sobre la recogida de beneficios y la liquidación de posiciones, y eliminando cualquier mención a conceptos de microestructura más opacos como el *bid-ask bounce*.

Al igual que antes, te lo dejo en inglés para que encaje directamente en tu documento:

---

## Momentum & Reversal Factor Dynamics

### Theoretical Foundations

The momentum effect is one of the most widely documented asset pricing anomalies in empirical finance (Jegadeesh & Titman, 1993). It posits that assets with high relative performance over the medium term ($3$ to $12$ months) tend to continue outperforming in subsequent periods, while historical losers continue to underperform.

To capture the underlying trend without distorting the signal, we decompose raw price momentum into two distinct, complementarily orthogonal dynamics:

1. **12-1 Intermediate Momentum ($\mathbf{M_{12,1}}$):**

$$M_{12,1,t} = \prod_{k=1}^{11} (1 + R_{t-21 \cdot k}) - 1 = \frac{P_{t-21}}{P_{t-252}} - 1$$

Calculated from month $t-12$ to month $t-1$ (purging the most recent $21$ trading days).

* **Exclusion of Month $t-1$:** Omitting the immediate trailing month is essential to eliminate short-term mean reversion and temporary price distortions. Over very short windows, prices often experience pullbacks driven by investors taking profits after a strong run, or by institutions liquidating large positions. These actions move the price purely due to temporary supply and demand imbalances, which can mask the true medium-term trend.


2. **1-Month Short-Term Reversal ($\mathbf{STR_{1}}$):**

$$STR_{1,t} = R_{t-21, t} = \frac{P_t}{P_{t-21}} - 1$$

Calculated strictly over the trailing $21$ trading days.

* **Rationale:** Isolates these short-term market **overreactions** and temporary liquidity frictions (e.g., the sudden price drops from large block liquidations). While $M_{12,1}$ provides a clean structural trend signal, $STR_{1}$ captures these localized mean-reversion dynamics, allowing the model to differentiate between temporary noise and actual momentum breakdown.

> **Objective:** Decouple medium-term price persistence from short-term market noise (profit-taking and liquidity frictions) to supply clean, non-contaminated trend features to downstream ML models.

--- 

## Volatility & Risk Asymmetry Factors

### Theoretical Foundations
Total volatility fails to distinguish between desirable upside variation and damaging downside tail risk. We construct four complementary risk metrics evaluated over short-term (21-day) and long-term (252-day) windows:

1. **Total Rolling Volatility ($\mathbf{\sigma_t}$):**
   Standard deviation of daily log returns over rolling lookback window $N$. Serves as an unoriented metric for total asset agitation.

2. **Low-Volatility Factor ($\mathbf{\text{LowVol}_t}$):**
   $$\text{LowVol}_t = -\sigma_t$$
   Encodes the **Low-Volatility Anomaly**, which demonstrates that low-risk stocks historically achieve higher risk-adjusted returns than high-risk stocks, contradicting standard CAPM predictions.
   * **Behavioral & Institutional Drivers:** Driven by investor preference for lottery-like payoffs (overpaying for high-beta assets) and institutional leverage constraints that force benchmarked managers into high-volatility stocks.

3. **Downside Volatility ($\mathbf{\sigma_{\text{down}, t}}$):**
   Standard deviation calculated strictly on negative return days ($R_t < 0$). Captures asymmetric tail risk, financial leverage effects, and panic-driven stop-loss cascades. Provides the foundation for Sortino ratio modeling.

4. **Upside Volatility ($\mathbf{\sigma_{\text{up}, t}}$):**
   Standard deviation calculated strictly on positive return days ($R_t > 0$). Isolates upside momentum surges and differentiates demand-driven volatility from structural asset distress.

---

Tu explicación es totalmente correcta. Has definido bien el mecanismo matemático, el problema real de ejecución (falta de contraparte que absorba la orden) y el concepto de la prima de iliquidez.

Además, el matiz que haces sobre el S&P 500 es crítico. En una entrevista, reconocer que "la iliquidez en el S&P 500 es relativa" demuestra que no solo aplicas fórmulas a ciegas, sino que entiendes el universo con el que estás trabajando (la abismal diferencia de volumen entre las *Magnificent 7* y el percentil más bajo).

Vamos a integrar exactamente esa reflexión tuya en el `.md`, porque enriquece mucho la justificación del factor. El bloque quedaría así:

---

## Market Microstructure & Illiquidity (Amihud Factor)

### Formulation & Microstructure Rationale

In modern asset pricing literature, illiquidity represents a systematic risk dimension where market participants demand a return premium to compensate for the risk of getting trapped in a position or facing high execution friction. In the absence of high-frequency Level-2 order book data, we measure price impact and market depth using the daily Amihud (2002) illiquidity ratio:

$$K_t = \frac{\vert{}R_t\vert{}}{\text{Dollar Volume}_t} = \frac{\vert{}R_t\vert{}}{P_t \cdot V_t}$$

* **Interpretation:** Measures the absolute percentage price change generated per dollar of trading volume. Higher values indicate thin, shallow order books where large orders cause substantial price impact due to a lack of immediate counterparty depth.
* **Context within the S&P 500:** While the S&P 500 universe is generally highly liquid (effectively infinite liquidity for non-institutional investors due to dealer and market maker presence), trading volumes span several orders of magnitude. The liquidity gap between the top constituents (e.g., the "Magnificent 7") and the bottom percentile is massive. We include this factor to capture this *relative* illiquidity, letting the model determine if an illiquidity premium still exists within large-cap equities.
* **Rolling Smoothing (21 Days):** Raw daily Amihud ratios contain extreme microstructure noise (e.g., holiday volume drops) that would induce excessive, unsustainable portfolio turnover and transaction costs. A 21-day moving average ($\approx 1$ trading month) provides the optimal balance by stabilizing signal turnover while preserving cross-sectional illiquidity premiums.

### Preprocessing Requirements

Because dollar volume acts in the denominator, the ratio is inherently sensitive to stochastic liquidity shocks. As a result, raw values span multiple orders of magnitude and exhibit extreme right-skewness. Downstream machine learning ingestion requires log transformation ($\ln(K_t + \epsilon)$) followed by cross-sectional Winsorization and Z-score standardization.

--- 