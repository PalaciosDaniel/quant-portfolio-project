
# 06. Factor Modeling & Model Selection Specifications — Theoretical Documentation

## Feature Isolation & Complete-Case Analysis

### Sample Alignment Across Normalization Pipelines
To evaluate model performance across normalization schemes (Cross-Sectional Z-Score vs. Percentile Rank), model evaluation must execute on identical observation sets $N_{\text{obs}}$.

1. **Complete-Case Filtering**:
   $$\mathcal{D}_{\text{clean}} = \{(X_{i,t}, y_{i,t}) \in \mathcal{D}_{\text{raw}} \mid \text{NaN} \notin (X_{i,t} \cup \{y_{i,t}\})\}$$
   Observations containing missing values in either the predictor matrix $X$ or forward-return target $y$ are dropped.

2. **Index Intersection Matching**:
   $$\mathcal{I}_{\text{common}} = \text{Index}(\mathcal{D}_{\text{clean}, Z}) \cap \text{Index}(\mathcal{D}_{\text{clean}, \text{Rank}})$$
   Enforcing $\mathcal{I}_{\text{common}}$ guarantees that performance differentials between Z-Score and Percentile Rank models stem exclusively from feature scaling mechanics rather than sample selection variance.

---

## Evaluation Metrics

### Error Metrics (RMSE, MAE)
RMSE and MAE quantify prediction error on the 21-day forward return ($y_{t+21}$). RMSE is used as the primary training loss: its quadratic penalty on large deviations is important in a financial setting to avoid models producing erratic predictions during high-volatility periods or earnings releases. MAE provides a linear, outlier-robust reference for typical-market-condition error. Comparing RMSE and MAE jointly audits for fat tails in residuals — a large gap between the two signals vulnerability to extreme deviations in the cross-sectional panel.

### Ranking-Based Financial Metrics (IC / Rank IC, Information Ratio, Hit Rate, t-statistic)
In cross-sectional modeling, the priority is correctly **ranking** assets by expected return, not point-accuracy of the predicted value. The daily Information Coefficient is computed via Spearman (**Rank IC**), the primary metric for its robustness to outliers, complemented by Pearson (**IC**) to audit for extreme distortions. The daily IC series is summarized via:
- **Information Ratio ($IR_{IC}$)**: stability of the generated alpha, penalizing prediction volatility.
- **Hit Rate (%)**: frequency of days with correct ranking direction.
- **t-statistic**: validates that predictive capacity is statistically significant ($|t| > 2.0$) rather than a sampling artifact.

### Primary vs. Secondary Metric Hierarchy
- **Primary Signal Quality**: Evaluated via **Mean Rank IC**, as the core goal is cross-sectional asset ranking rather than exact point-return prediction.
- **Temporal Consistency & Directional Accuracy**: Measured via **Rank IC IR** and **Hit Rate (%)**.
- **Robustness Audit**: Evaluated via **Median Rank IC** (to verify if the mean is skewed by extreme periods) and **Rank IC Std** (to measure temporal dispersion across folds).
- **Statistical Significance**: Validated via the **Rank IC t-statistic**, used as a complementary significance filter rather than a standalone selection criterion.
- **Secondary Role of Error Metrics**: **RMSE** and **MAE** remain secondary diagnostic tools reserved for evaluating absolute prediction scale error.

### Modular Scoring Function (`evaluate_predictions`)
A single evaluator function unifies error metrics (RMSE, MAE) and ranking metrics (IC, Rank IC, $IR_{IC}$, Hit Rate, t-stat) computation to avoid duplicated code across model optimization and the 21 CPCV folds. It supports dynamic selection of the correlation method (Spearman/Pearson) and standardizes metric naming, enabling direct comparison across experiments and centralized result logging.

---

## Default Model Benchmark: Candidates Considered

### Random Forest — Excluded on Computational Cost
Random Forest was evaluated as an initial benchmark candidate. A single CPCV fold takes ~38 seconds even with only 10 trees; across the full scheme (21 folds × 2 normalization sets), the total cost is disproportionate relative to XGBoost/LightGBM. It was excluded provisionally — not for inferior predictive capacity, but for cost relative to the other candidates — and may be reinstated later as an extension if results from the selected models justify the additional comparison.

### Key Baseline Findings & Empirical Insights
- **Percentile Rank Superiority**: The non-parametric **Percentile Rank** transformation demonstrates a universal benefit across all model architectures, systematically increasing **Rank IC Mean** compared to Z-Score standardization by mitigating tail extremes and uniformizing input signal distributions.
- **XGBoost Default Fragility**: XGBoost achieves the weakest baseline performance (Rank IC Mean of only 0.0110 – 0.0122). Its default tree depth (`max_depth=6`) tends to overfit noisy financial signals, positioning it as a primary candidate for hyperparameter optimization in subsequent stages.

---

## Multicriteria Model Ranking Methodology
To objectively rank the eight default-configuration experiments, a composite score combines three dimensions: raw cross-sectional ranking strength (**Rank IC Mean, 50%**), temporal consistency of the signal (**Rank IC Information Ratio, 30%**), and frequency of positive-IC periods (**Hit Rate, 20%**):

$$Score = 0.50 \cdot Rank(IC_{Mean}) + 0.30 \cdot Rank(IC_{IR}) + 0.20 \cdot Rank(HitRate)$$

Each metric is converted to its within-panel percentile rank before weighting, to standardize scales. The t-statistic is deliberately excluded from the score to avoid redundancy with the Information Ratio, and is retained instead as a complementary significance indicator.

### Candidate Selection Criteria (for optimization phase)
1. Experiments leading the overall Multicriteria Score ranking advance automatically.
2. Any non-linear architecture that, despite not topping the score, stands out sharply on a critical dimension (temporal consistency, directional hit rate, or statistical significance) is rescued.
3. At least one linear and one non-linear model is guaranteed to advance, to test whether optimization changes the hierarchy and conclusions observed in the default benchmark.

---

## Hyperparameter Optimization Methodology

### Optimization Engine & Target Metric
Automated tuning is implemented via an Optuna-based pipeline (`optimize_hyperparameters`). Rather than using an exhaustive grid search, the function employs a Bayesian **Tree-structured Parzen Estimator (TPE)** sampler to efficiently search the hyperparameter space. In each trial, the engine proposes a parameter combination and directly **maximizes out-of-fold Mean Rank IC**, ensuring that selected hyperparameters prioritize cross-sectional asset ranking capacity. The pipeline returns the complete Optuna study, the optimal parameters dictionary (`best_params`), and the best score achieved.

### CPCV Fold Subsampling for the Search Phase
Hyperparameter search runs on a fixed, randomly sampled 7-fold subset of the 21 CPCV folds (`search_splits`, `SEED`-fixed), reused identically across every candidate model. This keeps the comparison fair and neutral across architectures. The tuning phase only needs sufficient ordinal signal to rank configurations against each other — not the full statistical stability required for final evaluation. Once optimal parameters (`best_params`) are identified, they are re-evaluated across all 21 CPCV folds to construct the final OOS benchmark without subsampling bias.

### Model Search Space Design

#### LightGBM (Z-Score & Percentile Rank)
- **Search Space:** `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, and `colsample_bytree`, covering tree structure capacity, sampling, and regularization.
- **Fixed Constraints:** `n_estimators=100` is held constant to control computational budget.
- **Budget:** Evaluated across 20 trials on the 7 `search_splits`.

#### XGBoost (Percentile Rank)
- **Search Space:** `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, and `reg_lambda`, covering capacity, structural complexity penalties, and L1/L2 regularization.
- **Fixed Constraints:** `n_estimators=100` is held constant to control computational cost.
- **Budget:** Evaluated across 20 trials on the 7 `search_splits`.

#### Random Forest (Percentile Rank — Architecture Robustness Analysis)
Random Forest was reconsidered directly in the tuning phase (bypassing an independent default benchmark) after hyperparameter optimization materially reordered the rankings of boosting models.
- **Computational Cost & Scaling:** On the full dataset (~1.6M rows), Random Forest is considerably more expensive than XGBoost/LightGBM because `scikit-learn` evaluates splits exactly rather than via histogram binning. To manage this, search reuses the 7 `search_splits` and sets `n_jobs=4` — taking advantage of independent tree building (bagging) which parallelizes without the thread contention penalty seen in sequential boosting algorithms.
- **Regularization & Search Restrictions (`max_depth` 3–6, `max_samples` 0.2–0.5):** Restricting tree depth and bootstrap fraction is both an efficiency measure and a methodological choice. Given that XGBoost's optimal setup favored near-linear behavior under strong regularization, constraining RF to a comparable capacity range ensures a fair comparison across tree-based families rather than granting one architecture a larger capacity due to compute budget.
- **Trial Budget:** Evaluated initially with an exploratory 5-trial budget to check for promising ordinal signal, followed by a full 20-trial run matching XGBoost and LightGBM.

---

## Regularization & XGBoost Structural Simplification
Hyperparameter optimization for XGBoost reveals a regime of **severe regularization**. High penalties on `gamma` (minimum loss reduction to split) and `min_child_weight` force effective tree pruning. Given a compact feature space of only three variables, these constraints prevent the ensemble from constructing complex, noisy partitions. 

Consequently, XGBoost's post-tuning performance gain does not stem from exploiting highly complex non-linearities, but from structural simplification. By penalizing capacity, Optuna guides the model toward a highly regularized operation that mirrors the monotonicity and stability of linear Ridge regression while retaining the directional flexibility of gradient boosting.

---

## Feature Importance Methodology by Model Family

To audit internal model dependence prior to analyzing effect direction, relative feature importance is extracted based on each family's mathematical design:
- **Tree Ensembles (XGBoost, Random Forest):** Derived from native split metrics (Gini impurity reduction for RF, Gain for XGBoost), quantifying structural impact during space partitioning.
- **Linear Model (Ridge):** Derived from the absolute magnitude of standardized coefficients. Because features use the Percentile Rank representation, inputs share an identical scale range, allowing direct, unbiased weight comparison.

### Factor Ranking (Mean Rank) Methodology
Because importance metrics differ across model families (coefficients vs. Gain vs. impurity), values are converted to within-model ranks (1st, 2nd, 3rd). Computing the **Mean Rank** identifies factors consistently prioritized across architectures. This measures internal structural dependence, not standalone predictive power ($IC$); determining out-of-sample performance drivers requires complementary SHAP analysis.

---

## SHAP Diagnostic Methodology & Sampling Protocol

SHAP values complement aggregate importance measures by quantifying both the **magnitude and direction** of each factor's contribution to individual predictions via:
- **Summary Plots**: Joint view of global feature importance and effect direction across the feature value domain.
- **Dependence Plots**: Marginal effect of factor levels on predicted returns, isolating non-linearities, threshold effects, and saturation points.

Because the goal is **diagnostic** (verifying signal consistency) rather than unbiased error estimation, SHAP calculations do not require full CPCV scheme coverage. Computations are executed on a reduced **7-fold subset** (the `search_splits` from hyperparameter search) with a subsampled **3,000-row evaluation sample per fold**, capturing global importance patterns without excessive computational cost.

---

## Model-Specific SHAP Interpretations

### Random Forest (Bagging Mechanics)
- **Decision Boundary Style:** Operates via rigid, discrete binary splits rather than continuous gradients, resulting in step-function transitions toward positive SHAP values.
- **Liquidity & Momentum Signals:** Systematically penalizes high liquidity. In extreme low-momentum tails, bagging isolates specific non-linear interactions where distressed low-momentum stocks receive high predictive boosts when paired with exceptional upside volatility.

### XGBoost (Boosting Mechanics)
- **Selective Upside Volatility:** Displays no negative left tail for upside volatility; the model does not penalize low upside volatility but acts as a high-precision filter rewarding only the top decile.
- **Liquidity Penalties & Interaction Rescues:** Imposes explicit penalties on high-liquidity assets while deploying non-linear interactions to rescue bottom-momentum assets if supported by high upside volatility or attractive illiquidity.

---

## Economic Rationale & Cross-Model Synthesis

### 1. Factor Economic Rationale (Citations)
- **Upside Volatility**: Positive contribution concentrated in the right tail aligns with investor preference for positive skewness / lottery-like payoff structures.
- **Amihud Illiquidity (`log10_amihud`)**: Reproduces the classic illiquidity premium documented in Amihud (1986, 2002).
- **12-1 Momentum**: Confirms the trend-continuation anomaly documented in Jegadeesh & Titman (1993).

### 2. Standalone vs. Multivariate Factor Dynamics
Contrasting univariate strength with multivariate model weights reveals key interaction dynamics:
- **Momentum Interaction Dependency:** Momentum accounts for only 3.12% of importance in Ridge (Rank 3), but expands to ~22–24% in non-linear models (Rank 2 in both XGBoost and RF). Momentum contributes little in isolation; its predictive signal is activated primarily via conditional interactions with volatility and liquidity.
- **Upside Volatility Dominance:** Maintains a dominant, stable rank across all three architectures (57.15% / 43.21% / 56.12%, Rank 1 unanimously), proving its contribution is predominantly direct.
- **Illiquidity Redundancy:** Drops from 39.73% in Ridge to 21.73% in RF, indicating part of its univariate signal is absorbed by non-linear feature combinations.

### 3. Linear vs. Non-Linear Functional Forms
Dependence plots explain the out-of-sample performance edge of tree architectures over linear models:
- **Threshold Effect (Upside Volatility):** Returns surge strictly in the upper distribution tail rather than scaling linearly.
- **Saturation Effect (Momentum):** Positive contribution flattens at upper extremes, consistent with overbought exhaustion signals.
- **Plateau Filter (Illiquidity):** Functions as an eligibility screen where impact stabilizes once a minimum illiquidity threshold is crossed.

These functional shapes (thresholds, saturation, plateaus) represent structures that decision trees naturally isolate, establishing strong qualitative model validation prior to backtesting.