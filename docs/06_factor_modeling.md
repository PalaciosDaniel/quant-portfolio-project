
# 06. Factor Modeling & Model Selection Specifications — Theoretical Documentation

## Feature Isolation & Complete-Case Analysis

### Sample Alignment Across Normalization Pipelines
To accurately compare models trained on Z-Score vs. Percentile Rank normalizations, they must be evaluated on the exact same universe of assets ($N_{\text{obs}}$).

1. **Complete-Case Filtering**:

   $$I_{\text{clean}} = \{(X_{i,t}, y_{i,t}) \in D_{\text{raw}} \mid \text{NaN} \notin (X_{i,t} \cup \{y_{i,t}\})\}$$

   We drop any observations with missing values in either the features or the forward-return target.

2. **Index Intersection Matching**:

   $$I_{\text{common}} = \text{Index}(D_{\text{clean}, Z}) \cap \text{Index}(D_{\text{clean}, \text{Rank}})$$

   We keep only the indices (assets/days) present in both cleaned datasets. Enforcing this guarantees that any performance difference comes from the mathematical normalization itself, avoiding sample biases (e.g., one method dropping hard-to-predict illiquid assets and artificially improving its metrics).

---

## Evaluation Metrics

### Ranking-Based Financial Metrics (IC / Rank IC, Information Ratio, Hit Rate, t-statistic)
In cross-sectional modeling (like a market-neutral long/short portfolio), the priority is correctly **ranking** assets by expected return, not predicting the exact numerical value of that return. A model might have terrible absolute error but perfectly rank the best and worst assets.

- **Primary Signal Quality (Mean Rank IC)**: Computed via Spearman correlation, it measures ranking accuracy and is robust to outliers.
- **Temporal Consistency (Information Ratio, $IR_{IC}$)**: Measures the stability of the alpha signal over time, penalizing prediction volatility. A high mean Rank IC is dangerous if driven by a few lucky periods; IR ensures consistency.
- **Directional Accuracy (Hit Rate %)**: Defined strictly as the frequency of days where the Rank IC is positive (i.e., the model ordered the universe in the right direction, regardless of whether the market went up or down overall).
- **Statistical Significance (t-statistic)**: Validates that predictive capacity is statistically significant ($|t| > 2.0$) rather than a sampling artifact.

### Error Metrics (RMSE, MAE)
RMSE and MAE are used only as secondary diagnostics. RMSE acts as a primary training loss because its quadratic penalty prevents erratic predictions during high-volatility periods, but it does not dictate model selection.

### Modular Scoring Function (`evaluate_predictions`)
A single evaluator function unifies error metrics (RMSE, MAE) and ranking metrics computation to avoid duplicated code across model optimization and the 21 CPCV folds. It standardizes metric naming, enabling direct comparison across experiments and centralized result logging.

---

## Default Model Benchmark: Candidates Considered

### Random Forest — Excluded on Computational Cost
Random Forest was evaluated as an initial benchmark candidate. A single CPCV fold takes ~38 seconds even with only 10 trees. It was excluded provisionally — not for inferior predictive capacity, but for disproportionate computational cost relative to XGBoost/LightGBM.

### Key Baseline Findings & Empirical Insights
Financial data is characterized by an extremely low signal-to-noise ratio. 
- **Percentile Rank Superiority**: The non-parametric Percentile Rank transformation systematically increases Rank IC Mean across all model architectures by mitigating tail extremes.
- **XGBoost Default Fragility**: With default parameters (like `max_depth=6`), XGBoost performed poorly in the initial benchmark (Rank IC Mean ~0.011). Because our feature space is compact (only 3 factors), deep trees create meaningless micro-segments, memorizing market noise rather than finding structural trends.

---

## Multicriteria Model Ranking Methodology
To objectively rank the initial experiments, a composite score combines three dimensions:

$$Score = 0.50 \cdot Rank(IC_{Mean}) + 0.30 \cdot Rank(IC_{IR}) + 0.20 \cdot Rank(HitRate)$$

Each metric is converted to its within-panel percentile rank before weighting to standardize scales. 

### Candidate Selection Criteria (for optimization phase)
1. Experiments leading the overall Multicriteria Score ranking advance automatically.
2. Any non-linear architecture that stands out sharply on a critical dimension (temporal consistency or hit rate) is rescued.
3. At least one linear and one non-linear model is guaranteed to advance to test if optimization changes the baseline hierarchy.

---

## Hyperparameter Optimization Methodology

### Optimization Engine & Target Metric
Automated tuning uses an Optuna-based pipeline with a Bayesian Tree-structured Parzen Estimator (TPE) sampler. In each trial, the engine directly **maximizes out-of-fold Mean Rank IC**, ensuring that selected hyperparameters prioritize cross-sectional asset ranking capacity. 

### CPCV Fold Subsampling for the Search Phase
Hyperparameter search runs on a fixed, randomly sampled 7-fold subset of the 21 CPCV folds (`search_splits`). This provides sufficient ordinal signal to rank configurations efficiently. Once optimal parameters (`best_params`) are identified, they are re-evaluated across all 21 CPCV folds to construct the final out-of-sample benchmark.

### Regularization Strategy
During the Optuna optimization phase, we forced the non-linear models into a regime of severe regularization. By heavily penalizing complexity and limiting tree depth, the models act more like robust linear models (preventing them from fitting random noise) while retaining the ability to capture key non-linear interactions.

*(Note: Random Forest was reconsidered during tuning, with similarly constrained depths and bootstrap fractions to ensure a fair, compute-equivalent comparison against XGBoost and LightGBM).*

---

## Feature Importance & SHAP Analysis

To audit what the models are actually learning, we use two complementary approaches:

### 1. Mean Rank for Cross-Model Comparability
Since linear models and tree models use different native metrics for feature importance (Coefficients vs. Gain/Impurity), we convert them into a **Mean Rank** (1st, 2nd, 3rd). This allows us to compare structural dependence across entirely different architectures.

### 2. SHAP Values for Direction & Form
Mean Rank only tells us *how much* a factor matters. We use SHAP to understand *how* it matters. SHAP reveals the direction of the effect and the functional form (e.g., linear slopes, threshold effects, or saturation points) learned by the models.

SHAP computations are executed on a reduced 7-fold subset with a subsampled 3,000-row evaluation sample per fold, which captures global importance patterns without excessive computational cost.

---

## Economic Rationale & Cross-Model Synthesis

Comparing the linear model (Ridge) to the tree models reveals why non-linear architectures have an out-of-sample edge in this specific setup:

### The Momentum Interaction
In the Ridge model, 12-1 Momentum shows weak standalone importance (Rank 3, ~3%). However, in tree models, its importance jumps significantly (Rank 2, ~23%). This occurs because Momentum's predictive power is highly conditional: an asset with positive momentum needs to be accompanied by strong **Upside Volatility** to be selected by the decision trees.

### The Illiquidity Premium
The models systematically reward illiquidity, reproducing the classic Amihud premium. The tree models natively capture the order book physics: when an illiquid stock (low supply) gains momentum, the lack of selling depth forces the price upward much faster than in a highly liquid stock.

---