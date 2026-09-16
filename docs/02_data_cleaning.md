# 02. Data Validation and Cleaning — Theoretical Documentation

**Zero-Volume Day Handling Alternatives**

When encountering zero-volume days in raw equity price series, several handling options exist:

* **Row Deletion:**
* *Rejection Reason:* Deleting the entire row discards valid pricing data for the rest of the companies in the universe (e.g., the other 499 in the S&P 500). Conversely, dropping the data point solely for the affected stock breaks the matrix structure of the dataset. Trading days would no longer align across different tickers, destroying the panel alignment required for matrix and tensor operations.


* **Price/Volume Imputation (Forward Fill or Interpolation):**
* *Rejection Reason:* For large-cap equities, zero volume is rarely a data error; it typically indicates a trading halt or suspension. Interpolating or carrying prices forward assumes the asset was tradable. In a backtest environment, this misleads the simulator into executing orders at theoretical prices that were physically inaccessible in real life, artificially inflating performance metrics. It is an error as severe as ignoring the temporal execution lag of trades.


* **Retention with Downstream Filtering (Selected Strategy):**
* *Rationale:* Preserves the exact $T \times N$ panel structure. Zero-volume instances are retained in the master dataset to avoid disrupting the temporal alignment of the matrices. Illiquidity management is instead delegated to downstream modules (cross-sectional factor generation and portfolio construction), where the model's logic enforces restrictions to prevent trade execution on halted days.

---

**Pre-IPO Missingness and Dataset Limitations**

* **Unbalanced Panel Strategy (Pre-IPO):**
Missing values (NaNs) in the raw dataset occur strictly because certain companies had not yet gone public prior to the start of our historical window (2010). Rather than trimming the dataset to the 421 continuously listed companies—which would discard a significant portion of high-performing assets—the dataset maintains an unbalanced panel structure. Tickers enter the eligible cross-section dynamically on their listing date once valid lookback windows for factor calculations are satisfied.
* **Known Limitation: Survivorship Bias:**
It is necessary to acknowledge a structural limitation imposed by the data source. Because historical data was extracted via Yahoo Finance using the *current* constituents of the S&P 500, the dataset inherently suffers from survivorship bias. It omits companies that were part of the index during the 2010–2024 period but went bankrupt or were delisted prior to 2024. Fully resolving this bias requires reconstructing the daily historical index constituents using institutional-grade, paid data terminals (e.g., Bloomberg or CRSP), which falls outside the scope of this project.