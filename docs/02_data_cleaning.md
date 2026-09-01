# 02. Data Validation and Cleaning — Theoretical Documentation

## Zero-Volume Day Handling Alternatives
When encountering zero-volume days in raw equity price series, several handling options exist:

1. **Row Deletion:**
   - *Rationale:* Eliminates days with non-existent liquidity.
   - *Rejection Reason:* Disrupts global panel alignment across tickers, creating ragged time-series arrays that complicate tensor/matrix operations in factor models.

2. **Price/Volume Imputation (Forward Fill or Interpolation):**
   - *Rationale:* Smooths out missing activity.
   - *Rejection Reason:* Introduces synthetic trading signals and artificially inflates liquidity metrics.

3. **Retention with Downstream Filtering (Selected Strategy):**
   - *Rationale:* Preserves exact panel structure. Zero-volume instances are retained in the master dataset, while downstream cross-sectional factor generation and portfolio selection modules enforce volume thresholds (e.g., minimum 20-day median daily dollar volume) to restrict trade execution on illiquid or suspended days.

## Pre-IPO Missingness (Unbalanced Panel Strategy)
Missing values in the raw dataset occur strictly prior to historical IPO dates. Rather than trimming the panel to the 421 continuously listed companies (which would introduce sample selection bias), the dataset maintains an unbalanced panel structure. Tickers enter the eligible cross-section dynamically on their listing date once valid lookback windows for factor calculations are satisfied.