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