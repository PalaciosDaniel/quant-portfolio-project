METRIC_FORMATTERS = {
    "CAGR": lambda x: f"{x:.2%}",
    "Vol": lambda x: f"{x:.2%}",
    "Sharpe": lambda x: f"{x:.3f}",
}

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

# NOTE: assumed available in the notebook namespace: `BENCHMARK_RETURNS_PATH`,
# `FINAL_WEIGHTS_PATH`, `FINAL_CANDIDATE`, `net_returns` and `simple_returns_ext`.

# --- Evaluation window -------------------------------------------------------
BENCH_START = "2025-01-16"
BENCH_END = "2026-06-18"

# --- Beta estimation ---------------------------------------------------------
NW_MAX_LAGS = 21  # matches the rebalancing cycle
TRADING_DAYS_PER_YEAR = 252

# --- Outlier tests (§7) ------------------------------------------------------
N_DROP = 5
CROSS_SECTIONAL_COST_PROXY = 0.05  # fraction of the mean daily net return, used as a cost proxy

def load_benchmark_returns(path, start=BENCH_START, end=BENCH_END):
    """Benchmark daily returns over the evaluation window, as a date-indexed Series."""
    df = pd.read_parquet(path)
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    return df.set_index("date")["benchmark_return"]


def get_strategy_returns(net_returns, candidate, return_col="net_return_base"):
    """Daily net returns of one (model, portfolio) candidate, indexed by date."""
    mask = (net_returns["model"] == candidate["model"]) & (
        net_returns["portfolio"] == candidate["portfolio"]
    )
    df = net_returns[mask].copy()
    if "date" in df.columns:
        df = df.set_index("date")
    return df[return_col]


def align_series(strategy, benchmark):
    """Restrict both series to their common dates."""
    common_dates = strategy.index.intersection(benchmark.index).sort_values()
    return strategy.loc[common_dates], benchmark.loc[common_dates]


def fit_capm_newey_west(strategy, benchmark, max_lags=NW_MAX_LAGS, trading_days=TRADING_DAYS_PER_YEAR):
    """CAPM regression with Newey-West (HAC) standard errors."""
    X = sm.add_constant(benchmark.values)
    fit = sm.OLS(strategy.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})

    daily_alpha, beta = fit.params
    return {
        "beta": beta,
        "beta_tstat": fit.tvalues[1],
        "daily_alpha": daily_alpha,
        "annual_alpha": daily_alpha * trading_days,
        "alpha_pvalue": fit.pvalues[0],
        "r_squared": fit.rsquared,
    }


def print_capm_summary(capm, role, max_lags=NW_MAX_LAGS):
    print("=" * 90)
    print(f"CAPM BETA - {role} (Newey-West, lags={max_lags})")
    print("=" * 90)
    print(f"beta             = {capm['beta']:.3f}  (t={capm['beta_tstat']:.2f})")
    print(f"annualized alpha = {capm['annual_alpha']:.2%}  (p={capm['alpha_pvalue']:.4f})")
    print(f"R-squared        = {capm['r_squared']:.3f}")
    print("=" * 90)


def drop_best_days(returns, n_drop):
    """Replace the `n_drop` highest-return days with 0.0; also return those dates."""
    top_dates = returns.nlargest(n_drop).index
    penalized = returns.copy()
    penalized.loc[top_dates] = 0.0
    return penalized, top_dates


def quick_metrics(returns, trading_days=TRADING_DAYS_PER_YEAR):
    """CAGR, annualized volatility and Sharpe (CAGR / vol, no risk-free rate)."""
    years = len(returns) / trading_days
    cagr = (1.0 + returns).prod() ** (1.0 / years) - 1.0
    vol = returns.std(ddof=1) * np.sqrt(trading_days)
    sharpe = cagr / vol if vol > 0 else np.nan
    return cagr, vol, sharpe


def metrics_row(portfolio, state, returns):
    cagr, vol, sharpe = quick_metrics(returns)
    return {"Portfolio": portfolio, "State": state, "CAGR": cagr, "Vol": vol, "Sharpe": sharpe}


def best_days_test(strategy, benchmark, strategy_label, n_drop=N_DROP):
    """Original vs. missing-best-N-days metrics for benchmark and strategy.

    Returns the comparison table, the penalized series and the dropped dates.
    """
    bmk_pen, bmk_dates = drop_best_days(benchmark, n_drop)
    strat_pen, strat_dates = drop_best_days(strategy, n_drop)
    state_pen = f"Missing Best {n_drop} Days"

    table = pd.DataFrame(
        [
            metrics_row("S&P 500 (Benchmark)", "Original", benchmark),
            metrics_row("S&P 500 (Benchmark)", state_pen, bmk_pen),
            metrics_row(f"{strategy_label} (Strategy)", "Original", strategy),
            metrics_row(f"{strategy_label} (Strategy)", state_pen, strat_pen),
        ]
    )
    penalized = {"strategy": strat_pen, "benchmark": bmk_pen}
    dropped = {"strategy": strat_dates, "benchmark": bmk_dates}
    return table, penalized, dropped


def print_robustness_table(table, title, notes=()):
    print("=" * 90)
    print(title)
    print("=" * 90)
    print(table.to_string(index=False, formatters=METRIC_FORMATTERS))
    if notes:
        print("-" * 90)
        for note in notes:
            print(note)
    print("=" * 90)


def plot_wealth_index(strategy, strategy_pen, benchmark, benchmark_pen, strat_top_dates, n_drop=N_DROP):
    """Wealth index (base 100) of strategy and benchmark, original vs. penalized."""
    sns.set_theme(style="whitegrid", rc={"axes.spines.top": False, "axes.spines.right": False})

    wealth = lambda r: 100 * (1.0 + r).cumprod()
    w_strat, w_strat_pen = wealth(strategy), wealth(strategy_pen)
    w_bmk, w_bmk_pen = wealth(benchmark), wealth(benchmark_pen)

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(w_strat.index, w_strat, label="Strategy (Original)", color="#1f77b4", linewidth=2.5)
    ax.plot(w_strat_pen.index, w_strat_pen, label=f"Strategy (Missing Best {n_drop} Days)",
            color="#7fb1d3", linewidth=2.5)
    ax.plot(w_bmk.index, w_bmk, label="S&P 500 (Original)", color="#595959", linewidth=2)
    ax.plot(w_bmk_pen.index, w_bmk_pen, label=f"S&P 500 (Missing Best {n_drop} Days)",
            color="#b3b3b3", linewidth=2)

    # Best days only on the original strategy curve, to avoid clutter
    ax.scatter(strat_top_dates, w_strat.loc[strat_top_dates], color="#d62728", zorder=5, s=50,
               label=f"Top {n_drop} Days (Strategy)")

    ax.set_title(f"Impact of the Top {n_drop} Days on Compounding (Wealth Index)",
                 fontsize=16, pad=20, fontweight="bold")
    ax.set_ylabel("Portfolio Value (Base 100)", fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=0)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="lightgray", fontsize=11)

    plt.tight_layout()
    plt.show()


def cross_sectional_drop_best_stock(
    final_weights, candidate, simple_returns, strategy_net, cost_proxy=CROSS_SECTIONAL_COST_PROXY
):
    """Daily portfolio return after removing the best-contributing stock each day."""
    mask = (final_weights["model"] == candidate["model"]) & (
        final_weights["portfolio"] == candidate["portfolio"]
    )
    wide_weights = (
        final_weights[mask]
        .pivot(index="date", columns="ticker", values="weight")
        .reindex(simple_returns.index)
        .ffill()
        .fillna(0.0)
    )
    returns_aligned = simple_returns.reindex(columns=wide_weights.columns).fillna(0.0)

    # Contribution = weight(t-1) * return(t)
    contributions = wide_weights.shift(1) * returns_aligned
    gross_orig = contributions.sum(axis=1)
    gross_penalized = gross_orig - contributions.max(axis=1)

    common_dates = gross_orig.index.intersection(strategy_net.index)
    return gross_penalized.loc[common_dates] - strategy_net.loc[common_dates].mean() * cost_proxy