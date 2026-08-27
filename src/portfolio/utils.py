import pandas as pd

def compute_avg_portfolio_size(weights_df, model, portfolio_name):
    subset = weights_df[
        (weights_df["model"] == model) & (weights_df["portfolio"] == portfolio_name)
    ]
    active_counts = (
        subset[subset["weight"] > 1e-4]
        .groupby("date")["ticker"]
        .count()
    )
    return active_counts.mean()

MAX_TURNOVER = 0.30

def get_macro_stats(df):
    return df.groupby("weighting_scheme")["turnover"].agg(
        mean="mean",
        maximum="max",
        p95=lambda x: np.percentile(x, 95),
        violations=lambda x: (x > (MAX_TURNOVER + 1e-6)).sum(),
    )


# -------------------------------------------------------------------------
# Helper Function for Top N Extraction & Printing
# -------------------------------------------------------------------------

TOP_N_CANDIDATES = 5

DISPLAY_COLUMNS = [
    "frequency_days",
    "model",
    "portfolio",
    "CAGR",
    "annualized_volatility",
    "annualized_turnover",  # Critical metric for execution viability
    "Sharpe",
    "Sortino",
    "Calmar",
    "maximum_drawdown",
]

FORMATTERS = {
    "CAGR": lambda x: f"{x:.2%}",
    "annualized_volatility": lambda x: f"{x:.2%}",
    "annualized_turnover": lambda x: f"{x:.2%}",
    "Sharpe": lambda x: f"{x:.3f}",
    "Sortino": lambda x: f"{x:.3f}",
    "Calmar": lambda x: f"{x:.3f}",
    "maximum_drawdown": lambda x: f"{x:.2%}",
}

def display_top_candidates(
    data: pd.DataFrame, metric: str, title: str, n: int = TOP_N_CANDIDATES
) -> pd.DataFrame:
    top_df = data.sort_values(metric, ascending=False).head(n).copy()

    print("\n" + "=" * 110)
    print(f"TOP {n} — {title.upper()}")
    print(top_df[DISPLAY_COLUMNS].to_string(index=False, formatters=FORMATTERS))

    return top_df

# -------------------------------------------------------------------------
# Table from Benchmark B
# -------------------------------------------------------------------------

def format_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    formatted["frequency_days"] = formatted["frequency_days"].astype(int)
    formatted["CAGR"] = formatted["CAGR"].apply(lambda x: f"{x:.2%}")
    formatted["annualized_volatility"] = formatted[
        "annualized_volatility"
    ].apply(lambda x: f"{x:.2%}")
    formatted["annualized_turnover"] = formatted["annualized_turnover"].apply(
        lambda x: f"{x:.2%}"
    )
    formatted["Sharpe"] = formatted["Sharpe"].apply(lambda x: f"{x:.3f}")
    formatted["Sortino"] = formatted["Sortino"].apply(lambda x: f"{x:.3f}")
    formatted["Calmar"] = formatted["Calmar"].apply(lambda x: f"{x:.3f}")
    formatted["maximum_drawdown"] = formatted["maximum_drawdown"].apply(
        lambda x: f"{x:.2%}"
    )
    return formatted