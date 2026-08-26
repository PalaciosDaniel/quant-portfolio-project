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


def get_macro_stats(df):
    return df.groupby("weighting_scheme")["turnover"].agg(
        mean="mean",
        maximum="max",
        p95=lambda x: np.percentile(x, 95),
        violations=lambda x: (x > (MAX_TURNOVER + 1e-6)).sum(),
    )