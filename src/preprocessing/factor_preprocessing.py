# =============================================================================
# FUNCTIONS FOR FACTOR PREPROCESSING
# =============================================================================

def winsorize_cross_sectional(
    df,
    cols,
    p_low=0.01,
    p_high=0.99
):
    """
    Apply cross-sectional winsorization by date.

    For each factor and each trading day, values below the lower percentile
    are replaced by the lower threshold, while values above the upper
    percentile are replaced by the upper threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a 'date' column and the factor columns.
    cols : list of str
        Factor columns to winsorize.
    p_low : float, default=0.01
        Lower percentile used as the clipping threshold.
    p_high : float, default=0.99
        Upper percentile used as the clipping threshold.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with additional columns named
        '<factor>_win' containing the winsorized values.
    """

    df_out = df.copy()

    for col in cols:

        # Compute both daily percentile thresholds in a single groupby operation
        quantiles = (
            df_out
            .groupby("date")[col]
            .quantile([p_low, p_high])
            .unstack()
        )

        # Map daily thresholds back to each observation
        q_low = df_out["date"].map(quantiles[p_low])
        q_high = df_out["date"].map(quantiles[p_high])

        # Clip values outside the percentile range
        df_out[f"{col}_win"] = df_out[col].clip(lower=q_low, upper=q_high)

    return df_out


def standardize_cross_sectional(df, cols):
    """
    Apply cross-sectional Z-score standardization by date.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a 'date' column or index and factor columns.
    cols : list of str
        Factor columns to standardize.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with standardized columns ('<factor>_z').
    """
    df_out = df.copy()

    # Ensure 'date' is accessible whether it is in columns or in the index
    grouped = (
        df_out.groupby("date")
        if "date" in df_out.columns
        else df_out.groupby(level="date")
    )

    for col in cols:

        # Compute daily cross-sectional mean and standard deviation
        mean_t = grouped[col].transform("mean")
        std_t = grouped[col].transform("std")

        # Compute cross-sectional Z-score
        df_out[f"{col}_z"] = (df_out[col] - mean_t) / std_t

    return df_out


def rank_cross_sectional(df, cols):
    """
    Apply cross-sectional percentile ranking by date.

    For each trading day, factor values are converted into percentile ranks
    and centered around zero by subtracting 0.5.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a 'date' column or index and factor columns.
    cols : list of str
        Factor columns to rank.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with ranked columns ('<factor>_rank').
    """
    df_out = df.copy()

    # Ensure 'date' is accessible whether it is in columns or in the index
    grouped = (
        df_out.groupby("date")
        if "date" in df_out.columns
        else df_out.groupby(level="date")
    )

    for col in cols:

        # Compute daily cross-sectional percentile rank
        rank_t = grouped[col].rank(method="average", pct=True)

        # Center the percentile rank around zero
        df_out[f"{col}_rank"] = rank_t - 0.5

    return df_out


def compute_forward_return(prices, horizon=21):
    """
    Compute forward returns from adjusted close prices.

    Parameters
    ----------
    prices : pandas.DataFrame
        DataFrame containing adjusted close prices.
        Rows correspond to dates and columns to tickers.
    horizon : int, default=21
        Forward return horizon in trading days.

    Returns
    -------
    pandas.DataFrame
        Forward returns with the same shape as the input.
    """

    return prices.pct_change(horizon).shift(-horizon)