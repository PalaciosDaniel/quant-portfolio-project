
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def get_warmup_trading_days(df):
    """
    Calculates the number of trading days from the first date to the first date where there is at least one valid value.
    """
    dates = df.index.get_level_values("Date")
    trading_dates = dates.unique().sort_values()

    first_date = trading_dates.min()
    first_valid_date = dates[df.notna().any(axis=1)].min()

    return trading_dates.get_loc(first_valid_date) - trading_dates.get_loc(first_date)


def internal_nan_count_by_ticker(df):
    """
    Counts the number of internal NaN values for each ticker in the DataFrame.
    """
    internal_nan_counts = {}

    for ticker, series in df.items():
        first_valid_date = series.first_valid_index()

        # tickers with no valid values are ignored and not included in the result
        if first_valid_date is not None:
            internal_nan_counts[ticker] = (
                series.loc[first_valid_date:].isna().sum()
            )

    return pd.Series(
        internal_nan_counts,
        name="internal_nan_count",
    ).sort_values(ascending=False)


def internal_nans_by_quarter(df):
    df = df.sort_index()

    # True only for eligible values (after the first valid value for each ticker)
    eligible_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for ticker in df.columns:
        first_valid_date = df[ticker].first_valid_index()

        if first_valid_date is not None:
            eligible_mask.loc[first_valid_date:, ticker] = True

    daily_summary = pd.DataFrame(
        {
            "internal_nan_count": (df.isna() & eligible_mask).sum(axis=1),
            "eligible_values": eligible_mask.sum(axis=1),
        }
    )

    quarterly_summary = daily_summary.groupby(pd.Grouper(freq="QE")).sum()

    quarterly_summary["internal_nan_rate"] = (
        quarterly_summary["internal_nan_count"]
        / quarterly_summary["eligible_values"]
        * 100
    )

    return quarterly_summary


def factor_coverage_since_first_valid(df, total_companies=499):
    """
    Percentage of companies with the factor available since their
    first valid observation.
    """
    df = df.sort_index()

    available_mask = pd.DataFrame(
        False,
        index=df.index,
        columns=df.columns,
    )

    for ticker in df.columns:
        first_valid_date = df[ticker].first_valid_index()

        if first_valid_date is not None:
            available_mask.loc[first_valid_date:, ticker] = True

    return available_mask.sum(axis=1) / total_companies * 100