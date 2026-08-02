


def get_warmup_trading_days(df):
    """
    Calculates the number of trading days from the first date to the first date where there is at least one valid value.
    """
    dates = df.index.get_level_values("Date")
    trading_dates = dates.unique().sort_values()

    first_date = trading_dates.min()
    first_valid_date = dates[df.notna().any(axis=1)].min()

    return trading_dates.get_loc(first_valid_date) - trading_dates.get_loc(first_date)


def count_nans_after_first_valid(df, date_level="Date"):
    """
    Counts the number of NaN values in the DataFrame after the first date where there is at least one valid value.
    """
    dates = df.index.get_level_values(date_level)
    first_valid_date = dates[df.notna().any(axis=1)].min()

    df_after_first_valid = df.loc[
        dates >= first_valid_date
    ]

    return df_after_first_valid.isna().sum().sum()