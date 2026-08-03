# =============================================================================
# FUNCTIONS FOR VALIDATION OF HISTORICAL MARKET DATA
# =============================================================================

from __future__ import annotations
import yfinance as yf
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_float_dtype, is_integer_dtype


def describe_yfinance_data(df: pd.DataFrame) -> dict:
    """
    Summarize a yfinance DataFrame with MultiIndex columns
    (levels: Price and Ticker).

    Returns the number of assets, observations and date range.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Columns must be a MultiIndex with Price and Ticker levels.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex.")

    # Identify the Ticker level regardless of whether columns are
    # ordered as Price/Ticker or Ticker/Price.
    ticker_level = "Ticker" if "Ticker" in df.columns.names else 1

    n_assets = df.columns.get_level_values(ticker_level).nunique()

    # Remove missing dates and ensure chronological order.
    dates = df.index.dropna().sort_values()

    return {
        "n_assets": n_assets,
        "n_observations": len(dates),
        "start_date": dates.min(),
        "end_date": dates.max(),
    }

import pandas as pd


def summarize_date_gaps(df: pd.DataFrame) -> pd.Series:
    """
    Count calendar-day gaps between consecutive dates in the DataFrame index.

    A gap of 1 means consecutive calendar days, while a gap of 3 usually
    corresponds to the transition from Friday to Monday.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex.")

    # Remove missing dates and sort them chronologically before comparison.
    dates = df.index.dropna().sort_values()

    # Calculate the number of calendar days between consecutive observations.
    gaps_in_days = dates.to_series().diff().dt.days.dropna()

    # Count the occurrence of each gap length.
    gap_summary = gaps_in_days.value_counts().sort_index()

    # Make the result easier to interpret.
    gap_summary.index = [
        f"{gap:.0f} day" if gap == 1 else f"{gap:.0f} days"
        for gap in gap_summary.index
    ]
    gap_summary.name = "count"

    return gap_summary


def count_nans_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count total and internal NaN values for each ticker.

    Internal NaNs are missing values found between the first and last date
    where the ticker has at least one available value.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Columns must be a MultiIndex with Price and Ticker levels.")

    # Identify the Ticker level regardless of column level order.
    ticker_level = "Ticker" if "Ticker" in df.columns.names else 1

    tickers = df.columns.get_level_values(ticker_level).unique()
    results = []

    for ticker in tickers:
        # Extract all price fields associated with one ticker.
        ticker_data = df.xs(ticker, level=ticker_level, axis=1)

        # Count NaN values across every price field.
        total_nans = ticker_data.isna().sum().sum()

        # Identify dates where at least one field has valid information.
        valid_dates = ticker_data.notna().any(axis=1)

        if not valid_dates.any():
            internal_nans = 0
            nans_outside_trading_period = total_nans
        else:
            # Define the active period from the first to the last valid date.
            first_valid_position = valid_dates.to_numpy().argmax()
            last_valid_position = len(valid_dates) - 1 - valid_dates.iloc[::-1].to_numpy().argmax()

            active_period = ticker_data.iloc[
                first_valid_position:last_valid_position + 1
            ]

            # Count missing values occurring while the ticker was active.
            internal_nans = active_period.isna().sum().sum()
            nans_outside_trading_period = total_nans - internal_nans

        results.append({
            "Ticker": ticker,
            "total_nans": total_nans,
            "internal_nans": internal_nans,
            "nans_outside_trading_period": nans_outside_trading_period,
        })

    return (
    pd.DataFrame(results)
    .set_index("Ticker")
    .sort_values("total_nans", ascending=False)
    )


def has_duplicate_dates(df: pd.DataFrame) -> bool:
    """
    Check whether the DataFrame index contains duplicated dates.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    return df.index.duplicated().any()


def is_index_sorted(df: pd.DataFrame) -> bool:
    """
    Check whether the DataFrame index is sorted in ascending chronological order.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex.")

    return df.index.is_monotonic_increasing


def detect_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect invalid price and volume values in a yfinance DataFrame.

    Returns a DataFrame with one row per ticker and the number of issues
    found for each validation rule.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Columns must be a MultiIndex with Price and Ticker levels.")

    # Identify MultiIndex levels regardless of their column order.
    price_level = "Price" if "Price" in df.columns.names else 0
    ticker_level = "Ticker" if "Ticker" in df.columns.names else 1

    required_fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    available_fields = df.columns.get_level_values(price_level).unique()
    missing_fields = set(required_fields) - set(available_fields)

    if missing_fields:
        raise ValueError(f"Missing required fields: {sorted(missing_fields)}")

    tickers = df.columns.get_level_values(ticker_level).unique()
    results = []

    for ticker in tickers:
        # Extract data for one ticker independently of MultiIndex level order.
        ticker_data = df.xs(ticker, level=ticker_level, axis=1)

        # Count non-missing price values that are zero or negative.
        invalid_open = (ticker_data["Open"] <= 0).sum()
        invalid_high = (ticker_data["High"] <= 0).sum()
        invalid_low = (ticker_data["Low"] <= 0).sum()
        invalid_close = (ticker_data["Close"] <= 0).sum()
        invalid_adj_close = (ticker_data["Adj Close"] <= 0).sum()

        # Report zero and negative volumes separately.
        zero_volume = (ticker_data["Volume"] == 0).sum()
        negative_volume = (ticker_data["Volume"] < 0).sum()

        # Low must never be greater than High.
        low_greater_than_high = (
            (ticker_data["Low"] > ticker_data["High"])
            .where(ticker_data["Low"].notna() & ticker_data["High"].notna(), False)
            .sum()
        )

        # High must be greater than or equal to both Open and Close.
        high_less_than_open = (
            (ticker_data["High"] < ticker_data["Open"])
            .where(ticker_data["High"].notna() & ticker_data["Open"].notna(), False)
            .sum()
        )

        high_less_than_close = (
            (ticker_data["High"] < ticker_data["Close"])
            .where(ticker_data["High"].notna() & ticker_data["Close"].notna(), False)
            .sum()
        )

        # Low must be less than or equal to both Open and Close.
        low_greater_than_open = (
            (ticker_data["Low"] > ticker_data["Open"])
            .where(ticker_data["Low"].notna() & ticker_data["Open"].notna(), False)
            .sum()
        )

        low_greater_than_close = (
            (ticker_data["Low"] > ticker_data["Close"])
            .where(ticker_data["Low"].notna() & ticker_data["Close"].notna(), False)
            .sum()
        )

        results.append({
            "Ticker": ticker,
            "invalid_open": invalid_open,
            "invalid_high": invalid_high,
            "invalid_low": invalid_low,
            "invalid_close": invalid_close,
            "invalid_adj_close": invalid_adj_close,
            "zero_volume": zero_volume,
            "negative_volume": negative_volume,
            "low_greater_than_high": low_greater_than_high,
            "high_less_than_open": high_less_than_open,
            "high_less_than_close": high_less_than_close,
            "low_greater_than_open": low_greater_than_open,
            "low_greater_than_close": low_greater_than_close,
        })

    result = pd.DataFrame(results).set_index("Ticker")

    # Add a total count including all detected issues.
    result["total_issues"] = result.sum(axis=1)

    return result.sort_values("total_issues", ascending=False)


def get_trading_periods_by_ticker(
    df: pd.DataFrame,
    price_field: str = "Close"
) -> dict:
    """
    Identify the first and last trading date for each ticker based on
    non-missing values in the selected price field.

    Returns two groups:
    - full_period: tickers with data from the dataset's first to last date.
    - partial_period: tickers that start later, end earlier, or have no data.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Columns must be a MultiIndex with Price and Ticker levels.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex.")

    # Identify MultiIndex levels regardless of their column order.
    price_level = "Price" if "Price" in df.columns.names else 0
    ticker_level = "Ticker" if "Ticker" in df.columns.names else 1

    available_fields = df.columns.get_level_values(price_level).unique()
    if price_field not in available_fields:
        raise ValueError(f"'{price_field}' is not available in the dataset.")

    # Define the global date range of the dataset.
    dataset_start = df.index.min()
    dataset_end = df.index.max()

    tickers = df.columns.get_level_values(ticker_level).unique()
    results = []

    for ticker in tickers:
        # Extract the selected price series for the current ticker.
        ticker_prices = df.xs(ticker, level=ticker_level, axis=1)[price_field]

        # Keep only dates with an observed price.
        valid_dates = ticker_prices.dropna().index

        if len(valid_dates) == 0:
            first_trading_date = pd.NaT
            last_trading_date = pd.NaT
            traded_full_period = False
        else:
            first_trading_date = valid_dates.min()
            last_trading_date = valid_dates.max()

            # A ticker belongs to the full-period group if it has data
            # at both boundaries of the complete dataset.
            traded_full_period = (
                first_trading_date == dataset_start
                and last_trading_date == dataset_end
            )

        results.append({
            "Ticker": ticker,
            "first_trading_date": first_trading_date,
            "last_trading_date": last_trading_date,
            "traded_full_period": traded_full_period,
        })

    periods = pd.DataFrame(results).set_index("Ticker")

    full_period = periods[periods["traded_full_period"]].sort_index()

    partial_period = (
        periods[~periods["traded_full_period"]]
        .sort_values(
        "first_trading_date",
        ascending=False,
        na_position="last",
    )
)

    return {
    "full_period": full_period,
    "partial_period": partial_period,
    }


def check_yfinance_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check whether the DataFrame index and market fields have the expected
    yfinance data types.

    Expected types:
    - Date index: datetime64
    - Open, High, Low, Close: float64
    - Volume: int64 or float64
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Columns must be a MultiIndex with Price and Ticker levels.")

    # Identify the Price level regardless of the MultiIndex column order.
    price_level = "Price" if "Price" in df.columns.names else 0

    checks = []

    # Check the date index type.
    checks.append({
        "field": "Date",
        "expected_dtype": "datetime64",
        "actual_dtype": str(df.index.dtype),
        "is_valid": is_datetime64_any_dtype(df.index),
    })

    expected_float_fields = ["Open", "High", "Low", "Close"]

    for field in expected_float_fields:
        # Select all columns associated with the current price field.
        field_columns = df.loc[:, df.columns.get_level_values(price_level) == field]

        if field_columns.empty:
            checks.append({
                "field": field,
                "expected_dtype": "float64",
                "actual_dtype": "missing",
                "is_valid": False,
            })
            continue

        actual_dtypes = field_columns.dtypes.astype(str).unique().tolist()

        checks.append({
            "field": field,
            "expected_dtype": "float64",
            "actual_dtype": ", ".join(actual_dtypes),
            "is_valid": all(is_float_dtype(dtype) for dtype in field_columns.dtypes),
        })

    # Volume can be stored as either an integer or a floating-point type.
    volume_columns = df.loc[:, df.columns.get_level_values(price_level) == "Volume"]

    if volume_columns.empty:
        checks.append({
            "field": "Volume",
            "expected_dtype": "int64 or float64",
            "actual_dtype": "missing",
            "is_valid": False,
        })
    else:
        actual_dtypes = volume_columns.dtypes.astype(str).unique().tolist()

        checks.append({
            "field": "Volume",
            "expected_dtype": "int64 or float64",
            "actual_dtype": ", ".join(actual_dtypes),
            "is_valid": all(
                is_integer_dtype(dtype) or is_float_dtype(dtype)
                for dtype in volume_columns.dtypes
            ),
        })

    return pd.DataFrame(checks).set_index("field")


def create_data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a consolidated data-quality summary using the previously
    defined validation functions.
    """
    # Get the general dataset description.
    dataset_info = describe_yfinance_data(df)

    # Check date duplicates and chronological index order.
    duplicated_dates = has_duplicate_dates(df)
    sorted_index = is_index_sorted(df)

    # Split tickers according to their available trading period.
    trading_periods = get_trading_periods_by_ticker(df)
    full_period_tickers = len(trading_periods["full_period"])
    partial_period_tickers = len(trading_periods["partial_period"])

    # Build a readable summary table.
    summary = pd.DataFrame({
        "metric": [
            "Number of assets",
            "Number of observations",
            "Start date",
            "End date",
            "Duplicated dates",
            "Chronologically sorted index",
            "Tickers with data for the full period",
            "Tickers with partial trading periods",
        ],
        "value": [
            dataset_info["n_assets"],
            dataset_info["n_observations"],
            dataset_info["start_date"],
            dataset_info["end_date"],
            duplicated_dates,
            sorted_index,
            full_period_tickers,
            partial_period_tickers,
        ],
    })

    return summary