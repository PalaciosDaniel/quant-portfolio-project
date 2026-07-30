# =============================================================================
# FUNCTIONS FOR EXTRACTION OF HISTORICAL MARKET DATA
# =============================================================================

from __future__ import annotations
import yfinance as yf
import pandas as pd


def download_prices(
    tickers: list[str],
    START_DATE: str,
    END_DATE: str,
    INTERVAL: str = "1d",
    AUTO_ADJUST: bool = False,
) -> pd.DataFrame:
    """
    Download historical OHLCV data from Yahoo Finance.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols.
    START_DATE : str
        Start date (YYYY-MM-DD).
    END_DATE : str
        End date (YYYY-MM-DD).
    INTERVAL : str, default="1d"
        Data frequency.

    Returns
    -------
    pd.DataFrame
        OHLCV data with MultiIndex columns:
        Level 0 -> Price field
        Level 1 -> Ticker
    """

    prices = yf.download(
        tickers=tickers,
        start=START_DATE,
        end=END_DATE,
        interval=INTERVAL,
        auto_adjust=AUTO_ADJUST,
        progress=True,
        group_by="column",
        threads=True,
    )

    prices.columns.names = ["Price", "Ticker"]

    prices.sort_index(inplace=True)

    return prices


def validate_download(
    prices: pd.DataFrame,
    requested_tickers: list[str],
) -> None:
    """
    Validate the downloaded market data.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame returned by download_prices().
    requested_tickers : list[str]
        Original list of requested ticker symbols.

    Returns
    -------
    None
    """

    # 1. Tickers that are in columns of the downloaded DataFrame
    downloaded_tickers = (
        prices.columns
              .get_level_values("Ticker")
              .unique()
              .tolist()
    )

    # 2. Tickers that are not in the downloaded DataFrame (i.e., not returned by the API)
    missing_tickers = sorted(
        set(requested_tickers) - set(downloaded_tickers)
    )

    # 3. Tickers that came back but have 100% of their data as NaN
    # We use 'Close' as a reference (or you could evaluate the entire sub-table)
    empty_tickers = []
    if "Close" in prices.columns.get_level_values("Price"):
        close_prices = prices["Close"]
        for ticker in downloaded_tickers:
            if close_prices[ticker].isna().all():
                empty_tickers.append(ticker)
    
    empty_tickers = sorted(empty_tickers)

    # 4. Tickers that are valid (downloaded and not empty)
    valid_tickers = sorted(
        set(downloaded_tickers) - set(empty_tickers)
    )

    print("=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)

    print(f"Requested tickers : {len(requested_tickers)}")
    print(f"Valid downloads   : {len(valid_tickers)}")
    print(f"Missing tickers   : {len(missing_tickers)}")
    print(f"Empty tickers (NaN): {len(empty_tickers)}")

    if missing_tickers:
        print("\nMissing tickers (not returned by API):")
        for ticker in missing_tickers:
            print(f"  - {ticker}")

    if empty_tickers:
        print("\nEmpty tickers (100% NaN values):")
        for ticker in empty_tickers:
            print(f"  - {ticker}")

    if not missing_tickers and not empty_tickers:
        print("\nAll tickers downloaded and validated successfully.")


def remove_empty_tickers(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Remove tickers whose price history is completely missing.
    """

    close = prices["Close"]

    valid_tickers = close.columns[~close.isna().all()]

    return prices.loc[:, (slice(None), valid_tickers)]