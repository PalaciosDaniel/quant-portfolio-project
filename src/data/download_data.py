"""
Utilities for downloading historical market data.
"""

from __future__ import annotations
import yfinance as yf
import pandas as pd

# download historical OHLCV data from Yahoo Finance
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
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).
    interval : str, default="1d"
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


# validate the downloaded data
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

    # 1. Tickers que sí están presentes en las columnas
    downloaded_tickers = (
        prices.columns
              .get_level_values("Ticker")
              .unique()
              .tolist()
    )

    # 2. Tickers que directamente no vinieron en la respuesta
    missing_tickers = sorted(
        set(requested_tickers) - set(downloaded_tickers)
    )

    # 3. Tickers que vinieron pero tienen el 100% de sus datos como NaN
    # Usamos 'Close' como referencia (o podrías evaluar la sub-tabla entera)
    empty_tickers = []
    if "Close" in prices.columns.get_level_values("Price"):
        close_prices = prices["Close"]
        for ticker in downloaded_tickers:
            if close_prices[ticker].isna().all():
                empty_tickers.append(ticker)
    
    empty_tickers = sorted(empty_tickers)

    # 4. Tickers totalmente válidos (tienen columna y al menos algún dato)
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

#remove tickers whose price history is completely missing
def remove_empty_tickers(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Remove tickers whose price history is completely missing.
    """

    close = prices["Close"]

    valid_tickers = close.columns[~close.isna().all()]

    return prices.loc[:, (slice(None), valid_tickers)]