"""
Utilities for downloading historical market data.
"""

from __future__ import annotations

import yfinance as yf
import pandas as pd


def download_prices(
    tickers: list[str],
    START_DATE: str,
    END_DATE: str,
    INTERVAL: str = "1d",
    AUTO_ADJUST: bool = True,
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