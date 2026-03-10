"""
Project AutoQuant — YFinance MCP Tool
======================================
Async client for fetching US stock data from the yfinance API.
Returns standardized pandas DataFrames securely and cache-free.
"""

import asyncio
from typing import Any, Dict, Optional

import pandas as pd
import yfinance as yf
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Pydantic schemas for structured responses
# ---------------------------------------------------------------------------

class YFinanceError(BaseModel):
    """Structured error from YFinance."""
    message: str
    function: str
    symbol: str


# ---------------------------------------------------------------------------
# YFinance Client
# ---------------------------------------------------------------------------

class YFinanceClient:
    """
    Async client for the yfinance API to fetch live, cache-free market data.

    Implements the MCP (Model Context Protocol) tool pattern — each method
    is a discrete *tool* that an agent can invoke safely.
    """

    def __init__(self) -> None:
        pass

    # -- public tools --------------------------------------------------------

    async def get_daily_prices(
        self,
        symbol: str,
        outputsize: str = "full",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV prices for *symbol*.

        Returns a DataFrame with columns:
        ``open``, ``high``, ``low``, ``close``, ``volume``
        and a ``DatetimeIndex``.
        """
        def fetch():
            ticker = yf.Ticker(symbol)
            if start and end:
                return ticker.history(start=start, end=end)
            elif outputsize == "full":
                return ticker.history(period="max")
            else:
                return ticker.history(period="100d")

        try:
            df = await asyncio.to_thread(fetch)
        except Exception as e:
            raise ValueError(f"YFinance error fetching daily prices for {symbol}: {e}")

        if df.empty:
            raise ValueError(
                f"YFinance returned no daily data for {symbol}. "
            )

        df = df.copy()
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        # Keep only canonical columns
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]

        df.index = pd.to_datetime(df.index)
        
        # Strip timezone awareness to fix Timestamp comparisons
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.sort_index()
        df = df.astype(float)
        return df

    async def get_intraday(
        self,
        symbol: str,
        interval: str = "5m",
        outputsize: str = "full",
    ) -> pd.DataFrame:
        """
        Fetch intraday OHLCV prices for *symbol*.

        Parameters
        ----------
        interval : str
            One of ``1m``, ``2m``, ``5m``, ``15m``, ``30m``, ``60m``, ``90m``, ``1h``, ``1d``, ``5d``, ``1wk``, ``1mo``, ``3mo``.
        """
        def fetch():
            ticker = yf.Ticker(symbol)
            # yfinance limits intraday history (e.g. 1m to 7 days, 5m to 60 days)
            if interval == "1m":
                period = "7d"
            else:
                period = "60d"
                
            if outputsize != "full":
                period = "5d"

            return ticker.history(interval=interval, period=period)

        try:
            df = await asyncio.to_thread(fetch)
        except Exception as e:
            raise ValueError(f"YFinance error fetching intraday prices for {symbol}: {e}")

        if df.empty:
            raise ValueError(
                f"YFinance returned no intraday data for {symbol}/{interval}."
            )

        df = df.copy()
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        # Keep only canonical columns
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]

        df.index = pd.to_datetime(df.index)
        
        # Strip timezone awareness to fix Timestamp comparisons
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.sort_index()
        df = df.astype(float)
        return df
