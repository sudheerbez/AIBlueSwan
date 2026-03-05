"""
Project AutoQuant — Alpha Vantage MCP Tool
============================================
Async client for fetching US stock data from the Alpha Vantage API.
Implements rate limiting and returns standardized pandas DataFrames.
"""

import asyncio
import os
from typing import Any, Dict, Optional

import aiohttp
import pandas as pd
from pydantic import BaseModel

from src.utils.config import ALPHA_VANTAGE_BASE_URL


# ---------------------------------------------------------------------------
# Pydantic schemas for structured responses
# ---------------------------------------------------------------------------

class AlphaVantageError(BaseModel):
    """Structured error from Alpha Vantage API."""
    status_code: int
    message: str
    function: str
    symbol: str


# ---------------------------------------------------------------------------
# Alpha Vantage Client
# ---------------------------------------------------------------------------

class AlphaVantageClient:
    """
    Async client for the Alpha Vantage REST API.

    Implements the MCP (Model Context Protocol) tool pattern — each method
    is a discrete *tool* that an agent can invoke.

    Rate limiting
    -------------
    Free-tier Alpha Vantage allows 5 requests/minute and 500/day.
    A simple asyncio semaphore + sleep enforces the per-minute cap.

    Parameters
    ----------
    api_key : str, optional
        Overrides the ``ALPHA_VANTAGE_API_KEY`` environment variable.
    calls_per_minute : int
        Maximum API calls per 60-second window (default 5).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        calls_per_minute: int = 5,
    ) -> None:
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self.base_url = ALPHA_VANTAGE_BASE_URL
        self._semaphore = asyncio.Semaphore(calls_per_minute)
        self._rate_delay = 60.0 / max(calls_per_minute, 1)

    # -- public tools --------------------------------------------------------

    async def get_daily_prices(
        self,
        symbol: str,
        outputsize: str = "full",
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV prices for *symbol*.

        Returns a DataFrame with columns:
        ``open``, ``high``, ``low``, ``close``, ``volume``
        and a ``DatetimeIndex``.
        """
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": outputsize,
            "datatype": "json",
        }

        data = await self._request(params)
        ts_key = "Time Series (Daily)"

        if ts_key not in data:
            raise ValueError(
                f"Alpha Vantage returned no daily data for {symbol}. "
                f"Response keys: {list(data.keys())}"
            )

        df = pd.DataFrame.from_dict(data[ts_key], orient="index")
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.astype(float)
        return df

    async def get_intraday(
        self,
        symbol: str,
        interval: str = "5min",
        outputsize: str = "full",
    ) -> pd.DataFrame:
        """
        Fetch intraday OHLCV prices for *symbol*.

        Parameters
        ----------
        interval : str
            One of ``1min``, ``5min``, ``15min``, ``30min``, ``60min``.
        """
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "datatype": "json",
        }

        data = await self._request(params)
        ts_key = f"Time Series ({interval})"

        if ts_key not in data:
            raise ValueError(
                f"Alpha Vantage returned no intraday data for {symbol}/{interval}."
            )

        df = pd.DataFrame.from_dict(data[ts_key], orient="index")
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.astype(float)
        return df

    async def get_technical_indicator(
        self,
        symbol: str,
        indicator: str = "RSI",
        interval: str = "daily",
        time_period: int = 14,
        series_type: str = "close",
    ) -> pd.DataFrame:
        """
        Fetch a technical indicator time series.

        Parameters
        ----------
        indicator : str
            E.g. ``RSI``, ``MACD``, ``SMA``, ``EMA``, ``BBANDS``.
        """
        params = {
            "function": indicator,
            "symbol": symbol,
            "interval": interval,
            "time_period": str(time_period),
            "series_type": series_type,
            "datatype": "json",
        }

        data = await self._request(params)

        # The response key varies (e.g. "Technical Analysis: RSI")
        ta_key = next(
            (k for k in data if k.startswith("Technical Analysis")),
            None,
        )
        if ta_key is None:
            raise ValueError(
                f"Alpha Vantage returned no indicator data for {indicator}/{symbol}."
            )

        df = pd.DataFrame.from_dict(data[ta_key], orient="index").astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df

    # -- internals -----------------------------------------------------------

    async def _request(self, params: Dict[str, str]) -> Dict[str, Any]:
        """Execute a rate-limited GET request to Alpha Vantage."""
        params["apikey"] = self.api_key

        async with self._semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            # Simple rate-limit delay
            await asyncio.sleep(self._rate_delay)

        # Alpha Vantage returns error messages inside the JSON body
        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
        if "Note" in data:
            raise ValueError(
                f"Alpha Vantage rate-limit hit: {data['Note']}"
            )

        return data
