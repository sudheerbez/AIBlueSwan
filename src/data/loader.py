"""
Project AutoQuant — Unified Data Loader
========================================
Orchestrates Alpha Vantage, FMP, and yfinance to load price data for a
universe of tickers.  Caches results locally as CSV to reduce API calls.
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from src.utils.config import NASDAQ_100_TICKERS
from src.data.yfinance_client import YFinanceClient


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Unified data-loading facade.

    Resolution order for price data:
    1. **Local CSV cache** — fastest, avoids API usage.
    2. **yfinance** — free, no key required, great for daily OHLCV.
    3. **Alpha Vantage** — enriched with adjusted close & SMA/RSI.
    4. **FMP** — alternative source with fundamental overlays.

    Parameters
    ----------
    yf_client : YFinanceClient, optional
        Pre-configured YFinance client.
    """

    def __init__(
        self,
        yf_client: Optional[YFinanceClient] = None,
    ) -> None:

        self.yf_client = yf_client or YFinanceClient()

    # -- public API ----------------------------------------------------------

    def load_universe(
        self,
        tickers: Optional[List[str]] = None,
        start: str = "2020-01-01",
        end: Optional[str] = None,
        source: str = "yfinance",
    ) -> Dict[str, pd.DataFrame]:
        """
        Load daily OHLCV data for each *ticker* in the universe.

        Parameters
        ----------
        tickers : list[str], optional
            Symbols to load.  Defaults to ``NASDAQ_100_TICKERS``.
        start, end : str
            ISO-format date bounds (``end`` defaults to today).
        source : str
            Primary data source: ``"yfinance"`` (default).
        """
        tickers = tickers or NASDAQ_100_TICKERS
        end = end or datetime.now().strftime("%Y-%m-%d")

        universe: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = self._load_single(ticker, start, end, source)
            if df is not None and not df.empty:
                universe[ticker] = df

        return universe

    # -- single-ticker loader ------------------------------------------------

    def _load_single(
        self,
        ticker: str,
        start: str,
        end: str,
        source: str,
    ) -> Optional[pd.DataFrame]:
        """Load one ticker with live data structure."""
        # Fetch directly from yfinance without caching
        try:
            coro = self.yf_client.get_daily_prices(ticker, start=start, end=end)
            try:
                loop = asyncio.get_event_loop()
                df = loop.run_until_complete(coro)
            except RuntimeError:
                df = asyncio.run(coro)

            if df is not None and not df.empty:
                df = self._standardise(df)
                mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
                return df.loc[mask]

        except Exception as exc:
            print(f"[DataLoader] Warning: failed to load {ticker}: {exc}")

        return None

    # -- standardisation -----------------------------------------------------

    @staticmethod
    def _standardise(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure columns are lower-case ``open, high, low, close, volume``
        and index is a ``DatetimeIndex``.
        """
        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]

        # Rename common variations
        rename_map = {
            "adj close": "close",
            "adjusted_close": "close",
        }
        df = df.rename(columns=rename_map)

        # Keep only canonical columns
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Strip timezone awareness to fix Timestamp comparisons
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.sort_index()
        return df
