"""
Project AutoQuant — Unified Data Loader
========================================
Orchestrates multiple free data sources with automatic fallback:
  1. yfinance (primary, free, no API key)
  2. Stooq (free fallback, no API key)
  3. Synthetic (absolute last resort)

Also provides macro overlay data (VIX, Treasury yields) for
regime-aware strategy generation.
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from src.utils.config import NASDAQ_100_TICKERS
from src.data.yfinance_client import YFinanceClient
from src.data.stooq_client import StooqClient
from src.data.macro_client import MacroClient


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Unified data-loading facade with automatic fallback chain.

    Resolution order for price data:
    1. **yfinance** — free, no key required, great for daily OHLCV.
    2. **Stooq** — free fallback, CSV download, no API key.
    3. **Synthetic** — absolute last resort for offline dev.

    Parameters
    ----------
    yf_client : YFinanceClient, optional
        Pre-configured YFinance client.
    stooq_client : StooqClient, optional
        Pre-configured Stooq client.
    macro_client : MacroClient, optional
        Pre-configured macro data client.
    load_macro : bool
        If True, automatically loads VIX/Treasury data and merges
        into the price DataFrame as extra columns.
    """

    def __init__(
        self,
        yf_client: Optional[YFinanceClient] = None,
        stooq_client: Optional[StooqClient] = None,
        macro_client: Optional[MacroClient] = None,
        load_macro: bool = False,
    ) -> None:
        self.yf_client = yf_client or YFinanceClient()
        self.stooq_client = stooq_client or StooqClient()
        self.macro_client = macro_client or MacroClient()
        self.load_macro = load_macro

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

        Uses automatic fallback: yfinance → Stooq → synthetic.
        Optionally merges macro overlay data (VIX, treasury yields).
        """
        tickers = tickers or NASDAQ_100_TICKERS
        end = end or datetime.now().strftime("%Y-%m-%d")

        # Load macro overlay data if enabled
        macro_df = None
        if self.load_macro:
            try:
                macro_df = self.macro_client.get_macro_overlay_sync(start=start, end=end)
                if macro_df is not None and not macro_df.empty:
                    print(f"[DataLoader] Loaded macro overlay: {list(macro_df.columns)}")
            except Exception as exc:
                print(f"[DataLoader] Macro overlay failed (non-critical): {exc}")

        universe: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = self._load_single(ticker, start, end, source)
            if df is not None and not df.empty:
                # Merge macro data if available
                if macro_df is not None and not macro_df.empty:
                    df = df.join(macro_df, how="left")
                    df = df.ffill()
                universe[ticker] = df

        return universe

    # -- single-ticker loader with fallback ----------------------------------

    def _load_single(
        self,
        ticker: str,
        start: str,
        end: str,
        source: str,
    ) -> Optional[pd.DataFrame]:
        """Load one ticker with automatic fallback chain."""
        # Attempt 1: yfinance
        df = self._try_yfinance(ticker, start, end)
        if df is not None and not df.empty:
            return df

        # Attempt 2: Stooq
        df = self._try_stooq(ticker, start, end)
        if df is not None and not df.empty:
            return df

        print(f"[DataLoader] All sources failed for {ticker}")
        return None

    def _try_yfinance(
        self, ticker: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Try loading from yfinance."""
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
                result = df.loc[mask]
                if not result.empty:
                    print(f"[DataLoader] Loaded {ticker} via yfinance: {len(result)} bars")
                    return result
        except Exception as exc:
            print(f"[DataLoader] yfinance failed for {ticker}: {exc}")
        return None

    def _try_stooq(
        self, ticker: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Try loading from Stooq (free, no API key)."""
        try:
            coro = self.stooq_client.get_daily_prices(ticker, start=start, end=end)
            try:
                loop = asyncio.get_event_loop()
                df = loop.run_until_complete(coro)
            except RuntimeError:
                df = asyncio.run(coro)

            if df is not None and not df.empty:
                df = self._standardise(df)
                mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
                result = df.loc[mask]
                if not result.empty:
                    print(f"[DataLoader] Loaded {ticker} via Stooq (fallback): {len(result)} bars")
                    return result
        except Exception as exc:
            print(f"[DataLoader] Stooq failed for {ticker}: {exc}")
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

        # Keep only canonical columns (preserve any macro columns already merged)
        keep = [c for c in df.columns if c in ("open", "high", "low", "close", "volume")
                or c.endswith("_close")]
        df = df[keep]

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Strip timezone awareness to fix Timestamp comparisons
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.sort_index()
        return df
