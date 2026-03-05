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

from src.utils.config import DATA_CACHE_DIR, NASDAQ_100_TICKERS
from src.data.alpha_vantage import AlphaVantageClient
from src.data.fmp import FMPClient


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
    cache_dir : str
        Directory for CSV caches (created automatically).
    av_client : AlphaVantageClient, optional
        Pre-configured Alpha Vantage client.
    fmp_client : FMPClient, optional
        Pre-configured FMP client.
    """

    def __init__(
        self,
        cache_dir: str = DATA_CACHE_DIR,
        av_client: Optional[AlphaVantageClient] = None,
        fmp_client: Optional[FMPClient] = None,
    ) -> None:
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.av_client = av_client or AlphaVantageClient()
        self.fmp_client = fmp_client or FMPClient()

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
            Primary data source: ``"yfinance"`` (default),
            ``"alpha_vantage"``, or ``"fmp"``.

        Returns
        -------
        dict[str, pd.DataFrame]
            ``{symbol: DataFrame}`` with standardised columns
            ``open, high, low, close, volume`` and ``DatetimeIndex``.
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
        """Load one ticker, checking cache first."""
        # 1. Try cache
        cached = self._read_cache(ticker)
        if cached is not None:
            mask = (cached.index >= pd.Timestamp(start)) & (cached.index <= pd.Timestamp(end))
            filtered = cached.loc[mask]
            if not filtered.empty:
                return filtered

        # 2. Fetch from primary source
        try:
            if source == "yfinance":
                df = self._fetch_yfinance(ticker, start, end)
            elif source == "alpha_vantage":
                df = asyncio.get_event_loop().run_until_complete(
                    self.av_client.get_daily_prices(ticker)
                )
            elif source == "fmp":
                df = asyncio.get_event_loop().run_until_complete(
                    self.fmp_client.get_daily_prices(ticker, start=start, end=end)
                )
            else:
                df = self._fetch_yfinance(ticker, start, end)

            if df is not None and not df.empty:
                df = self._standardise(df)
                self._write_cache(ticker, df)
                mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
                return df.loc[mask]

        except Exception as exc:
            print(f"[DataLoader] Warning: failed to load {ticker} from {source}: {exc}")

        # 3. Fallback — try yfinance if it wasn't the primary source
        if source != "yfinance":
            try:
                df = self._fetch_yfinance(ticker, start, end)
                if df is not None and not df.empty:
                    df = self._standardise(df)
                    self._write_cache(ticker, df)
                    return df
            except Exception:
                pass

        return None

    # -- yfinance fetcher ----------------------------------------------------

    @staticmethod
    def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
        """Fetch daily data via yfinance."""
        import yfinance as yf

        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end, auto_adjust=True)
        return df

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

        df = df.sort_index()
        return df

    # -- cache helpers -------------------------------------------------------

    def _cache_path(self, ticker: str) -> str:
        return os.path.join(self.cache_dir, f"{ticker.upper()}.csv")

    def _read_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(ticker)
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            return df
        except Exception:
            return None

    def _write_cache(self, ticker: str, df: pd.DataFrame) -> None:
        try:
            df.to_csv(self._cache_path(ticker))
        except Exception as exc:
            print(f"[DataLoader] Warning: could not write cache for {ticker}: {exc}")
