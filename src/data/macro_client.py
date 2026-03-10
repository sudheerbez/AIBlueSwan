"""
Project AutoQuant — Free Macro Data Client
=============================================
Fetches macro/regime indicators (VIX, Treasury yields, etc.) via yfinance.
All free, no API keys. Used for regime-aware strategy generation.
"""

import asyncio
from typing import Dict, Optional

import pandas as pd

from src.data.yfinance_client import YFinanceClient


# Macro symbols available free via yfinance
MACRO_SYMBOLS = {
    "vix": "^VIX",           # CBOE Volatility Index
    "tnx": "^TNX",           # 10-Year Treasury Yield
    "irx": "^IRX",           # 13-Week Treasury Bill
    "dxy": "DX-Y.NYB",       # US Dollar Index
    "spy": "SPY",            # S&P 500 ETF (for breadth)
}


class MacroClient:
    """
    Fetches free macro/regime data for strategy enrichment.

    All data is sourced from yfinance (no API keys required).
    Returns a combined DataFrame with macro columns that can be
    merged into the main price DataFrame for regime-aware strategies.
    """

    def __init__(self) -> None:
        self.yf = YFinanceClient()

    async def get_macro_overlay(
        self,
        start: str = "2018-01-01",
        end: Optional[str] = None,
        symbols: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Fetch macro indicators and return a single DataFrame with columns:
        vix_close, tnx_close, irx_close, dxy_close, spy_close
        """
        symbols = symbols or MACRO_SYMBOLS
        frames = {}

        for label, ticker in symbols.items():
            try:
                df = await self.yf.get_daily_prices(ticker, start=start, end=end)
                if not df.empty and "close" in df.columns:
                    frames[f"{label}_close"] = df["close"]
            except Exception:
                pass

        if not frames:
            return pd.DataFrame()

        macro_df = pd.DataFrame(frames)
        macro_df = macro_df.sort_index()
        macro_df = macro_df.ffill()
        return macro_df

    def get_macro_overlay_sync(
        self,
        start: str = "2018-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Synchronous wrapper for get_macro_overlay."""
        coro = self.get_macro_overlay(start=start, end=end)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, coro).result()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)
