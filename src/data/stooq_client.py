"""
Project AutoQuant — Stooq Free Data Client
============================================
Free OHLCV data from Stooq.com — no API key required.
Direct CSV download, works as a fallback when yfinance fails.
"""

import asyncio
from typing import Optional

import pandas as pd
import requests


class StooqClient:
    """
    Free OHLCV data client using Stooq.com CSV endpoint.
    No API key required. Good for daily data on US equities.
    """

    BASE_URL = "https://stooq.com/q/d/l/"

    def __init__(self) -> None:
        pass

    async def get_daily_prices(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV via Stooq CSV endpoint.

        Stooq uses .US suffix for US equities.
        Index symbols (^NDX, ^GSPC) are mapped to Stooq format.
        """
        stooq_symbol = self._map_symbol(symbol)

        def fetch():
            params = {"s": stooq_symbol, "i": "d"}
            if start:
                params["d1"] = start.replace("-", "")
            if end:
                params["d2"] = end.replace("-", "")

            resp = requests.get(
                self.BASE_URL,
                params=params,
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()

            if "No data" in resp.text or len(resp.text.strip()) < 50:
                return pd.DataFrame()

            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            return df

        try:
            df = await asyncio.to_thread(fetch)
        except Exception as e:
            raise ValueError(f"Stooq error for {symbol}: {e}")

        if df.empty:
            raise ValueError(f"Stooq returned no data for {symbol}")

        df.columns = [c.lower().strip() for c in df.columns]

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]
        df = df.sort_index()
        df = df.astype(float)

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df

    @staticmethod
    def _map_symbol(symbol: str) -> str:
        """Map standard symbols to Stooq format."""
        index_map = {
            "^NDX": "^NDQ",
            "^GSPC": "^SPX",
            "^DJI": "^DJI",
            "^VIX": "^VIX",
            "^TNX": "^TNX",
        }
        if symbol in index_map:
            return index_map[symbol]
        # US equities need .US suffix
        if not symbol.startswith("^") and "." not in symbol:
            return f"{symbol}.US"
        return symbol
