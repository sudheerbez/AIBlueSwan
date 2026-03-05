"""
Project AutoQuant — Financial Modeling Prep (FMP) MCP Tool
===========================================================
Async client for fetching fundamental & screening data from the FMP API.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
from pydantic import BaseModel, Field

from src.utils.config import FMP_BASE_URL


# ---------------------------------------------------------------------------
# Pydantic schemas for structured responses
# ---------------------------------------------------------------------------

class FinancialStatement(BaseModel):
    """Single-period financial statement snapshot."""
    date: str
    symbol: str
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    eps: Optional[float] = None
    operating_income: Optional[float] = None
    total_assets: Optional[float] = None
    total_debt: Optional[float] = None
    free_cash_flow: Optional[float] = None


class StockScreenerResult(BaseModel):
    """Result row from the FMP stock screener."""
    symbol: str
    company_name: str = ""
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    beta: Optional[float] = None
    price: Optional[float] = None
    volume: Optional[int] = None


# ---------------------------------------------------------------------------
# FMP Client
# ---------------------------------------------------------------------------

class FMPClient:
    """
    Async client for the Financial Modeling Prep REST API.

    Each method is a discrete MCP tool that agents can invoke to gather
    fundamental data, screen stocks, or fetch NASDAQ-100 constituents.

    Parameters
    ----------
    api_key : str, optional
        Overrides the ``FMP_API_KEY`` environment variable.
    calls_per_minute : int
        Rate-limit cap (default 300 for paid tier; free tier is ~250/day).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        calls_per_minute: int = 300,
    ) -> None:
        self.api_key = api_key or os.getenv("FMP_API_KEY", "")
        self.base_url = FMP_BASE_URL
        self._semaphore = asyncio.Semaphore(calls_per_minute)
        self._rate_delay = 60.0 / max(calls_per_minute, 1)

    # -- public tools --------------------------------------------------------

    async def get_financial_statements(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> List[FinancialStatement]:
        """
        Fetch income-statement data for *symbol*.

        Parameters
        ----------
        period : str
            ``"annual"`` or ``"quarter"``.
        """
        endpoint = f"{self.base_url}/income-statement/{symbol}"
        params = {"period": period, "limit": str(limit)}

        data = await self._request(endpoint, params)

        statements: List[FinancialStatement] = []
        for row in data:
            statements.append(
                FinancialStatement(
                    date=row.get("date", ""),
                    symbol=row.get("symbol", symbol),
                    revenue=row.get("revenue"),
                    net_income=row.get("netIncome"),
                    eps=row.get("eps"),
                    operating_income=row.get("operatingIncome"),
                    total_assets=row.get("totalAssets"),
                    total_debt=row.get("totalDebt"),
                    free_cash_flow=row.get("freeCashFlow"),
                )
            )
        return statements

    async def get_stock_screener(
        self,
        market_cap_min: Optional[float] = None,
        market_cap_max: Optional[float] = None,
        sector: Optional[str] = None,
        exchange: str = "NASDAQ",
        limit: int = 100,
    ) -> List[StockScreenerResult]:
        """
        Screen stocks using FMP's screener endpoint.
        """
        endpoint = f"{self.base_url}/stock-screener"
        params: Dict[str, str] = {
            "exchange": exchange,
            "limit": str(limit),
        }
        if market_cap_min is not None:
            params["marketCapMoreThan"] = str(int(market_cap_min))
        if market_cap_max is not None:
            params["marketCapLowerThan"] = str(int(market_cap_max))
        if sector:
            params["sector"] = sector

        data = await self._request(endpoint, params)

        return [
            StockScreenerResult(
                symbol=row.get("symbol", ""),
                company_name=row.get("companyName", ""),
                market_cap=row.get("marketCap"),
                sector=row.get("sector"),
                industry=row.get("industry"),
                beta=row.get("beta"),
                price=row.get("price"),
                volume=row.get("volume"),
            )
            for row in data
        ]

    async def get_nasdaq100_constituents(self) -> List[str]:
        """
        Fetch the live list of NASDAQ-100 constituent symbols.

        Falls back to the hard-coded list in ``config.py`` if the API call
        fails.
        """
        endpoint = f"{self.base_url}/nasdaq_constituent"
        try:
            data = await self._request(endpoint, {})
            return [row["symbol"] for row in data if "symbol" in row]
        except Exception:
            from src.utils.config import NASDAQ_100_TICKERS
            return list(NASDAQ_100_TICKERS)

    async def get_daily_prices(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical daily prices for *symbol* via FMP.

        Returns a DataFrame with columns:
        ``open``, ``high``, ``low``, ``close``, ``volume``
        and a ``DatetimeIndex``.
        """
        endpoint = f"{self.base_url}/historical-price-full/{symbol}"
        params: Dict[str, str] = {}
        if start:
            params["from"] = start
        if end:
            params["to"] = end

        data = await self._request(endpoint, params)
        historical = data.get("historical", [])

        if not historical:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(historical)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df.rename(columns={
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume",
        })
        return df[["open", "high", "low", "close", "volume"]].astype(float)

    # -- internals -----------------------------------------------------------

    async def _request(
        self, endpoint: str, params: Dict[str, str]
    ) -> Any:
        """Execute a rate-limited GET request to FMP."""
        params["apikey"] = self.api_key

        async with self._semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            await asyncio.sleep(self._rate_delay)

        if isinstance(data, dict) and "Error Message" in data:
            raise ValueError(f"FMP error: {data['Error Message']}")

        return data
