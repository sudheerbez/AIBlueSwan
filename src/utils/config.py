"""
Project AutoQuant — Centralized Configuration
==============================================
All environment variables, constants, and validated settings for the pipeline.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Settings (Pydantic-validated)
# ---------------------------------------------------------------------------

class Settings(BaseModel):
    """Validated configuration loaded from environment variables."""

    # Data-source API keys
    alpha_vantage_api_key: str = Field(
        default_factory=lambda: os.getenv("ALPHA_VANTAGE_API_KEY", "")
    )
    fmp_api_key: str = Field(
        default_factory=lambda: os.getenv("FMP_API_KEY", "")
    )

    # LLM provider keys
    openai_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    anthropic_api_key: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    google_api_key: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY", "")
    )

    # LLM defaults
    default_llm_model: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_LLM_MODEL", "gpt-4o")
    )

    # Trading defaults
    initial_capital: float = 100_000.0
    commission_rate: float = 0.001        # 10 bps per trade
    slippage: float = 0.0005              # 5 bps slippage

    # Walk-Forward Optimization defaults
    wfo_train_days: int = 252             # ~1 year in-sample
    wfo_test_days: int = 63              # ~1 quarter out-of-sample

    # Performance thresholds
    min_sharpe_ratio: float = 1.5
    max_drawdown_limit: float = -0.20     # -20%
    max_iterations: int = 5

    class Config:
        env_prefix = "AUTOQUANT_"


# ---------------------------------------------------------------------------
# NASDAQ-100 Constituents (as of early 2025)
# ---------------------------------------------------------------------------

NASDAQ_100_TICKERS: list[str] = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD",
    "AMGN", "AMZN", "ANSS", "APP", "ARM", "ASML", "AVGO", "AZN", "BIIB",
    "BKNG", "BKR", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT",
    "CRWD", "CSCO", "CSGP", "CTAS", "CTSH", "DASH", "DDOG", "DLTR",
    "DXCM", "EA", "EXC", "FANG", "FAST", "FTNT", "GEHC", "GFS", "GILD",
    "GOOG", "GOOGL", "HON", "IDXX", "ILMN", "INTC", "INTU", "ISRG",
    "KDP", "KHC", "KLAC", "LIN", "LRCX", "LULU", "MAR", "MCHP", "MDB",
    "MDLZ", "MELI", "META", "MNST", "MRNA", "MRVL", "MSFT", "MU",
    "NFLX", "NVDA", "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR",
    "PDD", "PEP", "PYPL", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SMCI",
    "SNPS", "TEAM", "TMUS", "TSLA", "TTD", "TTWO", "TXN", "VRSK", "VRTX",
    "WBD", "WDAY", "XEL", "ZS",
]

# ---------------------------------------------------------------------------
# API Base URLs
# ---------------------------------------------------------------------------

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# ---------------------------------------------------------------------------
# Data Cache Directory
# ---------------------------------------------------------------------------

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")
