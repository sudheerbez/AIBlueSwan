"""AutoQuant — MCP data layer."""
from src.data.yfinance_client import YFinanceClient
from src.data.fmp import FMPClient
from src.data.stooq_client import StooqClient
from src.data.macro_client import MacroClient
from src.data.loader import DataLoader

__all__ = ["YFinanceClient", "FMPClient", "StooqClient", "MacroClient", "DataLoader"]
