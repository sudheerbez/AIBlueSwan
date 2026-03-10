"""
Project AutoQuant — Backtesting Engine
=======================================
Event-driven portfolio simulator that converts signal DataFrames into
equity curves and performance metrics.
"""

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.agents.base import BacktestResult, TradeRecord, OHLCVBar
from src.backtest.metrics import calculate_all_metrics


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Simulates a long/short equity strategy from a signal DataFrame.

    The engine expects a *signal function* that takes a price DataFrame
    and returns the same DataFrame with an added ``signal`` column:
    * ``+1`` → long entry
    * ``-1`` → short entry / exit long
    * ``0``  → no action / flat

    Parameters
    ----------
    initial_capital : float
        Starting portfolio value in USD.
    commission_rate : float
        Proportional transaction cost per trade (e.g. 0.001 = 10 bps).
    slippage : float
        Proportional slippage per trade (e.g. 0.0005 = 5 bps).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_rate: float = 0.001,
        slippage: float = 0.0005,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage

    # -- public API ----------------------------------------------------------

    def run(
        self,
        signal_fn: Callable[[pd.DataFrame], pd.DataFrame],
        price_data: pd.DataFrame,
    ) -> BacktestResult:
        """
        Execute the backtest.

        Parameters
        ----------
        signal_fn : callable
            ``(DataFrame) -> DataFrame`` with a ``signal`` column appended.
        price_data : pd.DataFrame
            OHLCV DataFrame with ``DatetimeIndex``.

        Returns
        -------
        BacktestResult
            Pydantic model containing all performance metrics.
        """
        df = price_data.copy()
        logs: List[str] = []

        # 1.  Generate signals
        try:
            df = signal_fn(df)
        except Exception as exc:
            logs.append(f"[ERROR] Signal generation failed: {exc}")
            return self._empty_result(logs)

        if "signal" not in df.columns:
            logs.append("[ERROR] signal_fn did not produce a 'signal' column.")
            return self._empty_result(logs)

        # 2.  Simulate trades
        equity_curve, trades, daily_returns, trade_log = self._simulate(df, logs)

        # 3.  Build OHLCV for chart rendering
        ohlcv_data = self._extract_ohlcv(df)

        # 4.  Calculate metrics
        returns_series = pd.Series(daily_returns, index=df.index[: len(daily_returns)])
        result = calculate_all_metrics(
            returns=returns_series,
            equity_curve=equity_curve,
            trades_count=trades,
        )
        result.logs = logs
        result.trade_log = trade_log
        result.ohlcv_data = ohlcv_data
        return result

    # -- simulation core -----------------------------------------------------

    def _simulate(
        self,
        df: pd.DataFrame,
        logs: List[str],
    ) -> tuple[List[float], int, List[float], List[TradeRecord]]:
        """
        Walk through the DataFrame bar-by-bar, tracking positions and equity.

        Returns
        -------
        equity_curve : list[float]
        trades_count : int
        daily_returns : list[float]
        trade_log : list[TradeRecord]
        """
        cash = self.initial_capital
        position = 0        # number of shares held (positive = long)
        equity_curve: List[float] = [cash]
        trades = 0
        daily_returns: List[float] = []
        trade_log: List[TradeRecord] = []
        entry_price = 0.0
        entry_cost = 0.0

        signals = df["signal"].fillna(0).values
        closes = df["close"].values if "close" in df.columns else df["Close"].values
        dates = df.index

        logs.append(f"[INFO] Backtest started | capital=${self.initial_capital:,.0f}")

        for i in range(1, len(df)):
            price = closes[i]
            prev_price = closes[i - 1]
            sig = int(signals[i])
            ts = str(dates[i].date()) if hasattr(dates[i], 'date') else str(dates[i])

            # -- Position management ----------------------------------------
            if sig == 1 and position <= 0:
                # Go long: close any short, then buy
                if position < 0:
                    cash += position * price * (1 + self.slippage + self.commission_rate)
                    position = 0
                    trades += 1
                shares_to_buy = int(cash // (price * (1 + self.slippage + self.commission_rate)))
                if shares_to_buy > 0:
                    cost = shares_to_buy * price * (1 + self.slippage + self.commission_rate)
                    cash -= cost
                    position = shares_to_buy
                    entry_price = price
                    entry_cost = cost
                    trades += 1
                    equity = cash + position * price
                    trade_log.append(TradeRecord(
                        timestamp=ts,
                        price=round(price, 2),
                        action="entry",
                        equity_after=round(equity, 2),
                    ))

            elif sig == -1 and position >= 0:
                # Exit long
                if position > 0:
                    proceeds = position * price * (1 - self.slippage - self.commission_rate)
                    pnl = proceeds - entry_cost
                    cash += proceeds
                    equity = cash
                    trade_log.append(TradeRecord(
                        timestamp=ts,
                        price=round(price, 2),
                        action="exit",
                        pnl=round(pnl, 2),
                        equity_after=round(equity, 2),
                    ))
                    position = 0
                    trades += 1

            # -- Equity mark-to-market ---------------------------------------
            equity = cash + position * price
            prev_equity = equity_curve[-1]
            daily_ret = (equity / prev_equity - 1) if prev_equity != 0 else 0.0
            daily_returns.append(daily_ret)
            equity_curve.append(equity)

        logs.append(f"[INFO] Backtest finished | trades={trades} | final_equity=${equity_curve[-1]:,.2f}")
        return equity_curve, trades, daily_returns, trade_log

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _extract_ohlcv(df: pd.DataFrame) -> List[OHLCVBar]:
        """Extract OHLCV data from DataFrame for frontend chart rendering."""
        bars = []
        for idx, row in df.iterrows():
            ts = str(idx.date()) if hasattr(idx, 'date') else str(idx)
            bars.append(OHLCVBar(
                time=ts,
                open=round(float(row.get("open", row.get("Open", 0))), 2),
                high=round(float(row.get("high", row.get("High", 0))), 2),
                low=round(float(row.get("low", row.get("Low", 0))), 2),
                close=round(float(row.get("close", row.get("Close", 0))), 2),
                volume=round(float(row.get("volume", row.get("Volume", 0))), 0),
            ))
        return bars

    def _empty_result(self, logs: List[str]) -> BacktestResult:
        """Return a zeroed-out result when the backtest cannot proceed."""
        return BacktestResult(
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            annualized_return=0.0,
            volatility=0.0,
            trades_count=0,
            wfo_score=0.0,
            equity_curve=[self.initial_capital],
            trade_log=[],
            ohlcv_data=[],
            logs=logs,
        )
