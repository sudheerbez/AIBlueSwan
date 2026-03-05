"""
Project AutoQuant — Backtesting Engine
=======================================
Event-driven portfolio simulator that converts signal DataFrames into
equity curves and performance metrics.
"""

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.agents.base import BacktestResult
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
        equity_curve, trades, daily_returns = self._simulate(df, logs)

        # 3.  Calculate metrics
        returns_series = pd.Series(daily_returns, index=df.index[: len(daily_returns)])
        result = calculate_all_metrics(
            returns=returns_series,
            equity_curve=equity_curve,
            trades_count=trades,
        )
        result.logs = logs
        return result

    # -- simulation core -----------------------------------------------------

    def _simulate(
        self,
        df: pd.DataFrame,
        logs: List[str],
    ) -> tuple[List[float], int, List[float]]:
        """
        Walk through the DataFrame bar-by-bar, tracking positions and equity.

        Returns
        -------
        equity_curve : list[float]
        trades_count : int
        daily_returns : list[float]
        """
        cash = self.initial_capital
        position = 0        # number of shares held (positive = long)
        equity_curve: List[float] = [cash]
        trades = 0
        daily_returns: List[float] = []

        signals = df["signal"].fillna(0).values
        closes = df["close"].values if "close" in df.columns else df["Close"].values

        logs.append(f"[INFO] Backtest started | capital=${self.initial_capital:,.0f}")

        for i in range(1, len(df)):
            price = closes[i]
            prev_price = closes[i - 1]
            sig = int(signals[i])

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
                    trades += 1

            elif sig == -1 and position >= 0:
                # Exit long
                if position > 0:
                    proceeds = position * price * (1 - self.slippage - self.commission_rate)
                    cash += proceeds
                    position = 0
                    trades += 1

            # -- Equity mark-to-market ---------------------------------------
            equity = cash + position * price
            prev_equity = equity_curve[-1]
            daily_ret = (equity / prev_equity - 1) if prev_equity != 0 else 0.0
            daily_returns.append(daily_ret)
            equity_curve.append(equity)

        logs.append(f"[INFO] Backtest finished | trades={trades} | final_equity=${equity_curve[-1]:,.2f}")
        return equity_curve, trades, daily_returns

    # -- helpers -------------------------------------------------------------

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
            logs=logs,
        )
