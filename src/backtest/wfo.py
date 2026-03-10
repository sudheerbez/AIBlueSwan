"""
Project AutoQuant — Walk-Forward Optimization (WFO)
====================================================
Implements rolling in-sample / out-of-sample validation to guard against
overfitting.  Each OOS window is backtested independently and the results
are aggregated into a single WFO score.
"""

from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from src.agents.base import BacktestResult, TradeRecord, OHLCVBar
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import (
    sharpe_ratio as calc_sharpe,
    max_drawdown as calc_mdd,
    annualized_return as calc_cagr,
    volatility as calc_vol,
)


# ---------------------------------------------------------------------------
# WalkForwardOptimizer
# ---------------------------------------------------------------------------

class WalkForwardOptimizer:
    """
    Walk-Forward Optimization framework.

    Splits the historical data into rolling windows:

    .. code-block:: text

        |--- Train (IS) ---|--- Test (OOS) ---|
                           |--- Train ---|--- Test ---|
                                         |--- Train ---|--- Test ---|

    Each OOS window is backtested with ``BacktestEngine``.  The final
    WFO score reflects the *stability* of the strategy across windows.

    Parameters
    ----------
    train_days : int
        Number of bars for the in-sample (training) window.
    test_days : int
        Number of bars for the out-of-sample (test) window.
    step_days : int, optional
        Number of bars to advance the window each step.  Defaults to
        ``test_days`` (non-overlapping OOS windows).
    initial_capital : float
        Starting capital for each OOS backtest.
    commission_rate, slippage : float
        Passed through to ``BacktestEngine``.
    """

    def __init__(
        self,
        train_days: int = 252,
        test_days: int = 63,
        step_days: Optional[int] = None,
        initial_capital: float = 100_000.0,
        commission_rate: float = 0.001,
        slippage: float = 0.0005,
    ) -> None:
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days or test_days
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
        Execute Walk-Forward Optimization across all available windows.

        Parameters
        ----------
        signal_fn : callable
            ``(DataFrame) -> DataFrame`` with a ``signal`` column.
        price_data : pd.DataFrame
            Full historical OHLCV DataFrame.

        Returns
        -------
        BacktestResult
            Aggregated metrics plus a ``wfo_score`` measuring stability.
        """
        windows = self._generate_windows(len(price_data))
        logs: List[str] = []
        window_sharpes: List[float] = []
        all_oos_returns: List[float] = []
        all_equity: List[float] = [self.initial_capital]
        all_trade_log: List[TradeRecord] = []
        total_trades = 0

        logs.append(
            f"[WFO] Starting Walk-Forward Optimization | "
            f"train={self.train_days}d, test={self.test_days}d, "
            f"windows={len(windows)}"
        )

        for idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
            # Pass all data from the start through the OOS end to the signal
            # function so indicators with long lookback periods (e.g. SMA-200)
            # can warm up properly.  Then slice only the OOS portion for
            # performance measurement.

            full_context = price_data.iloc[:test_end].copy()

            if full_context.empty:
                continue

            # Generate signals on the full context
            try:
                signalled_data = signal_fn(full_context)
            except Exception as exc:
                logs.append(f"[WFO] Window {idx + 1}: signal generation failed: {exc}")
                continue

            # Extract only the OOS portion for backtesting
            oos_data = signalled_data.iloc[test_start:test_end].copy()

            if oos_data.empty:
                continue

            # Run the backtest engine on the pre-signalled OOS data
            # We wrap it in a pass-through signal_fn since signals are already computed
            def _passthrough(df: pd.DataFrame) -> pd.DataFrame:
                return df

            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                commission_rate=self.commission_rate,
                slippage=self.slippage,
            )
            result = engine.run(_passthrough, oos_data)

            # Collect per-window stats
            window_sharpes.append(result.sharpe_ratio)
            total_trades += result.trades_count
            all_equity.extend(result.equity_curve[1:])  # skip duplicate start
            all_trade_log.extend(result.trade_log)

            # Reconstruct OOS returns from equity curve
            for j in range(1, len(result.equity_curve)):
                prev = result.equity_curve[j - 1]
                curr = result.equity_curve[j]
                ret = (curr / prev - 1) if prev != 0 else 0.0
                all_oos_returns.append(ret)

            logs.append(
                f"[WFO] Window {idx + 1}/{len(windows)} | "
                f"OOS Sharpe={result.sharpe_ratio:.3f} | "
                f"OOS Return={result.annualized_return:.2%} | "
                f"Trades={result.trades_count}"
            )

        # -- Aggregate results -----------------------------------------------
        if not all_oos_returns:
            logs.append("[WFO] No valid OOS windows produced.")
            return BacktestResult(
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                annualized_return=0.0,
                volatility=0.0,
                trades_count=0,
                wfo_score=0.0,
                equity_curve=[self.initial_capital],
                trade_log=[],
                ohlcv_data=BacktestEngine._extract_ohlcv(price_data),
                logs=logs,
            )

        oos_series = pd.Series(all_oos_returns)

        # WFO Score: mean(window Sharpes) penalised by their variance.
        # A perfectly stable strategy has WFO ≈ mean Sharpe;
        # an unstable one gets heavily penalised.
        mean_sharpe = float(np.mean(window_sharpes)) if window_sharpes else 0.0
        std_sharpe = float(np.std(window_sharpes)) if len(window_sharpes) > 1 else 0.0
        wfo_score = mean_sharpe - std_sharpe  # simple stability-adjusted score

        agg_result = BacktestResult(
            sharpe_ratio=calc_sharpe(oos_series),
            max_drawdown=calc_mdd(oos_series),
            annualized_return=calc_cagr(oos_series),
            volatility=calc_vol(oos_series),
            trades_count=total_trades,
            wfo_score=round(wfo_score, 4),
            equity_curve=all_equity,
            trade_log=all_trade_log,
            ohlcv_data=BacktestEngine._extract_ohlcv(price_data),
            logs=logs,
        )

        logs.append(
            f"[WFO] Complete | Agg Sharpe={agg_result.sharpe_ratio:.3f} | "
            f"WFO Score={wfo_score:.3f} | Windows={len(windows)}"
        )

        return agg_result

    # -- window generation ---------------------------------------------------

    def _generate_windows(
        self, n_bars: int
    ) -> List[tuple[int, int, int, int]]:
        """
        Generate ``(train_start, train_end, test_start, test_end)`` index
        tuples for all valid rolling windows.
        """
        windows: List[tuple[int, int, int, int]] = []
        start = 0

        while True:
            train_start = start
            train_end = train_start + self.train_days
            test_start = train_end
            test_end = test_start + self.test_days

            if test_end > n_bars:
                break

            windows.append((train_start, train_end, test_start, test_end))
            start += self.step_days

        return windows
