"""
Project AutoQuant — ValidationAgent
=====================================
Runs a historical backtest (with optional Walk-Forward Optimization)
using the factor code produced by the ImplementationAgent.
"""

import json
from typing import Optional

import pandas as pd
import numpy as np

from src.agents.base import BaseAgent, GraphState, FactorCode, BacktestResult
from src.backtest.engine import BacktestEngine
from src.backtest.wfo import WalkForwardOptimizer
from src.utils.executor import SafeCodeExecutor
from src.utils.config import Settings


# ---------------------------------------------------------------------------
# ValidationAgent
# ---------------------------------------------------------------------------

class ValidationAgent(BaseAgent):
    """
    Executes the generated factor code against historical data and
    produces a ``BacktestResult``.

    If ``use_wfo=True`` (default), runs Walk-Forward Optimization to
    produce a stability-adjusted score.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        use_wfo: bool = True,
    ) -> None:
        super().__init__(model_name)
        self.use_wfo = use_wfo
        self.settings = Settings()

    def run(self, state: GraphState) -> GraphState:
        factor_json = state.get("factor_code", "{}")
        factor_code = FactorCode.model_validate_json(factor_json)

        print(f"[ValidationAgent] Running backtest for '{factor_code.hypothesis_id}'...")

        # If code was flagged invalid by ImplementationAgent, skip backtest
        if not factor_code.is_valid:
            print("[ValidationAgent] Skipping backtest — code is invalid.")
            empty = BacktestResult(
                logs=[f"[SKIP] Code validation failed: {factor_code.validation_error}"],
            )
            state["backtest_results"] = empty.model_dump_json()
            state["status"] = "backtest_error"
            return state

        # Load price data
        price_data = self._load_data()

        # Build executable signal function from the factor code string
        signal_fn = self._build_signal_fn(factor_code.python_code)

        if signal_fn is None:
            error_result = BacktestResult(
                logs=["[ERROR] Could not compile signal_generator from factor code."],
            )
            state["backtest_results"] = error_result.model_dump_json()
            state["status"] = "backtest_error"
            return state

        # Run backtest (WFO or simple)
        try:
            if self.use_wfo and len(price_data) > (self.settings.wfo_train_days + self.settings.wfo_test_days):
                wfo = WalkForwardOptimizer(
                    train_days=self.settings.wfo_train_days,
                    test_days=self.settings.wfo_test_days,
                    initial_capital=self.settings.initial_capital,
                    commission_rate=self.settings.commission_rate,
                    slippage=self.settings.slippage,
                )
                result = wfo.run(signal_fn, price_data)
            else:
                engine = BacktestEngine(
                    initial_capital=self.settings.initial_capital,
                    commission_rate=self.settings.commission_rate,
                    slippage=self.settings.slippage,
                )
                result = engine.run(signal_fn, price_data)

            state["backtest_results"] = result.model_dump_json()
            state["status"] = "backtest_complete"
            print(
                f"[ValidationAgent] Backtest done | "
                f"Sharpe={result.sharpe_ratio:.3f} | "
                f"MDD={result.max_drawdown:.2%} | "
                f"Trades={result.trades_count}"
            )

        except Exception as exc:
            error_result = BacktestResult(
                logs=[f"[ERROR] Backtest runtime error: {exc}"],
            )
            state["backtest_results"] = error_result.model_dump_json()
            state["status"] = "backtest_error"
            print(f"[ValidationAgent] Backtest failed: {exc}")

        return state

    # -- helpers -------------------------------------------------------------

    def _load_data(self) -> pd.DataFrame:
        """
        Load price data for backtesting.

        Uses a representative NASDAQ-100 stock (AAPL) for the scaffold.
        In production, this would load the full universe via DataLoader.
        """
        try:
            from src.data.loader import DataLoader
            loader = DataLoader()
            universe = loader.load_universe(tickers=["AAPL"], start="2018-01-01")
            if "AAPL" in universe and not universe["AAPL"].empty:
                return universe["AAPL"]
        except Exception as exc:
            print(f"[ValidationAgent] DataLoader failed ({exc}), using synthetic data.")

        # Fallback: synthetic data for scaffold testing
        return self._generate_synthetic_data()

    @staticmethod
    def _generate_synthetic_data(n: int = 1260) -> pd.DataFrame:
        """Generate ~5 years of synthetic OHLCV data."""
        np.random.seed(42)
        dates = pd.date_range("2018-01-01", periods=n, freq="B")
        prices = 150 + np.random.randn(n).cumsum() * 2

        return pd.DataFrame({
            "open": prices + np.random.randn(n) * 0.5,
            "high": prices + abs(np.random.randn(n)) * 1.5,
            "low": prices - abs(np.random.randn(n)) * 1.5,
            "close": prices,
            "volume": np.random.randint(5_000_000, 50_000_000, n).astype(float),
        }, index=dates)

    @staticmethod
    def _build_signal_fn(code_str: str):
        """
        Compile the factor code string and extract the ``signal_generator``
        function.  Returns ``None`` on failure.
        """
        namespace: dict = {}
        try:
            import pandas as pd  # noqa: F811
            import numpy as np  # noqa: F811

            namespace = {
                "pd": pd, "pandas": pd,
                "np": np, "numpy": np,
                "__builtins__": __builtins__,
            }
            exec(code_str, namespace)  # noqa: S102

            fn = namespace.get("signal_generator")
            if callable(fn):
                return fn
        except Exception as exc:
            print(f"[ValidationAgent] Failed to compile signal_generator: {exc}")

        return None
