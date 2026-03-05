"""
Project AutoQuant — Performance Metrics
=========================================
Pure-function implementations of standard quantitative performance metrics.
All functions operate on a ``pd.Series`` of simple (arithmetic) returns.
"""

import numpy as np
import pandas as pd
from typing import Dict

from src.agents.base import BacktestResult


# ---------------------------------------------------------------------------
# Individual Metric Functions
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualised Sharpe Ratio.

    .. math::
        SR = \\frac{\\bar{r} - r_f}{\\sigma_r} \\cdot \\sqrt{N}
    """
    if returns.empty or returns.std() == 0:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualised Sortino Ratio (downside deviation only)."""
    if returns.empty:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if downside.empty or downside.std() == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / downside.std())


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum Drawdown (expressed as a negative fraction, e.g. −0.25 = −25 %).

    Computed from the cumulative return curve.
    """
    if returns.empty:
        return 0.0
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdowns = cum / running_max - 1
    return float(drawdowns.min())


def annualized_return(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compound annualised growth rate (CAGR)."""
    if returns.empty:
        return 0.0
    total_return = (1 + returns).prod()
    n_years = len(returns) / periods_per_year
    if n_years == 0:
        return 0.0
    return float(total_return ** (1 / n_years) - 1)


def volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Annualised volatility (standard deviation of returns)."""
    if returns.empty:
        return 0.0
    return float(returns.std() * np.sqrt(periods_per_year))


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calmar Ratio = Annualised Return / |Max Drawdown|."""
    ann_ret = annualized_return(returns, periods_per_year)
    mdd = max_drawdown(returns)
    if mdd == 0:
        return float("inf") if ann_ret > 0 else 0.0
    return float(ann_ret / abs(mdd))


# ---------------------------------------------------------------------------
# Aggregate calculator
# ---------------------------------------------------------------------------

def calculate_all_metrics(
    returns: pd.Series,
    equity_curve: list[float],
    trades_count: int = 0,
    wfo_score: float = 0.0,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> BacktestResult:
    """
    Compute all metrics and return a ``BacktestResult`` Pydantic model.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.
    equity_curve : list[float]
        Portfolio equity values over time.
    trades_count : int
        Number of round-trip trades executed.
    wfo_score : float
        Walk-Forward Optimization score (set by the WFO module).
    """
    return BacktestResult(
        sharpe_ratio=sharpe_ratio(returns, risk_free_rate, periods_per_year),
        max_drawdown=max_drawdown(returns),
        annualized_return=annualized_return(returns, periods_per_year),
        volatility=volatility(returns, periods_per_year),
        trades_count=trades_count,
        wfo_score=wfo_score,
        equity_curve=equity_curve,
        logs=[],
    )
