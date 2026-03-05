"""
Project AutoQuant — Agent Base Layer
=====================================
Pydantic schemas for inter-agent communication and LangGraph-compatible
``GraphState`` TypedDict for the state machine.
"""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic Schemas — Structured JSON for deterministic agent communication
# ═══════════════════════════════════════════════════════════════════════════

class Hypothesis(BaseModel):
    """A trading hypothesis proposed by the SynthesisAgent."""
    title: str
    rationale: str
    target_asset_class: str = "NASDAQ 100"
    frequency: str = "Daily"                      # Daily, Hourly, etc.
    factors: List[str] = Field(default_factory=list)  # e.g. ["RSI", "MACD"]
    formula_logic: str = ""


class FactorCode(BaseModel):
    """Generated Python code implementing a factor/signal strategy."""
    hypothesis_id: str
    python_code: str
    required_libraries: List[str] = Field(default_factory=list)
    is_valid: bool = False                         # Set by SafeCodeExecutor
    validation_error: Optional[str] = None


class BacktestResult(BaseModel):
    """Aggregate performance metrics from a backtest run."""
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    trades_count: int = 0
    wfo_score: float = 0.0                         # Walk-Forward score
    equity_curve: List[float] = Field(default_factory=list)
    logs: List[str] = Field(default_factory=list)


class Critique(BaseModel):
    """Structured critique from the AnalysisAgent."""
    is_success: bool = False
    decision: str = "evolve_hypothesis"            # evolve_hypothesis | fix_code | end
    suggestions: List[str] = Field(default_factory=list)
    potential_biases: List[str] = Field(default_factory=list)
    refined_hypothesis: Optional[Hypothesis] = None


class ErrorReport(BaseModel):
    """Structured error report for failed steps."""
    agent: str
    error_type: str
    error_message: str
    traceback: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
# LangGraph State — TypedDict for the StateGraph
# ═══════════════════════════════════════════════════════════════════════════

class GraphState(TypedDict, total=False):
    """
    Central state passed between LangGraph nodes.

    Each agent reads from and writes to this dictionary.  The ``decision``
    field is used by the AnalysisAgent's ``decision_router`` to choose the
    next node.
    """
    # -- Pipeline metadata ---------------------------------------------------
    iteration_count: int
    max_iterations: int
    capital: float
    universe: str                       # e.g. "NASDAQ-100"
    status: str                         # initialized | running | success | failed

    # -- Agent outputs (serialised JSON) ------------------------------------
    current_hypothesis: str             # JSON-serialised Hypothesis
    factor_code: str                    # JSON-serialised FactorCode
    backtest_results: str               # JSON-serialised BacktestResult
    critique: str                       # JSON-serialised Critique
    error: str                          # JSON-serialised ErrorReport

    # -- History for the feedback loop --------------------------------------
    hypothesis_history: List[str]       # List of serialised Hypotheses
    critique_history: List[str]         # List of serialised Critiques

    # -- Routing decision (set by AnalysisAgent) ----------------------------
    decision: str                       # evolve_hypothesis | fix_code | end


# ═══════════════════════════════════════════════════════════════════════════
# Legacy AgentState (kept for backward compatibility with simple loop)
# ═══════════════════════════════════════════════════════════════════════════

class AgentState(BaseModel):
    """The state of the multi-agent orchestration loop (legacy)."""
    iteration: int = 0
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    codes: List[FactorCode] = Field(default_factory=list)
    results: List[BacktestResult] = Field(default_factory=list)
    critiques: List[Critique] = Field(default_factory=list)
    current_step: str = "synthesis"


# ═══════════════════════════════════════════════════════════════════════════
# Abstract Base Agent
# ═══════════════════════════════════════════════════════════════════════════

class BaseAgent(ABC):
    """Base class for all pipeline agents."""

    def __init__(self, model_name: str = "gpt-4o") -> None:
        self.model_name = model_name

    @abstractmethod
    def run(self, state: GraphState) -> GraphState:
        """Process the pipeline state and return an updated copy."""
        ...
