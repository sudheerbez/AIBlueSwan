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
from src.utils.config import Settings


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


class TradeRecord(BaseModel):
    """A single trade entry or exit."""
    timestamp: str                                  # ISO date string
    price: float
    action: str                                     # "entry" or "exit"
    pnl: Optional[float] = None                     # Realized PnL (exit only)
    equity_after: float = 0.0


class OHLCVBar(BaseModel):
    """A single OHLCV bar for chart rendering."""
    time: str                                       # ISO date or YYYY-MM-DD
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class BacktestResult(BaseModel):
    """Aggregate performance metrics from a backtest run."""
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    trades_count: int = 0
    wfo_score: float = 0.0                         # Walk-Forward score
    equity_curve: List[float] = Field(default_factory=list)
    trade_log: List[TradeRecord] = Field(default_factory=list)
    ohlcv_data: List[OHLCVBar] = Field(default_factory=list)
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
    llm_provider: str                   # OpenAI | Anthropic | Gemini

    # -- Agent outputs (serialised JSON) ------------------------------------
    current_hypothesis: str             # JSON-serialised Hypothesis
    factor_code: str                    # JSON-serialised FactorCode
    backtest_results: str               # JSON-serialised BacktestResult
    critique: str                       # JSON-serialised Critique
    error: str                          # JSON-serialised ErrorReport

    # -- Best-so-far tracking (set by AnalysisAgent) -----------------------
    best_hypothesis: str                # JSON-serialised Hypothesis (best Sharpe)
    best_backtest_results: str          # JSON-serialised BacktestResult (best Sharpe)
    best_factor_code: str               # JSON-serialised FactorCode (best Sharpe)
    best_sharpe: float                  # Sharpe ratio of the best iteration

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

# Module-level counter so MockLLM cycles through templates across get_llm() calls
_mock_call_count = 0


class BaseAgent(ABC):
    """Base class for all pipeline agents."""

    def __init__(self) -> None:
        self.settings = Settings()

    def get_llm(self, provider: str, temperature: float = 0.7, model_name: str = None):
        """Instantiate the correct LangChain chat model based on provider."""
        import os
        # Lightweight mock LLM used for offline testing when no API key is available
        class MockResponse:
            def __init__(self, content: str):
                self.content = content

        class MockLLM:
            """
            Mock LLM that cycles through proven strategy templates.
            Produces actual high-Sharpe strategies instead of trivial ones.
            """

            def __init__(self):
                pass

            def invoke(self, messages):
                global _mock_call_count
                from src.strategies.templates import STRATEGY_TEMPLATES
                _mock_call_count += 1

                text = "\n".join(getattr(m, "content", str(m)) for m in messages)
                low = text.lower()

                # 1) Hypothesis generation (check FIRST — synthesis prompts also contain "python")
                if "hypothesis" in low and "json" in low:
                    idx = (_mock_call_count - 1) % len(STRATEGY_TEMPLATES)
                    template = STRATEGY_TEMPLATES[idx]
                    import json
                    hypothesis = {
                        "title": template["title"],
                        "rationale": template["rationale"],
                        "target_asset_class": "NASDAQ 100",
                        "frequency": "Daily",
                        "factors": template["factors"],
                        "formula_logic": template["formula_logic"],
                    }
                    return MockResponse(content=json.dumps(hypothesis))

                # 2) Analysis/critique (check BEFORE code — analysis prompts mention "python" too)
                if "critique" in low or ("backtest" in low and "sharpe" in low):
                    import re
                    sharpe_match = re.search(r"sharpe.*?:\s*([-\d.]+)", low)
                    sharpe_val = float(sharpe_match.group(1)) if sharpe_match else 0.0

                    if sharpe_val >= 1.5:
                        content = '{"is_success": true, "decision": "end", "suggestions": ["Strategy meets targets."], "potential_biases": []}'
                    elif sharpe_val >= 1.0:
                        content = '{"is_success": false, "decision": "evolve_hypothesis", "suggestions": ["Add drawdown control: exit when 20-day rolling DD > -5%.", "Add volatility normalization.", "Consider composite multi-factor approach."], "potential_biases": ["Possible curve-fitting"]}'
                    elif sharpe_val >= 0.5:
                        content = '{"is_success": false, "decision": "evolve_hypothesis", "suggestions": ["Switch to volatility-normalized time-series momentum (TSMOM) with SMA-200 filter.", "Add multi-factor composite scoring.", "Add 20-day drawdown control."], "potential_biases": ["Strategy may be too simple"]}'
                    else:
                        content = '{"is_success": false, "decision": "evolve_hypothesis", "suggestions": ["Switch to a proven momentum+drawdown control strategy.", "Use composite multi-factor z-score approach.", "Add SMA-200 trend filter and volatility regime filter."], "potential_biases": ["Fundamental strategy flaw"]}'
                    return MockResponse(content=content)

                # 3) Code generation: return proven template code
                if "signal_generator" in low or "python" in low or "write a single function" in low:
                    idx = (_mock_call_count - 1) % len(STRATEGY_TEMPLATES)
                    template = STRATEGY_TEMPLATES[idx]
                    return MockResponse(content=template["code"])

                # Default fallback
                content = '{"is_success": false, "decision": "evolve_hypothesis", "suggestions": ["Try volatility-normalized momentum with drawdown control."], "potential_biases": []}'
                return MockResponse(content=content)

        use_mock = os.getenv("USE_MOCK_LLM", "") == "1"
        if use_mock:
            return MockLLM()
        
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model_name or self.settings.default_llm_model, 
            base_url=self.settings.ollama_base_url,
            temperature=temperature
        )

    @staticmethod
    def clean_llm_output(text: str) -> str:
        """Strip markdown fences (like ```json ... ```) from LLM output, handling preambles."""
        text = text.strip()
        import re
        # Find everything between triple backticks, optional language specifier
        match = re.search(r"```[a-zA-Z]*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    @abstractmethod
    def run(self, state: GraphState) -> GraphState:
        """Process the pipeline state and return an updated copy."""
        ...
