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
    llm_provider: str                   # OpenAI | Anthropic | Gemini

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
            def __init__(self):
                pass

            def invoke(self, messages):
                # Simple heuristic to decide which mock response to return
                text = "\n".join(getattr(m, "content", str(m)) for m in messages)
                low = text.lower()
                # Prefer returning code when the prompt asks for a function
                if "signal_generator" in low or "python" in low or "write a single function" in low:
                    content = (
                        "def signal_generator(df):\n"
                        "    # Defensive, vectorized signal generator suitable for sandbox testing\n"
                        "    df = df.copy()\n"
                        "    # Ensure required columns exist\n"
                        "    for col in ('open','high','low','close','volume'):\n"
                        "        if col not in df.columns:\n"
                        "            raise ValueError(f'Missing column: {col}')\n"
                        "    # Simple momentum + volatility filter to produce some trades\n"
                        "    df['ma5'] = df['close'].rolling(5, min_periods=1).mean()\n"
                        "    df['ret'] = df['close'].pct_change().fillna(0)\n"
                        "    df['vol5'] = df['ret'].rolling(5, min_periods=1).std().fillna(0)\n"
                        "    df['signal'] = 0\n"
                        "    long_mask = (df['close'] > df['ma5']) & (df['vol5'] < df['vol5'].rolling(20, min_periods=1).quantile(0.75))\n"
                        "    df.loc[long_mask, 'signal'] = 1\n"
                        "    # Avoid lookahead: shift signals forward by 1 bar\n"
                        "    df['signal'] = df['signal'].shift(1).fillna(0).astype(int)\n"
                        "    return df\n"
                        "\n"
                        "# define a helper name expected by the sandbox (ensures top-level name exists)\n"
                        "signal_generator = signal_generator\n"
                    )
                    return MockResponse(content)
                if "generate a new trading hypothesis" in low or ("generate" in low and "hypothesis" in low):
                    content = '{"title": "Mock Momentum+Volatility", "rationale": "A simple mock hypothesis for testing.", "target_asset_class": "NASDAQ 100", "frequency": "Daily", "factors": ["mock_momentum", "mock_volatility"], "formula_logic": "if close > close.rolling(5).mean(): signal=1 else signal=0"}'
                    return MockResponse(content)
                # Default: a generic critique / instruction
                content = '{"is_success": false, "decision": "evolve_hypothesis", "suggestions": ["Increase robustness; reduce lookahead."], "potential_biases": []}'
                return MockResponse(content)

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
