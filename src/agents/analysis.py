"""
Project AutoQuant — AnalysisAgent
====================================
Evaluates backtest metrics, identifies biases, and routes the pipeline
to the appropriate next step (evolve hypothesis, fix code, or terminate).
"""

import json
import os
from typing import Optional

from src.agents.base import (
    BaseAgent,
    GraphState,
    BacktestResult,
    Critique,
    Hypothesis,
    FactorCode,
)
from src.utils.config import Settings


# ---------------------------------------------------------------------------
# System prompt for critique generation
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM_PROMPT = """\
You are a senior quant portfolio manager reviewing a backtest.

Given the performance metrics and strategy details, produce a structured
critique as JSON:
{
  "is_success": true/false,
  "decision": "evolve_hypothesis" | "fix_code" | "end",
  "suggestions": ["..."],
  "potential_biases": ["..."]
}

Decision rules:
- "end" if Sharpe > 1.5 AND MaxDD > -20% AND WFO score > 1.0
- "fix_code" if the backtest errored or produced 0 trades
- "evolve_hypothesis" otherwise

Be specific in suggestions: mention exact factors, timeframes, or
risk management improvements.
"""


# ---------------------------------------------------------------------------
# AnalysisAgent
# ---------------------------------------------------------------------------

class AnalysisAgent(BaseAgent):
    """
    Evaluates backtest metrics and produces a Critique with a routing
    decision for the LangGraph conditional edge.
    """

    def __init__(self, model_name: str = "gpt-4o") -> None:
        super().__init__(model_name)
        self.settings = Settings()

    def run(self, state: GraphState) -> GraphState:
        results_json = state.get("backtest_results", "{}")
        result = BacktestResult.model_validate_json(results_json)
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", 5)

        print(
            f"[AnalysisAgent] Evaluating | Sharpe={result.sharpe_ratio:.3f} | "
            f"MDD={result.max_drawdown:.2%} | WFO={result.wfo_score:.3f}"
        )

        critique = self._generate_critique(result, state)

        # Force termination if max iterations reached
        if iteration + 1 >= max_iter and critique.decision != "end":
            critique.decision = "end"
            critique.suggestions.append(
                f"Max iterations ({max_iter}) reached — terminating pipeline."
            )

        # Update state
        state["critique"] = critique.model_dump_json()
        state["decision"] = critique.decision
        state["iteration_count"] = iteration + 1

        # Append to critique history
        history = list(state.get("critique_history", []))
        history.append(critique.model_dump_json())
        state["critique_history"] = history

        status_map = {
            "end": "success" if critique.is_success else "failed",
            "evolve_hypothesis": "evolving",
            "fix_code": "fixing",
        }
        state["status"] = status_map.get(critique.decision, "running")

        print(f"[AnalysisAgent] Decision: {critique.decision} | Success: {critique.is_success}")
        return state

    # -- decision router for LangGraph conditional edges --------------------

    @staticmethod
    def decision_router(state: GraphState) -> str:
        """
        Returns the routing key for LangGraph conditional edges.

        Possible values: ``"evolve_hypothesis"``, ``"fix_code"``, ``"end"``.
        """
        return state.get("decision", "end")

    # -- critique generation -------------------------------------------------

    def _generate_critique(
        self,
        result: BacktestResult,
        state: GraphState,
    ) -> Critique:
        """Try LLM first; fall back to rule-based mock."""
        if os.getenv("OPENAI_API_KEY"):
            try:
                return self._llm_critique(result, state)
            except Exception as exc:
                print(f"[AnalysisAgent] LLM call failed ({exc}), using rule-based critique.")

        return self._rule_based_critique(result, state)

    def _llm_critique(self, result: BacktestResult, state: GraphState) -> Critique:
        """Generate critique via LLM."""
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatOpenAI(model=self.model_name, temperature=0.3)

        user_msg = (
            f"Backtest Results:\n"
            f"- Sharpe Ratio: {result.sharpe_ratio:.4f}\n"
            f"- Max Drawdown: {result.max_drawdown:.4f}\n"
            f"- Annualised Return: {result.annualized_return:.4f}\n"
            f"- Volatility: {result.volatility:.4f}\n"
            f"- Trades: {result.trades_count}\n"
            f"- WFO Score: {result.wfo_score:.4f}\n"
            f"- Logs: {'; '.join(result.logs[-5:])}\n\n"
            f"Hypothesis: {state.get('current_hypothesis', 'N/A')}\n"
        )

        response = llm.invoke([
            SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])

        return Critique.model_validate_json(response.content)

    def _rule_based_critique(
        self,
        result: BacktestResult,
        state: GraphState,
    ) -> Critique:
        """Deterministic rule-based critique for scaffold testing."""
        suggestions = []
        biases = []

        # Check for backtest errors
        has_error = any("[ERROR]" in log for log in result.logs)
        status = state.get("status", "")

        if has_error or status == "backtest_error":
            return Critique(
                is_success=False,
                decision="fix_code",
                suggestions=[
                    "The backtest encountered a runtime error.",
                    "Check the signal_generator function for bugs.",
                    "Ensure all required columns (open, high, low, close, volume) are accessed correctly.",
                ],
                potential_biases=["Cannot assess biases — code failed to execute."],
            )

        if result.trades_count == 0:
            return Critique(
                is_success=False,
                decision="fix_code",
                suggestions=[
                    "No trades were generated — the entry/exit conditions may be too restrictive.",
                    "Relax the RSI / SMA thresholds.",
                    "Check for NaN values in the signal column due to insufficient lookback period.",
                ],
                potential_biases=["No trades executed — cannot evaluate."],
            )

        # Assess metrics
        is_success = (
            result.sharpe_ratio > self.settings.min_sharpe_ratio
            and result.max_drawdown > self.settings.max_drawdown_limit  # MDD is negative
            and result.wfo_score > 1.0
        )

        if result.sharpe_ratio < 0.5:
            suggestions.append("Sharpe ratio is very low. Consider a fundamentally different strategy.")
        elif result.sharpe_ratio < self.settings.min_sharpe_ratio:
            suggestions.append(
                f"Sharpe ({result.sharpe_ratio:.2f}) below target ({self.settings.min_sharpe_ratio}). "
                "Try adding a volatility filter or stop-loss."
            )

        if result.max_drawdown < self.settings.max_drawdown_limit:
            suggestions.append(
                f"Max drawdown ({result.max_drawdown:.1%}) exceeds limit "
                f"({self.settings.max_drawdown_limit:.0%}). Add position sizing or risk parity."
            )

        if result.volatility > 0.30:
            suggestions.append("Excessive volatility. Consider hedging with a market-neutral overlay.")

        if result.wfo_score < 0.5:
            suggestions.append(
                "Low WFO score suggests overfitting to in-sample data. "
                "Simplify the factor model or use a longer lookback."
            )
            biases.append("Possible overfitting — WFO instability detected.")

        if not suggestions:
            suggestions.append("Strategy meets all performance targets.")

        biases.extend([
            "Survivorship bias — NASDAQ-100 reconstitutions not accounted for.",
            "Transaction costs modelled at flat rate; real costs vary by market cap.",
        ])

        return Critique(
            is_success=is_success,
            decision="end" if is_success else "evolve_hypothesis",
            suggestions=suggestions,
            potential_biases=biases,
        )
