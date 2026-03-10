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
- "end" if Sharpe > 2.0 AND MaxDD > -15% AND WFO score > 1.0
- "fix_code" if the backtest errored or produced 0 trades
- "evolve_hypothesis" otherwise

Be specific in suggestions: mention exact factors, timeframes, or
risk management improvements. Do NOT use multi-line strings like `\"\"\"` inside the JSON. All strings must be properly escaped.
If the previous run failed with a TypeError (e.g., must be real number, not Series), explicitly instruct the coder to replace normal `if/else` statements with vectorized operations (like `np.where`) in your suggestions.
"""


# ---------------------------------------------------------------------------
# AnalysisAgent
# ---------------------------------------------------------------------------

class AnalysisAgent(BaseAgent):
    """
    Evaluates backtest metrics and produces a Critique with a routing
    decision for the LangGraph conditional edge.
    """

    def __init__(self) -> None:
        super().__init__()
        self.settings = Settings()

    def run(self, state: GraphState) -> GraphState:
        results_json = state.get("backtest_results", "{}")
        result = BacktestResult.model_validate_json(results_json)
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", 5)
        llm_provider = state.get("llm_provider", "OpenAI")

        print(
            f"[AnalysisAgent] Evaluating | Sharpe={result.sharpe_ratio:.3f} | "
            f"MDD={result.max_drawdown:.2%} | WFO={result.wfo_score:.3f} | LLM={llm_provider}"
        )

        critique = self._generate_critique(result, state, llm_provider)

        # Force termination if max iterations reached
        if iteration + 1 >= max_iter and critique.decision != "end":
            critique.decision = "end"
            critique.suggestions.append(
                f"Max iterations ({max_iter}) reached - terminating pipeline."
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
        llm_provider: str,
    ) -> Critique:
        """Generate critique, selectively bypassing the LLM to save API quota."""
        
        # 1. Short-Circuit: Syntax Error or Crash (Save 1 LLM call)
        if result.trades_count == 0 or any("[ERROR]" in log or "[SKIP]" in log for log in result.logs):
            return Critique(
                is_success=False,
                decision="fix_code",
                suggestions=[
                    "The code failed to execute, hit a timeout, or produced 0 trades.",
                    f"Check validation logs: {'; '.join(result.logs[-3:])}",
                    "Fix any syntax or logic errors without changing the core hypothesis."
                ],
                potential_biases=[]
            )

        # 2. Short-Circuit: Perfect Strategy (Save 1 LLM call)
        if result.sharpe_ratio >= 1.5 and result.max_drawdown >= -0.15 and result.wfo_score >= 1.0:
            return Critique(
                is_success=True,
                decision="end",
                suggestions=["Strategy meets all performance targets. Pipeline complete."],
                potential_biases=[]
            )

        # 3. LLM Fallback: Strategy ran but needs evolution
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = self.get_llm(llm_provider, temperature=0.3)

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

        cleaned_json = self.clean_llm_output(response.content)
        try:
            return Critique.model_validate_json(cleaned_json)
        except Exception as e:
            print(f"[AnalysisAgent] JSON Error: {e}. Defaulting to evolve_hypothesis.")
            return Critique(
                is_success=False,
                decision="evolve_hypothesis",
                suggestions=[f"Recovered from invalid JSON critique. Keep trying."],
                potential_biases=["Formatting error in previous critique"]
            )
