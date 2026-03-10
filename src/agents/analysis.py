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
You are a senior quant portfolio manager reviewing a backtest. Your goal is to guide the strategy toward Sharpe > 1.5, Max DD > -15%, and WFO > 1.0.

Given the performance metrics and strategy details, produce a structured critique as JSON:
{
  "is_success": true/false,
  "decision": "evolve_hypothesis" | "fix_code" | "end",
  "suggestions": ["..."],
  "potential_biases": ["..."]
}

Decision rules:
- "end" if Sharpe >= 1.5 AND MaxDD >= -15% AND WFO >= 1.0
- "fix_code" if the backtest errored, produced 0 trades, or has a TypeError/ValueError
- "evolve_hypothesis" otherwise

CRITICAL IMPROVEMENT SUGGESTIONS — be specific about which proven techniques to add:

If Sharpe < 1.0:
- Suggest adding SMA-200 trend filter (reduces exposure in bear markets)
- Suggest volatility normalization (scale signals by inverse realized vol)
- Suggest switching to a momentum+drawdown control approach

If 1.0 <= Sharpe < 1.5:
- Suggest adding drawdown control (exit on 20-day rolling DD > -5%)
- Suggest combining 2-3 factors (momentum + mean-reversion + low-vol)
- Suggest tightening exits with Chandelier stop or ATR trailing stop

If Max Drawdown < -20%:
- Suggest regime filtering (only long when VIX < 25 or close > SMA-200)
- Suggest adding a volatility regime filter (exit in high-vol regimes)

If few trades (<10):
- Suggest loosening entry thresholds
- Suggest using forward-fill signal pattern to maintain positions

If previous code failed with TypeError (must be real number, not Series):
- EXPLICITLY instruct: replace ALL if/else statements with np.where()
- EXPLICITLY instruct: never use if df['x'] > y syntax

Do NOT use multi-line strings inside the JSON. All strings must be properly escaped.
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

        # Track best strategy across all iterations
        current_sharpe = result.sharpe_ratio
        best_sharpe = state.get("best_sharpe", float("-inf"))
        if current_sharpe > best_sharpe:
            state["best_sharpe"] = current_sharpe
            state["best_backtest_results"] = state.get("backtest_results", "")
            state["best_hypothesis"] = state.get("current_hypothesis", "")
            state["best_factor_code"] = state.get("factor_code", "")
            print(f"[AnalysisAgent] New best strategy! Sharpe={current_sharpe:.4f}")

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
                    "Fix any syntax or logic errors without changing the core hypothesis.",
                    "CRITICAL: Replace ALL if/else statements with np.where() or boolean masks.",
                    "Ensure entry conditions are not too restrictive (loosen thresholds if needed).",
                ],
                potential_biases=[]
            )

        # 2. Short-Circuit: Excellent Strategy — meets all targets
        if result.sharpe_ratio >= 1.5 and result.max_drawdown >= -0.15 and result.wfo_score >= 1.0:
            return Critique(
                is_success=True,
                decision="end",
                suggestions=["Strategy meets all performance targets (Sharpe >= 1.5, MDD >= -15%, WFO >= 1.0). Pipeline complete."],
                potential_biases=[]
            )

        # 3. Smart critique with specific improvement suggestions
        suggestions = []
        biases = []

        # Analyze Sharpe ratio
        if result.sharpe_ratio < 0.5:
            suggestions.extend([
                "Sharpe is very low. SWITCH to Layered Risk Momentum: 63-day momentum + TRIPLE risk gates (20-day DD circuit-breaker at -3.5%, SMA-200 trend gate, vol regime filter).",
                "Alternative: Risk-Parity Composite blending 4 z-scored signals (momentum 30%, inverted RSI 25%, Bollinger %B 25%, OBV divergence 20%) weighted by inverse vol, with DD circuit-breaker at -4%.",
                "CRITICAL: Add 20-day rolling drawdown control — exit immediately when DD exceeds -3.5% from recent peak.",
            ])
        elif result.sharpe_ratio < 1.0:
            suggestions.extend([
                f"Sharpe={result.sharpe_ratio:.2f} is moderate. Add LAYERED risk gates: DD circuit-breaker (-3.5%), SMA-200 trend filter, AND vol regime filter (20d vol < 80th pctile of 252d window).",
                "Add volatility regime filter: only trade when 20-day vol < 80th percentile of its 252-day window.",
                "Add 20-day rolling drawdown control: force exit when DD from peak exceeds -3.5%. This alone typically adds 0.5+ to Sharpe.",
            ])
        elif result.sharpe_ratio < 1.5:
            suggestions.extend([
                f"Sharpe={result.sharpe_ratio:.2f} is close to target. Add DD circuit-breaker (exit on 20-day DD > -3.5%) for the final Sharpe boost.",
                "Consider normalizing the signal by inverse realized volatility for vol-targeting.",
                "Tighten exits with ATR trailing stop (highest high - 3*ATR) to preserve profits.",
            ])

        # Analyze drawdowns
        if result.max_drawdown < -0.20:
            suggestions.extend([
                f"Max drawdown={result.max_drawdown:.1%} is too deep. Add TRIPLE risk gates: 20-day DD circuit-breaker at -3.5%, SMA-200 trend gate, vol regime filter.",
                "Add 20-day rolling drawdown control: exit ALL positions if DD > -3.5% from recent peak. This is the #1 DD reducer.",
            ])
            biases.append("Potential survivorship bias or lack of risk management in the strategy.")

        # Analyze trade count
        if result.trades_count < 10:
            suggestions.append(
                f"Only {result.trades_count} trades generated. Loosen entry thresholds or use forward-fill signal pattern."
            )

        # Analyze WFO stability
        if result.wfo_score < 1.0:
            suggestions.append(
                f"WFO score={result.wfo_score:.2f} indicates instability across time windows. Use a more robust multi-factor approach."
            )
            biases.append("Strategy may be overfit to a specific market regime.")

        # Default: call LLM for additional nuance
        if not suggestions:
            suggestions.append("The strategy needs improvement. Try a composite multi-factor approach with regime filtering.")

        # Determine decision
        if result.sharpe_ratio >= 1.5 and result.max_drawdown >= -0.15:
            # Close to passing — WFO might be the only issue
            decision = "evolve_hypothesis"
        else:
            decision = "evolve_hypothesis"

        # Optionally call LLM for richer critique
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            llm = self.get_llm(llm_provider, temperature=0.3)
            user_msg = (
                f"Backtest Results:\n"
                f"- Sharpe Ratio: {result.sharpe_ratio:.4f}\n"
                f"- Max Drawdown: {result.max_drawdown:.4f}\n"
                f"- Annualised Return: {result.annualized_return:.4f}\n"
                f"- Volatility: {result.volatility:.4f}\n"
                f"- Trades: {result.trades_count}\n"
                f"- WFO Score: {result.wfo_score:.4f}\n\n"
                f"Hypothesis: {state.get('current_hypothesis', 'N/A')}\n"
            )
            response = llm.invoke([
                SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ])
            cleaned_json = self.clean_llm_output(response.content)
            llm_critique = Critique.model_validate_json(cleaned_json)
            # Merge LLM suggestions with our smart defaults
            suggestions.extend(llm_critique.suggestions)
            biases.extend(llm_critique.potential_biases)
        except Exception:
            pass  # Smart defaults are sufficient

        return Critique(
            is_success=False,
            decision=decision,
            suggestions=suggestions[:8],  # Cap at 8 suggestions
            potential_biases=biases[:4],
        )
