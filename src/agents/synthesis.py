"""
Project AutoQuant — SynthesisAgent
====================================
Generates trading hypotheses and factor formulas.
Uses an LLM with the full critique history as context so each iteration
proposes a more refined hypothesis.
"""

import json
import os
from typing import Optional

from src.agents.base import BaseAgent, GraphState, Hypothesis


# ---------------------------------------------------------------------------
# System prompt for hypothesis generation
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM_PROMPT = """\
You are a senior quantitative researcher.  Your job is to propose a NEW
trading hypothesis for US equities (NASDAQ-100 universe, daily frequency).

A hypothesis consists of:
- title: short descriptive name
- rationale: economic reasoning (2–3 sentences)
- factors: list of technical/fundamental factors (e.g. RSI, EPS_growth)
- formula_logic: plain-English pseudocode for entry/exit rules

Rules:
1. The hypothesis MUST be different from any previously rejected hypotheses.
2. If prior critiques exist, address every suggestion explicitly.
3. Prefer factors that are uncorrelated with broad market beta.
4. Target Sharpe > 1.5, Max Drawdown < 20%.

Respond ONLY with valid JSON matching this schema:
{
  "title": "...",
  "rationale": "...",
  "target_asset_class": "NASDAQ 100",
  "frequency": "Daily",
  "factors": ["..."],
  "formula_logic": "..."
}
"""


# ---------------------------------------------------------------------------
# SynthesisAgent
# ---------------------------------------------------------------------------

class SynthesisAgent(BaseAgent):
    """
    Generates trading hypotheses and factor formulas.

    If an LLM API key is configured, uses langchain to call the model.
    Otherwise, falls back to a deterministic mock for scaffold testing.
    """

    def run(self, state: GraphState) -> GraphState:
        iteration = state.get("iteration_count", 0)
        critique_history = state.get("critique_history", [])

        print(f"[SynthesisAgent] Iteration {iteration + 1}: generating hypothesis...")

        hypothesis = self._generate_hypothesis(critique_history, iteration)

        # Serialise outputs into state
        state["current_hypothesis"] = hypothesis.model_dump_json()

        # Append to history
        history = list(state.get("hypothesis_history", []))
        history.append(hypothesis.model_dump_json())
        state["hypothesis_history"] = history

        state["status"] = "hypothesis_ready"
        print(f"[SynthesisAgent] Proposed: '{hypothesis.title}'")
        return state

    # -- LLM / mock ---------------------------------------------------------

    def _generate_hypothesis(
        self,
        critique_history: list[str],
        iteration: int,
    ) -> Hypothesis:
        """Try LLM first; fall back to mock."""
        if os.getenv("OPENAI_API_KEY"):
            try:
                return self._llm_generate(critique_history)
            except Exception as exc:
                print(f"[SynthesisAgent] LLM call failed ({exc}), using mock.")

        return self._mock_generate(critique_history, iteration)

    def _llm_generate(self, critique_history: list[str]) -> Hypothesis:
        """Generate hypothesis via OpenAI / langchain."""
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatOpenAI(model=self.model_name, temperature=0.7)

        user_content = "Generate a new trading hypothesis."
        if critique_history:
            user_content += "\n\nPrior critiques to address:\n"
            for c in critique_history[-3:]:  # last 3 critiques
                user_content += f"- {c}\n"

        response = llm.invoke([
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])

        return Hypothesis.model_validate_json(response.content)

    @staticmethod
    def _mock_generate(critique_history: list[str], iteration: int) -> Hypothesis:
        """Deterministic mock for testing without an LLM key."""
        strategies = [
            Hypothesis(
                title="RSI Mean Reversion on Oversold Tech",
                rationale="Stocks in NASDAQ 100 often revert to the mean after extreme dips below RSI 30, especially when the long-term trend (SMA-200) is intact.",
                factors=["RSI_14", "SMA_200"],
                formula_logic="Buy when RSI < 30 AND Price > SMA_200. Sell when RSI > 70.",
            ),
            Hypothesis(
                title="Momentum Breakout with Volume Confirmation",
                rationale="Stocks breaking above 52-week highs on above-average volume tend to sustain upward momentum for 20–60 days.",
                factors=["52W_HIGH", "VOLUME_RATIO", "ATR"],
                formula_logic="Buy when Close > 52W_High AND Volume > 1.5x 20d_avg_vol. Exit on trailing ATR stop (2x ATR).",
            ),
            Hypothesis(
                title="Earnings Surprise Drift",
                rationale="Post-earnings announcement drift (PEAD) shows stocks with positive EPS surprises outperform for 60 days.",
                factors=["EPS_SURPRISE", "REVENUE_GROWTH", "RSI_14"],
                formula_logic="Buy on positive EPS surprise > 5%. Hold for 60 days or sell if RSI > 80.",
            ),
            Hypothesis(
                title="Low Volatility Quality Factor",
                rationale="Low-volatility, high-quality stocks (high ROE, low debt) deliver superior risk-adjusted returns over time.",
                factors=["VOLATILITY_60D", "ROE", "DEBT_TO_EQUITY", "FREE_CASH_FLOW_YIELD"],
                formula_logic="Rank universe by composite score: 0.3*low_vol_rank + 0.3*high_roe_rank + 0.2*low_debt_rank + 0.2*high_fcf_yield_rank. Go long top decile, rebalance monthly.",
            ),
            Hypothesis(
                title="MACD Divergence with Bollinger Squeeze",
                rationale="MACD bullish divergence during a Bollinger Band squeeze often precedes explosive breakouts.",
                factors=["MACD", "MACD_SIGNAL", "BB_WIDTH", "RSI_14"],
                formula_logic="Buy when MACD crosses above signal AND BB_Width < 0.04 (squeeze). Sell when MACD crosses below signal OR RSI > 75.",
            ),
        ]
        return strategies[iteration % len(strategies)]
