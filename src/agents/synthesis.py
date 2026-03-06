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
You are an elite quantitative researcher at a top-tier hedge fund. Your objective is to discover highly innovative, unconventional, and extremely profitable trading strategies for US Equities (daily frequency).

You must think completely outside the box to maximize returns and Alpha. Go beyond basic indicators like RSI or SMA. Consider advanced statistical arbitrage, volatility clustering, regime-switching models, momentum variations, or synthetic alternative data proxies.

A hypothesis consists of:
- title: short descriptive name highlighting the advanced nature
- rationale: advanced economic or statistical reasoning (2–3 sentences)
- factors: list of technical/statistical factors (e.g. Hurst_Exponent, Rolling_ZScore)
- formula_logic: highly specific pseudocode for entry/exit/position-sizing

Rules:
1. The hypothesis MUST be radically different from previously rejected ones.
2. Address prior critiques to evolve the strategy into something sharper and more robust.
3. Target Maximum Returns and Sharpe > 2.0, Max Drawdown < 15%.
4. Be mathematically creative and rigorous.

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
        llm_provider = state.get("llm_provider", "OpenAI")

        print(f"[SynthesisAgent] Iteration {iteration + 1}: generating hypothesis using {llm_provider}...")

        hypothesis = self._generate_hypothesis(critique_history, iteration, llm_provider)

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
        llm_provider: str,
    ) -> Hypothesis:
        """Generate hypothesis via LLM exclusively."""
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = self.get_llm(llm_provider, temperature=0.7)

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
