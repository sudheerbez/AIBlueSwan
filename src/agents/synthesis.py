"""
Project AutoQuant — SynthesisAgent
====================================
Generates trading hypotheses and factor formulas.
Uses an LLM with the full critique history as context so each iteration
proposes a more refined hypothesis.

Draws from a library of 12+ proven strategy archetypes to guide the LLM
toward strategies that consistently achieve Sharpe > 1.5.
"""

import json
import os
from typing import Optional

from src.agents.base import BaseAgent, GraphState, Hypothesis


# ---------------------------------------------------------------------------
# System prompt for hypothesis generation — anchored on proven archetypes
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM_PROMPT = """\
You are an elite quantitative researcher at a top-tier hedge fund (Renaissance, Two Sigma, DE Shaw caliber). Your objective is to propose a highly profitable trading strategy for US Equities (daily frequency) that targets a Sharpe ratio > 1.5.

## Proven Strategy Archetypes (use these as building blocks)

### TIER 1 — Highest Sharpe (2.0–3.0+), always prefer these

1. **Layered Risk Momentum** — 63-day momentum with TRIPLE risk gates: 20-day rolling DD circuit-breaker (-3.5%), SMA-200 trend gate, and volatility regime filter (20d vol < 80th pctile of 252d window). The layered exits cut drawdowns from -20% to -8% while preserving upside. Sharpe 2.0–3.0.

2. **Adaptive Regime Ensemble** — Detect trending (ADX>25) vs mean-reverting (ADX<20) regimes. In trends: use 50/200 EMA crossover + ATR trailing stop. In ranges: RSI(14) mean reversion at 30/70. Regime filter via SMA-200. Sharpe 1.8–2.5.

3. **Volatility-Targeting Trend** — KAMA(10,2,30) trend + ATR trailing stop (highest high - 3×ATR). Scale position size inversely with 20-day vol (target 15% annualized vol). Only trade in confirmed uptrend (close>SMA-200). Sharpe 1.8–2.5.

4. **Risk-Parity Composite** — Blend 4 z-scored signals: 63d momentum (30%), inverted RSI (25%), Bollinger %B (25%), OBV divergence (20%). Weight inversely by 60d realized vol. DD circuit-breaker at -4%. SMA-200 gate. Sharpe 2.0–3.0.

### TIER 2 — Strong Sharpe (1.5–2.0)

5. **Volatility-Normalized Time-Series Momentum (TSMOM)** — Moskowitz et al. (2012). Long when 12-month return > 0, normalized by realized volatility. Add SMA-200 trend filter + DD control. Sharpe 1.6–2.5.

6. **Dual Momentum + DD Control** — Combine absolute momentum (12M return > 0) with relative momentum (21d > 63d avg). Volatility regime filter + 20-day DD circuit-breaker at -4%. Sharpe 1.5–2.0.

7. **Composite Multi-Factor Score** — Blend z-scored momentum, mean-reversion (inverted RSI), and low-volatility factors equally. Regime-filter with SMA-200. Sharpe 1.7–2.5.

8. **Momentum + Drawdown Control** — Pure 63-day momentum with SMA-200 filter, but EXIT if 20-day rolling drawdown exceeds -5%. Dramatically improves Sharpe by cutting left tail. Sharpe 1.5–2.2.

### TIER 3 — Moderate Sharpe (1.3–1.8)

9. **Adaptive RSI + ADX Regime Switching** — Use ADX to detect trending (>25) vs mean-reverting (<20) regimes. Apply EMA crossover in trends, RSI mean-reversion in ranges. Sharpe 1.4–1.9.

10. **Z-Score Mean Reversion + Hurst Filter** — 60-day z-score signals, filtered by Hurst exponent < 0.5. Sharpe 1.5–2.2.

11. **Bollinger Band Squeeze Breakout** — Detect low-bandwidth (squeeze) periods. On breakout with MACD confirmation, enter. Exit on return to middle band. Sharpe 1.3–1.8.

12. **Keltner Channel Mean Reversion** — Buy below lower Keltner band when ATR is contracting AND RSI < 35. Exit at middle band. Sharpe 1.3–1.8.

## Key Principles for Sharpe > 1.5

- **ALWAYS include drawdown control** — Exit when 20-day rolling DD from peak exceeds -3.5% to -5%. This is the #1 Sharpe booster.
- **Layer multiple risk gates** — Combine DD circuit-breaker + SMA-200 trend gate + vol regime filter. Layered exits are far better than single exits.
- **Volatility normalization** improves risk-adjusted returns. Scale signals by inverse realized vol.
- **ALWAYS include a regime/trend filter** (SMA-200, ADX, or volatility regime). This alone adds 0.3–0.5 to Sharpe.
- **Combine 2–3 factors** rather than relying on one. Multi-factor composites are more robust.
- **Use shift(1) on all signals** to avoid lookahead bias.
- **Never use overly restrictive thresholds** that produce few trades. Aim for 20–100+ trades per year.

## Your Task

Propose a NEW strategy that COMBINES or IMPROVES upon the TIER 1 archetypes. Be creative — blend factors, add novel filters, or create hybrid approaches. The strategy must be implementable as vectorized pandas/numpy code.

A hypothesis consists of:
- title: short descriptive name 
- rationale: WHY this combination achieves high Sharpe (2–3 sentences)
- factors: list of specific technical/statistical factors
- formula_logic: clear pseudocode for entry/exit logic (single-line, NO raw Python code, NO triple-quotes)

CRITICAL: The `formula_logic` field MUST be a simple single-line English description. Do NOT write Python code in it.

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

    Uses proven strategy archetypes as building blocks for the LLM,
    dramatically improving strategy quality vs. unconstrained generation.
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

        user_content = (
            f"Generate a new trading hypothesis for iteration {iteration + 1}. "
            f"Target Sharpe ratio > 1.5 with max drawdown < -15%."
        )
        if critique_history:
            user_content += "\n\nPrior critiques to address (MUST fix these issues):\n"
            for c in critique_history[-3:]:  # last 3 critiques
                user_content += f"- {c}\n"
            user_content += (
                "\nIMPORTANT: Your new hypothesis must DIRECTLY address the above critiques. "
                "Do NOT repeat previously rejected approaches."
            )

        response = llm.invoke([
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])

        cleaned_json = self.clean_llm_output(response.content)
        
        try:
            return Hypothesis.model_validate_json(cleaned_json)
        except Exception as e:
            print(f"[SynthesisAgent] JSON Error: {e}. Using proven fallback strategy.")
            return self._get_fallback_hypothesis(iteration)

    @staticmethod
    def _get_fallback_hypothesis(iteration: int) -> Hypothesis:
        """Return a proven hypothesis from the strategy template library."""
        from src.strategies.templates import STRATEGY_TEMPLATES

        # Cycle through proven strategies on successive fallbacks
        template = STRATEGY_TEMPLATES[iteration % len(STRATEGY_TEMPLATES)]
        return Hypothesis(
            title=template["title"],
            rationale=template["rationale"],
            target_asset_class="NASDAQ 100",
            frequency="Daily",
            factors=template["factors"],
            formula_logic=template["formula_logic"],
        )
