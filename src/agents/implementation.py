"""
Project AutoQuant — ImplementationAgent
=========================================
Translates a Hypothesis into executable Python factor code.
Validates the generated code via SafeCodeExecutor before returning.
"""

import json
import os

from src.agents.base import BaseAgent, GraphState, Hypothesis, FactorCode


# ---------------------------------------------------------------------------
# System prompt for code generation
# ---------------------------------------------------------------------------

IMPLEMENTATION_SYSTEM_PROMPT = """\
You are an elite Python quantitative developer. Given a highly advanced trading hypothesis, you must translate it into flawless, vectorized pandas/numpy code.

Write a SINGLE function called ``signal_generator`` that:
1. Accepts a pandas DataFrame ``df`` with columns: open, high, low, close, volume.
2. Constructs the complex statistical factors required by the hypothesis.
3. Adds a ``signal`` column where:
   - +1 = go long / buy
   - -1 = go short / exit long
   -  0 = no action (flat)
4. Returns the modified DataFrame with the ``signal`` column correctly shifted to avoid lookahead bias.

Rules:
- Use advanced vectorized pandas and numpy operations. Avoid slow iterrows loop at all costs.
- Include ALL helper functions inline (e.g. ``calculate_hurst``, ``rolling_zscore``).
- Handle NaN values gracefully (e.g., using .ffill().fillna(0)).
- CRITICAL PANDAS RULE: Never use `if/elif/else` with a pandas Series (e.g. `if df['close'] > x:`). ALWAYS use `np.where(condition, x, y)` or boolean masking (`df.loc[condition, 'signal'] = 1`). Never slice indexes incorrectly (no `df[-1]`).
- CRITICAL: DO NOT WRITE ANY `import` STATEMENTS. The following modules are already pre-injected into your global namespace: `pd` (pandas), `np` (numpy), `scipy`, and `ta`. Use them directly.
- Ensure your strategy ALWAYS generates trades. Do not use overly restrictive threshold conditions that are never met. If in doubt, fallback to a simple moving average crossover alongside your complex logic to guarantee a non-zero signal count.
- The function must be completely self-contained, deterministic, and highly optimized for backtesting.

EXAMPLE OUTPUT:
def signal_generator(df):
    df['sma10'] = df['close'].rolling(10).mean()
    df['sma30'] = df['close'].rolling(30).mean()
    df['signal'] = np.where(df['sma10'] > df['sma30'], 1, -1)
    df['signal'] = df['signal'].shift(1).fillna(0)
    return df

Respond ONLY with the raw Python code (no markdown fences).
"""


# ---------------------------------------------------------------------------
# ImplementationAgent
# ---------------------------------------------------------------------------

class ImplementationAgent(BaseAgent):
    """
    Writes and validates Python factor code for a given Hypothesis.
    """

    def run(self, state: GraphState) -> GraphState:
        hypothesis_json = state.get("current_hypothesis", "{}")
        hypothesis = Hypothesis.model_validate_json(hypothesis_json)
        llm_provider = state.get("llm_provider", "OpenAI")

        print(f"[ImplementationAgent] Coding hypothesis: '{hypothesis.title}' using {llm_provider}...")

        factor_code = self._generate_code(hypothesis, llm_provider)

        # Validate via sandbox
        factor_code = self._validate_code(factor_code)

        state["factor_code"] = factor_code.model_dump_json()
        state["status"] = "code_ready" if factor_code.is_valid else "code_error"

        if factor_code.is_valid:
            print(f"[ImplementationAgent] Code validated successfully.")
        else:
            print(f"[ImplementationAgent] Code validation failed: {factor_code.validation_error}")

        return state

    # -- code generation -----------------------------------------------------

    def _generate_code(self, hypothesis: Hypothesis, llm_provider: str) -> FactorCode:
        """Generate code via LLM exclusively."""
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = self.get_llm(
            llm_provider, 
            temperature=0.0, 
            model_name=self.settings.coder_llm_model
        )

        user_msg = (
            f"Hypothesis: {hypothesis.title}\n"
            f"Factors: {', '.join(hypothesis.factors)}\n"
            f"Logic: {hypothesis.formula_logic}"
        )

        response = llm.invoke([
            SystemMessage(content=IMPLEMENTATION_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])

        cleaned_code = self.clean_llm_output(response.content)
        return FactorCode(
            hypothesis_id=hypothesis.title,
            python_code=cleaned_code,
            required_libraries=["pandas", "numpy"],
        )

    # -- code validation -----------------------------------------------------

    @staticmethod
    def _validate_code(factor_code: FactorCode) -> FactorCode:
        """Run the code in SafeCodeExecutor to check for syntax/runtime errors."""
        from src.utils.executor import SafeCodeExecutor
        import pandas as pd
        import numpy as np

        executor = SafeCodeExecutor(timeout_seconds=10)

        # Create a small synthetic DataFrame for validation
        n = 300
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        np.random.seed(42)
        prices = 100 + np.random.randn(n).cumsum()
        test_df = pd.DataFrame({
            "open": prices + np.random.randn(n) * 0.5,
            "high": prices + abs(np.random.randn(n)),
            "low": prices - abs(np.random.randn(n)),
            "close": prices,
            "volume": np.random.randint(100_000, 10_000_000, n),
        }, index=dates)

        # Build validation code (run the signal_generator on test data)
        validation_code = factor_code.python_code + "\nresult = signal_generator(test_df)"

        exec_result = executor.execute(
            validation_code,
            extra_globals={"test_df": test_df},
        )

        if exec_result.success:
            factor_code.is_valid = True
            factor_code.validation_error = None
        else:
            factor_code.is_valid = False
            factor_code.validation_error = (
                f"{exec_result.error_type}: {exec_result.error_message}"
            )

        return factor_code
