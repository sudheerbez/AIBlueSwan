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
You are an expert Python quant developer.  Given a trading hypothesis,
write a SINGLE function called ``signal_generator`` that:

1. Accepts a pandas DataFrame ``df`` with columns: open, high, low, close, volume.
2. Adds a ``signal`` column where:
   - +1 = go long / buy
   - -1 = go short / exit long
   -  0 = no action
3. Returns the modified DataFrame.

Rules:
- Use ONLY pandas, numpy, and basic math.  No external API calls.
- Include ALL helper functions inline (e.g. ``calculate_rsi``).
- No I/O operations, no print statements, no imports of disallowed modules.
- The function must be self-contained and deterministic.

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

        print(f"[ImplementationAgent] Coding hypothesis: '{hypothesis.title}'...")

        factor_code = self._generate_code(hypothesis)

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

    def _generate_code(self, hypothesis: Hypothesis) -> FactorCode:
        """Try LLM first; fall back to mock."""
        if os.getenv("OPENAI_API_KEY"):
            try:
                return self._llm_generate(hypothesis)
            except Exception as exc:
                print(f"[ImplementationAgent] LLM call failed ({exc}), using mock.")

        return self._mock_generate(hypothesis)

    def _llm_generate(self, hypothesis: Hypothesis) -> FactorCode:
        """Generate code via LLM."""
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatOpenAI(model=self.model_name, temperature=0.0)

        user_msg = (
            f"Hypothesis: {hypothesis.title}\n"
            f"Factors: {', '.join(hypothesis.factors)}\n"
            f"Logic: {hypothesis.formula_logic}"
        )

        response = llm.invoke([
            SystemMessage(content=IMPLEMENTATION_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])

        return FactorCode(
            hypothesis_id=hypothesis.title,
            python_code=response.content.strip(),
            required_libraries=["pandas", "numpy"],
        )

    @staticmethod
    def _mock_generate(hypothesis: Hypothesis) -> FactorCode:
        """Generate representative mock code matching the hypothesis logic."""
        code = f'''\
def calculate_rsi(series, period=14):
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def signal_generator(df):
    """
    Strategy: {hypothesis.title}
    Logic:    {hypothesis.formula_logic}
    """
    df = df.copy()

    # Calculate factors
    df["rsi"] = calculate_rsi(df["close"], 14)
    df["sma_200"] = df["close"].rolling(window=200).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["volatility"] = df["close"].pct_change().rolling(20).std()

    # Generate signals
    df["signal"] = 0

    # Buy conditions
    buy_cond = (df["rsi"] < 30) & (df["close"] > df["sma_200"])
    df.loc[buy_cond, "signal"] = 1

    # Sell conditions
    sell_cond = (df["rsi"] > 70) | (df["close"] < df["sma_50"])
    df.loc[sell_cond, "signal"] = -1

    # Forward-fill positions (hold until exit signal)
    df["signal"] = df["signal"].replace(0, np.nan).ffill().fillna(0).astype(int)

    return df
'''
        return FactorCode(
            hypothesis_id=hypothesis.title,
            python_code=code,
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
