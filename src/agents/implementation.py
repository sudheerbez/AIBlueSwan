"""
Project AutoQuant — ImplementationAgent
=========================================
Translates a Hypothesis into executable Python factor code.
Validates the generated code via SafeCodeExecutor before returning.

Uses proven strategy templates as code examples to guide LLM code
generation toward robust, high-Sharpe implementations.
"""

import json
import os

from src.agents.base import BaseAgent, GraphState, Hypothesis, FactorCode


# ---------------------------------------------------------------------------
# System prompt for code generation — includes proven code patterns
# ---------------------------------------------------------------------------

IMPLEMENTATION_SYSTEM_PROMPT = """\
You are an elite Python quantitative developer. Given a trading hypothesis, you must translate it into flawless, vectorized pandas/numpy code that achieves Sharpe > 1.5.

Write a SINGLE function called ``signal_generator`` that:
1. Accepts a pandas DataFrame ``df`` with columns: open, high, low, close, volume.
2. Constructs the factors required by the hypothesis.
3. Adds a ``signal`` column where: +1 = long, -1 = exit/short, 0 = flat.
4. Returns the modified DataFrame with the ``signal`` column.

## MANDATORY RULES (violation = rejection)

1. **Vectorized only**: Use np.where(), .rolling(), .ewm(), boolean masks. NEVER use for/while loops, iterrows, or apply with lambda.
2. **No if/else on Series**: NEVER write `if df['x'] > y:`. ALWAYS use `np.where(condition, x, y)` or `df.loc[cond, col] = val`.
3. **No imports**: `pd`, `np`, `scipy`, `ta` are pre-injected. Do NOT write any import statements.
4. **Avoid lookahead**: ALWAYS add `df['signal'] = df['signal'].shift(1).fillna(0).astype(int)` as the LAST line before return.
5. **Handle NaN**: Use .fillna(0), .ffill(), or min_periods in rolling operations.
6. **Generate trades**: Ensure the strategy produces signals. If using a forward-fill pattern, ALWAYS seed initial buy/sell conditions that actually trigger.
7. **Use df.copy()**: Start with `df = df.copy()` to avoid SettingWithCopyWarning.

## PROVEN CODE PATTERNS (use these as building blocks)

### Pattern A: Momentum + Trend Filter (Sharpe ~1.6-2.0)
```
df['ret_252'] = df['close'].pct_change(252)
df['sma200'] = df['close'].rolling(200, min_periods=60).mean()
df['daily_ret'] = df['close'].pct_change()
df['rvol'] = df['daily_ret'].rolling(60, min_periods=20).std() * np.sqrt(252)
df['vol_mom'] = df['ret_252'] / df['rvol'].replace(0, 1e-10)
uptrend = df['close'] > df['sma200']
df['signal'] = np.where((df['vol_mom'] > 0) & uptrend, 1, -1)
```

### Pattern B: Multi-Factor Composite (Sharpe ~1.7-2.5)
```
# Z-score each factor over 252-day rolling window
mom = df['close'].pct_change(63)
z_mom = (mom - mom.rolling(252, min_periods=60).mean()) / mom.rolling(252, min_periods=60).std().replace(0, 1e-10)
# RSI mean-reversion factor
delta = df['close'].diff()
gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
rsi = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))
z_mr = ((50 - rsi) / 50)  # positive when oversold
composite = (z_mom + z_mr) / 2.0
df['signal'] = np.where(composite > 0.5, 1, np.where(composite < -0.5, -1, 0))
```

### Pattern C: Drawdown Control (critical for Sharpe)
```
df['roll_max'] = df['close'].rolling(20, min_periods=5).max()
df['roll_dd'] = df['close'] / df['roll_max'] - 1.0
# Override: force exit when drawdown > 5%
df.loc[df['roll_dd'] <= -0.05, 'signal'] = -1
```

### Pattern D: RSI Calculation (correct vectorized form)
```
delta = df['close'].diff()
gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
rs = gain / loss.replace(0, 1e-10)
df['rsi'] = 100 - (100 / (1 + rs))
```

### Pattern E: ATR Calculation
```
tr = pd.DataFrame({
    'hl': df['high'] - df['low'],
    'hc': (df['high'] - df['close'].shift(1)).abs(),
    'lc': (df['low'] - df['close'].shift(1)).abs()
}).max(axis=1)
df['atr'] = tr.rolling(14, min_periods=5).mean()
```

### Pattern F: Forward-fill signals (for hold-until-exit strategies)
```
df['signal'] = df['signal'].replace(0, np.nan).ffill().fillna(0).astype(int)
```

Respond ONLY with the raw Python code (no markdown fences, no explanations).
"""


# ---------------------------------------------------------------------------
# ImplementationAgent
# ---------------------------------------------------------------------------

class ImplementationAgent(BaseAgent):
    """
    Writes and validates Python factor code for a given Hypothesis.
    Uses proven code patterns and strategy templates as reference.
    """

    def run(self, state: GraphState) -> GraphState:
        hypothesis_json = state.get("current_hypothesis", "{}")
        hypothesis = Hypothesis.model_validate_json(hypothesis_json)
        llm_provider = state.get("llm_provider", "OpenAI")

        print(f"[ImplementationAgent] Coding hypothesis: '{hypothesis.title}' using {llm_provider}...")

        factor_code = self._generate_code(hypothesis, llm_provider)

        # Validate via sandbox
        factor_code = self._validate_code(factor_code)

        # If validation failed, try matching a proven template
        if not factor_code.is_valid:
            print("[ImplementationAgent] LLM code failed. Trying proven template match...")
            template_code = self._match_template(hypothesis)
            if template_code:
                factor_code.python_code = template_code
                factor_code = self._validate_code(factor_code)

        # If still invalid (e.g. template produced 0 trades), cycle through others
        if not factor_code.is_valid:
            from src.strategies.templates import STRATEGY_TEMPLATES
            print("[ImplementationAgent] Template match also failed. Cycling through templates...")
            for tmpl in STRATEGY_TEMPLATES:
                factor_code.python_code = tmpl["code"]
                factor_code = self._validate_code(factor_code)
                if factor_code.is_valid:
                    print(f"[ImplementationAgent] Found working template: {tmpl['name']}")
                    break

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
            f"Logic: {hypothesis.formula_logic}\n\n"
            f"Generate a signal_generator function implementing this hypothesis. "
            f"Use the proven code patterns from the system prompt. "
            f"Target Sharpe > 1.5. MUST include shift(1) as the last step."
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

    # -- template matching ---------------------------------------------------

    @staticmethod
    def _match_template(hypothesis: Hypothesis) -> str | None:
        """Try to find a matching proven strategy template for the hypothesis."""
        from src.strategies.templates import STRATEGY_TEMPLATES

        title_lower = hypothesis.title.lower()
        factors_lower = " ".join(hypothesis.factors).lower()
        logic_lower = hypothesis.formula_logic.lower()
        combined = f"{title_lower} {factors_lower} {logic_lower}"

        # Keyword matching to find the best template
        best_match = None
        best_score = 0

        keywords_map = {
            "layered_risk_momentum": ["layered", "triple protection", "circuit.*breaker", "risk momentum", "gate"],
            "risk_parity_composite": ["risk parity", "risk.*parity", "multi.*factor.*risk", "circuit breaker", "orthogonal"],
            "vol_targeting_trend": ["vol.*target", "volatility.*target", "adaptive exposure", "constant.*risk"],
            "adaptive_regime_ensemble": ["regime.*ensemble", "momentum.*reversion.*blend", "adx.*regime", "ensemble"],
            "dual_momentum": ["dual", "absolute momentum", "relative momentum", "12-month", "12m"],
            "keltner_mean_reversion": ["keltner", "atr squeeze", "mean reversion", "channel"],
            "adaptive_rsi_regime": ["adaptive", "regime", "adx", "switching"],
            "zscore_mean_reversion": ["z-score", "zscore", "hurst", "ornstein"],
            "multi_tf_trend": ["multi-timeframe", "triple ema", "trend alignment", "ema10.*ema50"],
            "kama_breakout": ["kama", "kaufman", "adaptive ma", "efficiency ratio"],
            "volume_price_trend": ["volume", "obv", "volume-weighted", "volume price"],
            "momentum_dd_control": ["drawdown control", "momentum.*drawdown", "dd control", "rolling drawdown"],
            "bb_squeeze_breakout": ["bollinger", "squeeze", "bb_width", "bandwidth"],
            "vol_adj_momentum": ["volatility-normalized", "vol.*momentum", "tsmom", "time series momentum"],
            "composite_multi_factor": ["composite", "multi-factor", "multi factor", "z.*score.*factor"],
            "chandelier_trend": ["chandelier", "donchian", "trailing stop", "trend following"],
        }

        for tmpl in STRATEGY_TEMPLATES:
            keywords = keywords_map.get(tmpl["name"], [])
            score = sum(1 for kw in keywords if kw in combined)
            if score > best_score:
                best_score = score
                best_match = tmpl

        if best_match and best_score > 0:
            print(f"[ImplementationAgent] Matched template: {best_match['name']}")
            return best_match["code"]

        # Default: cycle through templates to avoid repeating the same one
        attempt = getattr(ImplementationAgent, '_fallback_idx', 0)
        ImplementationAgent._fallback_idx = (attempt + 1) % len(STRATEGY_TEMPLATES)
        chosen = STRATEGY_TEMPLATES[attempt]
        print(f"[ImplementationAgent] Default fallback template: {chosen['name']}")
        return chosen["code"]

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
            # Check that the code actually generates trade signals
            try:
                result_df = exec_result.return_value
                if result_df is not None and hasattr(result_df, 'columns') and "signal" in result_df.columns:
                    signals = result_df["signal"]
                    n_trades = int((signals.diff().abs() > 0).sum())
                    unique_vals = set(int(v) for v in signals.dropna().unique())
                    has_longs = 1 in unique_vals
                    if n_trades < 2 or not has_longs:
                        factor_code.is_valid = False
                        factor_code.validation_error = (
                            f"Code produced only {n_trades} signal changes and "
                            f"unique signals={unique_vals}. Must generate trades with +1 entries."
                        )
                        return factor_code
                factor_code.is_valid = True
                factor_code.validation_error = None
            except Exception:
                factor_code.is_valid = True
                factor_code.validation_error = None
        else:
            factor_code.is_valid = False
            factor_code.validation_error = (
                f"{exec_result.error_type}: {exec_result.error_message}"
            )

        return factor_code
