# AI Blue Swan — Copilot Instructions

> Autonomous quantitative research engine: multi-agent LangGraph loop that proposes, codes, backtests, and refines US equity trading strategies.

---

## Architecture

```
main_orchestrator.py        ← LangGraph StateGraph (CLI entry point)
backend/app.py              ← FastAPI REST + WebSocket (real-time UI)
backend/runner.py           ← Async pipeline execution, event broadcasting
src/agents/                 ← 4 agents: Synthesis → Implementation → Validation → Analysis
src/backtest/               ← Event-driven portfolio simulator, metrics, WFO
src/data/                   ← yfinance (free) + Stooq (free fallback) + macro overlay (VIX/yields)
src/strategies/             ← 12 proven strategy templates (academically-backed)
src/utils/                  ← Config (Pydantic Settings), sandboxed code executor
frontend/                   ← React 19 + Vite + Tailwind + Recharts (WebSocket-driven)
```

**Agent loop**: Synthesis → Implementation → Validation → Analysis → (evolve_hypothesis | fix_code | end)

**State contract**: `GraphState` (TypedDict) in `src/agents/base.py`. Agents exchange Pydantic models serialized as JSON strings.

---

## Build & Run

### Python backend
```bash
pip install -r requirements.txt && pip install -r backend/requirements.txt
# CLI pipeline
python main_orchestrator.py --iterations 5 --capital 100000 --universe "NASDAQ-100"
# API server
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend
```bash
cd frontend && npm install
npm run dev       # http://localhost:5173
npm run build     # production build
npm run lint      # ESLint
```

### Environment variables
```bash
OLLAMA_BASE_URL="http://localhost:11434"
DEFAULT_LLM_MODEL="llama3.2"          # Synthesis + Analysis agents
CODER_LLM_MODEL="deepseek-coder"      # Implementation agent
USE_MOCK_LLM="1"                       # Offline mode (no Ollama needed)
FMP_API_KEY=""                          # Optional; yfinance is the default data source
AUTOQUANT_INITIAL_CAPITAL=100000.0
AUTOQUANT_MAX_ITERATIONS=5
```

No API keys required when using `USE_MOCK_LLM=1` + yfinance.

---

## Code Conventions

### Python
- **Type hints**: Always use them. Pydantic `BaseModel` for all data contracts; `TypedDict` for graph state.
- **Naming**: PascalCase classes; snake_case functions; agents end in `Agent`; private methods prefix `_`.
- **Async**: Use `asyncio.to_thread()` for blocking I/O (yfinance). Backend uses `graph.astream()` for streaming.
- **Imports**: stdlib → third-party → local. `from dotenv import load_dotenv` at top of entry points.
- **Sandboxing**: All LLM-generated code runs through `SafeCodeExecutor` — never use bare `exec()`.
- **Signal code**: Must be vectorized pandas/numpy. No Python loops over rows. Use `np.where()` not `if/else` with Series. Always `shift(1)` signals to avoid lookahead bias.

### Frontend (React + Vite + Tailwind)
- **Styling**: Tailwind CSS utility classes. `cn()` helper from `lib/utils.js` for conditional classes.
- **State**: React hooks (`useState`, `useEffect`). WebSocket subscription in `api.js`.
- **Components**: `frontend/src/components/` — each panel is self-contained (StrategyCard, MetricsPanel, EquityChart, ProgressFeed, CritiquePanel).
- **API**: Hard-coded `localhost:8000` base URLs in `api.js`.

---

## Key Domain Rules

1. **Walk-Forward Optimization**: WFO Score = mean(per-window Sharpes) − std(per-window Sharpes). A strategy passes when Sharpe ≥ 1.5, Max DD ≥ −15%, WFO ≥ 1.0.
2. **Sandboxed execution**: `SafeCodeExecutor` allows only `pandas`, `numpy`, `scipy`, `math`, `statistics`, `json`, `re`, `ta`, `datetime`. No `eval`, `exec`, `__import__`, `open`, `globals`.
3. **Signal contract**: Signal functions receive a DataFrame with columns `[open, high, low, close, volume]` and must return a Series of `{-1, 0, 1}`.
4. **Mock fallbacks**: When LLM fails to parse, `SynthesisAgent` returns a hardcoded "Fallback Momentum Strategy". When live data fails, synthetic OHLCV is generated.
5. **Backtest engine**: Bar-by-bar simulator. Applies commission (10 bps) + slippage (5 bps). Long-only (+1) or flat (−1/0).

---

## Pitfalls & Gotchas

- **LLM Series comparisons**: LLMs frequently generate `if signal > 0:` on a pandas Series — the ImplementationAgent prompt explicitly forbids this, but regeneration loops happen.
- **Timeout on Windows**: `SafeCodeExecutor` uses `signal.SIGALRM` (Unix only). No timeout enforcement on Windows.
- **Timezone stripping**: yfinance returns TZ-aware DatetimeIndex; data layer strips it. Don't re-introduce TZ-aware datetimes.
- **In-memory store**: `PipelineRun` objects in `backend/runner.py` are ephemeral — lost on restart. No database persistence.
- **Frontend WebSocket**: `onClose()` sets status to 'failed' even on graceful close — handle reconnection carefully.
- **CORS**: Backend allows all origins (`*`) — acceptable for local dev, not production.

---

## Testing

- No formal test suite. `test_executor.py` is a minimal smoke test for `SafeCodeExecutor`.
- To test offline: set `USE_MOCK_LLM=1` — all agents use `MockLLM` (deterministic, no network).
- Validate signal code with the synthetic 300-bar DataFrame in `ImplementationAgent._validate_code()`.

---

## File Reference

| File | Purpose |
|------|---------|
| `main_orchestrator.py` | LangGraph state machine: builds graph, invokes pipeline |
| `src/agents/base.py` | Pydantic schemas (`Hypothesis`, `FactorCode`, `BacktestResult`, `Critique`), `GraphState`, `BaseAgent` |
| `src/agents/synthesis.py` | LLM-driven hypothesis generation from critique history |
| `src/agents/implementation.py` | Code generation + sandbox validation |
| `src/agents/validation.py` | Backtest execution via WFO or simple engine |
| `src/agents/analysis.py` | Metric evaluation + decision routing (evolve/fix/end) |
| `src/backtest/engine.py` | Event-driven portfolio simulator |
| `src/backtest/metrics.py` | Sharpe, Sortino, Max DD, CAGR, Calmar, volatility |
| `src/backtest/wfo.py` | Walk-Forward Optimization with rolling windows |
| `src/data/yfinance_client.py` | Async yfinance wrapper (free, no API keys) |
| `src/data/stooq_client.py` | Free Stooq.com CSV data (fallback, no API keys) |
| `src/data/macro_client.py` | Free macro data (VIX, Treasury yields) for regime detection |
| `src/data/loader.py` | Unified data loader with yfinance → Stooq fallback + macro overlay |
| `src/strategies/templates.py` | 12 proven strategy templates with ready-to-run code |
| `src/utils/config.py` | Pydantic Settings, NASDAQ-100 tickers, constants |
| `src/utils/executor.py` | Sandboxed code execution with restricted builtins |
| `backend/app.py` | FastAPI endpoints + WebSocket |
| `backend/runner.py` | Async pipeline runner + event broadcasting |
| `frontend/src/App.jsx` | Main React app (config sidebar + dashboard panels) |
| `frontend/src/api.js` | REST + WebSocket client |
