"""
AutoQuant Backend — Pipeline Runner
=====================================
Wraps the LangGraph pipeline to run as a background task with
real-time progress events broadcast to WebSocket subscribers.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from src.agents.base import (
    GraphState, Hypothesis, FactorCode, BacktestResult, Critique,
)


# ---------------------------------------------------------------------------
# Pipeline Run Record
# ---------------------------------------------------------------------------

class PipelineRun:
    """In-memory record of a pipeline execution."""

    def __init__(
        self,
        run_id: str,
        capital: float,
        max_iterations: int,
        universe: str,
    ) -> None:
        self.run_id = run_id
        self.capital = capital
        self.max_iterations = max_iterations
        self.universe = universe
        self.status = "pending"           # pending | running | success | failed | cancelled
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.events: List[Dict[str, Any]] = []
        self.final_state: Optional[Dict[str, Any]] = None
        self._subscribers: List[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers = [s for s in self._subscribers if s is not q]

    async def _broadcast(self, event: Dict[str, Any]) -> None:
        self.events.append(event)
        for q in self._subscribers:
            await q.put(event)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "capital": self.capital,
            "max_iterations": self.max_iterations,
            "universe": self.universe,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "event_count": len(self.events),
            "final_state": self.final_state,
        }


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

_runs: Dict[str, PipelineRun] = {}


def get_run(run_id: str) -> Optional[PipelineRun]:
    return _runs.get(run_id)


def list_runs() -> List[Dict[str, Any]]:
    return [r.to_dict() for r in reversed(list(_runs.values()))]


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

async def start_pipeline(
    capital: float = 100_000.0,
    max_iterations: int = 5,
    universe: str = "NASDAQ-100",
    llm_provider: str = "Ollama",
) -> PipelineRun:
    """Create and launch a pipeline run as a background task."""
    run_id = str(uuid.uuid4())[:8]
    run = PipelineRun(run_id, capital, max_iterations, universe)
    run.llm_provider = llm_provider
        
    _runs[run_id] = run

    asyncio.create_task(_execute_pipeline(run))
    return run


async def _execute_pipeline(run: PipelineRun) -> None:
    """Run the LangGraph pipeline with progress events."""
    run.status = "running"
    run.started_at = datetime.now().isoformat()

    await run._broadcast({
        "type": "pipeline_started",
        "run_id": run.run_id,
        "timestamp": run.started_at,
        "config": {
            "capital": run.capital,
            "max_iterations": run.max_iterations,
            "universe": run.universe,
        },
    })

    try:
        import os
        from main_orchestrator import build_graph
        graph = build_graph()

        initial_state: GraphState = {
            "iteration_count": 0,
            "max_iterations": run.max_iterations,
            "capital": run.capital,
            "universe": run.universe,
            "llm_provider": run.llm_provider,
            "status": "initialized",
            "current_hypothesis": "",
            "factor_code": "",
            "backtest_results": "",
            "critique": "",
            "error": "",
            "hypothesis_history": [],
            "critique_history": [],
            "decision": "",
        }

        # Run graph step-by-step to capture per-node events
        prev_iteration = 0
        prev_status = ""

        # Use stream to get per-node updates
        async for event in graph.astream(initial_state):
            # LangGraph yields dict of {node_name: state_update}
            for node_name, state_update in event.items():
                iteration = state_update.get("iteration_count", prev_iteration)
                status = state_update.get("status", prev_status)

                agent_event = {
                    "type": "agent_update",
                    "run_id": run.run_id,
                    "timestamp": datetime.now().isoformat(),
                    "agent": node_name,
                    "iteration": iteration,
                    "status": status,
                }

                # Extract key data per agent
                if node_name == "SynthesisAgent" and state_update.get("current_hypothesis"):
                    try:
                        h = Hypothesis.model_validate_json(state_update["current_hypothesis"])
                        agent_event["hypothesis"] = {
                            "title": h.title,
                            "rationale": h.rationale,
                            "factors": h.factors,
                            "formula_logic": h.formula_logic,
                        }
                    except Exception:
                        pass

                if node_name == "ImplementationAgent" and state_update.get("factor_code"):
                    try:
                        fc = FactorCode.model_validate_json(state_update["factor_code"])
                        agent_event["code"] = {
                            "is_valid": fc.is_valid,
                            "validation_error": fc.validation_error,
                        }
                    except Exception:
                        pass

                if node_name == "ValidationAgent" and state_update.get("backtest_results"):
                    try:
                        br = BacktestResult.model_validate_json(state_update["backtest_results"])
                        agent_event["metrics"] = {
                            "sharpe_ratio": round(br.sharpe_ratio, 4),
                            "max_drawdown": round(br.max_drawdown, 4),
                            "annualized_return": round(br.annualized_return, 4),
                            "volatility": round(br.volatility, 4),
                            "trades_count": br.trades_count,
                            "wfo_score": round(br.wfo_score, 4),
                        }
                        agent_event["equity_curve"] = br.equity_curve
                    except Exception:
                        pass

                if node_name == "AnalysisAgent" and state_update.get("critique"):
                    try:
                        c = Critique.model_validate_json(state_update["critique"])
                        agent_event["critique"] = {
                            "is_success": c.is_success,
                            "decision": c.decision,
                            "suggestions": c.suggestions,
                            "potential_biases": c.potential_biases,
                        }
                    except Exception:
                        pass

                await run._broadcast(agent_event)
                prev_iteration = iteration
                prev_status = status

        # Store final state summary
        run.final_state = {
            "status": prev_status,
            "iterations": prev_iteration,
        }

        # Try to extract final metrics from the last ValidationAgent event
        for ev in reversed(run.events):
            if ev.get("metrics"):
                run.final_state["metrics"] = ev["metrics"]
                break
        for ev in reversed(run.events):
            if ev.get("equity_curve"):
                run.final_state["equity_curve"] = ev["equity_curve"]
                break
        for ev in reversed(run.events):
            if ev.get("hypothesis"):
                run.final_state["final_strategy"] = ev["hypothesis"]
                break

        run.status = "success" if prev_status == "success" else "failed"

    except Exception as exc:
        run.status = "failed"
        await run._broadcast({
            "type": "pipeline_error",
            "run_id": run.run_id,
            "timestamp": datetime.now().isoformat(),
            "error": str(exc),
        })

    finally:
        run.finished_at = datetime.now().isoformat()
        await run._broadcast({
            "type": "pipeline_finished",
            "run_id": run.run_id,
            "timestamp": run.finished_at,
            "status": run.status,
            "final_state": run.final_state,
        })
