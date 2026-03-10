"""
Project AutoQuant — Main Orchestrator
=======================================
LangGraph-based state machine that drives the multi-agent autonomous
quantitative research pipeline.

Usage::

    python main_orchestrator.py
    python main_orchestrator.py --iterations 10 --capital 200000
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END

from src.agents.base import GraphState
from src.agents.synthesis import SynthesisAgent
from src.agents.implementation import ImplementationAgent
from src.agents.validation import ValidationAgent
from src.agents.analysis import AnalysisAgent


# ═══════════════════════════════════════════════════════════════════════════
# Build the LangGraph state machine
# ═══════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """
    Construct and compile the multi-agent LangGraph workflow.

    Topology::

        ┌──────────────┐
        │ Synthesis     │◄────────────────────────────────┐
        │ Agent         │                                 │
        └──────┬───────┘                                 │
               │                                         │
               ▼                                         │
        ┌──────────────┐                                 │
        │ Implementation│◄───────── fix_code ────────┐   │
        │ Agent         │                            │   │ evolve_
        └──────┬───────┘                            │   │ hypothesis
               │                                    │   │
               ▼                                    │   │
        ┌──────────────┐                            │   │
        │ Validation    │                            │   │
        │ Agent         │                            │   │
        └──────┬───────┘                            │   │
               │                                    │   │
               ▼                                    │   │
        ┌──────────────┐    end                     │   │
        │ Analysis      │────────► END              │   │
        │ Agent         │───────────────────────────┘───┘
        └──────────────┘
    """
    synthesis_agent = SynthesisAgent()
    implementation_agent = ImplementationAgent()
    validation_agent = ValidationAgent()
    analysis_agent = AnalysisAgent()

    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("SynthesisAgent", synthesis_agent.run)
    workflow.add_node("ImplementationAgent", implementation_agent.run)
    workflow.add_node("ValidationAgent", validation_agent.run)
    workflow.add_node("AnalysisAgent", analysis_agent.run)

    # Linear edges
    workflow.set_entry_point("SynthesisAgent")
    workflow.add_edge("SynthesisAgent", "ImplementationAgent")
    workflow.add_edge("ImplementationAgent", "ValidationAgent")
    workflow.add_edge("ValidationAgent", "AnalysisAgent")

    # Conditional routing from AnalysisAgent
    workflow.add_conditional_edges(
        "AnalysisAgent",
        analysis_agent.decision_router,
        {
            "evolve_hypothesis": "SynthesisAgent",   # Refine the strategy
            "fix_code": "ImplementationAgent",         # Fix code bugs
            "end": END,                                # Terminate successfully
        },
    )

    return workflow.compile()


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project AutoQuant — Autonomous Quantitative Research Engine",
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=5,
        help="Maximum evolution iterations (default: 5)",
    )
    parser.add_argument(
        "--capital", "-c",
        type=float,
        default=100_000.0,
        help="Initial capital in USD (default: 100,000)",
    )
    parser.add_argument(
        "--universe", "-u",
        type=str,
        default="NASDAQ-100",
        help="Stock universe (default: NASDAQ-100)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║        PROJECT AUTOQUANT — Autonomous Quant Engine      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Universe:      {args.universe}")
    print(f"  Capital:       ${args.capital:,.0f}")
    print(f"  Max Iterations: {args.iterations}")
    print(f"  LLM Model:     {os.getenv('DEFAULT_LLM_MODEL', 'gpt-4o')}")
    print()

    # Build initial state
    initial_state: GraphState = {
        "iteration_count": 0,
        "max_iterations": args.iterations,
        "capital": args.capital,
        "universe": args.universe,
        "status": "initialized",
        "current_hypothesis": "",
        "factor_code": "",
        "backtest_results": "",
        "critique": "",
        "error": "",
        "hypothesis_history": [],
        "critique_history": [],
        "decision": "",
        "best_hypothesis": "",
        "best_backtest_results": "",
        "best_factor_code": "",
        "best_sharpe": float("-inf"),
    }

    # Compile and run the graph
    graph = build_graph()

    print("Starting autonomous pipeline...\n")

    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        print(f"\n[FATAL] Pipeline crashed: {exc}")
        sys.exit(1)

    # -- Report results ------------------------------------------------------
    # Use best-so-far results, falling back to last iteration if unset
    best_bt = final_state.get("best_backtest_results") or final_state.get("backtest_results", "")
    best_hyp = final_state.get("best_hypothesis") or final_state.get("current_hypothesis", "")
    last_bt = final_state.get("backtest_results", "")

    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Status:     {final_state.get('status', 'unknown')}")
    print(f"  Iterations: {final_state.get('iteration_count', 0)}")

    if best_bt:
        from src.agents.base import BacktestResult
        result = BacktestResult.model_validate_json(best_bt)
        print(f"\n  === Best Strategy (across all iterations) ===")
        print(f"  Sharpe:     {result.sharpe_ratio:.4f}")
        print(f"  Max DD:     {result.max_drawdown:.2%}")
        print(f"  Ann Return: {result.annualized_return:.2%}")
        print(f"  WFO Score:  {result.wfo_score:.4f}")
        print(f"  Trades:     {result.trades_count}")

    if best_hyp:
        from src.agents.base import Hypothesis
        hyp = Hypothesis.model_validate_json(best_hyp)
        print(f"\n  Strategy:   {hyp.title}")
        print(f"  Logic:      {hyp.formula_logic}")

    # Show last iteration too, if different from best
    if last_bt and last_bt != best_bt:
        from src.agents.base import BacktestResult as BR
        last_result = BR.model_validate_json(last_bt)
        print(f"\n  --- Last Iteration ---")
        print(f"  Sharpe: {last_result.sharpe_ratio:.4f} | Max DD: {last_result.max_drawdown:.2%} | Trades: {last_result.trades_count}")

    print()


if __name__ == "__main__":
    main()
