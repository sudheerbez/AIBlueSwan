"""
AutoQuant Backend — FastAPI Application
=========================================
REST + WebSocket API for the AutoQuant pipeline.
Designed for reuse by web, Android, and iOS clients.
"""

import asyncio
import json
import sys
import os

# Add project root to path so src/ imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from backend.runner import start_pipeline, get_run, list_runs


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AutoQuant API",
    description="Autonomous Quantitative Research Engine — REST + WebSocket API",
    version="1.0.0",
)

# CORS: allow all origins for web + future mobile clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/Response Schemas
# ---------------------------------------------------------------------------

class PipelineStartRequest(BaseModel):
    capital: float = Field(default=100_000.0, ge=1000, le=10_000_000)
    max_iterations: int = Field(default=5, ge=1, le=50)
    universe: str = Field(default="NASDAQ-100")


class PipelineStartResponse(BaseModel):
    run_id: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "autoquant-backend"}


@app.post("/api/pipeline/start", response_model=PipelineStartResponse)
async def start(req: PipelineStartRequest):
    """Start a new pipeline run."""
    run = await start_pipeline(
        capital=req.capital,
        max_iterations=req.max_iterations,
        universe=req.universe,
    )
    return PipelineStartResponse(
        run_id=run.run_id,
        status=run.status,
        message=f"Pipeline {run.run_id} started with ${req.capital:,.0f} capital.",
    )


@app.get("/api/pipeline/status/{run_id}")
async def status(run_id: str):
    """Get the current status of a pipeline run."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found.")
    return run.to_dict()


@app.get("/api/pipeline/results/{run_id}")
async def results(run_id: str):
    """Get the final results of a completed pipeline run."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found.")
    if run.status not in ("success", "failed"):
        raise HTTPException(202, "Pipeline still running.")
    return {
        "run_id": run.run_id,
        "status": run.status,
        "final_state": run.final_state,
        "events": run.events,
    }


@app.get("/api/pipeline/history")
async def history():
    """List all pipeline runs."""
    return {"runs": list_runs()}


# ---------------------------------------------------------------------------
# WebSocket — Real-time pipeline progress
# ---------------------------------------------------------------------------

@app.websocket("/ws/pipeline/{run_id}")
async def ws_pipeline(websocket: WebSocket, run_id: str):
    """
    Subscribe to real-time events from a running pipeline.
    Events are JSON objects with a ``type`` field.
    """
    await websocket.accept()

    run = get_run(run_id)
    if not run:
        await websocket.send_json({"type": "error", "message": f"Run {run_id} not found."})
        await websocket.close()
        return

    # Send all historical events first (catch-up)
    for event in run.events:
        await websocket.send_json(event)

    # If already finished, close
    if run.status in ("success", "failed", "cancelled"):
        await websocket.close()
        return

    # Subscribe to live events
    queue = run.subscribe()
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event)
            if event.get("type") == "pipeline_finished":
                break
    except WebSocketDisconnect:
        pass
    finally:
        run.unsubscribe(queue)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
