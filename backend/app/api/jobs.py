"""
Job Management API Endpoints

Handles:
- POST /run-backtest - Run backtest (sync or async)
- POST /optimize - Run optimizer
- POST /ai-agent - Run AI agent
- GET /ai-agent/progress/:jobId - Poll progress
- GET /ai-agent/progress-stream/:jobId - SSE progress stream
- GET /ai-agent/result/:jobId - Get final result
- POST /ai-agent/cancel/:jobId - Cancel job
"""

import os
import json
import time
import uuid
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.services.tasks import (
    run_backtest_task,
    optimize_task,
    ai_agent_task
)
from app.services.progress_store import (
    get_progress,
    get_result,
    set_cancel_flag,
    subscribe_progress,
    set_progress
)
from app.services.celery_app import celery

router = APIRouter()


# ==================== Request/Response Models ====================

class StrategyParams(BaseModel):
    """Strategy parameters - flexible dict"""
    class Config:
        extra = "allow"


class StrategyConfig(BaseModel):
    """Strategy configuration"""
    type: str
    params: Dict[str, Any]


class BacktestRequest(BaseModel):
    """Request body for /run-backtest"""
    strategy: Dict[str, Any]


class DateRange(BaseModel):
    """Date range - from/to can be null"""
    model_config = {"extra": "allow"}

    # Allow null values for from/to
    # Using 'from_' because 'from' is reserved keyword
    from_date: Optional[str] = Field(None, alias="from")
    to: Optional[str] = None


class OptimizeRequest(BaseModel):
    """Request body for /optimize and /ai-agent"""
    symbols: List[str]
    timeframes: List[str]
    strategy: StrategyConfig
    filters: Optional[Dict[str, Any]] = None
    properties: Optional[Dict[str, Any]] = None
    range: Optional[DateRange] = None
    topN: Optional[int] = 50
    minTFAgree: Optional[int] = 2
    stability: Optional[Dict[str, float]] = None


class JobResponse(BaseModel):
    """Response with job ID"""
    jobId: str


class ProgressResponse(BaseModel):
    """Progress response"""
    progress: int
    total: int
    status: str
    error: Optional[str] = None


# ==================== Validation ====================

def validate_optimize_config(cfg: Dict[str, Any]) -> Optional[str]:
    """
    Validate optimization configuration.
    Returns error message if invalid, None if valid.
    Same logic as server.js validateOptimizeCfg.
    """
    if not cfg or not isinstance(cfg, dict):
        return "Config must be JSON object"

    if not isinstance(cfg.get("symbols"), list) or len(cfg.get("symbols", [])) == 0:
        return "symbols must be non-empty array"

    if not isinstance(cfg.get("timeframes"), list) or len(cfg.get("timeframes", [])) == 0:
        return "timeframes must be non-empty array"

    strat = cfg.get("strategy", {})
    if not strat.get("type"):
        return "strategy.type is required"

    params = strat.get("params")
    if not params or not isinstance(params, dict):
        return "strategy.params is required"

    # Validate based on strategy type
    if strat["type"] == "ema_cross":
        if not isinstance(params.get("emaFast"), list) or not isinstance(params.get("emaSlow"), list):
            return "strategy.params.emaFast/emaSlow must be arrays"
        if len(params.get("emaFast", [])) == 0 or len(params.get("emaSlow", [])) == 0:
            return "emaFast/emaSlow must be non-empty arrays"

    elif strat["type"] == "rf_st_rsi":
        if not isinstance(params.get("st_atrPeriod"), list):
            return "strategy.params.st_atrPeriod must be array"
        if not isinstance(params.get("st_mult"), list):
            return "strategy.params.st_mult must be array"
        if not isinstance(params.get("rf_period"), list):
            return "strategy.params.rf_period must be array"
        if not isinstance(params.get("rf_mult"), list):
            return "strategy.params.rf_mult must be array"

    else:
        return f"Unknown strategy type: {strat['type']}"

    return None


def generate_job_id() -> str:
    """Generate unique job ID - same format as server.js"""
    timestamp = int(time.time() * 1000)
    random_suffix = uuid.uuid4().hex[:4]
    return f"job_{timestamp}_{random_suffix}"


# ==================== Endpoints ====================

@router.post("/run-backtest")
async def run_backtest(request: BacktestRequest, sync: bool = Query(False)):
    """
    Run backtest.

    Query params:
    - sync: If true, run synchronously and return result. Default false (async).

    Request body:
    - strategy: Strategy configuration

    Returns:
    - If sync=true: { success: true, result: {...} }
    - If sync=false: { jobId: "..." }
    """
    if not request.strategy:
        raise HTTPException(status_code=400, detail="strategy is required")

    job_id = generate_job_id()

    if sync:
        # Synchronous execution (for backward compatibility)
        # Note: This blocks the API - not recommended for production
        from app.services.engine_runner import run_backtest as run_backtest_sync
        try:
            result = run_backtest_sync(request.strategy, job_id)
            return {"success": True, "result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Async execution via Celery
        run_backtest_task.delay(request.strategy, job_id)
        return {"jobId": job_id}


@router.post("/optimize")
async def optimize(request: OptimizeRequest):
    """
    Run optimization.

    Request body: OptimizeRequest with symbols, timeframes, strategy, etc.

    Returns:
    - { jobId: "..." }
    """
    cfg = request.model_dump(by_alias=True)
    error = validate_optimize_config(cfg)
    if error:
        raise HTTPException(status_code=400, detail=error)

    job_id = generate_job_id()
    optimize_task.delay(cfg, job_id)

    return {"jobId": job_id}


@router.post("/ai-agent")
async def ai_agent(request: OptimizeRequest):
    """
    Run AI agent (optimization + best selection).

    Request body: OptimizeRequest with symbols, timeframes, strategy, etc.

    Returns:
    - { jobId: "..." }
    """
    cfg = request.model_dump(by_alias=True)
    error = validate_optimize_config(cfg)
    if error:
        raise HTTPException(status_code=400, detail=error)

    job_id = generate_job_id()

    # Initialize progress
    set_progress(job_id, {"progress": 0, "total": 0, "status": "running"})

    # Enqueue task
    ai_agent_task.delay(cfg, job_id)

    return {"jobId": job_id}


@router.get("/ai-agent/progress/{job_id}")
async def get_job_progress(job_id: str):
    """
    Get job progress by polling.

    Returns:
    - { progress: N, total: M, status: "running|done|error|canceled", error?: "..." }
    """
    progress = get_progress(job_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Job not found")

    return progress


@router.get("/ai-agent/progress-stream/{job_id}")
async def stream_job_progress(job_id: str):
    """
    Stream job progress via Server-Sent Events (SSE).

    Subscribes to Redis pub/sub channel for real-time progress updates.
    """
    # Check if job exists
    progress = get_progress(job_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        """Generate SSE events from Redis pub/sub"""
        pubsub = subscribe_progress(job_id)

        try:
            # Send initial progress
            current_progress = get_progress(job_id)
            if current_progress:
                yield {
                    "event": "progress",
                    "data": json.dumps(current_progress)
                }

                # If already done, send done event and stop
                if current_progress.get("status") in ["done", "error", "canceled"]:
                    yield {
                        "event": "done",
                        "data": json.dumps(current_progress)
                    }
                    return

            # Listen for updates
            for message in pubsub.listen():
                if message["type"] == "message":
                    data = message["data"]
                    try:
                        progress_data = json.loads(data) if isinstance(data, str) else data
                        yield {
                            "event": "progress",
                            "data": json.dumps(progress_data)
                        }

                        # Stop if done
                        if progress_data.get("status") in ["done", "error", "canceled"]:
                            yield {
                                "event": "done",
                                "data": json.dumps(progress_data)
                            }
                            break
                    except json.JSONDecodeError:
                        pass

        finally:
            pubsub.unsubscribe()
            pubsub.close()

    return EventSourceResponse(event_generator())


@router.get("/ai-agent/result/{job_id}")
async def get_job_result(job_id: str):
    """
    Get final job result.

    Returns:
    - If done: Full result JSON
    - If not done: 202 status with { status: "..." }
    - If not found: 404
    """
    progress = get_progress(job_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Job not found")

    if progress.get("status") != "done":
        # Return 202 Accepted with current status
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=202,
            content={"status": progress.get("status")}
        )

    result = get_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    return result


@router.post("/ai-agent/cancel/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a running job.

    Sets cancel flag in Redis and attempts to revoke Celery task.

    Returns:
    - { success: true }
    """
    # Set cancel flag
    set_cancel_flag(job_id)

    # Update progress
    set_progress(job_id, {
        "progress": 0,
        "total": 0,
        "status": "canceled",
        "error": "Canceled by user"
    })

    # Try to revoke Celery task (best effort)
    try:
        # Note: Celery task revocation requires task ID, not job ID
        # This is a best-effort cancellation
        celery.control.revoke(job_id, terminate=True)
    except Exception:
        pass  # Ignore revoke errors

    return {"success": True}
