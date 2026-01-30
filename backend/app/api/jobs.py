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
    ai_agent_task,
    quant_brain_task
)
from app.services.progress_store import (
    get_progress,
    get_result,
    set_cancel_flag,
    subscribe_progress,
    set_progress,
    set_task_id,
    get_task_id,
)
from app.services.celery_app import celery

router = APIRouter(prefix="/api", tags=["jobs"])


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
    """Request body for /optimize, /ai-agent, /quant-brain"""
    symbols: List[str]
    timeframes: List[str]
    strategy: StrategyConfig
    filters: Optional[Dict[str, Any]] = None
    properties: Optional[Dict[str, Any]] = None
    range: Optional[DateRange] = None
    topN: Optional[int] = 50
    minTFAgree: Optional[int] = 2
    stability: Optional[Dict[str, float]] = None
    # Quant Brain mode: "ultra", "fast", or "brain"
    mode: Optional[str] = "fast"


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

    elif strat["type"] == "rf_st_rsi_combined":
        # Combined strategy - validate entry params are arrays
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
        result = run_backtest_task.delay(request.strategy, job_id)
        # Store task_id mapping for proper revoke support
        set_task_id(job_id, result.id)
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
    result = optimize_task.delay(cfg, job_id)
    # Store task_id mapping for proper revoke support
    set_task_id(job_id, result.id)

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

    # Log the request for debugging
    import logging
    logger = logging.getLogger(__name__)
    strategy_type = cfg.get("strategy", {}).get("type", "unknown")
    symbols = cfg.get("symbols", [])
    timeframes = cfg.get("timeframes", [])
    logger.info(f"[AI-Agent] Received request: strategy={strategy_type}, symbols={symbols}, timeframes={timeframes}")

    error = validate_optimize_config(cfg)
    if error:
        raise HTTPException(status_code=400, detail=error)

    job_id = generate_job_id()

    # Initialize progress
    set_progress(job_id, {"progress": 0, "total": 0, "status": "running"})

    # Enqueue task
    result = ai_agent_task.delay(cfg, job_id)
    # Store task_id mapping for proper revoke support
    set_task_id(job_id, result.id)

    logger.info(f"[AI-Agent] Enqueued job {job_id} (task_id={result.id}) for strategy={strategy_type}")

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

    Uses polling with asyncio.sleep to avoid blocking the event loop.
    """
    import asyncio

    # Check if job exists
    progress = get_progress(job_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        """Generate SSE events by polling Redis"""
        last_progress = None
        last_status = None

        try:
            while True:
                # Get current progress from Redis
                current_progress = get_progress(job_id)

                if current_progress:
                    # Only send if progress changed
                    current_status = current_progress.get("status")
                    current_value = current_progress.get("progress", 0)

                    if (current_progress != last_progress or
                        current_status != last_status):
                        yield {
                            "event": "progress",
                            "data": json.dumps(current_progress)
                        }
                        last_progress = current_progress
                        last_status = current_status

                    # If done, send done event and stop
                    if current_status in ["done", "error", "canceled"]:
                        yield {
                            "event": "done",
                            "data": json.dumps(current_progress)
                        }
                        break

                # Poll interval - 200ms for responsive updates
                await asyncio.sleep(0.2)

        except asyncio.CancelledError:
            # Client disconnected
            pass

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
    # Set cancel flag - this will be checked by progress_cb in engine
    set_cancel_flag(job_id)

    # Update progress
    set_progress(job_id, {
        "progress": 0,
        "total": 0,
        "status": "canceled",
        "error": "Canceled by user"
    })

    # Try to revoke Celery task using proper task_id
    task_id = get_task_id(job_id)
    if task_id:
        try:
            # Use actual Celery task_id for revoke
            celery.control.revoke(task_id, terminate=True, signal='SIGTERM')
        except Exception:
            pass  # Ignore revoke errors - cancel flag will handle it

    return {"success": True}


@router.post("/quant-brain")
async def quant_brain(request: OptimizeRequest):
    """
    Run Quant AI Brain - Self-learning optimization engine.

    Features:
    - Long-term genome memory (ParamMemory)
    - Market regime classification
    - Evolutionary genome optimization
    - Coherence validation
    - Robustness filtering

    Request body: OptimizeRequest with symbols, timeframes, strategy, etc.

    Returns:
    - { jobId: "..." }
    """
    cfg = request.model_dump(by_alias=True)

    # Log the request for debugging
    import logging
    logger = logging.getLogger(__name__)
    strategy_type = cfg.get("strategy", {}).get("type", "unknown")
    symbols = cfg.get("symbols", [])
    timeframes = cfg.get("timeframes", [])
    logger.info(f"[Quant-Brain] Received request: strategy={strategy_type}, symbols={symbols}, timeframes={timeframes}")

    error = validate_optimize_config(cfg)
    if error:
        raise HTTPException(status_code=400, detail=error)

    job_id = generate_job_id()

    # Initialize progress
    set_progress(job_id, {"progress": 0, "total": 100, "status": "running", "phase": "initializing"})

    # Enqueue task
    result = quant_brain_task.delay(cfg, job_id)
    # Store task_id mapping for proper revoke support
    set_task_id(job_id, result.id)

    logger.info(f"[Quant-Brain] Enqueued job {job_id} (task_id={result.id}) for strategy={strategy_type}")

    return {"jobId": job_id}
