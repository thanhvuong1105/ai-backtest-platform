"""
Celery Tasks for AI Backtest Platform

Defines async tasks for:
- Running backtests
- Running optimizer
- Running AI agent
- Generating chart data

Now uses direct imports from engine package instead of subprocess.
"""

import os
import time
import logging
from typing import Dict, Any
from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import worker_ready

from app.services.celery_app import celery
from app.services.progress_store import (
    publish_progress,
    set_result,
    check_cancel_flag,
)

# Import engine functions directly
from engine import (
    run_backtest,
    run_optimizer,
    run_ai_agent,
    run_quant_brain,
    generate_chart_data,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler if not already configured
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)

# Configuration
JOB_TIMEOUT_SEC = int(os.getenv("JOB_TIMEOUT_SEC", 3600))
# Quant Brain needs longer timeout due to evolutionary optimization
# Set to 1 day to ensure no interruption during long optimization runs
QUANT_BRAIN_TIMEOUT_SEC = int(os.getenv("QUANT_BRAIN_TIMEOUT_SEC", 86400))  # 24 hours (1 day) default
PROGRESS_BATCH_SIZE = int(os.getenv("PROGRESS_BATCH_SIZE", 20))


@worker_ready.connect
def on_worker_ready(sender=None, **_kwargs):
    """
    Called when Celery worker is ready.
    Preload data into cache for faster first runs.
    """
    try:
        from engine.data_loader import preload_all_data
        logger.info(f"Worker {sender} ready - preloading data...")
        count = preload_all_data()
        logger.info(f"Preloaded {count} datasets")
    except Exception as e:
        logger.warning(f"Failed to preload data: {e}")


def create_progress_callback(job_id: str, batch_size: int = None):
    """
    Create a progress callback function for engine tasks.

    Uses batched updates to reduce Redis writes.

    Args:
        job_id: Job ID for progress tracking
        batch_size: Number of updates to batch before publishing (default from env)

    Returns:
        Callback function that publishes progress to Redis
    """
    last_log_time = [0]  # Use list to allow mutation in closure
    last_publish_time = [0]  # Track last publish time for UI updates
    last_publish_progress = [0]  # Track last published progress
    last_cancel_check = [0]  # Track last cancellation check time
    call_count = [0]  # Track callback calls for throttling
    LOG_INTERVAL = 5  # Log every 5 seconds
    PUBLISH_INTERVAL = 1.0  # Publish at least every 1 second for responsive UI
    CANCEL_CHECK_INTERVAL = 0.5  # Check cancellation every 0.5 seconds (not every call)
    batch = batch_size or PROGRESS_BATCH_SIZE

    def progress_cb(
        _job_id: str,
        progress: int,
        total: int,
        status: str,
        extra: Dict[str, Any]
    ):
        call_count[0] += 1
        now = time.time()

        # Throttled cancellation check - only check every 0.5 seconds
        # This reduces Redis reads from 100K+ to ~2K for a 1000-run optimization
        if now - last_cancel_check[0] >= CANCEL_CHECK_INTERVAL:
            last_cancel_check[0] = now
            if check_cancel_flag(job_id):
                logger.info(f"[{job_id}] Cancellation requested")
                raise InterruptedError("Job canceled by user")

        # Publish progress updates based on multiple conditions:
        # 1. At start (progress == 0)
        # 2. At completion (progress == total)
        # 3. Every N updates (batch)
        # 4. At least every 1 second for responsive UI
        time_since_publish = now - last_publish_time[0]
        should_publish = (
            progress - last_publish_progress[0] >= batch or
            progress == total or
            progress == 0 or
            time_since_publish >= PUBLISH_INTERVAL
        )

        if should_publish:
            publish_progress(job_id, {
                "progress": progress,
                "total": total,
                "status": status,
                **extra
            })
            last_publish_progress[0] = progress
            last_publish_time[0] = now

        # Throttled logging
        if now - last_log_time[0] >= LOG_INTERVAL:
            percent = (progress / total * 100) if total > 0 else 0
            logger.info(f"[{job_id}] Progress: {progress}/{total} ({percent:.1f}%)")
            last_log_time[0] = now

    return progress_cb


@celery.task(bind=True, name="app.services.tasks.run_backtest_task")
def run_backtest_task(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """
    Celery task to run backtest.

    Args:
        config: Strategy configuration
        job_id: Job ID

    Returns:
        Backtest result
    """
    start_time = time.time()
    logger.info(f"[{job_id}] Starting backtest task")
    logger.debug(f"[{job_id}] Config: {config}")

    try:
        # Initialize progress
        publish_progress(job_id, {
            "progress": 0,
            "total": 1,
            "status": "running"
        })

        # Run backtest directly
        result = run_backtest(config)

        # Store result and mark as done
        set_result(job_id, result)
        publish_progress(job_id, {
            "progress": 1,
            "total": 1,
            "status": "done"
        })

        elapsed = time.time() - start_time
        trades = result.get("summary", {}).get("totalTrades", 0)
        logger.info(f"[{job_id}] Backtest completed in {elapsed:.2f}s - {trades} trades")

        return result

    except SoftTimeLimitExceeded:
        elapsed = time.time() - start_time
        logger.error(f"[{job_id}] Backtest timed out after {elapsed:.2f}s")
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "error",
            "error": f"Job timed out after {JOB_TIMEOUT_SEC} seconds"
        })
        return {"error": "Timeout", "status": "error"}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.exception(f"[{job_id}] Backtest failed after {elapsed:.2f}s: {e}")
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "error",
            "error": str(e)
        })
        return {"error": str(e), "status": "error"}


@celery.task(bind=True, name="app.services.tasks.optimize_task")
def optimize_task(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """
    Celery task to run optimizer.

    Args:
        config: Optimization configuration
        job_id: Job ID

    Returns:
        Optimization result
    """
    start_time = time.time()
    symbols = config.get("symbols", [])
    timeframes = config.get("timeframes", [])
    strategy_type = config.get("strategy", {}).get("type", "unknown")

    logger.info(f"[{job_id}] Starting optimizer task - {strategy_type} on {symbols} x {timeframes}")

    try:
        # Initialize progress
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "running"
        })

        # Create progress callback
        progress_cb = create_progress_callback(job_id)

        # Run optimizer directly
        result = run_optimizer(config, job_id=job_id, progress_cb=progress_cb)

        # Store result and mark as done
        set_result(job_id, result)
        total = result.get("stats", {}).get("totalRuns", 1)
        publish_progress(job_id, {
            "progress": total,
            "total": total,
            "status": "done"
        })

        elapsed = time.time() - start_time
        stats = result.get("stats", {})
        logger.info(
            f"[{job_id}] Optimizer completed in {elapsed:.2f}s - "
            f"Total: {stats.get('totalRuns', 0)}, "
            f"Passed: {stats.get('passedRuns', 0)}, "
            f"Rejected: {stats.get('rejectedRuns', 0)}"
        )

        return result

    except InterruptedError:
        elapsed = time.time() - start_time
        logger.warning(f"[{job_id}] Optimizer canceled after {elapsed:.2f}s")
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "canceled",
            "error": "Canceled by user"
        })
        return {"error": "Canceled by user", "status": "canceled"}

    except SoftTimeLimitExceeded:
        elapsed = time.time() - start_time
        logger.error(f"[{job_id}] Optimizer timed out after {elapsed:.2f}s")
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "error",
            "error": f"Job timed out after {JOB_TIMEOUT_SEC} seconds"
        })
        return {"error": "Timeout", "status": "error"}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.exception(f"[{job_id}] Optimizer failed after {elapsed:.2f}s: {e}")
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "error",
            "error": str(e)
        })
        return {"error": str(e), "status": "error"}


@celery.task(bind=True, name="app.services.tasks.ai_agent_task")
def ai_agent_task(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """
    Celery task to run AI agent.

    Args:
        config: AI agent configuration
        job_id: Job ID

    Returns:
        AI agent result
    """
    start_time = time.time()
    symbols = config.get("symbols", [])
    timeframes = config.get("timeframes", [])
    strategy_type = config.get("strategy", {}).get("type", "unknown")

    logger.info(f"[{job_id}] Starting AI agent task - {strategy_type} on {symbols} x {timeframes}")

    try:
        # Initialize progress
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "running"
        })

        # Create progress callback
        progress_cb = create_progress_callback(job_id)

        # Run AI agent directly
        result = run_ai_agent(config, job_id=job_id, progress_cb=progress_cb)

        # Store result and mark as done
        set_result(job_id, result)
        total = result.get("total", 1)
        publish_progress(job_id, {
            "progress": total,
            "total": total,
            "status": "done"
        })

        elapsed = time.time() - start_time
        success = result.get("success", False)
        best = result.get("best", {})
        best_info = f"{best.get('symbol')} {best.get('timeframe')}" if best else "None"

        logger.info(
            f"[{job_id}] AI agent completed in {elapsed:.2f}s - "
            f"Success: {success}, Best: {best_info}, Total runs: {total}"
        )

        return result

    except InterruptedError:
        elapsed = time.time() - start_time
        logger.warning(f"[{job_id}] AI agent canceled after {elapsed:.2f}s")
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "canceled",
            "error": "Canceled by user"
        })
        return {"error": "Canceled by user", "status": "canceled"}

    except SoftTimeLimitExceeded:
        elapsed = time.time() - start_time
        logger.error(f"[{job_id}] AI agent timed out after {elapsed:.2f}s")
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "error",
            "error": f"Job timed out after {JOB_TIMEOUT_SEC} seconds"
        })
        return {"error": "Timeout", "status": "error"}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.exception(f"[{job_id}] AI agent failed after {elapsed:.2f}s: {e}")
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "error",
            "error": str(e)
        })
        return {"error": str(e), "status": "error"}


@celery.task(
    bind=True,
    name="app.services.tasks.quant_brain_task",
    soft_time_limit=QUANT_BRAIN_TIMEOUT_SEC,
    time_limit=QUANT_BRAIN_TIMEOUT_SEC + 60
)
def quant_brain_task(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """
    Celery task to run Quant AI Brain.

    Self-learning optimization with:
    - Long-term genome memory
    - Market regime classification
    - Evolutionary optimization
    - Robustness filtering

    Args:
        config: Quant Brain configuration
        job_id: Job ID

    Returns:
        Quant Brain result
    """
    start_time = time.time()
    symbols = config.get("symbols", [])
    timeframes = config.get("timeframes", [])
    strategy_type = config.get("strategy", {}).get("type", "unknown")

    logger.info(f"[{job_id}] Starting Quant Brain task - {strategy_type} on {symbols} x {timeframes}")

    try:
        # Initialize progress
        publish_progress(job_id, {
            "progress": 0,
            "total": 100,
            "status": "running",
            "phase": "initializing"
        })

        # Create progress callback
        progress_cb = create_progress_callback(job_id)

        # Run Quant Brain
        result = run_quant_brain(config, job_id=job_id, progress_cb=progress_cb)

        # Store result and mark as done
        set_result(job_id, result)
        publish_progress(job_id, {
            "progress": 100,
            "total": 100,
            "status": "done"
        })

        elapsed = time.time() - start_time
        success = result.get("success", False)
        regime = result.get("market_regime", "unknown")
        meta = result.get("meta", {})

        logger.info(
            f"[{job_id}] Quant Brain completed in {elapsed:.2f}s - "
            f"Success: {success}, Regime: {regime}, "
            f"Tested: {meta.get('total_tested', 0)}, "
            f"Robust: {meta.get('robustness_passed', 0)}"
        )

        return result

    except InterruptedError:
        elapsed = time.time() - start_time
        logger.warning(f"[{job_id}] Quant Brain canceled after {elapsed:.2f}s")
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "canceled",
            "error": "Canceled by user"
        })
        return {"error": "Canceled by user", "status": "canceled"}

    except SoftTimeLimitExceeded:
        elapsed = time.time() - start_time
        logger.error(f"[{job_id}] Quant Brain timed out after {elapsed:.2f}s")
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "error",
            "error": f"Job timed out after {QUANT_BRAIN_TIMEOUT_SEC} seconds"
        })
        return {"error": "Timeout", "status": "error"}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.exception(f"[{job_id}] Quant Brain failed after {elapsed:.2f}s: {e}")
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "error",
            "error": str(e)
        })
        return {"error": str(e), "status": "error"}


@celery.task(bind=True, name="app.services.tasks.chart_data_task")
def chart_data_task(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """
    Celery task to generate chart data.

    Args:
        config: Chart data configuration
        job_id: Job ID

    Returns:
        Chart data result
    """
    start_time = time.time()
    symbol = config.get("symbol", "unknown")
    timeframe = config.get("timeframe", "unknown")

    logger.info(f"[{job_id}] Starting chart data task - {symbol} {timeframe}")

    try:
        # Initialize progress
        publish_progress(job_id, {
            "progress": 0,
            "total": 1,
            "status": "running"
        })

        # Generate chart data directly
        result = generate_chart_data(config)

        # Store result and mark as done
        set_result(job_id, result)
        publish_progress(job_id, {
            "progress": 1,
            "total": 1,
            "status": "done"
        })

        elapsed = time.time() - start_time
        candles = len(result.get("candles", []))
        logger.info(f"[{job_id}] Chart data completed in {elapsed:.2f}s - {candles} candles")

        return result

    except SoftTimeLimitExceeded:
        elapsed = time.time() - start_time
        logger.error(f"[{job_id}] Chart data timed out after {elapsed:.2f}s")
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "error",
            "error": f"Job timed out after {JOB_TIMEOUT_SEC} seconds"
        })
        return {"error": "Timeout", "status": "error"}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.exception(f"[{job_id}] Chart data failed after {elapsed:.2f}s: {e}")
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "error",
            "error": str(e)
        })
        return {"error": str(e), "status": "error"}
