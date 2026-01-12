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
from typing import Dict, Any
from celery.exceptions import SoftTimeLimitExceeded

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
    generate_chart_data,
)

# Configuration
JOB_TIMEOUT_SEC = int(os.getenv("JOB_TIMEOUT_SEC", 1800))


def create_progress_callback(job_id: str):
    """
    Create a progress callback function for engine tasks.

    Args:
        job_id: Job ID for progress tracking

    Returns:
        Callback function that publishes progress to Redis
    """
    def progress_cb(
        _job_id: str,
        progress: int,
        total: int,
        status: str,
        extra: Dict[str, Any]
    ):
        # Check for cancellation
        if check_cancel_flag(job_id):
            raise InterruptedError("Job canceled by user")

        # Publish progress
        publish_progress(job_id, {
            "progress": progress,
            "total": total,
            "status": status,
            **extra
        })

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

        return result

    except SoftTimeLimitExceeded:
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "error",
            "error": f"Job timed out after {JOB_TIMEOUT_SEC} seconds"
        })
        return {"error": "Timeout", "status": "error"}

    except Exception as e:
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

        return result

    except InterruptedError:
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "canceled",
            "error": "Canceled by user"
        })
        return {"error": "Canceled by user", "status": "canceled"}

    except SoftTimeLimitExceeded:
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "error",
            "error": f"Job timed out after {JOB_TIMEOUT_SEC} seconds"
        })
        return {"error": "Timeout", "status": "error"}

    except Exception as e:
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

        return result

    except InterruptedError:
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "canceled",
            "error": "Canceled by user"
        })
        return {"error": "Canceled by user", "status": "canceled"}

    except SoftTimeLimitExceeded:
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "error",
            "error": f"Job timed out after {JOB_TIMEOUT_SEC} seconds"
        })
        return {"error": "Timeout", "status": "error"}

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
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

        return result

    except SoftTimeLimitExceeded:
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "error",
            "error": f"Job timed out after {JOB_TIMEOUT_SEC} seconds"
        })
        return {"error": "Timeout", "status": "error"}

    except Exception as e:
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "error",
            "error": str(e)
        })
        return {"error": str(e), "status": "error"}
