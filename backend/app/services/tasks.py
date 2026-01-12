"""
Celery Tasks for AI Backtest Platform

Defines async tasks for:
- Running backtests
- Running optimizer
- Running AI agent
- Generating chart data
"""

import os
import sys
import json
import select
import subprocess
from typing import Dict, Any
from pathlib import Path
from celery.exceptions import SoftTimeLimitExceeded

from app.services.celery_app import celery
from app.services.progress_store import (
    publish_progress,
    set_result,
    check_cancel_flag,
    set_progress,
    get_progress
)

# Configuration
JOB_TIMEOUT_SEC = int(os.getenv("JOB_TIMEOUT_SEC", 1800))
MAX_STDOUT_SIZE = int(os.getenv("MAX_STDOUT_SIZE", 20 * 1024 * 1024))

# Project root for engine scripts
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/app"))


def get_engine_path() -> Path:
    """Get the path to engine folder"""
    # In Docker, engine is mounted at /app/engine
    engine_path = PROJECT_ROOT / "engine"
    if engine_path.exists():
        return engine_path

    # Development fallback
    backend_path = Path(__file__).parent.parent.parent
    engine_path = backend_path.parent / "engine"
    if engine_path.exists():
        return engine_path

    raise FileNotFoundError("Engine folder not found")


def run_engine_subprocess(
    script_name: str,
    input_data: Dict[str, Any],
    job_id: str,
    env_vars: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Run engine script as subprocess with progress tracking.

    Args:
        script_name: Script filename (e.g., 'ai_agent.py')
        input_data: JSON input data
        job_id: Job ID for tracking
        env_vars: Additional environment variables

    Returns:
        Final result dict
    """
    engine_path = get_engine_path()
    script_path = engine_path / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    # Prepare environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(engine_path.parent)
    if env_vars:
        env.update(env_vars)

    # Initialize progress
    publish_progress(job_id, {
        "progress": 0,
        "total": 0,
        "status": "running"
    })

    # Start subprocess
    process = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(engine_path.parent),
        bufsize=0  # Unbuffered
    )

    try:
        # Send input
        input_json = json.dumps(input_data).encode()
        process.stdin.write(input_json)
        process.stdin.close()

        # Read output with non-blocking approach
        raw_output = ""
        stderr_output = ""
        buffer = ""

        # Make stdout non-blocking on Unix
        import fcntl
        flags = fcntl.fcntl(process.stdout, fcntl.F_GETFL)
        fcntl.fcntl(process.stdout, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        while True:
            # Check cancellation every iteration
            if check_cancel_flag(job_id):
                process.kill()
                publish_progress(job_id, {
                    "progress": 0,
                    "total": 0,
                    "status": "canceled",
                    "error": "Canceled by user"
                })
                return {"error": "Canceled by user", "status": "canceled"}

            # Check if process finished
            poll_result = process.poll()

            # Use select for non-blocking read with timeout
            try:
                readable, _, _ = select.select([process.stdout], [], [], 0.5)
            except (ValueError, OSError):
                # Process stdout closed
                readable = []

            if readable:
                try:
                    chunk = process.stdout.read(8192)
                    if chunk:
                        decoded = chunk.decode('utf-8', errors='replace')
                        raw_output += decoded
                        buffer += decoded

                        # Check size limit
                        if len(raw_output) > MAX_STDOUT_SIZE:
                            process.kill()
                            publish_progress(job_id, {
                                "progress": 0,
                                "total": 0,
                                "status": "error",
                                "error": "Output too large"
                            })
                            return {"error": "Output too large", "status": "error"}

                        # Parse complete lines for progress
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                                if isinstance(obj, dict) and "progress" in obj and "total" in obj and "success" not in obj:
                                    publish_progress(job_id, {
                                        "progress": obj["progress"],
                                        "total": obj["total"],
                                        "status": "running"
                                    })
                            except json.JSONDecodeError:
                                pass
                except (IOError, OSError):
                    pass

            # Exit loop if process finished and no more data
            if poll_result is not None and not readable:
                # Read any remaining data
                try:
                    remaining = process.stdout.read()
                    if remaining:
                        raw_output += remaining.decode('utf-8', errors='replace')
                except:
                    pass
                break

        # Read stderr
        try:
            stderr_output = process.stderr.read().decode('utf-8', errors='replace')
        except:
            stderr_output = ""

        # Check exit code
        if process.returncode != 0:
            error_msg = stderr_output[:1000] if stderr_output else "Script failed"
            publish_progress(job_id, {
                "progress": 0,
                "total": 0,
                "status": "error",
                "error": error_msg
            })
            return {"error": error_msg, "status": "error"}

        # Parse final result - the last valid JSON line that represents final output
        # ai_agent.py outputs progress lines first, then final JSON with "success"/"best" keys
        final_result = None
        lines = raw_output.strip().split("\n")

        # Try to find the final result from the last lines
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    # Check for final result markers from ai_agent.py
                    # ai_agent returns: success, fallback, message, best, alternatives, comment, top, all, total
                    if "best" in obj or "success" in obj or "summary" in obj or "candles" in obj or "alternatives" in obj:
                        final_result = obj
                        break
                    # Skip progress lines (have progress/total but no best/success)
                    if "progress" in obj and "total" in obj and "best" not in obj and "success" not in obj:
                        continue
            except json.JSONDecodeError:
                continue

        if final_result is None:
            # Try to parse the entire output as one JSON (in case there's no newline)
            try:
                final_result = json.loads(raw_output.strip())
            except json.JSONDecodeError:
                raise RuntimeError(f"No valid JSON result. Last 500 chars: {raw_output[-500:]}")

        # Store result
        set_result(job_id, final_result)
        total = final_result.get("total", 1)
        publish_progress(job_id, {
            "progress": total,
            "total": total,
            "status": "done"
        })

        return final_result

    except SoftTimeLimitExceeded:
        process.kill()
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "error",
            "error": f"Job timed out after {JOB_TIMEOUT_SEC} seconds"
        })
        return {"error": "Timeout", "status": "error"}

    finally:
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()


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
        return run_engine_subprocess(
            script_name="backtest_engine.py",
            input_data=config,
            job_id=job_id
        )
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
        return run_engine_subprocess(
            script_name="run_optimizer.py",
            input_data=config,
            job_id=job_id
        )
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
        return run_engine_subprocess(
            script_name="ai_agent.py",
            input_data=config,
            job_id=job_id,
            env_vars={"AI_PROGRESS": "1"}
        )
    except Exception as e:
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
        return run_engine_subprocess(
            script_name="chart_data.py",
            input_data=config,
            job_id=job_id
        )
    except Exception as e:
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "error",
            "error": str(e)
        })
        return {"error": str(e), "status": "error"}
