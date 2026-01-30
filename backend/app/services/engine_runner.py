"""
Engine Runner - Spawns and manages engine/*.py subprocess execution.

Handles:
- Spawning Python subprocess for engine scripts
- Streaming stdout and parsing JSON progress lines
- Publishing progress to Redis
- Timeout and output size limits
- Cancel handling via subprocess termination
"""

import os
import sys
import json
import signal
import subprocess
from typing import Optional, Dict, Any, Callable
from pathlib import Path

from app.services.progress_store import (
    publish_progress,
    set_result,
    check_cancel_flag,
    set_progress
)

# Configuration from environment
JOB_TIMEOUT_SEC = int(os.getenv("JOB_TIMEOUT_SEC", 1800))  # 30 minutes default
MAX_STDOUT_SIZE = int(os.getenv("MAX_STDOUT_SIZE", 20 * 1024 * 1024))  # 20MB default

# Project root - engine scripts are at PROJECT_ROOT/engine/
# In Docker, this will be /app (backend) but engine is at /app/engine
# We need to adjust path based on deployment
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).parent.parent.parent.parent))


def get_engine_path() -> Path:
    """Get the path to engine folder"""
    # Check if engine exists relative to backend
    engine_in_backend = PROJECT_ROOT / "engine"
    if engine_in_backend.exists():
        return engine_in_backend

    # Check parent directory (development mode)
    engine_in_parent = PROJECT_ROOT.parent / "engine"
    if engine_in_parent.exists():
        return engine_in_parent

    # Default to PROJECT_ROOT/engine
    return engine_in_backend


def run_engine_script(
    script_name: str,
    input_data: Dict[str, Any],
    job_id: str,
    env_vars: Optional[Dict[str, str]] = None,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Run an engine script as subprocess and handle output.

    Args:
        script_name: Name of the script (e.g., 'backtest_engine.py')
        input_data: JSON-serializable input to send via stdin
        job_id: Job ID for progress tracking
        env_vars: Additional environment variables
        on_progress: Optional callback for progress updates

    Returns:
        Final result dictionary

    Raises:
        TimeoutError: If job exceeds timeout
        RuntimeError: If job fails or output too large
        KeyboardInterrupt: If job is cancelled
    """
    engine_path = get_engine_path()
    script_path = engine_path / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Engine script not found: {script_path}")

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
    # Use -m to run as module for proper relative imports
    module_name = f"engine.{script_name.replace('.py', '')}"
    process = subprocess.Popen(
        [sys.executable, "-m", module_name],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(engine_path.parent)
    )

    # Send input
    input_json = json.dumps(input_data)
    process.stdin.write(input_json.encode())
    process.stdin.close()

    # Collect output
    raw_output = ""
    stderr_output = ""
    aborted = False
    final_result = None

    try:
        # Read stdout line by line with timeout handling
        import select

        while True:
            # Check for cancellation
            if check_cancel_flag(job_id):
                process.kill()
                publish_progress(job_id, {
                    "progress": 0,
                    "total": 0,
                    "status": "canceled",
                    "error": "Canceled by user"
                })
                raise KeyboardInterrupt("Job cancelled by user")

            # Check if process has finished
            poll = process.poll()
            if poll is not None:
                # Process finished, read remaining output
                remaining = process.stdout.read()
                if remaining:
                    raw_output += remaining.decode()
                break

            # Try to read with select (non-blocking with timeout)
            if sys.platform != 'win32':
                readable, _, _ = select.select([process.stdout], [], [], 1.0)
                if readable:
                    chunk = process.stdout.read(8192)
                    if chunk:
                        raw_output += chunk.decode()
            else:
                # Windows fallback - blocking read with small chunks
                chunk = process.stdout.read(8192)
                if chunk:
                    raw_output += chunk.decode()
                else:
                    break

            # Check output size limit
            if len(raw_output) > MAX_STDOUT_SIZE:
                aborted = True
                process.kill()
                publish_progress(job_id, {
                    "progress": 0,
                    "total": 0,
                    "status": "error",
                    "error": "Output too large (exceeded 20MB limit)"
                })
                raise RuntimeError("Output too large")

            # Parse progress lines
            lines = raw_output.split("\n")
            for line in lines[:-1]:  # Don't process incomplete last line
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        # Check if it's a progress update (has progress/total, no success key)
                        if "progress" in obj and "total" in obj and "success" not in obj:
                            progress_data = {
                                "progress": obj["progress"],
                                "total": obj["total"],
                                "status": "running"
                            }
                            publish_progress(job_id, progress_data)
                            if on_progress:
                                on_progress(progress_data)
                except json.JSONDecodeError:
                    pass  # Not JSON, ignore

        # Read stderr
        stderr_output = process.stderr.read().decode()

        # Check exit code
        if process.returncode != 0:
            error_msg = stderr_output or "Engine script failed"
            publish_progress(job_id, {
                "progress": 0,
                "total": 0,
                "status": "error",
                "error": error_msg[:1000]  # Limit error message size
            })
            raise RuntimeError(f"Engine failed: {error_msg}")

        # Parse final result from output
        lines = raw_output.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "success" in obj:
                    final_result = obj
                    break
                # For backtest_engine and chart_data, result doesn't have "success"
                # but has "summary" or "candles"
                if isinstance(obj, dict) and ("summary" in obj or "candles" in obj):
                    final_result = obj
                    break
            except json.JSONDecodeError:
                continue

        if final_result is None:
            # Try to parse the entire output as JSON
            try:
                final_result = json.loads(raw_output.strip())
            except json.JSONDecodeError:
                raise RuntimeError(f"No valid result JSON found. Output: {raw_output[-500:]}")

        # Store result and update progress
        set_result(job_id, final_result)
        total = final_result.get("total", 1)
        publish_progress(job_id, {
            "progress": total,
            "total": total,
            "status": "done"
        })

        return final_result

    except subprocess.TimeoutExpired:
        process.kill()
        publish_progress(job_id, {
            "progress": 0,
            "total": 0,
            "status": "error",
            "error": f"Job timed out after {JOB_TIMEOUT_SEC} seconds"
        })
        raise TimeoutError(f"Job timed out after {JOB_TIMEOUT_SEC} seconds")

    finally:
        # Ensure process is terminated
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()


def run_backtest(config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """
    Run backtest_engine.py with given config.

    Args:
        config: Strategy configuration
        job_id: Job ID for tracking

    Returns:
        Backtest result
    """
    return run_engine_script(
        script_name="backtest_engine.py",
        input_data=config,
        job_id=job_id
    )


def run_optimizer(config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """
    Run run_optimizer.py with given config.

    Args:
        config: Optimization configuration
        job_id: Job ID for tracking

    Returns:
        Optimization result
    """
    return run_engine_script(
        script_name="run_optimizer.py",
        input_data=config,
        job_id=job_id
    )


def run_ai_agent(config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """
    Run ai_agent.py with given config.
    Sets AI_PROGRESS=1 to enable progress output.

    Args:
        config: AI agent configuration
        job_id: Job ID for tracking

    Returns:
        AI agent result
    """
    return run_engine_script(
        script_name="ai_agent.py",
        input_data=config,
        job_id=job_id,
        env_vars={"AI_PROGRESS": "1"}
    )


def run_chart_data(config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """
    Run chart_data.py with given config.

    Args:
        config: Chart data configuration
        job_id: Job ID for tracking

    Returns:
        Chart data result
    """
    return run_engine_script(
        script_name="chart_data.py",
        input_data=config,
        job_id=job_id
    )
