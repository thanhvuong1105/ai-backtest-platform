"""
Chart Data API Endpoint

POST /chart-data - Generate chart data for debug/verify view
"""

import os
import json
import sys
import subprocess
from typing import Dict, Any, Optional, List
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["chart"])


class StrategyConfig(BaseModel):
    """Strategy configuration for chart data"""
    type: str
    params: Dict[str, Any]


class CapitalConfig(BaseModel):
    """Capital configuration"""
    initial: float = 10000
    orderPct: float = 100


class RiskConfig(BaseModel):
    """Risk configuration"""
    pyramiding: int = 1
    commission: float = 0.04
    slippage: float = 0.0


class DateRange(BaseModel):
    """Date range configuration"""
    start: Optional[str] = None
    end: Optional[str] = None


class ChartDataRequest(BaseModel):
    """Request body for /chart-data"""
    symbol: str
    timeframe: str
    strategy: StrategyConfig
    capital: Optional[CapitalConfig] = None
    risk: Optional[RiskConfig] = None
    range: Optional[Dict[str, str]] = None
    dateRange: Optional[DateRange] = None


def get_engine_path() -> Path:
    """Get the path to engine folder"""
    project_root = Path(os.getenv("PROJECT_ROOT", "/app"))

    # In Docker, engine is at /app/engine
    engine_path = project_root / "engine"
    if engine_path.exists():
        return engine_path

    # Development fallback
    backend_path = Path(__file__).parent.parent.parent
    engine_path = backend_path.parent / "engine"
    if engine_path.exists():
        return engine_path

    raise FileNotFoundError("Engine folder not found")


@router.post("/chart-data")
async def get_chart_data(request: ChartDataRequest):
    """
    Generate chart data for visualization.

    Request body:
    - symbol: Trading pair (e.g., "BTCUSDT")
    - timeframe: Candle timeframe (e.g., "1h")
    - strategy: Strategy configuration
    - capital: Capital settings (optional)
    - risk: Risk settings (optional)
    - range/dateRange: Date range filter (optional)

    Returns:
    - candles: OHLCV data with UTC+7 timestamps
    - indicators: Indicator values
    - markers: Entry/exit markers
    - trades: Trade list
    - summary: Summary statistics
    """
    # Validate required fields
    if not request.symbol:
        raise HTTPException(status_code=400, detail="symbol is required")
    if not request.timeframe:
        raise HTTPException(status_code=400, detail="timeframe is required")
    if not request.strategy or not request.strategy.type:
        raise HTTPException(status_code=400, detail="strategy.type is required")

    # Build config for engine
    config = {
        "symbol": request.symbol,
        "timeframe": request.timeframe,
        "strategy": {
            "type": request.strategy.type,
            "params": request.strategy.params
        },
        "capital": {
            "initial": request.capital.initial if request.capital else 10000,
            "orderPct": request.capital.orderPct if request.capital else 100
        },
        "risk": {
            "pyramiding": request.risk.pyramiding if request.risk else 1,
            "commission": request.risk.commission if request.risk else 0.04,
            "slippage": request.risk.slippage if request.risk else 0.0
        }
    }

    # Handle date range - support both 'range' and 'dateRange'
    if request.range:
        config["range"] = request.range
    elif request.dateRange:
        config["range"] = {
            "from": request.dateRange.start,
            "to": request.dateRange.end
        }

    # Run chart_data.py
    try:
        engine_path = get_engine_path()
        script_path = engine_path / "chart_data.py"

        if not script_path.exists():
            raise HTTPException(status_code=500, detail="chart_data.py not found")

        # Prepare environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(engine_path.parent)

        # Run subprocess
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(engine_path.parent)
        )

        # Send input and get output
        stdout, stderr = process.communicate(
            input=json.dumps(config).encode(),
            timeout=600  # 10 minute timeout for large datasets like 30m
        )

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Chart data generation failed"
            raise HTTPException(status_code=500, detail=error_msg[:500])

        # Parse result
        try:
            result = json.loads(stdout.decode())
            return result
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid JSON from chart_data.py: {stdout.decode()[:200]}"
            )

    except subprocess.TimeoutExpired:
        process.kill()
        raise HTTPException(status_code=500, detail="Chart data generation timed out")
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
