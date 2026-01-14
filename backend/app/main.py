"""
FastAPI Main Application
AI Backtest Platform - Backend API

Replaces the Node.js Express server with Python FastAPI.
Provides the same API contract for frontend compatibility.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.api.health import router as health_router
from app.api.parse_pine import router as parse_pine_router
from app.api.jobs import router as jobs_router
from app.api.chart import router as chart_router

# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("Starting AI Backtest Platform API...")
    yield
    # Shutdown
    print("Shutting down AI Backtest Platform API...")


# Create FastAPI app
app = FastAPI(
    title="AI Backtest Platform API",
    description="Backend API for AI-powered trading strategy backtesting and optimization",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration - allow all origins for development
# In production, you should restrict this to your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(parse_pine_router)
app.include_router(jobs_router)
app.include_router(chart_router)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("FASTAPI_HOST", "0.0.0.0")
    port = int(os.getenv("FASTAPI_PORT", 8000))

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True
    )
