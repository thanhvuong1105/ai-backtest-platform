"""
Health check endpoint
GET / -> status ok JSON
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def health_check():
    """Health check endpoint - matches original server.js behavior"""
    return {
        "status": "ok",
        "message": "AI Backtest Platform API is running"
    }
