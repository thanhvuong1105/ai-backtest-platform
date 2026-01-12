"""
Health check endpoint tests.
"""

import pytest


def test_health_check(client):
    """Test GET / returns status ok."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "message" in data


def test_health_check_message(client):
    """Test health check returns correct message."""
    response = client.get("/")
    data = response.json()

    assert "AI Backtest Platform API" in data["message"]
