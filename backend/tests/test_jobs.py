"""
Job management endpoint tests.

Note: These tests mock Celery tasks and Redis to avoid requiring external services.
For full integration tests, run with docker-compose up.
"""

import pytest
from unittest.mock import patch, MagicMock


def test_ai_agent_validation_symbols(client):
    """Test POST /ai-agent validates symbols field."""
    response = client.post(
        "/ai-agent",
        json={
            "symbols": [],  # Empty array
            "timeframes": ["1h"],
            "strategy": {
                "type": "ema_cross",
                "params": {
                    "emaFast": [12],
                    "emaSlow": [26]
                }
            }
        }
    )
    assert response.status_code == 400
    assert "symbols" in response.json()["detail"].lower()


def test_ai_agent_validation_timeframes(client):
    """Test POST /ai-agent validates timeframes field."""
    response = client.post(
        "/ai-agent",
        json={
            "symbols": ["BTCUSDT"],
            "timeframes": [],  # Empty array
            "strategy": {
                "type": "ema_cross",
                "params": {
                    "emaFast": [12],
                    "emaSlow": [26]
                }
            }
        }
    )
    assert response.status_code == 400
    assert "timeframes" in response.json()["detail"].lower()


def test_ai_agent_validation_strategy_type(client):
    """Test POST /ai-agent validates strategy.type field."""
    response = client.post(
        "/ai-agent",
        json={
            "symbols": ["BTCUSDT"],
            "timeframes": ["1h"],
            "strategy": {
                "type": "",  # Empty type
                "params": {
                    "emaFast": [12],
                    "emaSlow": [26]
                }
            }
        }
    )
    assert response.status_code == 400
    assert "strategy.type" in response.json()["detail"]


def test_ai_agent_validation_unknown_strategy(client):
    """Test POST /ai-agent rejects unknown strategy type."""
    response = client.post(
        "/ai-agent",
        json={
            "symbols": ["BTCUSDT"],
            "timeframes": ["1h"],
            "strategy": {
                "type": "unknown_strategy",
                "params": {
                    "someParam": [1, 2, 3]
                }
            }
        }
    )
    assert response.status_code == 400
    assert "Unknown strategy type" in response.json()["detail"]


def test_ai_agent_validation_ema_params(client):
    """Test POST /ai-agent validates ema_cross params."""
    response = client.post(
        "/ai-agent",
        json={
            "symbols": ["BTCUSDT"],
            "timeframes": ["1h"],
            "strategy": {
                "type": "ema_cross",
                "params": {
                    "emaFast": "not_array",  # Should be array
                    "emaSlow": [26]
                }
            }
        }
    )
    assert response.status_code == 400


@patch("app.services.tasks.ai_agent_task.delay")
@patch("app.services.progress_store.set_progress")
def test_ai_agent_enqueue_success(mock_set_progress, mock_task, client, sample_optimize_config):
    """Test POST /ai-agent successfully enqueues job."""
    mock_task.return_value = MagicMock()

    response = client.post("/ai-agent", json=sample_optimize_config)

    assert response.status_code == 200
    data = response.json()
    assert "jobId" in data
    assert data["jobId"].startswith("job_")

    # Verify task was called
    mock_task.assert_called_once()


@patch("app.services.progress_store.get_progress")
def test_progress_not_found(mock_get_progress, client):
    """Test GET /ai-agent/progress/:jobId with non-existent job."""
    mock_get_progress.return_value = None

    response = client.get("/ai-agent/progress/nonexistent_job_123")
    assert response.status_code == 404


@patch("app.services.progress_store.get_progress")
def test_progress_running(mock_get_progress, client):
    """Test GET /ai-agent/progress/:jobId with running job."""
    mock_get_progress.return_value = {
        "progress": 50,
        "total": 100,
        "status": "running"
    }

    response = client.get("/ai-agent/progress/job_123")
    assert response.status_code == 200

    data = response.json()
    assert data["progress"] == 50
    assert data["total"] == 100
    assert data["status"] == "running"


@patch("app.services.progress_store.get_progress")
@patch("app.services.progress_store.get_result")
def test_result_not_ready(mock_get_result, mock_get_progress, client):
    """Test GET /ai-agent/result/:jobId when job not done."""
    mock_get_progress.return_value = {
        "progress": 50,
        "total": 100,
        "status": "running"
    }

    response = client.get("/ai-agent/result/job_123")
    assert response.status_code == 202
    assert response.json()["status"] == "running"


@patch("app.services.progress_store.get_progress")
@patch("app.services.progress_store.get_result")
def test_result_ready(mock_get_result, mock_get_progress, client):
    """Test GET /ai-agent/result/:jobId when job is done."""
    mock_get_progress.return_value = {
        "progress": 100,
        "total": 100,
        "status": "done"
    }
    mock_get_result.return_value = {
        "success": True,
        "best": {"score": 1.5},
        "total": 100
    }

    response = client.get("/ai-agent/result/job_123")
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "best" in data


@patch("app.services.progress_store.set_cancel_flag")
@patch("app.services.progress_store.set_progress")
def test_cancel_job(mock_set_progress, mock_cancel_flag, client):
    """Test POST /ai-agent/cancel/:jobId."""
    response = client.post("/ai-agent/cancel/job_123")

    assert response.status_code == 200
    assert response.json()["success"] is True

    mock_cancel_flag.assert_called_once_with("job_123")


def test_optimize_validation(client):
    """Test POST /optimize validates config same as ai-agent."""
    response = client.post(
        "/optimize",
        json={
            "symbols": [],  # Invalid
            "timeframes": ["1h"],
            "strategy": {
                "type": "ema_cross",
                "params": {"emaFast": [12], "emaSlow": [26]}
            }
        }
    )
    assert response.status_code == 400
