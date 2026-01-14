"""
Parse Pine Script endpoint tests.
"""

import pytest


def test_parse_pine_success(client, sample_pine_code):
    """Test POST /parse-pine with valid Pine Script."""
    response = client.post(
        "/parse-pine",
        json={"pineCode": sample_pine_code}
    )
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "strategy" in data

    strategy = data["strategy"]
    assert "meta" in strategy
    assert "entry" in strategy
    assert "exit" in strategy


def test_parse_pine_empty_code(client):
    """Test POST /parse-pine with empty Pine Script."""
    response = client.post(
        "/parse-pine",
        json={"pineCode": ""}
    )
    # Empty string is still valid input, returns mock result
    # But explicitly empty might be rejected
    assert response.status_code in [200, 400]


def test_parse_pine_missing_code(client):
    """Test POST /parse-pine without pineCode field."""
    response = client.post(
        "/parse-pine",
        json={}
    )
    assert response.status_code == 422  # Validation error


def test_parse_pine_strategy_structure(client, sample_pine_code):
    """Test that parsed strategy has correct structure."""
    response = client.post(
        "/parse-pine",
        json={"pineCode": sample_pine_code}
    )
    data = response.json()
    strategy = data["strategy"]

    # Check meta structure
    assert "name" in strategy["meta"]
    assert "timeframe" in strategy["meta"]
    assert "symbols" in strategy["meta"]
    assert isinstance(strategy["meta"]["symbols"], list)

    # Check entry structure
    assert "long" in strategy["entry"]
    assert "short" in strategy["entry"]

    # Check exit structure
    assert "type" in strategy["exit"]
