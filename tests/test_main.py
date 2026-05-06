"""
AutoDrive Chatbot — Unit Tests (synchronous TestClient, no pytest-asyncio)
"""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from starlette.testclient import TestClient


# ── Mock helpers ─────────────────────────────────────────────────────

async def _mock_refresh():
    return 12


def make_mock_engine():
    engine = MagicMock()
    engine.retrieve_context = AsyncMock(
        return_value="[ID:2] 2023 Hyundai Creta | Price: ₹14.50 lakh"
    )
    async def mock_stream(*args, **kwargs):
        for token in ["The ", "Hyundai Creta ", "[CAR_ID:2]", " is great."]:
            yield token
    engine.stream_response = mock_stream
    return engine


# ── Fixture: patches stay active for entire test ─────────────────────

@pytest.fixture
def client():
    mock_engine = make_mock_engine()

    # start() keeps patches alive beyond this function's scope
    p1 = patch("src.main.force_inventory_refresh", _mock_refresh)
    p2 = patch("src.main._history_store", None)
    p1.start()
    p2.start()

    # Set engine directly so get_rag_engine() returns mock immediately
    import src.main as m
    original_engine = m._rag_engine
    m._rag_engine = mock_engine

    from src.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    # Teardown
    m._rag_engine = original_engine
    p1.stop()
    p2.stop()


# ── Tests ────────────────────────────────────────────────────────────

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_chat_empty_message(client):
    resp = client.post("/chat/stream", json={"session_id": "t", "message": " "})
    assert resp.status_code in (400, 422)


def test_chat_stream_returns_tokens(client):
    resp = client.post(
        "/chat/stream",
        json={"session_id": "s1", "message": "Show me SUVs under 20 lakh"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    events = [
        json.loads(line[6:])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    assert any(e.get("done") for e in events)
    tokens = "".join(e.get("token", "") for e in events)
    assert "Hyundai Creta" in tokens


def test_car_id_event_emitted(client):
    resp = client.post(
        "/chat/stream",
        json={"session_id": "s2", "message": "Tell me about the Creta"},
    )
    events = [
        json.loads(line[6:])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    car_events = [e for e in events if "car_id" in e]
    assert len(car_events) > 0
    assert car_events[0]["car_id"] == "2"


def test_inventory_refresh(client):
    resp = client.post("/inventory/refresh")
    assert resp.status_code == 200
    assert resp.json()["status"] == "refreshed"
