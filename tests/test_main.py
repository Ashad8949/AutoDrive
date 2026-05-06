"""
AutoDrive Chatbot — Unit Tests
Tests the FastAPI endpoints with mocked LLM/RAG components.
"""

import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient, ASGITransport


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Set minimal env vars so no real API keys are needed."""
    monkeypatch.setenv("GROQ_API_KEY", "sk-test-fake-groq-key")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.3")
    monkeypatch.setenv("RETRIEVER_K", "3")


@pytest.fixture
def mock_rag_engine():
    """Mock RAG engine — no real LLM or inventory calls."""
    engine = MagicMock()
    engine.retrieve_context = AsyncMock(
        return_value="[ID:2] 2023 Hyundai Creta | Price: ₹14.50 lakh | Fuel: Diesel"
    )

    async def mock_stream(*args, **kwargs):
        for token in ["The ", "Hyundai Creta ", "[CAR_ID:2]", " is a great choice."]:
            yield token

    engine.stream_response = mock_stream
    return engine


@pytest_asyncio.fixture
async def client(mock_rag_engine):
    """Test client with all external calls mocked."""
    # Mock inventory refresh so tests never hit the live API
    async def mock_refresh():
        return 12

    with patch("src.main.force_inventory_refresh", mock_refresh), \
         patch("src.main._rag_engine", mock_rag_engine), \
         patch("src.main._history_store", None):

        from src.main import app
        import src.main as main
        main._rag_engine = mock_rag_engine
        main._history_store = None

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


# ── Tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_chat_stream_returns_sse(client):
    resp = await client.post(
        "/chat/stream",
        json={"session_id": "test-session", "message": "Show me SUVs under 20 lakh"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    events = [
        line.replace("data: ", "")
        for line in resp.text.strip().split("\n")
        if line.startswith("data: ")
    ]
    assert len(events) > 0
    assert json.loads(events[-1]).get("done") is True

    tokens = [json.loads(e).get("token", "") for e in events[:-1]]
    assert "Hyundai Creta" in "".join(tokens)


@pytest.mark.asyncio
async def test_car_id_event_emitted(client):
    """[CAR_ID:X] tags should be stripped from tokens and emitted as car_id events."""
    resp = await client.post(
        "/chat/stream",
        json={"session_id": "test-car-id", "message": "Tell me about the Creta"},
    )
    events = [json.loads(l.replace("data: ", ""))
              for l in resp.text.strip().split("\n")
              if l.startswith("data: ")]
    car_id_events = [e for e in events if "car_id" in e]
    assert len(car_id_events) > 0
    assert car_id_events[0]["car_id"] == "2"


@pytest.mark.asyncio
async def test_chat_empty_message(client):
    resp = await client.post(
        "/chat/stream",
        json={"session_id": "test", "message": ""},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_inventory_refresh_endpoint(client):
    resp = await client.post("/inventory/refresh")
    assert resp.status_code == 200
    assert resp.json()["status"] == "refreshed"


@pytest.mark.asyncio
async def test_chat_non_streaming(client):
    resp = await client.post(
        "/chat",
        json={"session_id": "test-session-2", "message": "Tell me about SUVs"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert "session_id" in data
    assert len(data["response"]) > 0
