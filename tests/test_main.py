"""
AutoDrive Chatbot — Unit Tests
Tests the FastAPI endpoints with mocked LLM/RAG components.
"""

import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient, ASGITransport

# We need to mock the RAG engine BEFORE importing main
# to avoid triggering real LLM/embedding initialization


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Set minimal env vars for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.3")
    monkeypatch.setenv("RETRIEVER_K", "3")


@pytest.fixture
def mock_rag_engine():
    """Create a mock RAG engine that yields test tokens."""
    engine = MagicMock()
    engine.retrieve_context = AsyncMock(
        return_value="2024 Toyota Camry | Price: $28,500 | Fuel: Hybrid"
    )

    async def mock_stream(*args, **kwargs):
        for token in ["Hello", "! ", "I found a ", "Toyota Camry", " for you."]:
            yield token

    engine.stream_response = mock_stream
    return engine


@pytest_asyncio.fixture
async def client(mock_rag_engine):
    """Create test client with mocked RAG engine."""
    # Patch the global RAG engine and history before importing the app
    with patch("src.main._rag_engine", mock_rag_engine), \
         patch("src.main._history_store", None):
        from src.main import app

        # Force the mocked engine
        import src.main as main
        main._rag_engine = mock_rag_engine
        main._history_store = None  # Will use InMemory

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


# ── Tests ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    """Health endpoint should always return 200."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_chat_stream_returns_sse(client):
    """SSE endpoint should stream tokens."""
    resp = await client.post(
        "/chat/stream",
        json={"session_id": "test-session", "message": "Show me hybrid cars"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    # Parse SSE events
    text = resp.text
    events = [
        line.replace("data: ", "")
        for line in text.strip().split("\n")
        if line.startswith("data: ")
    ]
    assert len(events) > 0

    # Last event should be done=True
    last = json.loads(events[-1])
    assert last.get("done") is True

    # Other events should have tokens
    tokens = [json.loads(e).get("token", "") for e in events[:-1]]
    full = "".join(tokens)
    assert "Toyota Camry" in full


@pytest.mark.asyncio
async def test_chat_empty_message(client):
    """Empty message should return 400."""
    resp = await client.post(
        "/chat/stream",
        json={"session_id": "test", "message": ""},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_chat_non_streaming(client):
    """Non-streaming /chat endpoint should return full response."""
    resp = await client.post(
        "/chat",
        json={"session_id": "test-session-2", "message": "Tell me about SUVs"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert "session_id" in data
    assert len(data["response"]) > 0
