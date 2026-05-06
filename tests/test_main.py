"""
AutoDrive Chatbot — Unit Tests
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient, ASGITransport


# ── Shared async mock refresh ────────────────────────────────────────
async def _mock_refresh():
    return 12


# ── Mock RAG engine ──────────────────────────────────────────────────
def make_mock_engine():
    engine = MagicMock()
    engine.retrieve_context = AsyncMock(
        return_value="[ID:2] 2023 Hyundai Creta | Price: ₹14.50 lakh | Fuel: Diesel"
    )
    async def mock_stream(*args, **kwargs):
        for token in ["The ", "Hyundai Creta ", "[CAR_ID:2]", " is great."]:
            yield token
    engine.stream_response = mock_stream
    return engine


# ── Test client helper ───────────────────────────────────────────────
async def get_client():
    mock_engine = make_mock_engine()
    patches = [
        patch("src.main.force_inventory_refresh", _mock_refresh),
        patch("src.main._rag_engine", mock_engine),
        patch("src.main._history_store", None),
    ]
    for p in patches:
        p.start()

    import src.main as main
    main._rag_engine = mock_engine
    main._history_store = None

    from src.main import app
    return app, patches


# ── Tests ────────────────────────────────────────────────────────────

async def test_health():
    mock_engine = make_mock_engine()
    with patch("src.main.force_inventory_refresh", _mock_refresh), \
         patch("src.main._rag_engine", mock_engine), \
         patch("src.main._history_store", None):
        import src.main as m; m._rag_engine = mock_engine
        from src.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


async def test_chat_empty_message():
    mock_engine = make_mock_engine()
    with patch("src.main.force_inventory_refresh", _mock_refresh), \
         patch("src.main._rag_engine", mock_engine), \
         patch("src.main._history_store", None):
        import src.main as m; m._rag_engine = mock_engine
        from src.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/chat/stream", json={"session_id": "t", "message": ""})
    assert resp.status_code == 400


async def test_chat_stream_returns_tokens():
    mock_engine = make_mock_engine()
    with patch("src.main.force_inventory_refresh", _mock_refresh), \
         patch("src.main._rag_engine", mock_engine), \
         patch("src.main._history_store", None):
        import src.main as m; m._rag_engine = mock_engine
        from src.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/chat/stream",
                json={"session_id": "s1", "message": "Show me SUVs"},
            )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    events = [json.loads(l[6:]) for l in resp.text.splitlines() if l.startswith("data: ")]
    assert any(e.get("done") for e in events)
    tokens = "".join(e.get("token", "") for e in events)
    assert "Hyundai Creta" in tokens


async def test_car_id_event_emitted():
    mock_engine = make_mock_engine()
    with patch("src.main.force_inventory_refresh", _mock_refresh), \
         patch("src.main._rag_engine", mock_engine), \
         patch("src.main._history_store", None):
        import src.main as m; m._rag_engine = mock_engine
        from src.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/chat/stream",
                json={"session_id": "s2", "message": "Tell me about Creta"},
            )
    events = [json.loads(l[6:]) for l in resp.text.splitlines() if l.startswith("data: ")]
    car_events = [e for e in events if "car_id" in e]
    assert len(car_events) > 0
    assert car_events[0]["car_id"] == "2"


async def test_inventory_refresh():
    mock_engine = make_mock_engine()
    with patch("src.main.force_inventory_refresh", _mock_refresh), \
         patch("src.main._rag_engine", mock_engine), \
         patch("src.main._history_store", None):
        import src.main as m; m._rag_engine = mock_engine
        from src.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/inventory/refresh")
    assert resp.status_code == 200
    assert resp.json()["status"] == "refreshed"
