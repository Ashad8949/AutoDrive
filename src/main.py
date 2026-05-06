"""
AutoDrive Chatbot — FastAPI Application
Endpoints:
  POST /chat/stream   → SSE streaming chat (RAG + LLM)
  GET  /health        → Liveness probe
  GET  /ready         → Readiness probe
"""

from __future__ import annotations
import json
import os
import re
import uuid
import time
import logging
import contextvars
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import settings
from .rag import RAGEngine, force_inventory_refresh
from .history import get_history_store

# ── Structured logging ───────────────────────────────────────────────
# Each request gets a correlation ID injected into every log line.
# Azure Log Analytics can filter/group by correlation_id field.
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

class CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","request_id":"%(request_id)s","logger":"%(name)s","msg":"%(message)s"}',
)
for handler in logging.root.handlers:
    handler.addFilter(CorrelationFilter())

logger = logging.getLogger("chatbot")

# ── Background inventory refresh ────────────────────────────────────
import asyncio as _asyncio

async def _inventory_refresh_loop():
    """Refresh inventory every hour so new cars appear without restarting the app."""
    REFRESH_INTERVAL = int(os.getenv("INVENTORY_REFRESH_INTERVAL", "86400"))  # 24 hours
    while True:
        await _asyncio.sleep(REFRESH_INTERVAL)
        try:
            count = await force_inventory_refresh()
            logger.info("Scheduled inventory refresh: %d cars loaded", count)
        except Exception as exc:
            logger.warning("Scheduled inventory refresh failed: %s", exc)


# ── Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  AutoDrive Chatbot v1.0.0")
    logger.info(f"  LLM Provider : {settings.llm_provider}")
    logger.info(f"  Inventory    : {os.getenv('INVENTORY_API_URL', 'https://autodriveai.duckdns.org/api/cars')}")
    logger.info(f"  Mode         : {'AZURE' if settings.is_azure else 'LOCAL'}")
    logger.info(f"  History      : {'Redis' if settings.has_redis else 'In-Memory'}")
    logger.info(f"  Port         : {settings.PORT}")
    logger.info("=" * 60)
    # Warm up inventory cache at startup so first request is fast
    try:
        count = await force_inventory_refresh()
        logger.info("Inventory pre-loaded: %d cars", count)
    except Exception as exc:
        logger.warning("Inventory pre-load failed (will retry on first request): %s", exc)
    # Start hourly background refresh
    task = _asyncio.create_task(_inventory_refresh_loop())
    yield
    task.cancel()


# ── FastAPI App ─────────────────────────────────────────────────────
app = FastAPI(
    title="AutoDrive Chatbot",
    description="LLM-powered RAG chatbot for AutoDrive car dealership",
    version="1.0.0",
    lifespan=lifespan,
)

_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def request_telemetry(request: Request, call_next):
    """Attach correlation ID to every request; log method, path, status, latency."""
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
    token = _request_id.set(req_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000)
        logger.info(
            "%s %s → %d  (%dms)",
            request.method, request.url.path, response.status_code, latency_ms,
        )
        response.headers["X-Request-ID"] = req_id
        return response
    except Exception as exc:
        logger.error("Unhandled exception: %s", exc)
        raise
    finally:
        _request_id.reset(token)


# ── Request Model ──────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Session ID for conversation memory")
    message: str = Field(..., min_length=1, description="User message to the chatbot", json_schema_extra={"examples": ["Show me SUVs under $40k"]})

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"session_id": "test", "message": "Show me SUVs under $40k"}
            ]
        }
    }


# ── Lazy-initialized singletons ────────────────────────────────────
_rag_engine: RAGEngine | None = None
_history_store = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        logger.info(f"Initializing RAG engine (provider={settings.llm_provider})…")
        _rag_engine = RAGEngine()
        logger.info("RAG engine ready ✓")
    return _rag_engine


def get_history():
    global _history_store
    if _history_store is None:
        _history_store = get_history_store()
        kind = "Redis" if settings.has_redis else "In-Memory"
        logger.info(f"Chat history store: {kind} ✓")
    return _history_store


# ── UI Endpoint ─────────────────────────────────────────────────────
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import UploadFile, File

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def serve_ui():
    """Serves the Chat HTML widget."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Chat UI not found. Did you create static/index.html?"}


# ── Voice Transcription Endpoint ───────────────────────────────────
@app.post("/voice/transcribe")
async def voice_transcribe(audio: UploadFile = File(...)):
    """
    Accepts an audio file (webm/mp3/wav) and returns the transcript.
    Used by the voice assistant widget.
    """
    try:
        from .voice import transcribe_audio
        audio_bytes = await audio.read()
        transcript = transcribe_audio(audio_bytes, filename=audio.filename or "audio.webm")
        logger.info(f"Transcribed audio: {transcript[:80]}…")
        return {"transcript": transcript}
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Inventory Refresh Endpoint ──────────────────────────────────────
@app.post("/inventory/refresh")
async def inventory_refresh():
    """
    Force-refresh the car inventory from the live API.
    Useful during demos to ensure the latest cars are indexed.
    """
    try:
        count = await force_inventory_refresh()
        return {"status": "refreshed", "cars": count}
    except Exception as e:
        logger.error("Manual inventory refresh failed: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Health / Readiness Probes ───────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "slot": os.getenv("SLOT_NAME", "production"),
        "provider": settings.llm_provider,
        "version": "1.0.0",
    }


@app.get("/ready")
async def ready():
    try:
        # Verify RAG engine can be instantiated
        get_rag_engine()
        return {"status": "ready", "provider": settings.llm_provider}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "error": str(e)},
        )


# ── Chat Stream Endpoint ───────────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    """
    Streaming chat endpoint. Returns Server-Sent Events with tokens.
    """
    session_id = body.session_id
    user_msg = body.message

    # Detect explicit test drive request locally without LLM for instant response (optional trigger)
    book_trigger = "book a test drive" in user_msg.lower() or "schedule test drive" in user_msg.lower()

    if not user_msg.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Message cannot be empty."},
        )

    rag = get_rag_engine()
    history = get_history()

    async def generate():
        try:
            # 1. Retrieve relevant car context
            context = await rag.retrieve_context(user_msg)
            logger.info(
                f"[{session_id[:8]}] Retrieved context for: {user_msg[:50]}…"
            )

            # 2. Get conversation history
            chat_history = history.get_messages(session_id, last_n=10)

            # 3. Stream LLM response
            full_response = ""
            tag_buffer = ""
            in_tag = False

            async for token in rag.stream_response(user_msg, context, chat_history):
                # Buffer everything between [ and ] so we can intercept special tags.
                # Tags we strip from the visible stream:
                #   [ACTION: BOOK_TEST_DRIVE <id>]  → emits {"action": "..."}
                #   [CAR_ID:X]                       → emits {"car_id": "X"} for deep-links
                if "[" in token or in_tag:
                    in_tag = True
                    tag_buffer += token
                    if "]" in tag_buffer:
                        in_tag = False
                        m_action = re.search(r"\[ACTION:\s*(.*?)\]", tag_buffer)
                        m_car = re.search(r"\[CAR_ID:\s*(\w+)\]", tag_buffer)
                        if m_action:
                            yield f"data: {json.dumps({'action': m_action.group(1).strip()})}\n\n"
                        elif m_car:
                            yield f"data: {json.dumps({'car_id': m_car.group(1).strip()})}\n\n"
                        else:
                            # Not a special tag — send as plain text
                            yield f"data: {json.dumps({'token': tag_buffer})}\n\n"
                            full_response += tag_buffer
                        tag_buffer = ""
                else:
                    full_response += token
                    yield f"data: {json.dumps({'token': token})}\n\n"

            # 4. Save conversation turn
            history.add_user_message(session_id, user_msg)
            history.add_ai_message(session_id, full_response + tag_buffer)

            yield f"data: {json.dumps({'done': True})}\n\n"
            logger.info(
                f"[{session_id[:8]}] Response complete ({len(full_response)} chars)"
            )

        except Exception as e:
            logger.error(f"[{session_id[:8]}] Error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Non-streaming Chat (for simple testing) ─────────────────────────
@app.post("/chat")
async def chat(body: ChatRequest):
    """
    Non-streaming endpoint. Returns full response as JSON.
    """
    session_id = body.session_id
    user_msg = body.message

    rag = get_rag_engine()
    history = get_history()

    context = await rag.retrieve_context(user_msg)
    chat_history = history.get_messages(session_id, last_n=10)

    full_response = ""
    async for token in rag.stream_response(user_msg, context, chat_history):
        full_response += token

    history.add_user_message(session_id, user_msg)
    history.add_ai_message(session_id, full_response)

    return {
        "session_id": session_id,
        "response": full_response,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
